"""Bayesian hyper-parameter optimisation for the MLP-Tabular surrogate.

This script wraps the Ax `optimize` helper so it can be run entirely from the
command-line -- mirroring the style of `mlp_tabular_only.py`. All static
training options live in :class:`TrainArgs` (imported from
`engiopt.mlp_tabular_only`), while the search-space and optimisation
parameters live in :class:`OptArgs` below.

Example:
-------
>>> python -m engiopt.surrogate_model.bayes_optimize \
        --huggingface_repo IDEALLab/power_electronics_v0 \
        --target_col Voltage_Ripple \
        --total_trials 100 \
        --device mps
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
from typing import Any, Literal, TYPE_CHECKING

from ax import optimize
import tyro

from engiopt.surrogate_model.mlp_tabular_only import Args as TrainArgs
from engiopt.surrogate_model.mlp_tabular_only import main as train_main

if TYPE_CHECKING:
    from pathlib import Path

# -----------------------------------------------------------------------------
# Search-space & optimisation definition
# -----------------------------------------------------------------------------


@dataclass
class OptArgs:
    """CLI arguments for Bayesian optimisation.

    Most fields correspond 1-for-1 to :class:`engiopt.mlp_tabular_only.Args` and
    simply *override* its defaults. Anything not listed here will stay exactly
    the same as in `mlp_tabular_only`.
    """

    # ---------------- DATA / PROBLEM -----------------
    problem_id: str = "power_electronics"
    target_col: str = "Voltage_Ripple"
    params_cols: list[str] = field(
        default_factory=lambda: [
            *(f"initial_design_{i}" for i in range(10)),
        ]
    )
    flatten_columns: list[str] = field(default_factory=lambda: ["initial_design"])
    strip_column_spaces: bool = True

    # ---------------- TRAINING CONSTANTS -------------
    n_epochs: int = 50
    patience: int = 40
    n_ensembles: int = 1
    seed: int = 18
    track: bool = True
    wandb_project: str = "engiopt"
    wandb_entity: str = "engibench"
    log_target: bool = True
    scale_target: bool = True
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    test_model: bool = True
    save_model: bool = True
    model_output_dir: str = "my_models"

    # ---------------- AX OPTIMISER -------------------
    total_trials: int = 50
    minimise: bool = True

    # Search-space - overridable so you do *not* have to touch code to tinker.
    learning_rate_bounds: tuple[float, float] = (1e-5, 1e-3)
    hidden_layers_choices: list[int] = field(default_factory=lambda: [2, 3, 4, 5])
    hidden_size_choices: list[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])
    batch_size_choices: list[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    l2_lambda_bounds: tuple[float, float] = (1e-6, 1e-3)
    activation_choices: list[str] = field(default_factory=lambda: ["relu", "tanh"])

    # ---------------- HOUSEKEEPING -------------------
    results_path: Path | None = None  # If set, dump best-config JSON here.


# -----------------------------------------------------------------------------
# Helper - wrap training so it can be called by Ax
# -----------------------------------------------------------------------------


def _train_and_eval(hparams: dict[str, Any], fixed: OptArgs) -> float:
    """Instantiate :class:`TrainArgs` from *fixed* values and `hparams`."""
    train_args = TrainArgs(
        # Static values - pulled from the user-supplied OptArgs ----------------
        problem_id=fixed.problem_id,
        target_col=fixed.target_col,
        log_target=fixed.log_target,
        params_cols=fixed.params_cols,
        flatten_columns=fixed.flatten_columns,
        strip_column_spaces=fixed.strip_column_spaces,
        n_epochs=fixed.n_epochs,
        patience=fixed.patience,
        n_ensembles=fixed.n_ensembles,
        seed=fixed.seed,
        scale_target=fixed.scale_target,
        track=fixed.track,
        wandb_project=fixed.wandb_project,
        wandb_entity=fixed.wandb_entity,
        save_model=fixed.save_model,
        model_output_dir=fixed.model_output_dir,
        test_model=fixed.test_model,
        device=fixed.device,
        # --------------------------------------------------------------------
        # The actual hyper-parameters under optimisation ----------------------
        learning_rate=float(hparams["learning_rate"]),
        hidden_layers=int(hparams["hidden_layers"]),
        hidden_size=int(hparams["hidden_size"]),
        batch_size=int(hparams["batch_size"]),
        l2_lambda=float(hparams["l2_lambda"]),
        activation=str(hparams["activation"]),
    )

    # Delegate to the regular training routine - returns best *val* loss.
    best_val = train_main(train_args)
    return float(best_val)


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------


def optimise(opt_args: OptArgs) -> None:
    """Run Bayesian optimisation and optionally dump best config to disk."""
    # Translate OptArgs search-space definition to Ax format ----------------
    space = [
        {
            "name": "learning_rate",
            "type": "range",
            "bounds": list(opt_args.learning_rate_bounds),
            "log_scale": True,
        },
        {
            "name": "hidden_layers",
            "type": "choice",
            "values": opt_args.hidden_layers_choices,
        },
        {
            "name": "hidden_size",
            "type": "choice",
            "values": opt_args.hidden_size_choices,
        },
        {
            "name": "batch_size",
            "type": "choice",
            "values": opt_args.batch_size_choices,
        },
        {
            "name": "l2_lambda",
            "type": "range",
            "bounds": list(opt_args.l2_lambda_bounds),
            "log_scale": True,
        },
        {
            "name": "activation",
            "type": "choice",
            "value_type": "str",
            "values": opt_args.activation_choices,
        },
    ]

    best_params, best_vals, experiment, _ = optimize(
        parameters=space,
        evaluation_function=lambda hp: _train_and_eval(hp, opt_args),
        minimize=opt_args.minimise,
        total_trials=opt_args.total_trials,
    )

    print("\n=== BEST RESULTS ===")
    print("Parameters :", best_params)
    print("Metric val :", best_vals)

    if opt_args.results_path is not None:
        opt_args.results_path.parent.mkdir(parents=True, exist_ok=True)
        with opt_args.results_path.open("w", encoding="utf-8") as fp:
            json.dump({"best_parameters": best_params, "best_values": best_vals}, fp, indent=2)
        print(f"[INFO] Wrote best configuration to {opt_args.results_path}")


# -----------------------------------------------------------------------------
# CLI entry-point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    cli_args = tyro.cli(OptArgs)
    optimise(cli_args)
