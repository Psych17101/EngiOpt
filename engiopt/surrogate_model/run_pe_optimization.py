# ruff: noqa: TRY003,TRY301,PLR0913,PLR0915
"""run_pe_optimization.py.

Command-line script to run the multi-objective power-electronics design
optimization using surrogate pipelines together with pymoo.

The script mirrors the style of your other utilities (dataclass + tyro CLI)
and logs every generation to Weights & Biases so you can explore:
  • an interactive scatter of the Pareto front with a generation slider
  • a parallel-coordinates plot of the decision variables per generation

Example usage
-------------
python run_pe_optimization.py \
    --model_gain_path "my_entity/engiopt/your_run_name_model:latest" \
    --model_ripple_path "my_entity/engiopt/your_other_run_name_model:latest" \
    --device mps \
    --pop_size 500 \
    --n_gen 100 \
    --seed 1 \
    --track \
    --wandb_project engibench
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
import time
from typing import Literal, TYPE_CHECKING

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import torch
import tyro

import wandb

# ---------------------------------------------------------------------------
#  Local imports - ensure that the pickled ModelPipeline can be un-pickled
#  no matter whether it was saved from the top-level model_pipeline module
#  or from the package path engiopt.model_pipeline.
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(REPO_ROOT)  # favour local checkout

import model_pipeline as _engiopt_model_pipeline  # noqa: E402
from pymoo_pe_problem import MyPowerElecProblem  # noqa: E402

# Make "model_pipeline" & "pymoo_pe_problem" aliases so that pickle finds them
sys.modules.setdefault("model_pipeline", _engiopt_model_pipeline)
sys.modules.setdefault("pymoo_pe_problem", sys.modules["pymoo_pe_problem"])

from model_pipeline import ModelPipeline  # noqa: E402  (after alias)

if TYPE_CHECKING:
    from pymoo.core.algorithm import Algorithm
    from pymoo.core.result import Result


# ---------------------------------------------------------------------------
#  CLI arguments
# ---------------------------------------------------------------------------
@dataclass
class Args:
    # Surrogate pipelines
    model_gain_path: str = "my_entity/engiopt/your_run_name_model:latest"
    model_ripple_path: str = "my_entity/engiopt/your_other_run_name_model:latest"

    # Optimisation hyperparameters
    pop_size: int = 500
    n_gen: int = 100
    seed: int = 1

    # Hardware
    device: Literal["cpu", "cuda", "mps"] = "cpu"

    # Logging / I/O
    track: bool = True
    wandb_project: str = "engiopt"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    output_dir: str = "results"
    save_csv: bool = True
    log_every: int = 1  # gens between logs
    algo: str = "moo_ga"
    problem_id: str = "power_electronics"


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def get_device(name: str) -> torch.device:
    """Return the appropriate torch.device for the given name."""
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Invalid or unavailable device '{name}'")


# -------------------------   W&B callback  ------------------------------
class WandbLogCallback:
    """Log pareto history table and slider-ready scatter each generation."""

    def __init__(self, log_every: int, *, track: bool) -> None:
        self.log_every = max(1, log_every)
        self.track = track
        self.history: wandb.Table | None = None
        self.columns: list[str] | None = None

    def _init_table(self, n_obj: int, n_var: int) -> None:
        # Columns: generation, f0..f{n_obj-1}, x0..x{n_var-1}
        self.columns = ["generation"] + [f"f{i}" for i in range(n_obj)] + [f"x{i}" for i in range(n_var)]
        self.history = wandb.Table(columns=self.columns)

    def __call__(self, algorithm: Algorithm) -> None:
        """Callback invoked by pymoo to log data each generation."""
        if not self.track:
            return
        gen = algorithm.n_gen
        # guard against Pyright thinking this might be Optional[int]
        if gen is None or gen % self.log_every != 0:
            return

        # Classic SolutionSet API
        sol_set = algorithm.opt
        f_vals = sol_set.get("F")  # returns an (n_points, n_obj) array
        x_vals = sol_set.get("X")  # returns an (n_points, n_var) array

        # Initialize table columns once
        if self.columns is None:
            self._init_table(f_vals.shape[1], x_vals.shape[1])

        # Prepare data for current generation
        data = [[gen, *f.tolist(), *x.tolist()] for f, x in zip(f_vals, x_vals)]

        # Create a table and scatter plot
        current_table = wandb.Table(columns=self.columns, data=data)
        scatter = wandb.plot.scatter(
            current_table,
            x="f0",
            y="f1",
            title="Pareto Front",
        )

        # Log metrics and visualizations
        wandb.log(
            {
                "generation": gen,
                "pareto_front": scatter,
                "min_f1": f_vals[:, 0].min(),
                "min_f2": f_vals[:, 1].min(),
            },
            step=gen,
        )


# -------------------------   CSV & TXT persistence  -----------------------------
def save_front(res: Result, output_dir: str) -> tuple[str, str, str, str, str]:
    """Save Pareto front objectives and decision variables to CSV and TXT, and return file paths."""
    os.makedirs(output_dir, exist_ok=True)

    # F csv
    f_csv = os.path.join(output_dir, "pareto_F.csv")
    np.savetxt(
        f_csv,
        res.F,
        delimiter=",",
        header="Objective_r,Objective_abs_g_minus_0.25",
        comments="",
    )

    # F txt
    f_txt = os.path.join(output_dir, "pareto_F.txt")
    np.savetxt(f_txt, res.F, fmt="%.6e", delimiter=",")

    # X csv
    x_csv = os.path.join(output_dir, "pareto_X.csv")
    cols = [f"x{i}" for i in range(res.X.shape[1])]
    pd.DataFrame(res.X, columns=cols).to_csv(x_csv, index=False)

    # X txt
    x_txt = os.path.join(output_dir, "pareto_X.txt")
    np.savetxt(x_txt, res.X, fmt="%.6e", delimiter=",")

    # combined csv
    front_csv = os.path.join(output_dir, "pareto_front.csv")
    df = pd.read_csv(x_csv)
    df[["f0", "f1"]] = res.F
    df.to_csv(front_csv, index=False)

    return f_csv, x_csv, front_csv, f_txt, x_txt


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main(args: Args) -> None:
    """Parse arguments, run the optimization, and handle logging and persistence."""
    device = get_device(args.device)
    print(f"[INFO] Device: {device}")

    run_name = args.wandb_run_name or f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            save_code=True,
        )
        wandb.define_metric("generation")
        wandb.define_metric("*", step_metric="generation")

    # load models from weights and biases
    assert wandb.run is not None, f"W&B run not found for run_name={run_name} in {args.wandb_entity}/{args.wandb_project}"
    gain_art = wandb.run.use_artifact(args.model_gain_path, type="model")
    gain_dir = gain_art.download()
    # find the .pkl inside that dir (we assume there's exactly one)
    gain_file = next(f for f in os.listdir(gain_dir) if f.endswith(".pkl"))
    pipeline_g = ModelPipeline.load(os.path.join(gain_dir, gain_file))

    ripple_art = wandb.run.use_artifact(args.model_ripple_path, type="model")
    ripple_dir = ripple_art.download()
    ripple_file = next(f for f in os.listdir(ripple_dir) if f.endswith(".pkl"))
    pipeline_r = ModelPipeline.load(os.path.join(ripple_dir, ripple_file))

    problem = MyPowerElecProblem(
        pipeline_r=pipeline_r,
        pipeline_g=pipeline_g,
        device=device,
    )
    algo = NSGA2(pop_size=args.pop_size)
    term = get_termination("n_gen", args.n_gen)

    res = minimize(
        problem,
        algo,
        term,
        seed=args.seed,
        verbose=True,
        callback=WandbLogCallback(args.log_every, track=args.track),
    )

    if args.save_csv:
        _, _, _, f_txt, x_txt = save_front(res, args.output_dir)
        print(f"[INFO] Saved CSV and TXT to {args.output_dir}")
        if args.track:
            # Only upload the TXT versions as a W&B artifact
            txt_art = wandb.Artifact(f"{run_name}_pareto_txt", type="pymoo_results")
            txt_art.add_file(f_txt)
            txt_art.add_file(x_txt)
            wandb.log_artifact(txt_art)

    if args.track:
        wandb.finish()


if __name__ == "__main__":
    try:
        main(tyro.cli(Args))
    except Exception:
        print("[ERROR] run_pe_optimization.py failed")
        raise
