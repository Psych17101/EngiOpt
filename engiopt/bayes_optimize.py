# ruff: noqa: D205

"""This script performs Bayesian hyperparameter optimization for an MLP model.

It uses the Ax library to optimize a few hyperparameters while reusing the configuration
defaults provided by mlp_tabular_only.py via its Args dataclass.

Usage:
    python bayes_optimize.py --total_trials 10 [additional Tyro args for training]
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

from ax import optimize
import tyro

# Ensure the current directory is in the path (if needed for local imports).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Patch legacy module names so that imports in mlp_tabular_only work correctly.
from engiopt.mlp_tabular_only import Args
from engiopt.mlp_tabular_only import main
import engiopt.model_pipeline
import engiopt.surrogate_utils
import engiopt.training_utils
import engiopt.vae_model  # noqa: F401


def parse_args() -> tuple[argparse.Namespace, Args]:
    """Parse command-line arguments for both optimization (total_trials) and training parameters (via Tyro).

    Returns:
        A tuple containing:
            - The optimization arguments parsed by argparse.
            - The training configuration parsed by Tyro into an instance of Args.
    """
    parser = argparse.ArgumentParser(description="Bayesian Optimization for MLP Model Hyperparameters")
    parser.add_argument(
        "--total_trials",
        type=int,
        default=50,
        help="Total number of Bayesian optimization trials to run.",
    )

    # Separate the known optimization args from any extra args for Tyro.
    opt_args, remaining_args = parser.parse_known_args()

    # Tyro will parse training args from the remaining (unknown) arguments.
    base_args = tyro.cli(Args, args=remaining_args)
    return opt_args, base_args


OPT_ARGS, BASE_ARGS = parse_args()


def train_and_evaluate_model(hyperparams: dict[str, float | int | str]) -> float:
    """Create a new configuration from BASE_ARGS, update it with hyperparameters from Ax,
    run training via main(args), and return the best validation loss.

    Args:
        hyperparams: A dictionary of hyperparameter values provided by Ax.

    Returns:
        The best validation loss (float) for the given hyperparameter combination.
    """
    try:
        print(f"\nStarting trial with Ax hyperparameters: {hyperparams}")
        # Use a deep copy so each trial starts with the same base configuration.
        args = copy.deepcopy(BASE_ARGS)

        # Update only the fields that Ax is optimizing.
        args.learning_rate = hyperparams["learning_rate"]
        args.hidden_layers = int(hyperparams["hidden_layers"])
        args.hidden_size = int(hyperparams["hidden_size"])
        args.batch_size = int(hyperparams["batch_size"])
        args.l2_lambda = hyperparams["l2_lambda"]
        args.activation = hyperparams["activation"]
    except Exception as exc:  # noqa: BLE001
        print(f"Error in trial: {exc}")
        return float("inf")
    else:
        best_val_loss = main(args)
        print(f"Trial completed. Best validation loss: {best_val_loss}")
        return best_val_loss


def main_script() -> None:
    """The main entry point for running Ax optimization. Pulls hyperparameter ranges,
    calls Ax's optimize() using train_and_evaluate_model, and prints the best result.
    """
    best_parameters, best_values, experiment, model = optimize(
        parameters=[
            {
                "name": "learning_rate",
                "type": "range",
                "bounds": [1e-5, 1e-3],
                "log_scale": True,
            },
            {
                "name": "hidden_layers",
                "type": "choice",
                "values": [2, 3, 4, 5],
            },
            {
                "name": "hidden_size",
                "type": "choice",
                "values": [16, 32, 64, 128, 256],
            },
            {
                "name": "batch_size",
                "type": "choice",
                "values": [8, 16, 32, 64, 128],
            },
            {
                "name": "l2_lambda",
                "type": "range",
                "bounds": [1e-6, 1e-3],
                "log_scale": True,
            },
            {
                "name": "activation",
                "type": "choice",
                "value_type": "str",
                "values": ["relu", "tanh"],
            },
        ],
        evaluation_function=train_and_evaluate_model,
        minimize=True,
        total_trials=OPT_ARGS.total_trials,
    )

    print("Best Parameters:", best_parameters)
    print("Best Validation Loss:", best_values)


if __name__ == "__main__":
    main_script()
