"""Run power electronics optimization module.

This module sets up and runs the power electronics optimization.
"""

import argparse
import os
import sys

import model_pipeline
from model_pipeline import ModelPipeline
import numpy as np
import pandas as pd
from pymoo.core.mixed import MixedVariableGA
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.optimize import minimize
import pymoo_pe_problem
from pymoo_pe_problem import MyPowerElecProblem
import torch

# Set sys.modules for local module resolution
sys.modules["model_pipeline"] = model_pipeline
sys.modules["pymoo_pe_problem"] = pymoo_pe_problem


def main() -> None:
    """Main function to run power electronics optimization."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run power electronics optimization")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for computation (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--pop_size",
        type=int,
        default=500,
        help="Population size for the genetic algorithm",
    )
    parser.add_argument(
        "--n_gen",
        type=int,
        default=100,
        help="Number of generations to run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Set the device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # 1) Load your trained ensemble pipelines for r and g.
    # (Adjust the file paths to where your pipelines are saved.)
    pipeline_r = ModelPipeline.load("my_models/final_pipeline_power_electronics_v0_1.csv__2025-03-26-11-03-52_tgt_r.pkl")
    pipeline_g = ModelPipeline.load("my_models/final_pipeline_power_electronics_v0_1.csv__2025-03-26-10-44-11_tgt_g.pkl")

    # 2) Create the pymoo problem instance with your pipelines.
    problem = MyPowerElecProblem(pipeline_r, pipeline_g, device=device)

    # 3) Create a mixed-variable GA algorithm.
    # Use MixedVariableGA with a multi-objective survival operator.
    algorithm = MixedVariableGA(pop_size=args.pop_size, survival=RankAndCrowding())

    # 4) Run the optimization.
    res = minimize(problem, algorithm, termination=("n_gen", args.n_gen), seed=args.seed, verbose=True)

    # 5) Inspect the results.
    print("Pareto front found:")
    print("Decision variables (X):")
    print(res.X)
    print("Objective values (F):")
    print(res.F)

    # Optionally, choose a best solution by summing objectives:
    best_idx = np.argmin(np.sum(res.F, axis=1))
    print("Best solution index:", best_idx)
    print("Best solution's design variables:", res.X[best_idx])
    print("Best solution's objectives:", res.F[best_idx])

    # 6) Save the Pareto front to CSV files.
    os.makedirs("results", exist_ok=True)

    # Save objectives:
    np.savetxt(
        "results/pareto_F.csv",
        res.F,
        delimiter=",",
        header="Objective_r,Objective_abs_g_minus_0.25",
        comments="",
    )

    # For decision variables, convert each solution (which is a dict) to a row.
    # Here, we assume that res.X is an array of dicts.
    x_dicts = []
    for sol in res.X:
        # sol is a numpy structured array or dict; we convert it to a dict
        sol_dict = {key: sol[key] for key in sol.dtype.names} if hasattr(sol, "dtype") else sol
        x_dicts.append(sol_dict)

    df_x = pd.DataFrame(x_dicts)
    df_x.to_csv("results/pareto_X.csv", index=False)

    # Also save a combined file with both decision variables and objectives.
    df_front = df_x.copy()
    df_front["f_r"] = res.F[:, 0]
    df_front["f_abs_g_minus_0.25"] = res.F[:, 1]
    df_front.to_csv("results/pareto_front.csv", index=False)

    print("Pareto front saved to:")
    print("  results/pareto_F.csv  (objectives)")
    print("  results/pareto_X.csv  (decision variables)")
    print("  results/pareto_front.csv  (combined)")


if __name__ == "__main__":
    main()
