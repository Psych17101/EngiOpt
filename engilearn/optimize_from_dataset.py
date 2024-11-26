"""This is a just a dummy example file to show how to use the benchmarking framework."""

from dataclasses import dataclass

from engibench.core import Problem
from engibench.utils.all_problems import all_problems
import numpy as np
import tyro


@dataclass
class Args:
    """Command-line arguments."""

    problem: str = "airfoil2d_v0"
    """Problem identifier."""
    seed: int = 1
    """Random seed."""
    mpicores: int = 1
    """Number of MPI cores."""


if __name__ == "__main__":
    args = tyro.cli(Args)
    problem: Problem = all_problems[args.problem].build()
    problem.reset(seed=args.seed)

    candidate_design = np.array(problem.dataset["test"]["initial"][0])
    print(f"Initial design: {candidate_design}")

    obj_values, design = problem.optimize(starting_point=candidate_design, mpicores=args.mpicores)
    print(f"Objective values: {obj_values}")
    print(f"Optimized design: {design}")
