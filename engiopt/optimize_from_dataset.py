"""This is a just a dummy example file to show how to use the benchmarking framework."""

from dataclasses import dataclass

from engibench.core import Problem
from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import tyro


@dataclass
class Args:
    """Command-line arguments."""

    problem: str = "heatconduction2d"
    """Problem identifier."""
    seed: int = 1
    """Random seed."""


if __name__ == "__main__":
    args = tyro.cli(Args)
    problem: Problem = BUILTIN_PROBLEMS[args.problem]()
    problem.reset(seed=args.seed)

    candidate_design = np.array(problem.dataset["train"]["optimal_design"][0])
    print(f"Initial design: {candidate_design}")

    obj_values, design = problem.optimize(starting_point=candidate_design)
    print(f"Objective values: {obj_values}")
    print(f"Optimized design: {design}")
