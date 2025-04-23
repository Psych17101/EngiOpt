"""Transformations for the data."""

from collections.abc import Callable

from engibench.core import Problem
from gymnasium import spaces
import torch as th


def flatten_dict_factory(problem: Problem, device: th.device) -> Callable:
    """Factory function to create a flatten_dict function."""

    def flatten_dict(x):  # noqa: ANN001, ANN202
        """Convert each design in the batch to a flattened tensor."""
        flattened = []
        for design in x:
            # Move to CPU for numpy conversion, then back to device
            design_cpu = {k: v.cpu().numpy() if isinstance(v, th.Tensor) else v for k, v in design.items()}
            flattened_array = spaces.flatten(problem.design_space, design_cpu)
            flattened.append(th.tensor(flattened_array, device=device))
        return th.stack(flattened)

    return flatten_dict
