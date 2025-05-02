"""This module provides metrics for evaluating generative model designs.

Maximum Mean Discrepancy (MMD), Determinantal Point Process (DPP) diversity, and
optimality gap calculations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gymnasium import spaces
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from datasets import Dataset
    from engibench import OptiStep
    from engibench.core import Problem


def mmd(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Compute the Maximum Mean Discrepancy (MMD) between two sets of samples.

    Args:
        x (np.ndarray): Array of shape (n, l, w) for generative model designs.
        y (np.ndarray): Array of shape (m, l, w) for dataset designs.
        sigma (float): Bandwidth parameter for the Gaussian kernel.

    Returns:
        float: The MMD value.
    """
    x_flat = x.reshape(x.shape[0], -1)
    y_flat = y.reshape(y.shape[0], -1)

    k_xx = np.exp(-cdist(x_flat, x_flat, "sqeuclidean") / (2 * sigma**2))
    k_yy = np.exp(-cdist(y_flat, y_flat, "sqeuclidean") / (2 * sigma**2))
    k_xy = np.exp(-cdist(x_flat, y_flat, "sqeuclidean") / (2 * sigma**2))

    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()


def dpp_diversity(x: np.ndarray, sigma: float = 1.0) -> float:
    """Compute the Determinantal Point Process (DPP) diversity for a set of samples.

    Args:
        x (np.ndarray): Array of shape (n, l, w) for generative model designs.
        sigma (float): Bandwidth parameter for the Gaussian kernel.

    Returns:
        float: The DPP diversity value.
    """
    x_flat = x.reshape(x.shape[0], -1)
    pairwise_sq_dists = cdist(x_flat, x_flat, "sqeuclidean")
    similarity_matrix = np.exp(-pairwise_sq_dists / (2 * sigma**2))

    # Regularize the matrix slightly to avoid numerical issues
    reg_matrix = similarity_matrix + 1e-6 * np.eye(x.shape[0])

    try:
        return np.linalg.det(reg_matrix)
    except np.linalg.LinAlgError:
        return 0.0  # fallback in case of numerical issues


def optimality_gap(opt_history: list[OptiStep], baseline: float) -> list[float]:
    """Compute the optimality gap of an optimization history.

    Args:
        opt_history (list[OptiStep]): The optimization history.
        baseline (float): The baseline value to compare against.

    Returns:
        list[float]: The optimality gap at each step in opt_history.
    """
    return [opt.obj_values - baseline for opt in opt_history]


def metrics(
    problem: Problem,
    gen_designs: npt.NDArray,
    dataset_designs: npt.NDArray,
    sampled_conditions: Dataset | None = None,
    sigma: float = 1.0,
) -> dict[str, float]:
    """Compute various metrics for evaluating generative model designs.

    Args:
        problem: The optimization problem to evaluate.
        gen_designs (np.ndarray): Array of shape (n_samples, l, w) for generative model designs (potentially flattened for dict spaces).
        dataset_designs (np.ndarray): Array of shape (n_samples, l, w) for dataset designs (these are not flattened).
        sampled_conditions (Dataset): Dataset of sampled conditions for optimization. If None, no conditions are used.
        sigma (float): Bandwidth parameter for the Gaussian kernel (in mmd and dpp calculation).

    Returns:
        dict[str, float]: A dictionary containing the computed metrics:
            - "iog": Average Initial Optimality Gap (float).
            - "cog": Average Cumulative Optimality Gap (float).
            - "fog": Average Final Optimality Gap (float).
            - "mmd": Maximum Mean Discrepancy (float).
            - "dpp": Determinantal Point Process diversity (float).
    """
    n_samples = len(gen_designs)

    cog_list = []
    iog_list = []
    fog_list = []
    for i in range(n_samples):
        conditions = sampled_conditions[i] if sampled_conditions is not None else None
        if isinstance(problem.design_space, spaces.Dict):
            # Need to unflatten the design to be used for optimization or simulation
            unflattened_design = spaces.unflatten(problem.design_space, gen_designs[i])
        else:
            unflattened_design = gen_designs[i]
        _, opt_history = problem.optimize(unflattened_design, config=conditions)
        reference_optimum = problem.simulate(dataset_designs[i], config=conditions)
        opt_history_gaps = optimality_gap(opt_history, reference_optimum)

        iog_list.append(opt_history_gaps[0])
        cog_list.append(np.sum(opt_history_gaps))
        fog_list.append(opt_history_gaps[-1])

    # Compute the average Initial Optimality Gap (IOG), Cumulative Optimality Gap (COG), and Final Optimality Gap (FOG)
    average_iog: float = float(np.mean(iog_list))  # Average of initial optimality gaps
    average_cog: float = float(np.mean(cog_list))  # Average of cumulative optimality gaps
    average_fog: float = float(np.mean(fog_list))  # Average of final optimality gaps

    # Compute the Maximum Mean Discrepancy (MMD) between generated and dataset designs
    # We compute the MMD on the flattened designs
    flattened_ds_designs: list[npt.NDArray] = []
    for design in dataset_designs:
        if isinstance(problem.design_space, spaces.Dict):
            flattened = spaces.flatten(problem.design_space, design)
            flattened_ds_designs.append(np.array(flattened))
        else:
            flattened_ds_designs.append(design)
    flattened_ds_designs_array: npt.NDArray = np.array(flattened_ds_designs)
    mmd_value: float = mmd(gen_designs, flattened_ds_designs_array, sigma=sigma)

    # Compute the Determinantal Point Process (DPP) diversity for generated designs
    # We compute the DPP on the flattened designs
    dpp_value: float = dpp_diversity(gen_designs, sigma=sigma)

    # Return all computed metrics as a dictionary
    return {
        "iog": average_iog,
        "cog": average_cog,
        "fog": average_fog,
        "mmd": mmd_value,
        "dpp": dpp_value,
    }
