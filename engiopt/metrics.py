from engibench import OptiStep
from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np


def mmd(X: np.ndarray, Y: np.ndarray, sigma=1.0) -> float:
    """
    Compute the Maximum Mean Discrepancy (MMD) between two sets of samples.

    Args:
        X (np.ndarray): Array of shape (n, l, w) for generative model designs.
        Y (np.ndarray): Array of shape (m, l, w) for dataset designs.
        sigma (float): Bandwidth parameter for the Gaussian kernel.

    Returns:
        float: The MMD value.
    """

    def gaussian_kernel(x, y, sigma=1.0):
        """Compute the Gaussian kernel between two samples."""
        return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))
    
    n, l, w = X.shape
    m, l, w = Y.shape

    # Flatten the images to (n, l*w) and (m, l*w)
    X_flat = X.reshape(n, -1)
    Y_flat = Y.reshape(m, -1)

    # Compute pairwise kernel values
    XX = np.mean([gaussian_kernel(x, x_, sigma) for x in X_flat for x_ in X_flat])
    YY = np.mean([gaussian_kernel(y, y_, sigma) for y in Y_flat for y_ in Y_flat])
    XY = np.mean([gaussian_kernel(x, y, sigma) for x in X_flat for y in Y_flat])

    # Compute MMD
    return XX + YY - 2 * XY

def dpp_diversity(X: np.ndarray, sigma=1.0) -> float:
    """
    Compute the Determinantal Point Process (DPP) diversity for a set of samples.

    Args:
        X (np.ndarray): Array of shape (n, l, w) for generative model designs.
        sigma (float): Bandwidth parameter for the Gaussian kernel.

    Returns:
        float: The DPP diversity value.
    """

    n, l, w = X.shape

    # Flatten the images to (n, l*w)
    X_flat = X.reshape(n, -1)

    # Compute the similarity matrix using the Gaussian kernel
    similarity_matrix = np.array([
        [np.exp(-np.linalg.norm(x - x_) ** 2 / (2 * sigma ** 2)) for x_ in X_flat]
        for x in X_flat
    ])

    # Compute the determinant of the similarity matrix
    diversity = np.linalg.det(similarity_matrix + np.eye(n) * 1e-6)  # Add small value for numerical stability

    return diversity

def optimality_gap(opt_history: list[OptiStep], baseline: float) -> list[float]:
    """Compute the optimality gap of an optimization history.

    Args:
        opt_history (list[OptiStep]): The optimization history.
        baseline (float): The baseline value to compare against.

    Returns:
        list[float]: The optimality gap at each step in opt_history.
    """
    return [opt.obj_values - float(baseline) for opt in opt_history]

def metrics(
    problem: BUILTIN_PROBLEMS,
    gen_designs: np.ndarray, 
    dataset_designs: np.ndarray, 
    sampled_conditions: list = None,
    sigma = 1.0 
) -> dict[str, float]:
    """
    Compute various metrics for evaluating generative model designs.

    Args:
        problem: The optimization problem to evaluate.
        gen_designs (np.ndarray): Array of shape (n_samples, l, w) for generative model designs.
        dataset_designs (np.ndarray): Array of shape (n_samples, l, w) for dataset designs.
        sampled_conditions (list): List of sampled conditions for optimization. If None, no conditions are used.
        sigma (float): Bandwidth parameter for the Gaussian kernel (in mmd and dpp calculation).

    Returns:
        dict[str, float]: A dictionary containing the computed metrics:
            - "average_IOG": Average Initial Optimality Gap (float).
            - "average_COG": Average Cumulative Optimality Gap (float).
            - "average_FOG": Average Final Optimality Gap (float).
            - "mmd_value": Maximum Mean Discrepancy (float).
            - "dpp_value": Determinantal Point Process diversity (float).
    """
    n_samples = len(gen_designs)

    COG_list = []
    IOG_list = []
    FOG_list = []
    for i in range(n_samples):
        conditions = sampled_conditions[i] if sampled_conditions is not None else None
        _, opt_history = problem.optimize(gen_designs[i], conditions)
        reference_optimum = problem.simulate(dataset_designs[i])
        opt_history_gaps = optimality_gap(opt_history, reference_optimum)

        IOG_list.append(opt_history_gaps[0])
        COG_list.append(np.sum(opt_history_gaps))
        FOG_list.append(opt_history_gaps[-1])

    # Compute the average Initial Optimality Gap (IOG), Cumulative Optimality Gap (COG), and Final Optimality Gap (FOG)
    average_IOG: float = np.mean(IOG_list)  # Average of initial optimality gaps
    average_COG: float = np.mean(COG_list)  # Average of cumulative optimality gaps
    average_FOG: float = np.mean(FOG_list)  # Average of final optimality gaps

    # Compute the Maximum Mean Discrepancy (MMD) between generated and dataset designs
    mmd_value: float = mmd(gen_designs, dataset_designs, sigma=sigma)

    # Compute the Determinantal Point Process (DPP) diversity for generated designs
    dpp_value: float = dpp_diversity(gen_designs, sigma=sigma)

    # Return all computed metrics as a dictionary
    return {
        "IOG": average_IOG,
        "COG": average_COG,
        "FOG": average_FOG,
        "mmd": mmd_value,
        "dpp": dpp_value,
    }
