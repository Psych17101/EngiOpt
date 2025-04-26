# ruff: noqa: TRY003
# ruff: noqa: TRY301
# ruff: noqa: PLR0913
# ruff: noqa: PLR0915


"""MLP model for tabular data training and evaluation.

This module provides functionality for training and evaluating MLP models on tabular data.
It includes data preprocessing, model training, and evaluation capabilities.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from dataclasses import field
import os
import random
import sys
import time
from typing import Any, Literal

# Local modules
from engibench.utils.all_problems import BUILTIN_PROBLEMS
from model_pipeline import DataPreprocessor
from model_pipeline import ModelPipeline
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import torch
from torch import cuda
from torch.utils.data import DataLoader
from training_utils import PlainTabularDataset
from training_utils import train_one_model
import tyro

import wandb

# Ensure the current directory is in the path (if needed for local imports).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _parse_list_from_string(value: str, field_name: str) -> list:
    """Parse a string representation of a list into an actual list.

    Args:
        value: The string to parse.
        field_name: The name of the field being parsed (for error messages).

    Returns:
        list: The parsed list.

    Raises:
        TypeError: If the parsed value is not a list.
        ValueError: If the string cannot be parsed.
    """
    try:
        parsed_value = ast.literal_eval(value)
        if isinstance(parsed_value, list):
            return parsed_value
        else:
            raise TypeError(f"Expected list for {field_name}")
    except Exception as e:
        raise ValueError(f"Invalid format for {field_name}") from e


def _parse_list_from_single_item_list(value_list: list, field_name: str) -> list:
    """Parse a list containing a single string item that might be a string representation of a list.

    Args:
        value_list: A list containing a single string item.
        field_name: The name of the field being parsed (for error messages).

    Returns:
        list: The parsed list or the original list if parsing is not needed.
    """
    if not value_list or not isinstance(value_list[0], str):
        return value_list

    first = value_list[0].strip()
    if first and first[0] in ("[", "{"):
        try:
            parsed_value = ast.literal_eval(first)
            if isinstance(parsed_value, list):
                return parsed_value
        except Exception as e:
            raise ValueError(f"Invalid format for {field_name}") from e

    return value_list


@dataclass
class Args:
    """Arguments for the MLP training script on tabular data."""

    # ---------------- PROBLEM DEFINITION -----------------
    problem_id: str = "power_electronics"

    # ---------------- TARGET -----------------
    target_col: str = "target"
    log_target: bool = False  # If True, apply log transform to the target

    # ---------------- PREPROCESSING -----------------
    flatten_columns: list[str] = field(default_factory=list)
    strip_column_spaces: bool = False
    subset_condition: str | None = None

    # If you only want certain columns to be used as features, specify them here:
    params_cols: list[str] = field(default_factory=list)

    # ---------------- MODEL ARCHITECTURE -----------------
    hidden_layers: int = 2
    hidden_size: int = 64
    activation: Literal[
        "relu",
        "leakyrelu",
        "prelu",
        "rrelu",
        "tanh",
        "sigmoid",
        "elu",
        "selu",
        "gelu",
        "celu",
        "none",
    ] = "relu"

    # ---------------- OPTIMIZATION -----------------
    optimizer: Literal[
        "sgd",
        "adam",
        "adamw",
        "rmsprop",
        "adagrad",
        "adadelta",
        "adamax",
        "asgd",
        "lbfgs",
    ] = "adam"
    learning_rate: float = 1e-3
    lr_decay: float = 1.0
    lr_decay_step: int = 1
    n_epochs: int = 50
    batch_size: int = 32
    patience: int = 10
    l2_lambda: float = 1e-3

    # ---------------- LOGGING & GENERAL -----------------
    scale_target: bool = True
    track: bool = True
    wandb_project: str = "engiopt"
    wandb_entity: str | None = None
    seed: int = 42
    n_ensembles: int = 1
    algo: str = "mlp_tabular"
    save_model: bool = False
    model_output_dir: str = "results"
    test_model: bool = False
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Parse string-based list args and create output dirs."""
        os.makedirs(self.model_output_dir, exist_ok=True)

        # Process flatten_columns
        if isinstance(self.flatten_columns, str):
            self.flatten_columns = _parse_list_from_string(self.flatten_columns, "--flatten_columns")
        elif isinstance(self.flatten_columns, list) and len(self.flatten_columns) == 1:
            self.flatten_columns = _parse_list_from_single_item_list(self.flatten_columns, "--flatten_columns")

        # Process params_cols
        if isinstance(self.params_cols, str):
            self.params_cols = _parse_list_from_string(self.params_cols, "--params_cols")
        elif isinstance(self.params_cols, list) and len(self.params_cols) == 1:
            self.params_cols = _parse_list_from_single_item_list(self.params_cols, "--params_cols")


def get_device(args: Args) -> torch.device:
    """Determine the best available device for PyTorch operations.

    Returns:
        torch.device: The device to use for PyTorch operations, prioritizing:
            1. MPS (Metal Performance Shaders) for Apple Silicon
            2. CUDA for NVIDIA GPUs
            3. CPU as fallback
    """
    if args.device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif args.device == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError(f"Invalid device: {args.device}")


def split_data(
    x_features_all: np.ndarray, y_all: np.ndarray, args: Args
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training, validation, and test sets.

    Args:
        x_features_all: Feature data as a numpy array.
        y_all: Target data as a numpy array.
        args: Configuration arguments containing splitting parameters.

    Returns:
        tuple: A tuple containing (x_train, x_val, x_test, y_train, y_val, y_test)
               where each element is a numpy array of the corresponding split.
    """
    x_temp, x_test, y_temp, y_test = train_test_split(
        x_features_all, y_all, test_size=args.test_size, random_state=args.split_random_state
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=args.val_size_of_train, random_state=args.split_random_state
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def scale_data(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    args: Args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Scale feature and target data using RobustScaler.

    Args:
        x_train: Training feature data.
        x_val: Validation feature data.
        x_test: Test feature data.
        y_train: Training target data.
        y_val: Validation target data.
        y_test: Test target data.
        args: Configuration arguments containing scaling parameters.

    Returns:
        tuple: A tuple containing (x_train_s, x_val_s, x_test_s, y_train_s, y_val_s, y_test_s, custom_scalers)
               where each element is the scaled data and custom_scalers is a dictionary of fitted scalers.
    """
    custom_scalers: dict[str, Any] = {}
    scaler_x = RobustScaler()
    x_train_s = scaler_x.fit_transform(x_train)
    x_val_s = scaler_x.transform(x_val)
    x_test_s = scaler_x.transform(x_test)
    custom_scalers["scaler_x"] = scaler_x

    if args.scale_target:
        scaler_y = RobustScaler()
        y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        custom_scalers["scaler_y"] = scaler_y
    else:
        y_train_s, y_val_s, y_test_s = y_train, y_val, y_test

    return x_train_s, x_val_s, x_test_s, y_train_s, y_val_s, y_test_s, custom_scalers


def train_ensemble(
    args: Args, train_loader: DataLoader, val_loader: DataLoader, device: torch.device
) -> tuple[list, list, list]:
    """Train an ensemble of models with different random seeds.

    Args:
        args: Configuration arguments containing training parameters.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: PyTorch device to use for training.

    Returns:
        tuple: A tuple containing (ensemble_models, ensemble_val_losses, seeds)
               where ensemble_models is a list of trained models,
               ensemble_val_losses is a list of validation losses for each model,
               and seeds is a list of random seeds used for each model.
    """
    ensemble_models = []
    ensemble_val_losses = []
    seeds = [args.seed + i for i in range(args.n_ensembles)]
    for seed_i in seeds:
        print(f"=== Training model for seed={seed_i} ===")
        torch.manual_seed(seed_i)
        np.random.seed(seed_i)
        random.seed(seed_i)
        cuda.deterministic = True
        cuda.benchmark = False

        model_i, (train_losses_i, val_losses_i), best_val_loss_i = train_one_model(
            args, train_loader, val_loader, device=device
        )
        if args.track:
            for epoch_idx, (tr_loss, va_loss) in enumerate(zip(train_losses_i, val_losses_i)):
                wandb.log({"train_loss": tr_loss, "val_loss": va_loss, "epoch": epoch_idx, "seed": seed_i})
        ensemble_models.append(model_i)
        ensemble_val_losses.append(best_val_loss_i)
    return ensemble_models, ensemble_val_losses, seeds


def evaluate_ensemble(
    args: Args, ensemble_models: list, test_loader: DataLoader, device: torch.device, custom_scalers: dict[str, Any]
) -> None:
    """Evaluate an ensemble of models on test data.

    Args:
        args: Configuration arguments containing evaluation parameters.
        ensemble_models: List of trained models to evaluate.
        test_loader: DataLoader for test data.
        device: PyTorch device to use for evaluation.
        custom_scalers: Dictionary of fitted scalers for inverse transformation.

    Returns:
        None: Prints evaluation metrics and logs them to wandb if tracking is enabled.
    """
    predictions_list = []
    truths_list = []
    for model_e in ensemble_models:
        model_e.eval()
        preds_e = []
        trues_e = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch_device = x_batch.to(device)
                y_batch_device = y_batch.to(device)
                pred = model_e(x_batch_device).squeeze(-1)
                preds_e.append(pred.cpu().numpy())
                trues_e.append(y_batch_device.cpu().numpy())
        preds_e = np.concatenate(preds_e)
        trues_e = np.concatenate(trues_e)
        predictions_list.append(preds_e)
        truths_list.append(trues_e)
    test_trues = truths_list[0]
    all_preds = np.stack(predictions_list, axis=0)
    test_preds_ensemble = np.mean(all_preds, axis=0)

    if args.scale_target and "scaler_y" in custom_scalers:
        test_preds_ensemble = custom_scalers["scaler_y"].inverse_transform(test_preds_ensemble.reshape(-1, 1)).flatten()
        test_trues = custom_scalers["scaler_y"].inverse_transform(test_trues.reshape(-1, 1)).flatten()

    if args.log_target:
        test_preds_ensemble = np.exp(test_preds_ensemble)
        test_trues = np.exp(test_trues)

    print("[INFO] Test samples (avg ensemble):")
    for i in range(min(5, len(test_preds_ensemble))):
        print(f"  Sample {i}: Pred={test_preds_ensemble[i]:.4f}, True={test_trues[i]:.4f}")
    mse = np.mean((test_preds_ensemble - test_trues) ** 2)
    print(f"[INFO] Test MSE: {mse:.6f}")
    if args.track:
        wandb.log({"test_mse": mse})


def main(args: Args) -> float:
    """Main function to run the MLP training and evaluation pipeline.

    This function orchestrates the entire process:
    1. Loads and preprocesses data
    2. Splits data into train/val/test sets (if not already split)
    3. Scales the data
    4. Trains an ensemble of models
    5. Evaluates the models on test data
    6. Saves the model pipeline if requested

    Args:
        args: Configuration arguments for the entire pipeline.

    Returns:
        float: The best validation loss achieved by any model in the ensemble.
    """
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    df_train = problem.dataset["train"].to_pandas()
    df_val = problem.dataset["val"].to_pandas()
    df_test = problem.dataset["test"].to_pandas()

    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"

    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            save_code=True,
            name=run_name,
        )

    device = get_device(args)
    print(f"[INFO] Using device: {device}")

    # Preprocess each split
    preprocessor = DataPreprocessor(vars(args))
    processed_dict_train, df_train = preprocessor.transform_inputs(df_train, fit_params=True)
    processed_dict_val, df_val = preprocessor.transform_inputs(df_val, fit_params=False)
    processed_dict_test, df_test = preprocessor.transform_inputs(df_test, fit_params=False)

    # Extract features and targets
    if args.params_cols:
        x_train = df_train[args.params_cols].values
        x_val = df_val[args.params_cols].values
        x_test = df_test[args.params_cols].values
    else:
        x_train = processed_dict_train["X"]
        x_val = processed_dict_val["X"]
        x_test = processed_dict_test["X"]

    y_train = df_train[args.target_col].values
    y_val = df_val[args.target_col].values
    y_test = df_test[args.target_col].values

    # Scale data
    x_train_s, x_val_s, x_test_s, y_train_s, y_val_s, y_test_s, custom_scalers = scale_data(
        x_train, x_val, x_test, y_train, y_val, y_test, args
    )

    # Create datasets and dataloaders
    train_dataset = PlainTabularDataset(x_train_s, y_train_s)
    val_dataset = PlainTabularDataset(x_val_s, y_val_s)
    test_dataset = PlainTabularDataset(x_test_s, y_test_s)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Train ensemble
    ensemble_models, ensemble_val_losses, seeds = train_ensemble(args, train_loader, val_loader, device)
    best_model_idx = int(np.argmin(ensemble_val_losses))
    print(
        f"[INFO] Best model index={best_model_idx}, seed={seeds[best_model_idx]}, "
        f"val_loss={ensemble_val_losses[best_model_idx]:.4f}"
    )

    # Save pipeline if requested
    pipeline_metadata = {"args": vars(args)}
    pipeline = ModelPipeline(
        models=ensemble_models,
        scalers=custom_scalers,
        structured=False,  # no shape-based logic
        log_target=args.log_target,
        metadata=pipeline_metadata,
        preprocessor=preprocessor,
    )
    pipeline_filename = os.path.join(args.model_output_dir, f"final_pipeline_{run_name}_{args.target_col}.pkl")
    if args.save_model:
        pipeline.save(pipeline_filename, device=device)
        print(f"[INFO] Saved pipeline to {pipeline_filename}")
        if args.track:
            artifact = wandb.Artifact(f"{run_name}_model", type="model")
            artifact.add_file(pipeline_filename)
            wandb.log_artifact(artifact)
            print("[INFO] Uploaded model artifact to W&B.")

    # Evaluate on test set if requested
    if args.test_model:
        evaluate_ensemble(args, ensemble_models, test_loader, device, custom_scalers)

    if args.track:
        wandb.finish()

    return ensemble_val_losses[best_model_idx]


if __name__ == "__main__":
    try:
        args = tyro.cli(Args)
        main(args)
    except Exception:
        print("An error occurred during execution:")
        raise
