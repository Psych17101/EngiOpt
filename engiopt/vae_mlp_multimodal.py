# ruff: noqa: TRY003
# ruff: noqa: TRY301
"""VAE-MLP multimodal training script.

This module provides functionality for training either a plain MLP or a structured
VAE+Surrogate (Shape2ShapeVAE) model. It supports dataset loading from HuggingFace,
handling multiple ensembles, W&B logging, and usage of modular helper components.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from dataclasses import field
import os
import random
import time

# Keep only what's needed from typing:
from typing import Any, Literal

from datasets import load_dataset

# Local modules
from model_pipeline import DataPreprocessor
from model_pipeline import ModelPipeline
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from training_utils import PlainTabularDataset
from training_utils import Shape2ShapeWithParamsDataset
from training_utils import train_one_model
import tyro

import wandb


@dataclass
class Args:
    """Arguments for the VAE-MLP multimodal training script."""

    huggingface_repo: str = "IDEALLab/power_electronics_v0"
    huggingface_split: str = "train"

    data_dir: str = "./data"
    data_input: str = "airfoil_data.csv"

    # Columns for structured data
    init_col: str = "initial_design"
    opt_col: str = "optimal_design"

    target_col: str = "cl_val"
    log_target: bool = False  # If True, apply np.log to target during preprocessing

    # Additional columns, flattening, etc.
    params_cols: list[str] = field(default_factory=lambda: ["mach", "reynolds"])
    flatten_columns: list[str] = field(default_factory=lambda: ["initial_design", "optimal_design"])
    strip_column_spaces: bool = False
    subset_condition: str | None = None
    nondim_map: str | None = None

    split_random_state: int = 999

    # MLP or structured VAE?
    structured: bool = True
    hidden_layers: int = 2
    hidden_size: int = 64
    latent_dim: int = 32

    activation: Literal["relu", "leakyrelu", "prelu", "rrelu", "tanh", "sigmoid", "elu", "selu", "gelu", "celu", "none"] = (
        "relu"
    )
    optimizer: Literal["sgd", "adam", "adamw", "rmsprop", "adagrad", "adadelta", "adamax", "asgd", "lbfgs"] = "adam"
    learning_rate: float = 1e-3

    lr_decay: float = 1.0
    lr_decay_step: int = 1

    n_epochs: int = 50
    batch_size: int = 32
    patience: int = 10
    l2_lambda: float = 1e-3

    # VAE Surrogate additional terms
    gamma: float = 1.0
    lambda_lv: float = 1e-2

    test_size: float = 0.2
    val_size_of_train: float = 0.25
    scale_target: bool = True

    # W&B + logging
    track: bool = True
    wandb_project: str = "engiopt"
    wandb_entity: str | None = None

    seed: int = 42
    n_ensembles: int = 1

    problem_id: str = "engiopt"
    algo: str = os.path.basename(__file__)[: -len(".py")]

    save_model: bool = False
    plot_loss: bool = True
    model_output_dir: str = "results"
    test_model: bool = False
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Post-initialization checks and parsing."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_output_dir, exist_ok=True)

        # Parse flatten_columns, params_cols if they're strings
        for field_name in ["flatten_columns", "params_cols"]:
            value = getattr(self, field_name)
            if isinstance(value, str):
                try:
                    parsed_value = ast.literal_eval(value)
                    if isinstance(parsed_value, list):
                        setattr(self, field_name, parsed_value)
                    else:
                        raise TypeError(  # ruff: noqa: TRY003, TRY301
                            f"Expected list for --{field_name}, got {parsed_value}"
                        )
                except Exception as e:
                    raise ValueError(  # ruff: noqa: TRY003, TRY301
                        f"Invalid format for --{field_name}: {value}"
                    ) from e
            elif isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
                first = value[0].strip()
                if first and first[0] in ("[", "{"):
                    try:
                        parsed_value = ast.literal_eval(value[0])
                        if isinstance(parsed_value, list):
                            setattr(self, field_name, parsed_value)
                    except Exception as e:
                        raise ValueError(  # ruff: noqa: TRY003, TRY301
                            f"Invalid format for --{field_name}: {value}"
                        ) from e


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
        raise ValueError(  # ruff: noqa: TRY003, TRY301
            f"Invalid device: {args.device}"
        )


def load_data(args: Args) -> pd.DataFrame:
    """Load the dataset from either HuggingFace or CSV/Parquet."""
    if args.huggingface_repo:
        print(f"[INFO] Loading dataset from HuggingFace: {args.huggingface_repo} (split={args.huggingface_split})")
        ds = load_dataset(args.huggingface_repo, split=args.huggingface_split)
        df_loaded = ds.to_pandas()
    else:
        data_path = os.path.join(args.data_dir, args.data_input)
        if not os.path.isfile(data_path):
            raise FileNotFoundError(  # ruff: noqa: TRY003, TRY301
                f"{data_path} does not exist"
            )
        ext = os.path.splitext(data_path)[1].lower()
        if ext == ".csv":
            df_loaded = pd.read_csv(data_path)
        elif ext == ".parquet":
            df_loaded = pd.read_parquet(data_path)
        else:
            raise ValueError(  # ruff: noqa: TRY003, TRY301
                "data_input must be CSV or Parquet"
            )

    print("[INFO] DataFrame head:")
    print(df_loaded.head())
    return df_loaded


def preprocess_data(args: Args, df: pd.DataFrame) -> tuple[DataPreprocessor, dict, pd.DataFrame]:
    """Instantiate `DataPreprocessor`, transform inputs, and return (preprocessor, processed_dict, df)."""
    preprocessor = DataPreprocessor(vars(args))
    processed_dict, df_out = preprocessor.transform_inputs(df, fit_params=True)
    return preprocessor, processed_dict, df_out


def check_shape_columns(args: Args, df: pd.DataFrame) -> tuple[bool, np.ndarray]:
    """Check shape columns if in structured mode and confirm target column presence."""
    have_shape_cols = (
        args.init_col != ""
        and args.opt_col != ""
        and any(c.startswith(args.init_col + "_") for c in df.columns)
        and any(c.startswith(args.opt_col + "_") for c in df.columns)
    )
    if args.structured and not have_shape_cols:
        raise ValueError(  # ruff: noqa: TRY003, TRY301
            "Structured mode but no shape columns found. Check init_col/opt_col settings."
        )

    if args.target_col not in df.columns:
        raise ValueError(  # ruff: noqa: TRY003, TRY301
            f"Missing target_col in DataFrame: {args.target_col}"
        )

    y_all = df[args.target_col].values
    return have_shape_cols, y_all


def split_inputs(
    args: Args,
    *,  # Force keyword arguments
    have_shape_cols: bool,
    processed_dict: dict[str, Any],
    y_all: np.ndarray,
) -> dict[str, Any]:
    """Split the data into train/val/test sets, returning a dictionary with the needed arrays."""
    if have_shape_cols and args.structured:
        x_init_all = processed_dict["x_init"]
        x_opt_all = processed_dict["x_opt"]
        params_all = processed_dict["params"]

        (
            xinit_temp,
            xinit_test,
            xopt_temp,
            xopt_test,
            params_temp,
            params_test,
            y_temp,
            y_test,
        ) = train_test_split(
            x_init_all,
            x_opt_all,
            params_all,
            y_all,
            test_size=args.test_size,
            random_state=args.split_random_state,
        )
        (
            xinit_train,
            xinit_val,
            xopt_train,
            xopt_val,
            params_train,
            params_val,
            y_train,
            y_val,
        ) = train_test_split(
            xinit_temp,
            xopt_temp,
            params_temp,
            y_temp,
            test_size=args.val_size_of_train,
            random_state=args.split_random_state,
        )

        return {
            "structured": True,
            "xinit_train": xinit_train,
            "xinit_val": xinit_val,
            "xinit_test": xinit_test,
            "xopt_train": xopt_train,
            "xopt_val": xopt_val,
            "xopt_test": xopt_test,
            "params_train": params_train,
            "params_val": params_val,
            "params_test": params_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }
    else:
        x_features_all = processed_dict["X"]
        x_temp, x_test, y_temp, y_test = train_test_split(
            x_features_all,
            y_all,
            test_size=args.test_size,
            random_state=args.split_random_state,
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_temp,
            y_temp,
            test_size=args.val_size_of_train,
            random_state=args.split_random_state,
        )
        return {
            "structured": False,
            "x_train": x_train,
            "x_val": x_val,
            "x_test": x_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }


# ruff: noqa: PLR0915
def scale_data(
    args: Args, split_dict: dict[str, Any]
) -> tuple[
    dict[str, Any],
    int | None,
    int,
    int,
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
]:
    """Scale data differently for structured vs. unstructured. Return scalers, dims, and the Datasets."""
    custom_scalers: dict[str, Any] = {}
    num_cont, num_cat = 0, 0  # Adjust if needed

    if split_dict["structured"]:
        scaler_init = RobustScaler()
        scaler_opt = RobustScaler()
        scaler_params = RobustScaler()

        xinit_train_s = scaler_init.fit_transform(split_dict["xinit_train"])
        xinit_val_s = scaler_init.transform(split_dict["xinit_val"])
        xinit_test_s = scaler_init.transform(split_dict["xinit_test"])

        xopt_train_s = scaler_opt.fit_transform(split_dict["xopt_train"])
        xopt_val_s = scaler_opt.transform(split_dict["xopt_val"])
        xopt_test_s = scaler_opt.transform(split_dict["xopt_test"])

        params_train_s = scaler_params.fit_transform(split_dict["params_train"])
        params_val_s = scaler_params.transform(split_dict["params_val"])
        params_test_s = scaler_params.transform(split_dict["params_test"])

        y_train = split_dict["y_train"]
        y_val = split_dict["y_val"]
        y_test = split_dict["y_test"]

        if args.scale_target:
            scaler_y = RobustScaler()
            y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
            y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
            custom_scalers["scaler_y"] = scaler_y
        else:
            y_train_s, y_val_s, y_test_s = y_train, y_val, y_test

        train_dataset = Shape2ShapeWithParamsDataset(xinit_train_s, xopt_train_s, params_train_s, y_train_s)
        val_dataset = Shape2ShapeWithParamsDataset(xinit_val_s, xopt_val_s, params_val_s, y_val_s)
        test_dataset = Shape2ShapeWithParamsDataset(xinit_test_s, xopt_test_s, params_test_s, y_test_s)

        shape_dim = xinit_train_s.shape[1]

        custom_scalers["scaler_init"] = scaler_init
        custom_scalers["scaler_opt"] = scaler_opt
        custom_scalers["scaler_params"] = scaler_params

    else:
        scaler_x = RobustScaler()

        x_train_s = scaler_x.fit_transform(split_dict["x_train"])
        x_val_s = scaler_x.transform(split_dict["x_val"])
        x_test_s = scaler_x.transform(split_dict["x_test"])

        y_train = split_dict["y_train"]
        y_val = split_dict["y_val"]
        y_test = split_dict["y_test"]

        if args.scale_target:
            scaler_y = RobustScaler()
            y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
            y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
            custom_scalers["scaler_y"] = scaler_y
        else:
            y_train_s, y_val_s, y_test_s = y_train, y_val, y_test

        train_dataset = PlainTabularDataset(x_train_s, y_train_s)
        val_dataset = PlainTabularDataset(x_val_s, y_val_s)
        test_dataset = PlainTabularDataset(x_test_s, y_test_s)

        shape_dim = None
        custom_scalers["scaler_x"] = scaler_x

    return (
        custom_scalers,
        shape_dim,
        num_cont,
        num_cat,
        train_dataset,
        val_dataset,
        test_dataset,
    )


# ruff: noqa: PLR0913
def test_ensemble(
    args: Args,
    ensemble_models: list[torch.nn.Module],
    test_loader: DataLoader,
    custom_scalers: dict[str, Any],
    *,  # Force keyword arguments
    have_shape_cols: bool,
    device: torch.device,
) -> None:
    """Optionally evaluate the ensemble on the test set and log MSE."""
    predictions_list = []
    truths_list = []

    for model_e in ensemble_models:
        model_e.eval()
        preds_e = []
        trues_e = []
        with torch.no_grad():
            for batch_data in test_loader:
                # structured => x_init, x_opt, param, cl
                # unstructured => X_batch, y_batch
                if have_shape_cols and args.structured:
                    x_init_b, x_opt_b, params_b, y_b = batch_data
                    x_init_b = x_init_b.to(device)
                    params_b = params_b.to(device)
                    y_b = y_b.to(device)

                    # model's forward returns (x_opt_pred, mu, logvar, z, cl_pred)
                    _, _, _, _, cl_pred = model_e(x_init_b, params_b)
                    preds_e.append(cl_pred.view(-1).cpu().numpy())
                    trues_e.append(y_b.view(-1).cpu().numpy())
                else:
                    x_b, y_b = batch_data
                    x_b = x_b.to(device)
                    y_b = y_b.to(device)
                    pred = model_e(x_b).squeeze(-1)
                    preds_e.append(pred.cpu().numpy())
                    trues_e.append(y_b.cpu().numpy())

        preds_e = np.concatenate(preds_e)
        trues_e = np.concatenate(trues_e)
        predictions_list.append(preds_e)
        truths_list.append(trues_e)

    # Average ensemble
    test_trues = truths_list[0]
    all_preds = np.stack(predictions_list, axis=0)
    test_preds_ensemble = np.mean(all_preds, axis=0)

    # If scaled, invert
    if args.scale_target and "scaler_y" in custom_scalers:
        test_preds_ensemble = custom_scalers["scaler_y"].inverse_transform(test_preds_ensemble.reshape(-1, 1)).flatten()
        test_trues = custom_scalers["scaler_y"].inverse_transform(test_trues.reshape(-1, 1)).flatten()

    # If log-targeted, exponentiate
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
    """Main function to train and evaluate the model."""
    # --------------------------------------------------
    # Setup
    # --------------------------------------------------
    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    device = get_device(args)

    # --------------------------------------------------
    # Optional W&B Init
    # --------------------------------------------------
    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            save_code=True,
            name=run_name,
        )

    # --------------------------------------------------
    # 1) Load the dataset
    # --------------------------------------------------
    df = load_data(args)

    # --------------------------------------------------
    # 2) Preprocess
    # --------------------------------------------------
    preprocessor, processed_dict, df = preprocess_data(args, df)

    # --------------------------------------------------
    # 3) Check shape columns, verify target col
    # --------------------------------------------------
    have_shape_cols, y_all = check_shape_columns(args, df)

    # --------------------------------------------------
    # 4) Split
    # --------------------------------------------------
    split_dict = split_inputs(
        args,
        have_shape_cols=have_shape_cols,
        processed_dict=processed_dict,
        y_all=y_all,
    )

    # --------------------------------------------------
    # 5) Scale + create Datasets
    # --------------------------------------------------
    (
        custom_scalers,
        shape_dim,
        num_cont,
        num_cat,
        train_dataset,
        val_dataset,
        test_dataset,
    ) = scale_data(args, split_dict)

    # --------------------------------------------------
    # 6) DataLoaders
    # --------------------------------------------------
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --------------------------------------------------
    # 7) Ensemble logic
    # --------------------------------------------------
    ensemble_models = []
    ensemble_val_losses = []
    seeds = [args.seed + i for i in range(args.n_ensembles)]

    for seed_i in seeds:
        print(f"=== Training model for seed={seed_i} ===")
        # Fix random seeds for reproducibility
        torch.manual_seed(seed_i)
        np.random.seed(seed_i)
        random.seed(seed_i)
        cudnn.deterministic = True
        cudnn.benchmark = False

        # Train one model
        model_i, (train_losses_i, val_losses_i), best_val_loss_i = train_one_model(
            args,
            train_loader,
            val_loader,
            have_shape_cols,
            shape_dim,
            num_cont,
            num_cat,
            device,
        )

        ensemble_models.append(model_i)
        ensemble_val_losses.append(best_val_loss_i)

        # Optionally log each epoch to W&B
        if args.track:
            for epoch_idx, (tr_l, va_l) in enumerate(zip(train_losses_i, val_losses_i)):
                wandb.log({"train_loss": tr_l, "val_loss": va_l, "epoch": epoch_idx, "seed": seed_i})

    # --------------------------------------------------
    # 8) Pick best model
    # --------------------------------------------------
    best_model_idx = int(np.argmin(ensemble_val_losses))
    print(
        f"[INFO] Best model index={best_model_idx}, seed={seeds[best_model_idx]}, "
        f"val_loss={ensemble_val_losses[best_model_idx]:.4f}"
    )

    # --------------------------------------------------
    # 9) Build final pipeline
    # --------------------------------------------------
    pipeline_metadata = {"args": vars(args)}
    pipeline = ModelPipeline(
        models=ensemble_models,
        scalers=custom_scalers,
        structured=args.structured,
        log_target=args.log_target,
        metadata=pipeline_metadata,
        preprocessor=preprocessor,
    )

    run_filename = f"final_pipeline_{run_name}_{args.target_col}.pkl"
    pipeline_filename = os.path.join(args.model_output_dir, run_filename)

    if args.save_model:
        pipeline.save(pipeline_filename, device=device)
        print(f"[INFO] Saved pipeline to {pipeline_filename}")

        if args.track:
            artifact = wandb.Artifact(f"{run_name}_model", type="model")
            artifact.add_file(pipeline_filename)
            wandb.log_artifact(artifact)
            print("[INFO] Uploaded model artifact to W&B.")

    # --------------------------------------------------
    # 10) Optional test evaluation
    # --------------------------------------------------
    if args.test_model:
        test_ensemble(
            args,
            ensemble_models,
            test_loader,
            custom_scalers,
            have_shape_cols=have_shape_cols,
            device=device,
        )

    # --------------------------------------------------
    # 11) Finish W&B
    # --------------------------------------------------
    if args.track:
        wandb.finish()

    return ensemble_val_losses[best_model_idx]


if __name__ == "__main__":
    try:
        cli_args = tyro.cli(Args)
        main(cli_args)
    except Exception:
        print("An error occurred during execution:")
        raise
