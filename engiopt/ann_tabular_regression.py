#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example PyTorch script for tabular regression with optional:
- Flattening of list columns
- Train/val/test split
- Data scaling
- Flexible MLP architecture
- Early stopping
- Weights & Biases tracking
- Command-line interface via Tyro
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import ast
import wandb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

###############################################################################
# 1) Argument Parsing
###############################################################################

@dataclass
class Args:
    """Command-line arguments for the tabular regression script."""
    # -----------------
    # Data & Columns
    # -----------------
    data_path: str = "Engibench/data/airfoil_data.csv"
    """Path to CSV/Parquet file containing the dataset."""
    input_cols: List[str] = field(default_factory=lambda: ["optimized", "mach", "reynolds", "alpha"])
    """List of columns to use as features (may contain list columns to flatten)."""
    target_col: str = "cl_val"
    """Name of the target column."""
    flatten_columns: List[str] = field(default_factory=lambda: ["optimized"])
    """Columns that we know contain list-like data (e.g., coordinates) to flatten.
    If empty, we won't do any flattening. See also `auto_detect_flatten`."""
    auto_detect_flatten: bool = False
    """If True, we auto-detect any column whose first row is a list/array and flatten it.
    If you also specified `flatten_columns`, both approaches are applied."""

    # -----------------
    # Model / Training
    # -----------------
    hidden_layers: int = 2
    """Number of hidden layers in the MLP."""
    hidden_size: int = 64
    """Size (width) of each hidden layer."""
    activation: Literal["relu", "leakyrelu"] = "relu"
    """Activation function."""
    optimizer: Literal["adam", "sgd"] = "adam"
    """Which optimizer to use."""
    learning_rate: float = 1e-3
    """Learning rate."""
    n_epochs: int = 50
    """Number of training epochs (upper bound, if early stopping doesn’t stop earlier)."""
    batch_size: int = 32
    """Batch size."""
    patience: int = 10
    """Early stopping patience (epochs without improvement)."""
    
    # -----------------
    # Splitting & Scaling
    # -----------------
    test_size: float = 0.2
    """Fraction of data to hold out for final test."""
    val_size_of_train: float = 0.25
    """Fraction of train split to use for validation. 
    Example: 0.25 means 25% of the training data is used for validation 
    => total data usage is  (1 - test_size)* val_size_of_train for val."""
    scale_target: bool = True
    """Whether to apply standard scaling to the target as well."""
    
    # -----------------
    # Logging & Repro
    # -----------------
    track: bool = True
    """Whether to enable W&B experiment tracking."""
    wandb_project: str = "engiopt"
    """W&B project name."""
    wandb_entity: Optional[str] = None
    """W&B entity name (team or username)."""
    seed: int = 42
    """Random seed for reproducibility."""
    save_model: bool = False
    """If True, saves the best model (during training) and logs as W&B artifact."""

    # -----------------
    # Visualization
    # -----------------
    plot_loss: bool = True
    """Whether to show a Matplotlib figure of training & validation loss."""

    def __post_init__(self):
        """Ensure correct parsing of list arguments."""
        for field_name in ["input_cols", "flatten_columns"]:
            value = getattr(self, field_name)
            if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
                try:
                    parsed_value = ast.literal_eval(value[0])
                    if isinstance(parsed_value, list) and all(isinstance(x, str) for x in parsed_value):
                        setattr(self, field_name, parsed_value)
                except Exception:
                    raise ValueError(f"Invalid format for --{field_name}: {value}. Expected a list format like ['col1', 'col2']")



        # Validate float values
        if not (0 < self.test_size < 1):
            raise ValueError(f"Invalid --test_size: {self.test_size}. Must be between 0 and 1.")

        if not (0 < self.val_size_of_train < 1):
            raise ValueError(f"Invalid --val_size_of_train: {self.val_size_of_train}. Must be between 0 and 1.")

        if self.test_size + self.val_size_of_train >= 1:
            raise ValueError(f"Invalid train/val/test split. Too much data allocated to validation and test.")

        # Ensure positive integers for model parameters
        for field_name in ["hidden_layers", "hidden_size", "batch_size", "patience", "n_epochs"]:
            value = getattr(self, field_name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"Invalid value for --{field_name}: {value}. Must be a positive integer.")



###############################################################################
# 2) Flattening Utilities (For List Columns)
###############################################################################
def recursive_flatten(val):
    """
    Recursively flatten a nested list/tuple into a single list of values.
    E.g. [[x0, x1], [y0, y1]] -> [x0, x1, y0, y1].
    """
    if not isinstance(val, (list, tuple)):
        return [val]
    result = []
    for item in val:
        result.extend(recursive_flatten(item))
    return result

def flatten_list_columns(df: pd.DataFrame, columns_to_flatten: List[str]) -> pd.DataFrame:
    """
    Flatten only the specified columns in `columns_to_flatten` that contain list/array data.
    Expands them into multiple numeric columns.
    """
    new_cols_list = []
    drop_cols = []
    
    for col in columns_to_flatten:
        if col not in df.columns:
            print(f"⚠️ Warning: Column '{col}' not found in DataFrame. Skipping...")
            continue
        
        first_value = df[col].iloc[0]

        # If stored as a string, convert back to list using `ast.literal_eval`
        if isinstance(first_value, str):
            try:
                first_value = ast.literal_eval(first_value)
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            except:
                print(f"⚠️ Warning: Column '{col}' contains string data that cannot be parsed as a list. Skipping...")
                continue

        # Check if first row is a list/tuple
        if isinstance(first_value, (list, tuple)):
            print(f"\n🔍 Debug: Flattening column '{col}', detected list structure.")

            # Flatten each row
            flattened_rows = [recursive_flatten(val) for val in df[col]]

            # Ensure consistent length across rows
            lengths = [len(row) for row in flattened_rows]
            unique_lengths = set(lengths)
            if len(unique_lengths) > 1:
                raise ValueError(
                    f"🚨 Column '{col}' has rows of varying lengths {unique_lengths}. "
                    "Cannot flatten consistently without special handling."
                )

            # Create new DataFrame with expanded numeric columns
            n = lengths[0]
            new_col_names = [f"{col}_{i}" for i in range(n)]
            expanded_df = pd.DataFrame(flattened_rows, columns=new_col_names)

            # Store new columns and mark the original for dropping
            new_cols_list.append(expanded_df)
            drop_cols.append(col)

    if new_cols_list:
        df = pd.concat([df.drop(columns=drop_cols).reset_index(drop=True)] + new_cols_list, axis=1)

    print("\n✅ Debug: Flattening completed. New DataFrame columns:", df.columns.tolist())
    return df

def auto_detect_and_flatten(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically detect columns whose first row is a list/tuple,
    flatten them, and return the updated DataFrame.
    """
    columns_to_flatten = []
    for col in df.columns:
        if isinstance(df[col].iloc[0], (list, tuple)):
            columns_to_flatten.append(col)
    if columns_to_flatten:
        df = flatten_list_columns(df, columns_to_flatten)
    return df


###############################################################################
# 3) PyTorch Dataset
###############################################################################

class TabularDataset(torch.utils.data.Dataset):
    """Simple dataset for (X, y) arrays."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


###############################################################################
# 4) MLP Model Construction (Variable Hidden Layers / Activation)
###############################################################################

def make_mlp(
    input_dim: int,
    hidden_layers: int,
    hidden_size: int,
    activation: Literal["relu", "leakyrelu"] = "relu",
    output_dim: int = 1
) -> nn.Sequential:
    """
    Creates an MLP with a specified number of hidden layers,
    hidden size, activation, and single scalar output (for regression).
    """
    layers = []
    in_dim = input_dim
    
    # Choose activation
    if activation == "relu":
        act_fn = nn.ReLU
    elif activation == "leakyrelu":
        act_fn = lambda: nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    
    for _ in range(hidden_layers):
        layers.append(nn.Linear(in_dim, hidden_size))
        layers.append(act_fn())
        in_dim = hidden_size
    
    # Final layer for regression
    layers.append(nn.Linear(in_dim, output_dim))
    model = nn.Sequential(*layers)
    return model


###############################################################################
# 5) Main Training Logic
###############################################################################

def main(args: Args) -> None:
    print("\n Debug: Parsed input columns from arguments:", args.input_cols)
    # -----------------
    # 5.1) Reproducibility
    # -----------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # -----------------
    # 5.2) Logging Setup
    # -----------------
    run_name = f"{os.path.basename(args.data_path)}__seed{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            save_code=True,
            name=run_name
        )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")


    # -----------------
    # 5.3) Load DataFrame
    # -----------------
    ext = os.path.splitext(args.data_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(args.data_path)
    elif ext == ".parquet":
        df = pd.read_parquet(args.data_path)
    else:
        raise ValueError("Data path must be a CSV or Parquet file.")
    
    print("\n Debug: Available columns in dataset:")
    print(df.columns.tolist())  # Print all columns in the dataset

    # Debug print BEFORE flattening
    print("\n Debug: Columns BEFORE flattening:", df.columns.tolist())

    # (Optional) Auto-detect flatten
    if args.auto_detect_flatten:
        df = auto_detect_and_flatten(df)
        print("\n Debug: auto_detect_and_flatten executed")
    
    # (Optional) Flatten specific columns
    if args.flatten_columns:
        df = flatten_list_columns(df, args.flatten_columns)
        print("\n Debug: flatten_list_columns executed")

    # Debug print AFTER flattening
    print("\n Debug: Columns AFTER flattening:", df.columns.tolist())

    # -----------------
    # 5.4) Create X, y
    # -----------------
    # If your input_cols now contain multiple subcolumns (e.g., optimized_0, optimized_1...), 
    # you may want to find them by prefix. Alternatively, if your columns are fully known:
    #   X = df[args.input_cols].values
    # But let's handle a more dynamic scenario: if user has "optimized" in input_cols but 
    # we flattened them to optimized_0, optimized_1, etc., we detect them by prefix.
    
    # We'll build a final list of columns. If a requested col is found exactly, we use it.
    # If not, but we see that prefix_{i} columns exist, we add them.
    # Debug print BEFORE processing
    print("\n Debug: Checking input columns after flattening...")
    print(f"Original requested input columns: {args.input_cols}")
    final_input_cols = []
    for col in args.input_cols:
        if col in df.columns:
            final_input_cols.append(col)
        else:
            # Check if we have flattened columns that start with col_
            matches = [c for c in df.columns if c.startswith(col + "_")]
            if matches:
                final_input_cols.extend(matches)

    # Debug print AFTER processing
    print("\n Debug: Final selected input columns:", final_input_cols)
    
    if not final_input_cols:
        raise ValueError("No valid input columns found after flattening/detection.")
    
    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in dataset.")

    X_all = df[final_input_cols].values
    y_all = df[args.target_col].values

    # -----------------
    # 5.5) Split Data
    # -----------------
    # 1) Split off test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all, y_all, test_size=args.test_size, random_state=args.seed
    )
    # 2) Split train & val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=args.val_size_of_train, random_state=args.seed
    )

    # -----------------
    # 5.6) Scale Data
    # -----------------
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled   = scaler_X.transform(X_val)
    X_test_scaled  = scaler_X.transform(X_test)

    if args.scale_target:
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()
        y_val_scaled   = scaler_y.transform(y_val.reshape(-1,1)).flatten()
        y_test_scaled  = scaler_y.transform(y_test.reshape(-1,1)).flatten()
    else:
        y_train_scaled, y_val_scaled, y_test_scaled = y_train, y_val, y_test

    # -----------------
    # 5.7) Create Dataloaders
    # -----------------
    train_dataset = TabularDataset(X_train_scaled, y_train_scaled)
    val_dataset   = TabularDataset(X_val_scaled,   y_val_scaled)
    test_dataset  = TabularDataset(X_test_scaled,  y_test_scaled)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    # -----------------
    # 5.8) Build Model
    # -----------------
    input_dim = X_train_scaled.shape[1]
    model = make_mlp(
        input_dim=input_dim,
        hidden_layers=args.hidden_layers,
        hidden_size=args.hidden_size,
        activation=args.activation,
        output_dim=1
    ).to(device)

    # Choose optimizer
    if args.optimizer == "adam":
        optimizer_fn = optim.Adam
    elif args.optimizer == "sgd":
        optimizer_fn = optim.SGD
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    optimizer = optimizer_fn(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    # -----------------
    # 5.9) Train with Early Stopping
    # -----------------
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_weights = None

    train_losses = []
    val_losses = []

    for epoch in range(args.n_epochs):
        # ---------- TRAIN ----------
        model.train()
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch).squeeze(-1)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * X_batch.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # ---------- VALIDATION ----------
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(X_batch).squeeze(-1)
                loss = criterion(preds, y_batch)
                running_val_loss += loss.item() * X_batch.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_weights = model.state_dict()
        else:
            epochs_no_improve += 1

        if args.track:
            wandb.log({
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "epoch": epoch
            })

        print(f"Epoch [{epoch+1}/{args.n_epochs}] - "
              f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        if epochs_no_improve >= args.patience:
            print("Early stopping triggered.")
            break

    print(f"\nBest Val Loss: {best_val_loss:.4f} at epoch {best_epoch+1}")

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # -----------------
    # 5.10) (Optional) Save Model
    # -----------------
    if args.save_model:
        model_fname = f"best_model_{run_name}.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "scaler_X": scaler_X,
            "scaler_y": scaler_y if args.scale_target else None
        }, model_fname)
        print(f"Saved best model to {model_fname}")
        if args.track:
            artifact = wandb.Artifact("best_model", type="model")
            artifact.add_file(model_fname)
            wandb.log_artifact(artifact)

    # -----------------
    # 5.11) Plot Loss Curves
    # -----------------
    if args.plot_loss:
        plt.figure(figsize=(8,5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.tight_layout()
        plot_file = f"loss_curve_{run_name}.png"
        plt.savefig(plot_file)
        if args.track:
            wandb.log({"loss_curve": wandb.Image(plot_file)})
        plt.show()

    # -----------------
    # 5.12) Final Test Evaluation
    # -----------------
    model.eval()
    test_preds_list = []
    test_targets_list = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch).squeeze(-1)
            test_preds_list.append(preds.cpu().numpy())
            test_targets_list.append(y_batch.cpu().numpy())

    test_preds_scaled = np.concatenate(test_preds_list)
    test_targets_scaled = np.concatenate(test_targets_list)

    # If we scaled y, we must inverse-transform
    if args.scale_target:
        test_preds = scaler_y.inverse_transform(test_preds_scaled.reshape(-1,1)).flatten()
        test_targets = scaler_y.inverse_transform(test_targets_scaled.reshape(-1,1)).flatten()
    else:
        test_preds = test_preds_scaled
        test_targets = test_targets_scaled

    test_mse = np.mean((test_preds - test_targets)**2)
    test_rmse = np.sqrt(test_mse)
    print(f"\nTest MSE:  {test_mse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")

    # Show some samples
    print("\nSample predictions vs. true:")
    for i in range(min(5, len(test_preds))):
        print(f"  Pred: {test_preds[i]:.4f}, True: {test_targets[i]:.4f}, Diff: {test_preds[i] - test_targets[i]:.4f}")

    if args.track:
        wandb.log({
            "test_mse": test_mse,
            "test_rmse": test_rmse,
        })
        wandb.finish()

###############################################################################
# 6) Entry Point
###############################################################################

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)

    """
    # Example use:
    # Note "no-" is used to mean Flase on the booleans
    python ../EngiOpt/engiopt/ann_tabular_regression.py \
        --data_path "../Engibench/data/airfoil_data.csv" \
        --input_cols '["optimized", "mach", "reynolds", "alpha"]' \
        --target_col "cl_val" \
        --flatten_columns '["optimized"]' \
        --no-auto_detect_flatten \
        --hidden_layers 2 \
        --hidden_size 64 \
        --activation relu \
        --optimizer adam \
        --learning_rate 1e-3 \
        --n_epochs 50 \
        --batch_size 32 \
        --patience 10 \
        --scale_target \
        --no-track \
        --wandb_project "engiopt_ann_tabular_regression" \
        --seed 42 \
        --save_model \

    """
