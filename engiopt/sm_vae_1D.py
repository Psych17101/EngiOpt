#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example PyTorch script that:
- Performs tabular regression with a simple MLP,
- Supports flattening of list columns,
- Splits data into train/val/test,
- Scales features (and optionally target),
- Performs early stopping,
- Tracks run via Weights & Biases (optional),
- And saves both the model & loss curves into an output folder with
  a user-friendly timestamp-based naming scheme.

In this version, we define two arguments for the data path:
    1) data_dir (default: "Engibench/data")
    2) data_input (default: "airfoil_data.csv")
We then join them with os.path.join() to form the final dataset path.
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example PyTorch script that:
- Performs tabular regression,
- Supports flattening of list columns,
- Splits data into train/val/test,
- Scales features (and optionally target),
- Performs early stopping,
- Tracks runs via Weights & Biases (optional),
- And saves both the model & loss curves into an output folder.

In this version, you can choose between:
    (a) A plain MLP surrogate (for unstructured tabular data), or
    (b) A VAE + surrogate model (for structured tabular data)
The choice is made via the '--structured' argument.
"""

import os
import time
import ast
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import wandb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

###############################################################################
# 1) Argument Parsing
###############################################################################

@dataclass
class Args:
    """Command-line arguments for the tabular regression script with VAE surrogate."""
    # -----------------
    # Data & Columns
    # -----------------
    data_dir: str = "./data"
    data_input: str = "airfoil_data.csv"
    input_cols: List[str] = field(default_factory=lambda: ["optimized", "mach", "reynolds", "alpha"])
    target_col: str = "cl_val"
    flatten_columns: List[str] = field(default_factory=lambda: ["optimized"])
    
    # -----------------
    # Model / Training
    # -----------------
    structured: bool = True
    """If True, use a VAE + surrogate model; if False, use a plain MLP surrogate."""
    hidden_layers: int = 2
    hidden_size: int = 64  # used only for plain MLP model
    latent_dim: int = 32   # used only for VAE model
    activation: Literal[
        "relu", "leakyrelu", "prelu", "rrelu", "tanh",
        "sigmoid", "elu", "selu", "gelu", "celu", "none"
    ] = "relu"
    optimizer: Literal[
        "sgd", "adam", "adamw", "rmsprop",
        "adagrad", "adadelta", "adamax", "asgd", "lbfgs"
    ] = "adam"
    learning_rate: float = 1e-3
    n_epochs: int = 50
    batch_size: int = 32
    patience: int = 10
    beta: float = 1.0   # KL divergence weight (for VAE)
    gamma: float = 1.0  # Surrogate loss weight (for VAE)
    
    # -----------------
    # Splitting & Scaling
    # -----------------
    test_size: float = 0.2
    val_size_of_train: float = 0.25
    scale_target: bool = True

    # -----------------
    # Logging & Repro
    # -----------------
    track: bool = True
    wandb_project: str = "sm_vae_1D"
    wandb_entity: Optional[str] = None
    seed: int = 42
    save_model: bool = False

    # -----------------
    # Visualization & Output
    # -----------------
    plot_loss: bool = True
    model_output_dir: str = "results"

    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_output_dir, exist_ok=True)
        # Convert string lists if necessary
        for field_name in ["input_cols", "flatten_columns"]:
            value = getattr(self, field_name)
            if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
                try:
                    parsed_value = ast.literal_eval(value[0])
                    if isinstance(parsed_value, list) and all(isinstance(x, str) for x in parsed_value):
                        setattr(self, field_name, parsed_value)
                except Exception:
                    raise ValueError(
                        f"Invalid format for --{field_name}: {value}. "
                        "Expected a list format like ['col1','col2']"
                    )

###############################################################################
# 2) Flattening Utilities
###############################################################################

def recursive_flatten(val):
    """Recursively flatten a nested list/tuple into a single list of values."""
    if not isinstance(val, (list, tuple)):
        return [val]
    result = []
    for item in val:
        result.extend(recursive_flatten(item))
    return result

def flatten_list_columns(df: pd.DataFrame, columns_to_flatten: List[str]) -> pd.DataFrame:
    new_cols_list = []
    drop_cols = []
    for col in columns_to_flatten:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not in DataFrame, skipping flatten.")
            continue
        
        first_val = df[col].iloc[0]
        if isinstance(first_val, str):
            try:
                _ = ast.literal_eval(first_val)
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            except:
                print(f"Warning: Column '{col}' cannot be parsed as a list. Skipping.")
                continue

        if isinstance(df[col].iloc[0], (list, tuple)):
            flattened_rows = [recursive_flatten(x) for x in df[col]]
            lengths = [len(row) for row in flattened_rows]
            if len(set(lengths)) > 1:
                raise ValueError(f"Cannot flatten: Column '{col}' has varying lengths among rows.")
            n = lengths[0]
            new_col_names = [f"{col}_{i}" for i in range(n)]
            expanded_df = pd.DataFrame(flattened_rows, columns=new_col_names)
            new_cols_list.append(expanded_df)
            drop_cols.append(col)
    if new_cols_list:
        df = pd.concat([df.drop(columns=drop_cols).reset_index(drop=True)] + new_cols_list, axis=1)
    return df

###############################################################################
# 3) PyTorch Dataset
###############################################################################

class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

###############################################################################
# 4) Model Definitions
###############################################################################

# 4A. Plain MLP Surrogate Model (for unstructured tabular data)
def make_activation(name: str):
    if name == "relu":
        return nn.ReLU()
    elif name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif name == "prelu":
        return nn.PReLU()
    elif name == "rrelu":
        return nn.RReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "elu":
        return nn.ELU()
    elif name == "selu":
        return nn.SELU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "celu":
        return nn.CELU()
    elif name == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation: {name}")

def make_mlp(input_dim: int, hidden_layers: int, hidden_size: int, activation: str, output_dim: int = 1) -> nn.Sequential:
    layers = []
    in_dim = input_dim
    act_fn = make_activation(activation)
    for _ in range(hidden_layers):
        layers.append(nn.Linear(in_dim, hidden_size))
        layers.append(act_fn)
        in_dim = hidden_size
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)

# 4B. VAE + Surrogate Model for Structured Tabular Data
class TabularVAEWithSurrogate(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        # Encoder: outputs both mu and logvar (concatenated)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)
        )
        # Decoder: reconstruct input from latent z
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        # Surrogate branch: maps latent code (mu) to target
        self.surrogate = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        return self.decoder(z)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        # For surrogate prediction, we use the deterministic latent code (mu)
        target_pred = self.surrogate(mu)
        return x_recon, mu, logvar, z, target_pred

###############################################################################
# 5) Loss Functions
###############################################################################

def vae_surrogate_loss(x, x_recon, mu, logvar, pred, target, beta=1.0, gamma=1.0):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")
    KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    sur_loss = nn.functional.mse_loss(pred, target, reduction="mean")
    total_loss = recon_loss + beta * KL + gamma * sur_loss
    return total_loss, recon_loss, KL, sur_loss

###############################################################################
# 6) Main Training & Evaluation
###############################################################################

def main(args: Args) -> None:
    # A. Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # B. Weights & Biases Logging
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    data_path = os.path.join(args.data_dir, args.data_input)
    run_name = f"{args.data_input}__seed{args.seed}__{time_str}"
    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            save_code=True,
            name=run_name
        )

    # C. Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # D. Load Dataset
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"The specified data file does not exist: {data_path}")
    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(data_path)
    elif ext == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        raise ValueError("data_input must be a CSV or Parquet file.")

    if args.flatten_columns:
        df = flatten_list_columns(df, args.flatten_columns)

    final_input_cols = []
    for col in args.input_cols:
        if col in df.columns:
            final_input_cols.append(col)
        else:
            matches = [c for c in df.columns if c.startswith(col + "_")]
            if matches:
                final_input_cols.extend(matches)
    if not final_input_cols:
        raise ValueError("No valid input columns found after flattening/detection.")
    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in dataset.")

    X_all = df[final_input_cols].values
    y_all = df[args.target_col].values

    # E. Train/Val/Test Split
    X_temp, X_test, y_temp, y_test = train_test_split(X_all, y_all, test_size=args.test_size, random_state=args.seed)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=args.val_size_of_train, random_state=args.seed)

    # F. Scaling
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled   = scaler_X.transform(X_val)
    X_test_scaled  = scaler_X.transform(X_test)
    if args.scale_target:
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled   = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled  = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    else:
        y_train_scaled, y_val_scaled, y_test_scaled = y_train, y_val, y_test

    # G. DataLoaders
    train_dataset = TabularDataset(X_train_scaled, y_train_scaled)
    val_dataset   = TabularDataset(X_val_scaled, y_val_scaled)
    test_dataset  = TabularDataset(X_test_scaled, y_test_scaled)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # H. Construct Model
    input_dim = X_train_scaled.shape[1]
    if args.structured:
        # Use VAE + surrogate for structured tabular data
        model = TabularVAEWithSurrogate(input_dim=input_dim, latent_dim=args.latent_dim).to(device)
    else:
        # Use plain MLP surrogate
        model = make_mlp(input_dim=input_dim, hidden_layers=args.hidden_layers,
                         hidden_size=args.hidden_size, activation=args.activation, output_dim=1).to(device)

    # I. Choose Optimizer and Criterion
    if args.optimizer == "sgd":
        optimizer_ = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adam":
        optimizer_ = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adamw":
        optimizer_ = optim.AdamW(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "rmsprop":
        optimizer_ = optim.RMSprop(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adagrad":
        optimizer_ = optim.Adagrad(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adadelta":
        optimizer_ = optim.Adadelta(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adamax":
        optimizer_ = optim.Adamax(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "asgd":
        optimizer_ = optim.ASGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "lbfgs":
        optimizer_ = optim.LBFGS(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    if not args.structured:
        criterion = nn.MSELoss()

    # J. Training Loop with Early Stopping
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_weights = None
    train_losses = []
    val_losses = []

    for epoch in range(args.n_epochs):
        model.train()
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer_.zero_grad()
            if args.structured:
                # VAE + surrogate forward pass
                x_recon, mu, logvar, z, pred = model(X_batch)
                loss, recon_loss, KL, sur_loss = vae_surrogate_loss(X_batch, x_recon, mu, logvar, pred, y_batch, beta=args.beta, gamma=args.gamma)
            else:
                pred = model(X_batch).squeeze(-1)
                loss = nn.functional.mse_loss(pred, y_batch, reduction="mean")
            loss.backward()
            optimizer_.step()
            running_train_loss += loss.item() * X_batch.size(0)
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                if args.structured:
                    x_recon, mu, logvar, z, pred = model(X_batch)
                    loss, _, _, _ = vae_surrogate_loss(X_batch, x_recon, mu, logvar, pred, y_batch, beta=args.beta, gamma=args.gamma)
                else:
                    pred = model(X_batch).squeeze(-1)
                    loss = nn.functional.mse_loss(pred, y_batch, reduction="mean")
                running_val_loss += loss.item() * X_batch.size(0)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

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
                "epoch": epoch,
            })
        print(f"Epoch [{epoch+1}/{args.n_epochs}] - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        if epochs_no_improve >= args.patience:
            print("Early stopping triggered.")
            break

    print(f"\nBest Val Loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # K. Save Model and Plot Loss Curves
    os.makedirs(args.model_output_dir, exist_ok=True)
    if args.save_model:
        model_fname = f"best_model_{run_name}.pth"
        model_path = os.path.join(args.model_output_dir, model_fname)
        torch.save({
            "model_state_dict": model.state_dict(),
            "scaler_X": scaler_X,
            "scaler_y": scaler_y if args.scale_target else None,
        }, model_path)
        print(f"Saved best model to {model_path}")
        if args.track:
            artifact = wandb.Artifact("best_model", type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
    if args.plot_loss:
        plt.figure(figsize=(8,5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.tight_layout()
        plot_fname = f"loss_curve_{run_name}.png"
        plot_path = os.path.join(args.model_output_dir, plot_fname)
        plt.savefig(plot_path)
        print(f"Saved loss curve to {plot_path}")
        if args.track:
            wandb.log({"loss_curve": wandb.Image(plot_path)})
        plt.show()

    # L. Final Test Evaluation
    model.eval()
    test_preds_list = []
    test_targets_list = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            if args.structured:
                _, _, _, _, preds = model(X_batch)
            else:
                preds = model(X_batch).squeeze(-1)
            test_preds_list.append(preds.cpu().numpy())
            test_targets_list.append(y_batch.cpu().numpy())
    test_preds_scaled = np.concatenate(test_preds_list)
    test_targets_scaled = np.concatenate(test_targets_list)
    if args.scale_target:
        test_preds = scaler_y.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()
        test_targets = scaler_y.inverse_transform(test_targets_scaled.reshape(-1, 1)).flatten()
    else:
        test_preds = test_preds_scaled
        test_targets = test_targets_scaled
    test_mse = np.mean((test_preds - test_targets)**2)
    test_rmse = np.sqrt(test_mse)
    print(f"\nTest MSE:  {test_mse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print("\nSample predictions vs. true:")
    for i in range(min(5, len(test_preds))):
        diff = test_preds[i] - test_targets[i]
        print(f"  Pred: {test_preds[i]:.4f}, True: {test_targets[i]:.4f}, Diff: {diff:.4f}")
    if args.track:
        wandb.log({"test_mse": test_mse, "test_rmse": test_rmse})
        wandb.finish()

###############################################################################
# 7) Entry Point
###############################################################################

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)



# ---------------------------------------------------------------------------
# Usage Example
# ---------------------------------------------------------------------------
"""
python sm_vae_1D.py \
    --data_dir "../EngiOpt/data" \
    --data_input "airfoil_data.csv" \
    --input_cols '["optimized","mach","reynolds","alpha"]' \
    --target_col "cl_val" \
    --flatten_columns '["optimized"]' \
    --structured  \
    --latent_dim 32 \
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
    --seed 42 \
    --save_model \
    --model_output_dir "my_models"

"""

