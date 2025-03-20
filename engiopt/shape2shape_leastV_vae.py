#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shape-to-Shape VAE + Surrogate Example

This script trains a model that maps an initial airfoil shape 
to an optimized airfoil shape and predicts a performance metric (cl_val).
It supports:
  - Flattening list columns (e.g. for airfoil coordinates, or param columns)
  - Splitting data into train/val/test and scaling features
  - Early stopping and optional Weights & Biases tracking
  - Two modes: 
      (a) Structured mode (VAE + surrogate that uses the latent code concatenated with extra parameters)
      (b) Unstructured mode (plain MLP surrogate that uses [init + params] → performance)
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

import torch.nn.functional as F

###############################################################################
# 1) Argument Parsing
###############################################################################

@dataclass
class Args:
    data_dir: str = "./data"
    data_input: str = "airfoil_data.csv"
    init_col: str = "initial_design"
    opt_col: str = "optimal_design"
    target_col: str = "cl_val"
    # If you want certain param columns (mach, reynolds, or any list-of-lists) to be flattened,
    # add them to flatten_columns as well. E.g. '["initial_design","optimal_design","mach"]'.
    params_cols: List[str] = field(default_factory=lambda: ["mach", "reynolds"])
    flatten_columns: List[str] = field(default_factory=lambda: ["initial_design", "optimal_design"])
    structured: bool = True  
    hidden_layers: int = 2
    hidden_size: int = 64
    latent_dim: int = 32
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
    # Removed beta and beta_warmup_fraction since they were for KL divergence.
    gamma: float = 1.0
    lambda_lv: float = 1e-2   # New hyperparameter for least volume penalty weight.
    test_size: float = 0.2
    val_size_of_train: float = 0.25
    scale_target: bool = True
    track: bool = True
    wandb_project: str = "shape2shape_vae"
    wandb_entity: Optional[str] = None
    seed: int = 42
    save_model: bool = False
    plot_loss: bool = True
    model_output_dir: str = "results"
    test_model: bool = False

    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_output_dir, exist_ok=True)
        # Handle cases where --flatten_columns or --params_cols might be passed as a single string
        for field_name in ["flatten_columns", "params_cols"]:
            value = getattr(self, field_name)
            if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
                try:
                    parsed_value = ast.literal_eval(value[0])
                    if isinstance(parsed_value, list) and all(isinstance(x, str) for x in parsed_value):
                        setattr(self, field_name, parsed_value)
                except Exception:
                    raise ValueError(
                        f"Invalid format for --{field_name}: {value}. Expected a list like ['col1','col2']"
                    )

###############################################################################
# 2) Utilities
###############################################################################

def recursive_flatten(val):
    if not isinstance(val, (list, tuple)):
        return [val]
    result = []
    for item in val:
        result.extend(recursive_flatten(item))
    return result

def flatten_list_columns(df: pd.DataFrame, columns_to_flatten: List[str]) -> pd.DataFrame:
    """Recursively flattens any columns in `columns_to_flatten` that contain lists-of-lists."""
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
        # Reset index before concatenation to keep alignment
        df = pd.concat([df.drop(columns=drop_cols).reset_index(drop=True)] + new_cols_list, axis=1)
    return df

###############################################################################
# 3) Datasets
###############################################################################

class Shape2ShapeWithParamsDataset(torch.utils.data.Dataset):
    """
    Structured mode: returns (x_init, x_opt, params, target).
    """
    def __init__(self, init_array, opt_array, params_array, cl_array):
        self.X_init = torch.from_numpy(init_array).float()
        self.X_opt  = torch.from_numpy(opt_array).float()
        self.params = torch.from_numpy(params_array).float()
        self.y      = torch.from_numpy(cl_array).float()
        if self.X_init.shape[1] != self.X_opt.shape[1]:
            raise ValueError("initial_design and optimal_design must have the same dimension.")
    def __len__(self):
        return len(self.X_init)
    def __getitem__(self, idx):
        return self.X_init[idx], self.X_opt[idx], self.params[idx], self.y[idx]

class PlainTabularDataset(torch.utils.data.Dataset):
    """
    Unstructured mode: returns (X, y).
    Possibly X = [init, params] (concatenated).
    """
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

###############################################################################
# 4) Model Definitions
###############################################################################

def make_activation(name: str):
    if name == "relu":
        return nn.ReLU()
    elif name == "leakyrelu":
        return nn.LeakyReLU(0.2)
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

def make_mlp(input_dim: int, hidden_layers: int, hidden_size: int,
             activation: str, output_dim: int = 1) -> nn.Sequential:
    layers = []
    in_dim = input_dim
    act_fn = make_activation(activation)
    for _ in range(hidden_layers):
        layers.append(nn.Linear(in_dim, hidden_size))
        layers.append(act_fn)
        in_dim = hidden_size
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)

class Shape2ShapeVAE(nn.Module):
    """
    Structured mode:
      - Encode x_init → (mu, logvar)
      - Reparameterize → z
      - Decode z → x_opt_pred
      - Surrogate: [z + param_vec] → cl_pred
    """
    def __init__(self, shape_dim: int, param_dim: int, latent_dim: int = 32,
                 surrogate_hidden_layers: int = 2, surrogate_hidden_size: int = 64,
                 surrogate_activation: str = "relu"):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder outputs both mu and logvar
        self.encoder = nn.Sequential(
            nn.Linear(shape_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(latent_dim, 64)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(64, 128)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(128, shape_dim))
        )
        # Surrogate network
        self.surrogate = make_mlp(
            input_dim=(latent_dim + param_dim),
            hidden_layers=surrogate_hidden_layers,
            hidden_size=surrogate_hidden_size,
            activation=surrogate_activation,
            output_dim=1
        )

    def encode(self, x_init):
        h = self.encoder(x_init)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x_init, param_vec):
        mu, logvar = self.encode(x_init)
        z = self.reparameterize(mu, logvar)
        x_opt_pred = self.decode(z)
        # Surrogate sees [z, param_vec]
        sur_input = torch.cat([z, param_vec], dim=1)
        cl_pred = self.surrogate(sur_input)
        return x_opt_pred, mu, logvar, z, cl_pred

###############################################################################
# 5) Loss Function
###############################################################################

def least_volume_loss(z, eta=1e-3):
    """
    Computes the least volume loss as the geometric mean of the latent dimensions' std.
    """
    std_z = torch.std(z, dim=0) + eta  # avoid log(0)
    volume = torch.exp(torch.mean(torch.log(std_z)))
    return volume

def shape2shape_loss(x_opt_true, x_opt_pred, mu, logvar, z, cl_pred, cl_true,
                     lambda_lv=1e-2, gamma=1.0):
    """
    For the least volume autoencoder:
      - MSE recon on x_opt
      - Surrogate MSE on cl_val
      - plus a least-volume penalty on z
    """
    cl_pred = cl_pred.view(-1)
    cl_true = cl_true.view(-1)
    recon_loss = F.smooth_l1_loss(x_opt_pred, x_opt_true, reduction="mean")
    lv_loss = least_volume_loss(z)
    sur_loss = F.smooth_l1_loss(cl_pred, cl_true, reduction="mean")
    total_loss = recon_loss + lambda_lv * lv_loss + gamma * sur_loss
    return total_loss, recon_loss, lv_loss, sur_loss

###############################################################################
# 6) Main Training & Evaluation
###############################################################################

def main(args: Args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"The specified data file does not exist: {data_path}")
    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(data_path)
    elif ext == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        raise ValueError("data_input must be a CSV or Parquet file.")

    # 1) Flatten any columns specified
    if args.flatten_columns:
        df = flatten_list_columns(df, args.flatten_columns)
        print("After flattening, df.columns:", df.columns.tolist())

    # 2) Verify we can find init, opt, target columns
    if not any(col.startswith(args.init_col) for col in df.columns):
        raise ValueError(f"Missing required column(s) with prefix: {args.init_col}")
    if not any(col.startswith(args.opt_col) for col in df.columns):
        raise ValueError(f"Missing required column(s) with prefix: {args.opt_col}")
    if args.target_col not in df.columns:
        raise ValueError(f"Missing required target column: {args.target_col}")

    # 3) Gather flattened init & opt columns
    X_init_all = df[[c for c in df.columns if c.startswith(args.init_col + "_")]].values
    X_opt_all  = df[[c for c in df.columns if c.startswith(args.opt_col + "_")]].values
    if X_init_all.shape[1] == 0 or X_opt_all.shape[1] == 0:
        raise ValueError("Could not find flattened columns for initial or optimal designs.")

    y_all = df[args.target_col].values

    if args.structured:
        # 4) For structured mode, we require param_cols in the data
        for pcol in args.params_cols:
            if pcol not in df.columns:
                raise ValueError(f"Missing parameter column: {pcol}")
        params_all = df[args.params_cols].values

        # 5) Split into train/val/test
        Xinit_temp, Xinit_test, Xopt_temp, Xopt_test, params_temp, params_test, y_temp, y_test = train_test_split(
            X_init_all, X_opt_all, params_all, y_all,
            test_size=args.test_size, random_state=args.seed
        )
        Xinit_train, Xinit_val, Xopt_train, Xopt_val, params_train, params_val, y_train, y_val = train_test_split(
            Xinit_temp, Xopt_temp, params_temp, y_temp,
            test_size=args.val_size_of_train, random_state=args.seed
        )

        # 6) Scale each portion
        scaler_init = StandardScaler()
        scaler_opt  = StandardScaler()
        scaler_params = StandardScaler()

        Xinit_train_scaled = scaler_init.fit_transform(Xinit_train)
        Xinit_val_scaled   = scaler_init.transform(Xinit_val)
        Xinit_test_scaled  = scaler_init.transform(Xinit_test)

        Xopt_train_scaled  = scaler_opt.fit_transform(Xopt_train)
        Xopt_val_scaled    = scaler_opt.transform(Xopt_val)
        Xopt_test_scaled   = scaler_opt.transform(Xopt_test)

        params_train_scaled = scaler_params.fit_transform(params_train)
        params_val_scaled   = scaler_params.transform(params_val)
        params_test_scaled  = scaler_params.transform(params_test)

    else:
        # Unstructured MLP mode:
        # If user gave params_cols, let's horizontally stack them with X_init_all
        # so the MLP can see [init + params].
        if len(args.params_cols) > 0:
            missing_params = [p for p in args.params_cols if p not in df.columns]
            if missing_params:
                raise ValueError(f"Missing param column(s) for unstructured mode: {missing_params}")
            # Extract param columns
            params_all = df[args.params_cols].values
            # Combine them horizontally with X_init
            X_features_all = np.hstack([X_init_all, params_all])
        else:
            X_features_all = X_init_all

        X_temp, X_test, y_temp, y_test = train_test_split(
            X_features_all, y_all,
            test_size=args.test_size, random_state=args.seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=args.val_size_of_train, random_state=args.seed
        )

        # Scale
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled   = scaler_X.transform(X_val)
        X_test_scaled  = scaler_X.transform(X_test)

    # Scale target if requested
    if args.scale_target:
        scaler_y = StandardScaler()
        if args.structured:
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()
            y_val_scaled   = scaler_y.transform(y_val.reshape(-1,1)).flatten()
            y_test_scaled  = scaler_y.transform(y_test.reshape(-1,1)).flatten()
        else:
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()
            y_val_scaled   = scaler_y.transform(y_val.reshape(-1,1)).flatten()
            y_test_scaled  = scaler_y.transform(y_test.reshape(-1,1)).flatten()
    else:
        y_train_scaled, y_val_scaled, y_test_scaled = y_train, y_val, y_test

    # Build datasets
    if args.structured:
        train_dataset = Shape2ShapeWithParamsDataset(Xinit_train_scaled, Xopt_train_scaled,
                                                     params_train_scaled, y_train_scaled)
        val_dataset   = Shape2ShapeWithParamsDataset(Xinit_val_scaled, Xopt_val_scaled,
                                                     params_val_scaled, y_val_scaled)
        test_dataset  = Shape2ShapeWithParamsDataset(Xinit_test_scaled, Xopt_test_scaled,
                                                     params_test_scaled, y_test_scaled)
    else:
        train_dataset = PlainTabularDataset(X_train_scaled, y_train_scaled)
        val_dataset   = PlainTabularDataset(X_val_scaled,   y_val_scaled)
        test_dataset  = PlainTabularDataset(X_test_scaled,  y_test_scaled)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    # Build model
    if args.structured:
        shape_dim = Xinit_train_scaled.shape[1]
        param_dim = params_train_scaled.shape[1]
        model = Shape2ShapeVAE(
            shape_dim=shape_dim,
            param_dim=param_dim,
            latent_dim=args.latent_dim,
            surrogate_hidden_layers=args.hidden_layers,
            surrogate_hidden_size=args.hidden_size,
            surrogate_activation=args.activation
        ).to(device)
    else:
        # input_dim is [init + params] if user specified params_cols
        if len(args.params_cols) > 0:
            input_dim = X_train_scaled.shape[1]  # after the horizontal stack
        else:
            input_dim = X_train_scaled.shape[1]  # just init shape
        model = make_mlp(
            input_dim=input_dim,
            hidden_layers=args.hidden_layers,
            hidden_size=args.hidden_size,
            activation=args.activation,
            output_dim=1
        ).to(device)

    # Build optimizer
    def make_optimizer(name: str, params, lr: float):
        if name == "sgd":
            return optim.SGD(params, lr=lr)
        elif name == "adam":
            return optim.Adam(params, lr=lr)
        elif name == "adamw":
            return optim.AdamW(params, lr=lr)
        elif name == "rmsprop":
            return optim.RMSprop(params, lr=lr)
        elif name == "adagrad":
            return optim.Adagrad(params, lr=lr)
        elif name == "adadelta":
            return optim.Adadelta(params, lr=lr)
        elif name == "adamax":
            return optim.Adamax(params, lr=lr)
        elif name == "asgd":
            return optim.ASGD(params, lr=lr)
        elif name == "lbfgs":
            return optim.LBFGS(params, lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {name}")

    optimizer_ = make_optimizer(args.optimizer, model.parameters(), args.learning_rate)
    mse_criterion = nn.MSELoss()

    ############################################################################
    # Training Loop
    ############################################################################
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_weights = None
    train_losses = []
    val_losses = []

    for epoch in range(args.n_epochs):
        model.train()
        running_train_loss = 0.0

        if args.structured:
            # Structured mode: shape2shape_loss
            for x_init_batch, x_opt_batch, param_batch, cl_batch in train_loader:
                x_init_batch = x_init_batch.to(device)
                x_opt_batch  = x_opt_batch.to(device)
                param_batch  = param_batch.to(device)
                cl_batch     = cl_batch.to(device)
                optimizer_.zero_grad()

                x_opt_pred, mu, logvar, z, cl_pred = model(x_init_batch, param_batch)
                loss, recon_loss, lv_loss, sur_loss = shape2shape_loss(
                    x_opt_true=x_opt_batch,
                    x_opt_pred=x_opt_pred,
                    mu=mu,
                    logvar=logvar,
                    z=z,
                    cl_pred=cl_pred,
                    cl_true=cl_batch,
                    lambda_lv=args.lambda_lv,
                    gamma=args.gamma
                )
                loss.backward()
                optimizer_.step()
                running_train_loss += loss.item() * x_init_batch.size(0)

        else:
            # Unstructured mode: plain MSE for predicted cl_val
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer_.zero_grad()
                preds = model(X_batch).squeeze(-1)
                loss = mse_criterion(preds, y_batch)
                loss.backward()
                optimizer_.step()
                running_train_loss += loss.item() * X_batch.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            if args.structured:
                for x_init_batch, x_opt_batch, param_batch, cl_batch in val_loader:
                    x_init_batch = x_init_batch.to(device)
                    x_opt_batch  = x_opt_batch.to(device)
                    param_batch  = param_batch.to(device)
                    cl_batch     = cl_batch.to(device)

                    x_opt_pred, mu, logvar, z, cl_pred = model(x_init_batch, param_batch)
                    val_loss, _, _, _ = shape2shape_loss(
                        x_opt_true=x_opt_batch,
                        x_opt_pred=x_opt_pred,
                        mu=mu,
                        logvar=logvar,
                        z=z,
                        cl_pred=cl_pred,
                        cl_true=cl_batch,
                        lambda_lv=args.lambda_lv,
                        gamma=args.gamma
                    )
                    running_val_loss += val_loss.item() * x_init_batch.size(0)
            else:
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    preds = model(X_batch).squeeze(-1)
                    loss = mse_criterion(preds, y_batch)
                    running_val_loss += loss.item() * X_batch.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # Early stopping logic
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

        print(f"Epoch [{epoch+1}/{args.n_epochs}] - "
              f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        if epochs_no_improve >= args.patience:
            print("Early stopping triggered.")
            break

    print(f"\nBest Val Loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Save model if requested
    if args.save_model:
        model_fname = f"best_model_{run_name}.pth"
        model_path = os.path.join(args.model_output_dir, model_fname)
        save_dict = {"model_state_dict": model.state_dict()}
        if args.structured:
            save_dict["scaler_init"] = scaler_init
            save_dict["scaler_opt"] = scaler_opt
            save_dict["scaler_params"] = scaler_params
        else:
            save_dict["scaler_X"] = scaler_X
        if args.scale_target:
            save_dict["scaler_y"] = scaler_y
        torch.save(save_dict, model_path)
        print(f"Saved best model to {model_path}")
        if args.track:
            artifact = wandb.Artifact("best_model", type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

    # Plot loss curves
    if args.plot_loss:
        plt.figure(figsize=(8, 5))
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

    # ----------------- Visualization & Final Test Evaluation -----------------
    if args.structured:
        # Show a few samples of init vs. predicted/true optimized shape
        model.eval()
        num_samples = 3
        sample_indices = np.random.choice(len(test_dataset), num_samples, replace=False)

        def invert_and_pair(scaler, tensor):
            """
            Helper to re-invert a flattened shape from [x0, x1, ..., y0, y1, ...]
            and return Nx2 array of coordinate pairs.
            """
            arr = scaler.inverse_transform(tensor.cpu().numpy().reshape(1, -1)).flatten()
            n = arr.shape[0] // 2
            x_coords = arr[:n]
            y_coords = arr[n:]
            return np.column_stack((x_coords, y_coords))
        
        for idx in sample_indices:
            x_init, x_opt, params, cl_true = test_dataset[idx]
            x_init_batch = x_init.unsqueeze(0).to(device)
            params_batch = params.unsqueeze(0).to(device)

            with torch.no_grad():
                x_opt_pred, mu, logvar, z, cl_pred = model(x_init_batch, params_batch)

            x_init_np = invert_and_pair(scaler_init, x_init)
            x_opt_np = invert_and_pair(scaler_opt, x_opt)
            x_opt_pred_np = invert_and_pair(scaler_opt, x_opt_pred.squeeze(0))

            cl_pred_np = cl_pred.detach().cpu().numpy().reshape(-1,1)
            if args.scale_target:
                cl_pred_unscaled = scaler_y.inverse_transform(cl_pred_np).flatten()[0]
            else:
                cl_pred_unscaled = cl_pred_np.flatten()[0]
            cl_true_unscaled = cl_true  # if scale_target, it is already scaled, but we can unscale if needed

            fig, ax = plt.subplots(figsize=(6,6))
            ax.plot(x_init_np[:,0], x_init_np[:,1], 'bo-', label='Initial Airfoil')
            ax.plot(x_opt_np[:,0], x_opt_np[:,1], 'go-', label='True Optimized Airfoil')
            ax.plot(x_opt_pred_np[:,0], x_opt_pred_np[:,1], 'r--', label='Reconstructed Optimized Airfoil')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"Sample {idx} (cl_val: true={float(cl_true_unscaled):.2f}, "
                         f"pred={float(cl_pred_unscaled):.2f})")
            ax.legend()
            fig_filename = os.path.join(args.model_output_dir, f"sample_{idx}_overlay.png")
            fig.savefig(fig_filename)
            plt.close(fig)
            print(f"Saved overlay figure for sample {idx} to {fig_filename}")

    # Final test predictions
    if args.test_model:
        model.eval()
        test_preds = []
        test_trues = []
        with torch.no_grad():
            if args.structured:
                # Evaluate surrogate
                for x_init_batch, x_opt_batch, param_batch, cl_batch in test_loader:
                    x_init_batch = x_init_batch.to(device)
                    param_batch  = param_batch.to(device)
                    cl_batch     = cl_batch.to(device)

                    _, mu, logvar, z, cl_pred = model(x_init_batch, param_batch)
                    cl_pred_np = cl_pred.cpu().numpy().reshape(-1, 1)
                    if args.scale_target:
                        cl_pred_unscaled = scaler_y.inverse_transform(cl_pred_np).flatten()
                    else:
                        cl_pred_unscaled = cl_pred_np.flatten()

                    cl_true_np = cl_batch.cpu().numpy().reshape(-1, 1)
                    if args.scale_target:
                        cl_true_unscaled = scaler_y.inverse_transform(cl_true_np).flatten()
                    else:
                        cl_true_unscaled = cl_true_np.flatten()

                    test_preds.extend(cl_pred_unscaled)
                    test_trues.extend(cl_true_unscaled)
            else:
                # Plain MLP
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    preds = model(X_batch).squeeze(-1)

                    cl_pred_np = preds.cpu().numpy().reshape(-1, 1)
                    if args.scale_target:
                        cl_pred_unscaled = scaler_y.inverse_transform(cl_pred_np).flatten()
                    else:
                        cl_pred_unscaled = cl_pred_np.flatten()

                    cl_true_np = y_batch.cpu().numpy().reshape(-1, 1)
                    if args.scale_target:
                        cl_true_unscaled = scaler_y.inverse_transform(cl_true_np).flatten()
                    else:
                        cl_true_unscaled = cl_true_np.flatten()

                    test_preds.extend(cl_pred_unscaled)
                    test_trues.extend(cl_true_unscaled)

        print("\nTest Predictions vs. True (first 10):")
        for i, (p, t) in enumerate(zip(test_preds, test_trues)):
            if i < 10:
                print(f"  Sample {i:3d} → Predicted={p:.4f}, True={t:.4f}")

    if args.track:
        wandb.finish()

###############################################################################
# 7) Entry Point
###############################################################################

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
