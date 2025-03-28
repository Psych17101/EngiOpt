#!/usr/bin/env python

"""Shape-to-Shape VAE + Surrogate Model Training and Evaluation.

This script demonstrates a multi-stage pipeline for training a Variational Autoencoder (VAE) 
on shape-based design data, optionally followed by a surrogate model to predict a target 
metric (e.g., lift coefficient). It can also operate in a "plain MLP" mode where shape 
information is absent.

Main Features:
  1. **Data Preprocessing**:
     - Reads CSV or Parquet data containing shape and/or tabular columns.
     - Splits data into train, validation, and test sets.
     - Optionally scales inputs and target values using robust scalers.

  2. **Flexible Model Architecture**:
     - Structured mode (Shape2ShapeVAE): Encodes initial shape into a latent code, 
       decodes to predict an optimized shape, and uses a surrogate network for target prediction.
     - Plain MLP: Trains a simple feedforward neural network on tabular data only.

  3. **Hyperparameter Control**:
     - Parameters such as hidden size, number of layers, choice of optimizer, learning rate, 
       etc. are specified via command-line arguments (using Tyro CLI).

  4. **Ensembling**:
     - The `seed` argument can be a list of integers; each seed trains a separate instance of 
       the model. At inference time, predictions are averaged to produce an ensemble result.

  5. **Tracking & Visualization**:
     - If `track` is enabled, logs training/validation metrics to Weights & Biases (wandb).
     - Optionally plots and saves training curves, predicted vs. true scatter plots, 
       and shape overlay comparisons (structured mode only).

  6. **Usage**:
     - Invoke via `python script_name.py --help` for command-line options.
     - Typical arguments include data file paths, hyperparameters, and boolean flags 
       for structured vs. unstructured modes.

Key Steps:
  - Parse command-line arguments.
  - Read and preprocess data (flatten lists, scale features).
  - Build PyTorch datasets and dataloaders for train/val/test.
  - Train and validate model(s) for one or multiple random seeds.
  - If multiple seeds are provided, ensemble predictions are produced.
  - Save final pipeline (model + scalers) and generate optional plots.

Note:
  - Requires `pandas`, `numpy`, `torch`, `matplotlib`, `tyro`, `wandb`, and `sklearn`.
  - Expects a consistent format for shape columns if `structured` is enabled.

"""

# Standard library imports
import ast
from dataclasses import dataclass
from dataclasses import field
import os
import random
import time
from typing import List, Literal, Optional, Tuple

from matplotlib import pyplot as plt
from model_pipeline import DataPreprocessor

# Local application/library specific imports
from model_pipeline import ModelPipeline

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import tyro

import wandb


###############################################################################
# 1) Argument Parsing
###############################################################################
@dataclass
class Args:
    data_dir: str = "./data"
    data_input: str = "airfoil_data.csv"

    # If your data has shape columns, set these. If no shape, leave them empty.
    init_col: str = "initial_design"
    opt_col: str = "optimal_design"

    target_col: str = "cl_val"
    log_target: bool = False

    # Param columns (some might be continuous, some might have <5 unique => one-hot).
    params_cols: List[str] = field(default_factory=lambda: ["mach", "reynolds"])

    # Columns to flatten if they’re lists
    flatten_columns: List[str] = field(default_factory=lambda: ["initial_design", "optimal_design"])
    strip_column_spaces: bool = False
    subset_condition: Optional[str] = None  # e.g. "r > 0"

    nondim_map: Optional[str] = None  # e.g. '{"colA":"refColA","colB":"refColB"}'

    split_random_state: int = 999

    # VAE + surrogate vs. plain MLP
    structured: bool = True
    hidden_layers: int = 2
    hidden_size: int = 64
    latent_dim: int = 32
    activation: Literal[
        "relu","leakyrelu","prelu","rrelu","tanh",
        "sigmoid","elu","selu","gelu","celu","none"
    ] = "relu"
    optimizer: Literal[
        "sgd","adam","adamw","rmsprop","adagrad",
        "adadelta","adamax","asgd","lbfgs"
    ] = "adam"
    learning_rate: float = 1e-3

    # LR scheduler
    lr_decay: float = 1.0
    lr_decay_step: int = 1

    n_epochs: int = 50
    batch_size: int = 32
    patience: int = 10
    l2_lambda: float = 1e-3

    gamma: float = 1.0
    lambda_lv: float = 1e-2

    test_size: float = 0.2
    val_size_of_train: float = 0.25
    scale_target: bool = True

    track: bool = True
    wandb_project: str = "vae_mlp_multimodal"
    wandb_entity: Optional[str] = None

    seed: List[int] = field(default_factory=lambda: [42])
    save_model: bool = False
    plot_loss: bool = True
    model_output_dir: str = "results"
    test_model: bool = False

    def __post_init__(self):
        # If user passed a single int for seed, convert to list
        if isinstance(self.seed, int):
            self.seed = [self.seed]

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_output_dir, exist_ok=True)

        # For "params_cols" and "flatten_columns", parse them if they were passed as strings
        for field_name in ["flatten_columns", "params_cols"]:
            value = getattr(self, field_name)
            if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
                try:
                    parsed_value = ast.literal_eval(value[0])
                    if isinstance(parsed_value, list) and all(isinstance(x,str) for x in parsed_value):
                        setattr(self, field_name, parsed_value)
                except Exception:
                    raise ValueError(f"Invalid format for --{field_name}: {value}")

###############################################################################
# 2) Utilities
###############################################################################
# e.g. flatten_list_columns, etc., if you want them. But we rely on DataPreprocessor now.
def compute_l2_penalty(model: nn.Module) -> torch.Tensor:
    """Computes the L2 penalty (sum of squared parameters) for a given model.

    This is often added to the total loss to encourage smaller parameter magnitudes 
    (i.e., L2 regularization or weight decay).

    Args:
        model (nn.Module): The PyTorch model for which to compute the L2 penalty.

    Returns:
        torch.Tensor: A scalar tensor representing the sum of the squared parameter values.
    """
    device = next(model.parameters()).device
    l2_norm = torch.zeros(1, device=device)
    for param in model.parameters():
        l2_norm += (param ** 2).sum()
    return l2_norm


def invert_and_pair(scaler, tensor: torch.Tensor) -> np.ndarray:
    """Inverts scaling for a 1D flattened shape array and splits it into (x, y) coordinates.

    This is used primarily for visualizing shapes in 2D after they've been scaled 
    and flattened for the model input.

    Args:
        scaler: A fitted scaler (e.g., sklearn.preprocessing.RobustScaler) with 
            an `inverse_transform` method.
        tensor (torch.Tensor): A 1D tensor representing flattened shape coordinates.

    Returns:
        np.ndarray: A 2D array of shape (n_points, 2) where each row is an (x, y) pair.
    """
    arr = scaler.inverse_transform(tensor.cpu().numpy().reshape(1, -1)).flatten()
    n = arr.shape[0] // 2
    x_coords = arr[:n]
    y_coords = arr[n:]
    return np.column_stack((x_coords, y_coords))

###############################################################################
# 3) Datasets
###############################################################################
class Shape2ShapeWithParamsDataset(torch.utils.data.Dataset):
    """Dataset for structured shape data with parameters and target values.

    Attributes:
        X_init: Initial design shapes.
        X_opt: Optimized design shapes.
        params: Additional parameter values.
        y: Target values.
    """
    def __init__(self, init_array, opt_array, params_array, cl_array):
        self.X_init = torch.from_numpy(init_array).float()
        self.X_opt  = torch.from_numpy(opt_array).float()
        self.params = torch.from_numpy(params_array).float()
        self.y      = torch.from_numpy(cl_array).float()
        if self.X_init.shape[1] != self.X_opt.shape[1]:
            raise ValueError("init_design and opt_design must have same dimension.")
    def __len__(self) -> int:
        return len(self.X_init)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_init[idx], self.X_opt[idx], self.params[idx], self.y[idx]

class PlainTabularDataset(torch.utils.data.Dataset):
    """A simple dataset for plain tabular data (no shape columns).

    Args:
        X: NumPy array of input features.
        y: NumPy array of target values.
    """
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self) -> int:
        return len(self.X)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

###############################################################################
# 4) Model Definitions
###############################################################################

def make_activation(name: str) -> nn.Module:
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

class HybridSurrogate(nn.Module):
    """A hybrid surrogate network that processes continuous parameters and categorical
    (one-hot encoded) parameters separately, then combines them with the latent code.
    """
    def __init__(self, latent_dim: int, cont_dim: int, cat_dim: int,
                 hidden_layers: int = 2, hidden_size: int = 64, activation: str = "relu"):
        super().__init__()
        act_fn = make_activation(activation)
        # Continuous branch: process latent + continuous parameters
        in_dim_cont = latent_dim + cont_dim
        cont_layers = []
        for _ in range(hidden_layers):
            cont_layers.append(nn.Linear(in_dim_cont, hidden_size))
            cont_layers.append(act_fn)
            in_dim_cont = hidden_size
        self.cont_net = nn.Sequential(*cont_layers)
        # Categorical branch: process categorical parameters
        if cat_dim > 0:
            cat_layers = []
            in_dim_cat = cat_dim
            for _ in range(hidden_layers):
                cat_layers.append(nn.Linear(in_dim_cat, hidden_size))
                cat_layers.append(act_fn)
                in_dim_cat = hidden_size
            self.cat_net = nn.Sequential(*cat_layers)
        else:
            self.cat_net = None
        # Combined branch: combine outputs from both branches
        combined_in = hidden_size * (2 if self.cat_net is not None else 1)
        self.combined = nn.Sequential(
            nn.Linear(combined_in, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1)
        )
    def forward(self, latent: torch.Tensor, cont_params: torch.Tensor, cat_params: torch.Tensor):
        # Process continuous branch (concatenate latent and continuous parameters)
        cont_out = self.cont_net(torch.cat([latent, cont_params], dim=1))
        if self.cat_net is not None:
            cat_out = self.cat_net(cat_params)
            combined = torch.cat([cont_out, cat_out], dim=1)
        else:
            combined = cont_out
        return self.combined(combined)

class Shape2ShapeVAE(nn.Module):
    """Structured mode VAE with surrogate.
    If categorical parameters are present (i.e. cat_dim > 0), a hybrid surrogate is used.
    """
    def __init__(self, shape_dim: int, cont_dim: int, cat_dim: int, latent_dim: int = 32,
                 surrogate_hidden_layers: int = 2, surrogate_hidden_size: int = 64,
                 surrogate_activation: str = "relu"):
        super().__init__()
        self.latent_dim = latent_dim
        self.cont_dim = cont_dim
        self.cat_dim = cat_dim
        # Encoder
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
        # Surrogate: if categorical parameters exist, use the hybrid surrogate.
        if self.cat_dim > 0:
            self.use_hybrid = True
            self.surrogate = HybridSurrogate(latent_dim, self.cont_dim, self.cat_dim,
                                             hidden_layers=surrogate_hidden_layers,
                                             hidden_size=surrogate_hidden_size,
                                             activation=surrogate_activation)
        else:
            self.use_hybrid = False
            self.surrogate = make_mlp(input_dim=latent_dim + self.cont_dim,
                                       hidden_layers=surrogate_hidden_layers,
                                       hidden_size=surrogate_hidden_size,
                                       activation=surrogate_activation,
                                       output_dim=1)
    def encode(self, x_init: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x_init)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    def forward(self, x_init: torch.Tensor, param_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x_init)
        z = self.reparameterize(mu, logvar)
        x_opt_pred = self.decode(z)
        # In structured mode, param_vec is of shape (batch, total) where
        # total = cont_dim + cat_dim. If hybrid, split them.
        if self.use_hybrid:
            cont_params = param_vec[:, :self.cont_dim]
            cat_params = param_vec[:, self.cont_dim:]
            cl_pred = self.surrogate(z, cont_params, cat_params)
        else:
            cl_pred = self.surrogate(torch.cat([z, param_vec], dim=1))
        return x_opt_pred, mu, logvar, z, cl_pred

###############################################################################
# 5) Loss Function
###############################################################################

def least_volume_loss(z: torch.Tensor, eta: float = 1e-3) -> torch.Tensor:
    """Computes a 'least volume' regularization term for the latent code.

    The idea is to encourage the latent space to be compact by calculating 
    an approximate volume based on the standard deviations of each latent dimension.

    Args:
        z (torch.Tensor): The latent code tensor, typically shape (batch_size, latent_dim).
        eta (float, optional): A small constant to avoid taking the log of zero. Defaults to 1e-3.

    Returns:
        torch.Tensor: A scalar tensor representing the least-volume term.
    """
    std_z = torch.std(z, dim=0) + eta
    volume = torch.exp(torch.mean(torch.log(std_z)))
    return volume

def shape2shape_loss(x_opt_true: torch.Tensor, x_opt_pred: torch.Tensor,
                     mu: torch.Tensor, logvar: torch.Tensor, z: torch.Tensor,
                     cl_pred: torch.Tensor, cl_true: torch.Tensor,
                     lambda_lv: float = 1e-2, gamma: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the total loss for the Shape2ShapeVAE, including:
      - Reconstruction loss between the true optimized shape and the predicted shape.
      - A 'least volume' regularization term for the latent space.
      - A surrogate loss between the predicted target and true target.
      - Optionally scaled by `gamma` for the surrogate and `lambda_lv` for the latent volume term.

    Args:
        x_opt_true (torch.Tensor): Ground-truth optimized shape, shape (batch_size, shape_dim).
        x_opt_pred (torch.Tensor): Predicted optimized shape from the VAE decoder.
        mu (torch.Tensor): Mean of the latent distribution from the encoder.
        logvar (torch.Tensor): Log-variance of the latent distribution from the encoder.
        z (torch.Tensor): Latent code sampled via reparameterization.
        cl_pred (torch.Tensor): Predicted target metric (e.g., lift coefficient).
        cl_true (torch.Tensor): True target metric.
        lambda_lv (float, optional): Scaling factor for the least-volume term. Defaults to 1e-2.
        gamma (float, optional): Scaling factor for the surrogate loss. Defaults to 1.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        - total_loss: Sum of reconstruction, least-volume, and surrogate losses (with scaling).
        - recon_loss: The MSE reconstruction loss for shape prediction.
        - lv_loss: The least-volume loss on the latent space.
        - sur_loss: The MSE loss between predicted and true target values.
    """
    cl_pred = cl_pred.view(-1)
    cl_true = cl_true.view(-1)
    recon_loss = F.mse_loss(x_opt_pred, x_opt_true, reduction="mean")
    lv_loss    = least_volume_loss(z)
    sur_loss   = F.mse_loss(cl_pred, cl_true, reduction="mean")
    total_loss = recon_loss + lambda_lv * lv_loss + gamma * sur_loss
    return total_loss, recon_loss, lv_loss, sur_loss

###############################################################################
# 6) Train/Eval helpers
###############################################################################

def make_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    """Extend or modify as needed if you want more optimizer options.
    For brevity, only demonstrates 'sgd' and 'adam'.
    """
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

###############################################################################
# 7) Main Training & Evaluation
###############################################################################

def train_one_model(args: Args,
                    train_loader: torch.utils.data.DataLoader,
                    val_loader: torch.utils.data.DataLoader,
                    have_shape_cols: bool,
                    shape_dim: Optional[int],
                    num_cont: int,
                    num_cat: int,
                    device: torch.device) -> Tuple[nn.Module, Tuple[List[float], List[float]], float]:
    """Given the train/val data loaders, create and train one model,
    returning the best model (state dict) and training curves.
    """
    if have_shape_cols and args.structured:
        # Shape2Shape + Surrogate
        model = Shape2ShapeVAE(
            shape_dim=shape_dim,
            cont_dim=num_cont,
            cat_dim=num_cat,
            latent_dim=args.latent_dim,
            surrogate_hidden_layers=args.hidden_layers,
            surrogate_hidden_size=args.hidden_size,
            surrogate_activation=args.activation
        ).to(device)

        def train_step_vae_mlp(x_init_batch, x_opt_batch, param_batch, cl_batch):
            """Performs a single training step for the Shape2ShapeVAE + surrogate model.

            1) Moves the batch tensors to the specified device.
            2) Runs a forward pass to get reconstructed shape and predicted target.
            3) Calculates shape reconstruction loss, least-volume loss, and surrogate loss.
            4) Adds L2 penalty to the total loss.
            5) Returns the total loss and intermediate losses for logging.

            Args:
                x_init_batch (torch.Tensor): Initial shape batch.
                x_opt_batch (torch.Tensor): True optimized shape batch.
                param_batch (torch.Tensor): Parameter batch (continuous and/or categorical).
                cl_batch (torch.Tensor): True target batch.

            Returns:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - total_loss: Combined loss including reconstruction, surrogate, and L2 penalty.
                - recon_loss: Shape reconstruction loss.
                - lv_loss: Least-volume loss in latent space.
                - sur_loss: Surrogate MSE between predicted and true targets.
            """
            x_init_batch = x_init_batch.to(device)
            x_opt_batch  = x_opt_batch.to(device)
            param_batch  = param_batch.to(device)
            cl_batch     = cl_batch.to(device)
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
            # Add L2 penalty from all model parameters
            l2_pen = compute_l2_penalty(model)
            loss = loss + args.l2_lambda * l2_pen
            return loss, recon_loss, lv_loss, sur_loss

    else:
        # Plain MLP
        # Determine input_dim from any sample
        input_dim = next(iter(train_loader))[0].shape[1]
        model = make_mlp(
            input_dim=input_dim,
            hidden_layers=args.hidden_layers,
            hidden_size=args.hidden_size,
            activation=args.activation,
            output_dim=1
        ).to(device)

        def train_step_mlp(X_batch, y_batch):
            """Performs a single training step for the plain MLP model.

            1) Moves the batch tensors to the specified device.
            2) Runs a forward pass to get predicted target.
            3) Calculates MSE loss.
            4) Adds L2 penalty to the total loss.
            5) Returns the total loss and placeholder zeros for other metrics (which don't apply here).

            Args:
                X_batch (torch.Tensor): Input features for the MLP.
                y_batch (torch.Tensor): True target values.

            Returns:
                Tuple[torch.Tensor, int, int, int]:
                - total_loss: MSE loss plus L2 penalty.
                - 0, 0, 0: Placeholders for reconstruction, least-volume, and surrogate losses (unused here).
            """
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch).squeeze(-1)
            loss = F.mse_loss(preds, y_batch)
            l2_pen = compute_l2_penalty(model)
            loss = loss + args.l2_lambda * l2_pen
            return loss, 0, 0, 0

    optimizer_ = make_optimizer(args.optimizer, model.parameters(), args.learning_rate)

    # Create the learning rate scheduler (if lr_decay < 1.0, decay will occur)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_, gamma=args.lr_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_weights = None
    train_losses = []
    val_losses = []

    for epoch in range(args.n_epochs):
        model.train()
        running_train_loss = 0.0
        n_train = len(train_loader.dataset)

        for batch_data in train_loader:
            optimizer_.zero_grad()
            if have_shape_cols and args.structured:
                (x_init_batch, x_opt_batch, param_batch, cl_batch) = batch_data
                loss, recon_loss, lv_loss, sur_loss = train_step_vae_mlp(x_init_batch, x_opt_batch, param_batch, cl_batch)
            else:
                (X_batch, y_batch) = batch_data
                loss, _, _, _ = train_step_mlp(X_batch, y_batch)

            batch_size_ = len(batch_data[0])
            loss.backward()
            optimizer_.step()
            running_train_loss += loss.item() * batch_size_

        epoch_train_loss = running_train_loss / n_train
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        n_val = len(val_loader.dataset)
        with torch.no_grad():
            for batch_data in val_loader:
                if have_shape_cols and args.structured:
                    (x_init_batch, x_opt_batch, param_batch, cl_batch) = batch_data
                    loss_val, _, _, _ = train_step_vae_mlp(x_init_batch, x_opt_batch, param_batch, cl_batch)
                else:
                    (X_batch, y_batch) = batch_data
                    loss_val, _, _, _ = train_step_mlp(X_batch, y_batch)

                batch_size_ = len(batch_data[0])
                running_val_loss += loss_val.item() * batch_size_

        epoch_val_loss = running_val_loss / n_val
        val_losses.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_weights = model.state_dict()
        else:
            epochs_no_improve += 1

        # End of epoch: update learning rate if needed.
        if (epoch + 1) % args.lr_decay_step == 0:
            scheduler.step()

        if args.track:
            wandb.log({
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "epoch": epoch,
            })

        print(f"[Epoch {epoch+1}/{args.n_epochs}] Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        if epochs_no_improve >= args.patience:
            print("Early stopping triggered.")
            break

    print(f"Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, (train_losses, val_losses), best_val_loss

###############################################################################
# 8) The main function
###############################################################################
def main(args: Args) -> float:
    """Orchestrates the entire workflow: data loading, preprocessing, model training,
    evaluation, optional ensembling, and final saving of the pipeline.

    Steps:
        1) Initializes Weights & Biases (wandb) run if tracking is enabled.
        2) Selects and reports the available computation device (CPU, CUDA, MPS, etc.).
        3) Loads CSV or Parquet data, then applies data preprocessing/flattening.
        4) Splits data into train/val/test sets, fits scalers on training data, and builds Datasets/Dataloaders.
        5) Trains one or multiple models depending on the list of seeds.
        6) Ensembles the models if multiple seeds are provided.
        7) Saves the final pipeline and optionally plots training curves.
        8) (Structured mode only) Visualizes shape predictions for a few random samples.
        9) Optionally evaluates the trained ensemble on the test set to produce final predictions.

    Args:
        args (Args): Configuration dataclass instance containing file paths, hyperparameters,
            and other settings.

    Raises:
        FileNotFoundError: If the specified data file cannot be found.
        ValueError: If invalid data format is encountered or columns are missing.

    Returns:
        float: The best validation loss achieved by the ensemble of models.
    """
    # 1) Possibly do wandb.init
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    data_path = os.path.join(args.data_dir, args.data_input)
    run_name = f"{args.data_input}__{time_str}"

    if args.track:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args),
                   save_code=True, name=run_name)

    # 2) Choose device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 3) Read the raw CSV/Parquet
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"{data_path} does not exist")
    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(data_path)
    elif ext == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        raise ValueError("data_input must be CSV or Parquet")

    print(df.head())

    # 4) Create DataPreprocessor, transform the entire dataset in training mode
    preprocessor = DataPreprocessor(vars(args))
    processed_dict, df = preprocessor.transform_inputs(df, fit_params=True)

    # 5) Check if shape columns exist
    have_shape_cols = (
        args.init_col != "" and args.opt_col != "" and
        any(c.startswith(args.init_col + "_") for c in df.columns) and
        any(c.startswith(args.opt_col + "_") for c in df.columns)
    )
    if args.structured and not have_shape_cols:
        raise ValueError("Structured mode but no shape columns found. Check your init_col/opt_col settings.")

    if args.target_col not in df.columns:
        raise ValueError(f"Missing target_col in DataFrame: {args.target_col}")
    y_all = df[args.target_col].values

    # 6) Build entire arrays for shape & params (structured) or X (unstructured)
    if have_shape_cols and args.structured:
        X_init_all = processed_dict["x_init"]
        X_opt_all  = processed_dict["x_opt"]
        params_all = processed_dict["params"]

        # We'll determine cont/cat dimension from param_all's shape:
        # but to find how many columns were one-hot, you'd look at param_df shape, etc.
        # For clarity, let's just set these later once we know the shape.

    else:
        # unstructured MLP
        X_features_all = processed_dict["X"]

    # 7) Train/Val/Test split
    #    The user’s code does a test split first, then a val split from the remainder.
    split_random_state = args.split_random_state

    if have_shape_cols and args.structured:
        # shape + param
        Xinit_temp, Xinit_test, Xopt_temp, Xopt_test, params_temp, params_test, y_temp, y_test = \
            train_test_split(X_init_all, X_opt_all, params_all, y_all,
                             test_size=args.test_size, random_state=split_random_state)
        Xinit_train, Xinit_val, Xopt_train, Xopt_val, params_train, params_val, y_train, y_val = \
            train_test_split(Xinit_temp, Xopt_temp, params_temp, y_temp,
                             test_size=args.val_size_of_train, random_state=split_random_state)
    else:
        # unstructured
        X_temp, X_test, y_temp, y_test = train_test_split(X_features_all, y_all,
                                                          test_size=args.test_size,
                                                          random_state=split_random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp,
                                                          test_size=args.val_size_of_train,
                                                          random_state=split_random_state)

    # 8) Fit scalers on training, transform train/val/test
    #    We'll store them in a dict (custom_scalers) to pass to the pipeline.

    custom_scalers = {}
    if have_shape_cols and args.structured:
        # shape-based approach
        # 8a) robust-scaler for x_init, x_opt, param
        scaler_init = RobustScaler()
        scaler_opt  = RobustScaler()
        scaler_params = RobustScaler()

        Xinit_train_s = scaler_init.fit_transform(Xinit_train)
        Xinit_val_s   = scaler_init.transform(Xinit_val)
        Xinit_test_s  = scaler_init.transform(Xinit_test)

        Xopt_train_s  = scaler_opt.fit_transform(Xopt_train)
        Xopt_val_s    = scaler_opt.transform(Xopt_val)
        Xopt_test_s   = scaler_opt.transform(Xopt_test)

        params_train_s = scaler_params.fit_transform(params_train)
        params_val_s   = scaler_params.transform(params_val)
        params_test_s  = scaler_params.transform(params_test)

        # If we also scale the target:
        if args.scale_target:
            scaler_y = RobustScaler()
            y_train_s = scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()
            y_val_s   = scaler_y.transform(y_val.reshape(-1,1)).flatten()
            y_test_s  = scaler_y.transform(y_test.reshape(-1,1)).flatten()
            custom_scalers["scaler_y"] = scaler_y
        else:
            y_train_s, y_val_s, y_test_s = y_train, y_val, y_test

        # Build datasets
        from torch.utils.data import DataLoader
        train_dataset = Shape2ShapeWithParamsDataset(Xinit_train_s, Xopt_train_s, params_train_s, y_train_s)
        val_dataset   = Shape2ShapeWithParamsDataset(Xinit_val_s,   Xopt_val_s,   params_val_s,   y_val_s)
        test_dataset  = Shape2ShapeWithParamsDataset(Xinit_test_s,  Xopt_test_s,  params_test_s,  y_test_s)

        shape_dim = Xinit_train_s.shape[1]  # dimension of shape
        # figure out # cont/cat from the shape of params_train_s
        num_params = params_train_s.shape[1]
        # we could track how many were cont vs. cat if you want
        num_cont = None  # optional
        num_cat  = None  # optional

        # store them
        custom_scalers["scaler_init"]   = scaler_init
        custom_scalers["scaler_opt"]    = scaler_opt
        custom_scalers["scaler_params"] = scaler_params

    else:
        # unstructured MLP
        scaler_X = RobustScaler()
        X_train_s = scaler_X.fit_transform(X_train)
        X_val_s   = scaler_X.transform(X_val)
        X_test_s  = scaler_X.transform(X_test)

        if args.scale_target:
            scaler_y = RobustScaler()
            y_train_s = scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()
            y_val_s   = scaler_y.transform(y_val.reshape(-1,1)).flatten()
            y_test_s  = scaler_y.transform(y_test.reshape(-1,1)).flatten()
            custom_scalers["scaler_y"] = scaler_y
        else:
            y_train_s, y_val_s, y_test_s = y_train, y_val, y_test

        from torch.utils.data import DataLoader
        train_dataset = PlainTabularDataset(X_train_s, y_train_s)
        val_dataset   = PlainTabularDataset(X_val_s,   y_val_s)
        test_dataset  = PlainTabularDataset(X_test_s,  y_test_s)

        shape_dim = None
        num_params = X_train_s.shape[1]
        num_cont = None
        num_cat  = None

        custom_scalers["scaler_X"] = scaler_X

    # Build data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    ################################################################
    # 9) Train ensemble (if multiple seeds)
    ################################################################
    from torch.backends import cudnn
    ensemble_models = []
    ensemble_val_losses = []

    for seed_i in args.seed:
        print(f"=== Training model for seed={seed_i} ===")
        torch.manual_seed(seed_i)
        np.random.seed(seed_i)
        random.seed(seed_i)
        cudnn.deterministic = True
        cudnn.benchmark = False

        # Call your "train_one_model" function
        model_i, (train_losses_i, val_losses_i), best_val_loss_i = train_one_model(
            args, train_loader, val_loader,
            have_shape_cols, shape_dim,
            num_cont if num_cont else 0, num_cat if num_cat else 0,
            device
        )
        ensemble_models.append(model_i)
        ensemble_val_losses.append(best_val_loss_i)

    best_model_idx = int(np.argmin(ensemble_val_losses))
    print(f"Best model index={best_model_idx}, seed={args.seed[best_model_idx]}, val_loss={ensemble_val_losses[best_model_idx]:.4f}")

    ################################################################
    # 10) Build final pipeline and save
    ################################################################
    pipeline_metadata = {"args": vars(args)}
    # Attach the same data preprocessor we used (with final_param_columns set)
    pipeline = ModelPipeline(
        models=ensemble_models,
        scalers=custom_scalers,
        structured=args.structured,
        log_target=args.log_target,
        metadata=pipeline_metadata,
        preprocessor=preprocessor
    )

    pipeline_filename = os.path.join(args.model_output_dir,
                                     f"final_pipeline_{run_name}_tgt_{args.target_col}.pkl")
    pipeline.save(pipeline_filename, device=device)
    print(f"Saved pipeline to {pipeline_filename}")

    # Optionally plot training curves for the best seed if there's only 1 seed
    if args.plot_loss and len(args.seed) == 1:
        plt.figure()
        plt.plot(train_losses_i, label="Train Loss")
        plt.plot(val_losses_i, label="Val Loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curves")
        plot_name = os.path.join(args.model_output_dir, f"loss_{run_name}.png")
        plt.savefig(plot_name)
        plt.show()
        print(f"Saved loss curve to {plot_name}")


    ################################################################
    # Visualization (structured only): shape overlays from best model
    ################################################################
    # If multiple seeds, we’ll just show overlays from the best model (for illustration).
    if have_shape_cols and args.structured:
        # We’ll re-use the best_model for shape overlay
        best_model = ensemble_models[best_model_idx]
        best_model.eval()

        # Grab the scalers
        scaler_init  = custom_scalers["scaler_init"]
        scaler_opt   = custom_scalers["scaler_opt"]

        

        # Show a few random test examples
        num_samples = 3
        sample_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        for idx in sample_indices:
            x_init, x_opt, params, cl_true = test_dataset[idx]
            x_init_batch = x_init.unsqueeze(0).to(device)
            params_batch = params.unsqueeze(0).to(device)

            with torch.no_grad():
                x_opt_pred, mu, logvar, z, cl_pred = best_model(x_init_batch, params_batch)

            x_init_np = invert_and_pair(scaler_init, x_init)
            x_opt_np = invert_and_pair(scaler_opt, x_opt)
            x_opt_pred_np = invert_and_pair(scaler_opt, x_opt_pred.squeeze(0))

            cl_pred_np = cl_pred.detach().cpu().numpy().reshape(-1,1)
            if args.scale_target:
                cl_pred_unscaled = custom_scalers["scaler_y"].inverse_transform(cl_pred_np).flatten()[0]
            else:
                cl_pred_unscaled = cl_pred_np.flatten()[0]
            cl_true_unscaled = cl_true.item()  # scaled or unscaled depends on your usage

            fig, ax = plt.subplots(figsize=(6,6))
            ax.plot(x_init_np[:,0], x_init_np[:,1], 'bo-', label='Initial Airfoil')
            ax.plot(x_opt_np[:,0], x_opt_np[:,1], 'go-', label='True Optimized Airfoil')
            ax.plot(x_opt_pred_np[:,0], x_opt_pred_np[:,1], 'r--', label='Reconstructed Optimized Airfoil')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"Sample {idx} (cl_val: true={float(cl_true_unscaled):.2f}, pred={float(cl_pred_unscaled):.2f})")
            ax.legend()
            fig_filename = os.path.join(args.model_output_dir, f"sample_{idx}_{args.target_col}_overlay.png")
            fig.savefig(fig_filename)
            plt.close(fig)
            print(f"Saved overlay figure for sample {idx} to {fig_filename}")

    ################################################################
    # Test evaluation: ensemble predictions
    ################################################################
    if args.test_model:
        # Evaluate each model in ensemble on the entire test set,
        # then average predictions.
        predictions_list = []
        truths_list = []

        # Collect all predictions from each model
        for e_idx, model_e in enumerate(ensemble_models):
            model_e.eval()
            preds_e = []
            trues_e = []
            with torch.no_grad():
                for batch_data in test_loader:
                    if have_shape_cols and args.structured:
                        (x_init_batch, x_opt_batch, param_batch, cl_batch) = batch_data
                        x_init_batch = x_init_batch.to(device)
                        param_batch  = param_batch.to(device)
                        cl_batch     = cl_batch.to(device)
                        _, _, _, _, cl_pred = model_e(x_init_batch, param_batch)
                        cl_pred = cl_pred.view(-1)
                        preds_e.append(cl_pred.cpu().numpy())
                        trues_e.append(cl_batch.cpu().numpy())
                    else:
                        (X_batch, y_batch) = batch_data
                        X_batch = X_batch.to(device)
                        y_batch = y_batch.to(device)
                        pred = model_e(X_batch).squeeze(-1)
                        preds_e.append(pred.cpu().numpy())
                        trues_e.append(y_batch.cpu().numpy())

            preds_e = np.concatenate(preds_e)
            trues_e = np.concatenate(trues_e)
            predictions_list.append(preds_e)
            truths_list.append(trues_e)

        # All seeds should see the same 'true' values, so we just pick from the first model
        test_trues = truths_list[0]

        # Average predictions over seeds
        # shape => (#seeds, #test_samples)
        all_preds = np.stack(predictions_list, axis=0)
        test_preds_ensemble = np.mean(all_preds, axis=0)

        # If target was scaled, invert it
        if args.scale_target and "scaler_y" in custom_scalers:
            test_preds_ensemble = custom_scalers["scaler_y"].inverse_transform(test_preds_ensemble.reshape(-1, 1)).flatten()
            test_trues = custom_scalers["scaler_y"].inverse_transform(test_trues.reshape(-1, 1)).flatten()

        # If target was log-transformed, exponentiate -> we assume all values are strictly positive
        if args.log_target:
            test_preds_ensemble = np.exp(test_preds_ensemble)
            test_trues = np.exp(test_trues)

        print("Test samples (avg ensemble):")
        for i in range(min(50, len(test_preds_ensemble))):  # show first 5
            print(f"Sample {i:3d}: Pred={test_preds_ensemble[i]:.4f}, True={test_trues[i]:.4f}")

        # Plot predicted vs. true for the ensemble
        plt.figure(figsize=(8, 6))
        plt.scatter(test_trues, test_preds_ensemble, alpha=0.7, label="Data points")
        plt.xlabel("True Values")
        plt.ylabel("Avg Ensemble Predicted Values")
        plt.title(f"Predicted vs True Values: {run_name}")
        min_val = min(np.min(test_trues), np.min(test_preds_ensemble))
        max_val = max(np.max(test_trues), np.max(test_preds_ensemble))
        plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
        plt.legend()
        plot_filename = os.path.join(args.model_output_dir, f"pred_vs_true_{run_name}_{args.target_col}_ensemble.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved predicted vs true plot to {plot_filename}")

    if args.track:
        wandb.finish()

    return ensemble_val_losses[best_model_idx]


###############################################################################
# 8) Entry Point
###############################################################################
if __name__ == "__main__":
    try:
        args = tyro.cli(Args)
        main(args)
    except Exception as e:
        print("An error occurred during execution:")
        raise
