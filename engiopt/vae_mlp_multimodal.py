#!/usr/bin/env python

"""
vae_mlp_multimodal.py

Updated script to train a plain MLP or structured VAE+Surrogate, 
with optional dataset loading from HuggingFace, single seed + multiple ensembles, 
new W&B run naming, and improved activation/optimizer mapping.
"""

import ast
from dataclasses import dataclass, field
import os
import random
import time
from typing import List, Literal, Optional, Tuple, Dict

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

import torch
from torch import nn, optim
import torch.nn.functional as F

import tyro
import wandb

from datasets import load_dataset


# Local imports
from model_pipeline import DataPreprocessor, ModelPipeline

# 1) Mappings for Activations and Optimizers
ACTIVATIONS = {
    "relu": nn.ReLU(),
    "leakyrelu": nn.LeakyReLU(0.2),
    "prelu": nn.PReLU(),
    "rrelu": nn.RReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "gelu": nn.GELU(),
    "celu": nn.CELU(),
    "none": nn.Identity(),
}

OPTIMIZERS = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
    "adadelta": optim.Adadelta,
    "adamax": optim.Adamax,
    "asgd": optim.ASGD,
    "lbfgs": optim.LBFGS
}

###############################################################################
# 2) Argument Parsing
###############################################################################
@dataclass
class Args:
    # Optionally pull data from Hugging Face
    huggingface_repo: str = "IDEALLab/power_electronics_v0"
    huggingface_split: str = "train"

    # Local file fallback
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

    # We use weight_decay from the optimizer as L2 penalty
    l2_lambda: float = 1e-3

    gamma: float = 1.0
    lambda_lv: float = 1e-2

    test_size: float = 0.2
    val_size_of_train: float = 0.25
    scale_target: bool = True

    # W&B / tracking
    track: bool = True
    wandb_project: str = "engiopt"  # changed default
    wandb_entity: Optional[str] = None

    # Merging single seed + ensemble count
    seed: int = 42
    n_ensembles: int = 1

    # Problem ID / algo for run name
    problem_id: str = "myproblem"
    algo: str = "myalgo"

    # Save / test / etc.
    save_model: bool = False
    plot_loss: bool = True  # Will remove local plot logic, but keep the flag
    model_output_dir: str = "results"
    test_model: bool = False

    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_output_dir, exist_ok=True)
        # For "flatten_columns" and "params_cols", only parse if the value is a string,
        # or if it's a single-element list whose first element looks like a string literal.
        for field_name in ["flatten_columns", "params_cols"]:
            value = getattr(self, field_name)
            if isinstance(value, str):
                try:
                    parsed_value = ast.literal_eval(value)
                    if isinstance(parsed_value, list):
                        setattr(self, field_name, parsed_value)
                    else:
                        raise ValueError(f"Expected list for --{field_name}, got {parsed_value}")
                except Exception as e:
                    raise ValueError(f"Invalid format for --{field_name}: {value}") from e
            elif isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
                first = value[0].strip()
                if first and first[0] in ("[", "{"):
                    try:
                        parsed_value = ast.literal_eval(value[0])
                        if isinstance(parsed_value, list):
                            setattr(self, field_name, parsed_value)
                    except Exception as e:
                        raise ValueError(f"Invalid format for --{field_name}: {value}") from e
        # Otherwise, leave the value as is.




###############################################################################
# 3) Simple utility
###############################################################################
def make_optimizer(name: str, params, lr: float, weight_decay: float = 0.0) -> optim.Optimizer:
    if name not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {name}")
    return OPTIMIZERS[name](params, lr=lr, weight_decay=weight_decay)

def make_activation(activation: str) -> nn.Module:
    if activation not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {activation}")
    return ACTIVATIONS[activation]

###############################################################################
# 4) Datasets
###############################################################################
class Shape2ShapeWithParamsDataset(torch.utils.data.Dataset):
    def __init__(self, init_array, opt_array, params_array, cl_array):
        self.X_init = torch.from_numpy(init_array).float()
        self.X_opt  = torch.from_numpy(opt_array).float()
        self.params = torch.from_numpy(params_array).float()
        self.y      = torch.from_numpy(cl_array).float()
        if self.X_init.shape[1] != self.X_opt.shape[1]:
            raise ValueError("init_design and opt_design must have same dimension.")
    def __len__(self) -> int:
        return len(self.X_init)
    def __getitem__(self, idx: int):
        return self.X_init[idx], self.X_opt[idx], self.params[idx], self.y[idx]

class PlainTabularDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self) -> int:
        return len(self.X)
    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

###############################################################################
# 5) Models
###############################################################################
class HybridSurrogate(nn.Module):
    def __init__(self, latent_dim: int, cont_dim: int, cat_dim: int,
                 hidden_layers: int = 2, hidden_size: int = 64, activation: str = "relu"):
        super().__init__()
        act_fn = make_activation(activation)
        in_dim_cont = latent_dim + cont_dim
        cont_layers = []
        for _ in range(hidden_layers):
            cont_layers.append(nn.Linear(in_dim_cont, hidden_size))
            cont_layers.append(act_fn)
            in_dim_cont = hidden_size
        self.cont_net = nn.Sequential(*cont_layers)

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

        combined_in = hidden_size * (2 if self.cat_net is not None else 1)
        self.combined = nn.Sequential(
            nn.Linear(combined_in, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1)
        )

    def forward(self, latent: torch.Tensor, cont_params: torch.Tensor, cat_params: torch.Tensor):
        cont_out = self.cont_net(torch.cat([latent, cont_params], dim=1))
        if self.cat_net is not None:
            cat_out = self.cat_net(cat_params)
            combined = torch.cat([cont_out, cat_out], dim=1)
        else:
            combined = cont_out
        return self.combined(combined)

class Shape2ShapeVAE(nn.Module):
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
        # Surrogate
        if cat_dim > 0:
            self.use_hybrid = True
            self.surrogate = HybridSurrogate(latent_dim, cont_dim, cat_dim,
                                             hidden_layers=surrogate_hidden_layers,
                                             hidden_size=surrogate_hidden_size,
                                             activation=surrogate_activation)
        else:
            self.use_hybrid = False
            act_fn = make_activation(surrogate_activation)
            layers = []
            in_dim = latent_dim + cont_dim
            for _ in range(surrogate_hidden_layers):
                layers.append(nn.Linear(in_dim, surrogate_hidden_size))
                layers.append(act_fn)
                in_dim = surrogate_hidden_size
            layers.append(nn.Linear(in_dim, 1))
            self.surrogate = nn.Sequential(*layers)

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

    def forward(self, x_init: torch.Tensor, param_vec: torch.Tensor):
        mu, logvar = self.encode(x_init)
        z = self.reparameterize(mu, logvar)
        x_opt_pred = self.decode(z)
        if self.use_hybrid:
            cont_params = param_vec[:, :self.cont_dim]
            cat_params = param_vec[:, self.cont_dim:]
            cl_pred = self.surrogate(z, cont_params, cat_params)
        else:
            cl_pred = self.surrogate(torch.cat([z, param_vec], dim=1))
        return x_opt_pred, mu, logvar, z, cl_pred

###############################################################################
# 6) Losses
###############################################################################
def least_volume_loss(z: torch.Tensor, eta: float = 1e-3) -> torch.Tensor:
    std_z = torch.std(z, dim=0) + eta
    volume = torch.exp(torch.mean(torch.log(std_z)))
    return volume

def shape2shape_loss(x_opt_true: torch.Tensor, x_opt_pred: torch.Tensor,
                     mu: torch.Tensor, logvar: torch.Tensor, z: torch.Tensor,
                     cl_pred: torch.Tensor, cl_true: torch.Tensor,
                     lambda_lv: float = 1e-2, gamma: float = 1.0):
    cl_pred = cl_pred.view(-1)
    cl_true = cl_true.view(-1)
    recon_loss = F.smooth_l1_loss(x_opt_pred, x_opt_true, reduction="mean")
    lv_loss    = least_volume_loss(z)
    sur_loss   = F.smooth_l1_loss(cl_pred, cl_true, reduction="mean")
    total_loss = recon_loss + lambda_lv * lv_loss + gamma * sur_loss
    return total_loss, recon_loss, lv_loss, sur_loss

###############################################################################
# 7) Training Routine
###############################################################################
def train_one_model(args: Args,
                    train_loader: torch.utils.data.DataLoader,
                    val_loader: torch.utils.data.DataLoader,
                    have_shape_cols: bool,
                    shape_dim: Optional[int],
                    num_cont: int,
                    num_cat: int,
                    device: torch.device) -> Tuple[nn.Module, Tuple[List[float], List[float]], float]:

    if have_shape_cols and args.structured:
        # shape2shape
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
            return loss, recon_loss, lv_loss, sur_loss

    else:
        # plain MLP
        input_dim = next(iter(train_loader))[0].shape[1]
        act_fn = make_activation(args.activation)
        layers = []
        in_dim = input_dim
        for _ in range(args.hidden_layers):
            layers.append(nn.Linear(in_dim, args.hidden_size))
            layers.append(act_fn)
            in_dim = args.hidden_size
        layers.append(nn.Linear(in_dim, 1))
        model = nn.Sequential(*layers).to(device)

        def train_step_mlp(X_batch, y_batch):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch).squeeze(-1)
            loss = F.smooth_l1_loss(preds, y_batch)
            return loss

    # build optimizer with weight_decay as L2
    optimizer_ = make_optimizer(args.optimizer, model.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.l2_lambda)

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
                loss, rL, lvL, sL = train_step_vae_mlp(x_init_batch, x_opt_batch, param_batch, cl_batch)
            else:
                (X_batch, y_batch) = batch_data
                loss = train_step_mlp(X_batch, y_batch)

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
                    loss_val = train_step_mlp(X_batch, y_batch)
                batch_size_ = len(batch_data[0])
                running_val_loss += loss_val.item() * batch_size_

        epoch_val_loss = running_val_loss / n_val
        val_losses.append(epoch_val_loss)

        if args.track:
            wandb.log({
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "epoch": epoch,
            })

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_weights = model.state_dict()
        else:
            epochs_no_improve += 1

        if (epoch + 1) % args.lr_decay_step == 0:
            scheduler.step()

        print(f"[Epoch {epoch+1}/{args.n_epochs}] Train: {epoch_train_loss:.4f}, Val: {epoch_val_loss:.4f}")
        if epochs_no_improve >= args.patience:
            print("Early stopping triggered.")
            break

    print(f"Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, (train_losses, val_losses), best_val_loss

###############################################################################
# 8) Main
###############################################################################
def main(args: Args) -> float:
    # Build wandb run name
    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"

    if args.track:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args),
                   save_code=True, name=run_name)

    # Decide device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    # Load DataFrame from huggingface or local file
    if args.huggingface_repo:
        print(f"[INFO] Loading dataset from HuggingFace: {args.huggingface_repo} (split={args.huggingface_split})")
        ds = load_dataset(args.huggingface_repo, split=args.huggingface_split)
        df = ds.to_pandas()
    else:
        # fallback to local CSV/Parquet
        data_path = os.path.join(args.data_dir, args.data_input)
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"{data_path} does not exist")
        ext = os.path.splitext(data_path)[1].lower()
        print(f"[INFO] Loading local file {data_path}")
        if ext == ".csv":
            df = pd.read_csv(data_path)
        elif ext == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            raise ValueError("data_input must be CSV or Parquet")

    print("[INFO] DataFrame head:")
    print(df.head())

    # Preprocess
    preprocessor = DataPreprocessor(vars(args))
    processed_dict, df = preprocessor.transform_inputs(df, fit_params=True)

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

    # Build arrays
    if have_shape_cols and args.structured:
        X_init_all = processed_dict["x_init"]
        X_opt_all  = processed_dict["x_opt"]
        params_all = processed_dict["params"]
    else:
        X_features_all = processed_dict["X"]

    # Train/val/test split
    if have_shape_cols and args.structured:
        Xinit_temp, Xinit_test, Xopt_temp, Xopt_test, params_temp, params_test, y_temp, y_test = \
            train_test_split(X_init_all, X_opt_all, params_all, y_all,
                             test_size=args.test_size, random_state=args.split_random_state)
        Xinit_train, Xinit_val, Xopt_train, Xopt_val, params_train, params_val, y_train, y_val = \
            train_test_split(Xinit_temp, Xopt_temp, params_temp, y_temp,
                             test_size=args.val_size_of_train, random_state=args.split_random_state)
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(X_features_all, y_all,
                                                          test_size=args.test_size,
                                                          random_state=args.split_random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp,
                                                          test_size=args.val_size_of_train,
                                                          random_state=args.split_random_state)

    # Scalings
    custom_scalers = {}
    from torch.utils.data import DataLoader
    if have_shape_cols and args.structured:
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

        if args.scale_target:
            scaler_y = RobustScaler()
            y_train_s = scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()
            y_val_s   = scaler_y.transform(y_val.reshape(-1,1)).flatten()
            y_test_s  = scaler_y.transform(y_test.reshape(-1,1)).flatten()
            custom_scalers["scaler_y"] = scaler_y
        else:
            y_train_s, y_val_s, y_test_s = y_train, y_val, y_test

        train_dataset = Shape2ShapeWithParamsDataset(Xinit_train_s, Xopt_train_s, params_train_s, y_train_s)
        val_dataset   = Shape2ShapeWithParamsDataset(Xinit_val_s,   Xopt_val_s,   params_val_s,   y_val_s)
        test_dataset  = Shape2ShapeWithParamsDataset(Xinit_test_s,  Xopt_test_s,  params_test_s,  y_test_s)

        shape_dim = Xinit_train_s.shape[1]
        num_params = params_train_s.shape[1]
        # If you want to separate cont vs cat dimension, you'd do so here. For simplicity, pass 0,0
        num_cont, num_cat = 0, 0  # Adjust if you want

        custom_scalers["scaler_init"]   = scaler_init
        custom_scalers["scaler_opt"]    = scaler_opt
        custom_scalers["scaler_params"] = scaler_params

    else:
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

        train_dataset = PlainTabularDataset(X_train_s, y_train_s)
        val_dataset   = PlainTabularDataset(X_val_s,   y_val_s)
        test_dataset  = PlainTabularDataset(X_test_s,  y_test_s)

        shape_dim = None
        num_params = X_train_s.shape[1]
        num_cont, num_cat = 0, 0

        custom_scalers["scaler_X"] = scaler_X

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    ###########################################################
    # Build the ensemble from seed..seed + n_ensembles - 1
    ###########################################################
    ensemble_models = []
    ensemble_val_losses = []
    seeds = [args.seed + i for i in range(args.n_ensembles)]

    from torch.backends import cudnn

    for seed_i in seeds:
        print(f"=== Training model for seed={seed_i} ===")
        torch.manual_seed(seed_i)
        np.random.seed(seed_i)
        random.seed(seed_i)
        cudnn.deterministic = True
        cudnn.benchmark = False

        model_i, (train_losses_i, val_losses_i), best_val_loss_i = train_one_model(
            args, train_loader, val_loader,
            have_shape_cols, shape_dim,
            num_cont, num_cat,
            device
        )
        ensemble_models.append(model_i)
        ensemble_val_losses.append(best_val_loss_i)

    best_model_idx = int(np.argmin(ensemble_val_losses))
    print(f"[INFO] Best model index={best_model_idx}, seed={seeds[best_model_idx]}, val_loss={ensemble_val_losses[best_model_idx]:.4f}")

    # Build final pipeline
    pipeline_metadata = {"args": vars(args)}
    pipeline = ModelPipeline(
        models=ensemble_models,
        scalers=custom_scalers,
        structured=args.structured,
        log_target=args.log_target,
        metadata=pipeline_metadata,
        preprocessor=preprocessor
    )

    pipeline_filename = os.path.join(args.model_output_dir,
                                     f"final_pipeline_{run_name}_{args.target_col}.pkl")

    # Save locally if needed
    if args.save_model:
        pipeline.save(pipeline_filename, device=device)
        print(f"Saved pipeline to {pipeline_filename}")
        # Also log to W&B as an artifact
        if args.track:
            artifact = wandb.Artifact(f"{run_name}_model", type="model")
            artifact.add_file(pipeline_filename)
            wandb.log_artifact(artifact)
            print("[INFO] Uploaded model artifact to W&B.")

    ############################################################################
    # If requested, evaluate on test set with ensemble
    ############################################################################
    if args.test_model:
        predictions_list = []
        truths_list = []

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
                        preds_e.append(cl_pred.view(-1).cpu().numpy())
                        trues_e.append(cl_batch.view(-1).cpu().numpy())
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

        test_trues = truths_list[0]  # same for all seeds
        all_preds = np.stack(predictions_list, axis=0)
        test_preds_ensemble = np.mean(all_preds, axis=0)

        # Unscale if needed
        if args.scale_target and "scaler_y" in custom_scalers:
            test_preds_ensemble = custom_scalers["scaler_y"].inverse_transform(test_preds_ensemble.reshape(-1, 1)).flatten()
            test_trues = custom_scalers["scaler_y"].inverse_transform(test_trues.reshape(-1, 1)).flatten()

        # If log target, exponentiate
        if args.log_target:
            test_preds_ensemble = np.exp(test_preds_ensemble)
            test_trues = np.exp(test_trues)

        # Print a few
        print("[INFO] Test samples (avg ensemble):")
        for i in range(min(5, len(test_preds_ensemble))):
            print(f"  Sample {i}: Pred={test_preds_ensemble[i]:.4f}, True={test_trues[i]:.4f}")

        # If you'd like, you can log an MSE to W&B:
        mse = np.mean((test_preds_ensemble - test_trues) ** 2)
        print(f"[INFO] Test MSE: {mse:.6f}")
        if args.track:
            wandb.log({"test_mse": mse})

    if args.track:
        wandb.finish()

    return ensemble_val_losses[best_model_idx]

###############################################################################
# 9) Entry Point
###############################################################################
if __name__ == "__main__":
    try:
        args = tyro.cli(Args)
        main(args)
    except Exception as e:
        print("An error occurred during execution:")
        raise
