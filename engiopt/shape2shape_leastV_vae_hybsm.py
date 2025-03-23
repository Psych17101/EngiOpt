#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shape-to-Shape VAE + Surrogate Example (Optionally skipping shapes entirely)

Now we allow:
- structured mode with shape columns
- unstructured MLP with or without shape columns
- possibility of ignoring shape columns by passing --init_col "" and --opt_col ""
- automatic detection of categorical parameter columns (fewer than 5 unique values) that get one-hot encoded
- a hybrid surrogate in structured mode that processes continuous and categorical parameters separately
"""

import os
import time
import ast
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

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

    # Set these to "" if you have NO shape columns at all
    init_col: str = "initial_design"
    opt_col: str = "optimal_design"

    target_col: str = "cl_val"

    log_target: bool = False  # If True, we'll log-transform the target before training

    # If you want to flatten param columns, add them here
    params_cols: List[str] = field(default_factory=lambda: ["mach", "reynolds"])
    flatten_columns: List[str] = field(default_factory=lambda: ["initial_design", "optimal_design"])
    strip_column_spaces: bool = False
    subset_condition: Optional[str] = None  # e.g., "r > 0"

    nondim_map: Optional[str] = None  # e.g., '{"C2": "C1", ...}'

    structured: bool = True  # Controls VAE+surrogate vs. plain MLP
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
    l2_lambda: float = 1e-3  # Set a default; can be tuned via command line


    gamma: float = 1.0
    lambda_lv: float = 1e-2
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
    """Recursively flattens any columns in `columns_to_flatten`."""
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

def strip_column_spaces(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leading and trailing whitespace from all column names."""
    df.columns = [col.strip() for col in df.columns]
    return df

def nondimensionalize(df: pd.DataFrame, nondim_map: dict) -> pd.DataFrame:
    """
    Divides each column specified in nondim_map by its reference column.
    For example, if nondim_map = {"C2": "C1"}, then df["C2"] = df["C2"] / df["C1"].
    """
    df_nd = df.copy()
    for col, ref in nondim_map.items():
        if col not in df_nd.columns:
            raise ValueError(f"Column {col} not found in DataFrame for nondimensionalization.")
        if ref not in df_nd.columns:
            raise ValueError(f"Reference column {ref} not found in DataFrame for nondimensionalization.")
        df_nd[col] = df_nd[col] / df_nd[ref]
    return df_nd

def process_params_split(df: pd.DataFrame, param_cols: List[str]) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits parameter columns into continuous and categorical parts.
    If a column has fewer than 5 unique values, one-hot encode it;
    otherwise, leave it as is.
    Returns a tuple (cont_df, cat_df). If no column qualifies for a part, returns an empty DataFrame.
    """
    cont_list = []
    cat_list = []
    for col in param_cols:
        if col not in df.columns:
            raise ValueError(f"Parameter column {col} not found in DataFrame.")
        if df[col].nunique() < 5:
            # One-hot encode categorical column
            dummies = pd.get_dummies(df[col], prefix=col)
            cat_list.append(dummies)
        else:
            cont_list.append(df[[col]])
    cont_df = pd.concat(cont_list, axis=1) if cont_list else pd.DataFrame(index=df.index)
    cat_df = pd.concat(cat_list, axis=1) if cat_list else pd.DataFrame(index=df.index)
    return cont_df, cat_df

def compute_l2_penalty(model: nn.Module) -> torch.Tensor:
    l2_norm = 0.0
    for param in model.parameters():
        l2_norm += torch.sum(param ** 2)
    return l2_norm


###############################################################################
# 3) Datasets
###############################################################################

class Shape2ShapeWithParamsDataset(torch.utils.data.Dataset):
    """
    (x_init, x_opt, params, target)
    In structured mode, the params are the concatenation of continuous and categorical values.
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
    (X, y) only
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

class HybridSurrogate(nn.Module):
    """
    A hybrid surrogate network that processes continuous parameters and categorical
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
    """
    Structured mode VAE with surrogate.
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

def least_volume_loss(z, eta=1e-3):
    std_z = torch.std(z, dim=0) + eta
    volume = torch.exp(torch.mean(torch.log(std_z)))
    return volume

def shape2shape_loss(x_opt_true, x_opt_pred, mu, logvar, z, cl_pred, cl_true,
                     lambda_lv=1e-2, gamma=1.0):
    cl_pred = cl_pred.view(-1)
    cl_true = cl_true.view(-1)
    recon_loss = F.smooth_l1_loss(x_opt_pred, x_opt_true, reduction="mean")
    lv_loss    = least_volume_loss(z)
    sur_loss   = F.smooth_l1_loss(cl_pred, cl_true, reduction="mean")
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

    # Choose device
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
        print(df.head())
    elif ext == ".parquet":
        df = pd.read_parquet(data_path)
        print(df.head())
    else:
        raise ValueError("data_input must be a CSV or Parquet file.")
    
    if args.strip_column_spaces:
        df = strip_column_spaces(df)

    if args.subset_condition is not None:
        try:
            df = df.query(args.subset_condition)
            print(f"Applied subset condition: {args.subset_condition}")
        except Exception as e:
            raise ValueError(f"Error applying subset condition '{args.subset_condition}': {e}")
    
    if args.nondim_map is not None:
        try:
            nondim_dict = ast.literal_eval(args.nondim_map)
            if not isinstance(nondim_dict, dict):
                raise ValueError("nondim_map must be a dictionary, e.g., '{\"C2\": \"C1\"}'")
        except Exception as e:
            raise ValueError(f"Invalid nondim_map: {args.nondim_map}. Error: {e}")
        df = nondimensionalize(df, nondim_dict)
        print("Applied nondimensionalization using map:", nondim_dict)

    if args.log_target:
        df[args.target_col] = np.log(df[args.target_col])
        print(f"Applied log-transform to column '{args.target_col}'")

    if args.flatten_columns:
        df = flatten_list_columns(df, args.flatten_columns)
        print("After flattening, df.columns:", df.columns.tolist())

    have_shape_cols = (
        args.init_col != "" and args.opt_col != "" and
        any(col.startswith(args.init_col + "_") for col in df.columns) and
        any(col.startswith(args.opt_col + "_") for col in df.columns)
    )

    if args.structured and not have_shape_cols:
        raise ValueError("Structured mode selected but no shape columns found! "
                         "Set --init_col= and --opt_col= to empty or use unstructured instead.")

    if args.target_col not in df.columns:
        raise ValueError(f"Missing target column: {args.target_col}")
    y_all = df[args.target_col].values

    ################################################################
    # Structured mode: use shape columns and process parameters with split.
    ################################################################
    if have_shape_cols and args.structured:
        X_init_all = df[[c for c in df.columns if c.startswith(args.init_col + "_")]].values
        X_opt_all  = df[[c for c in df.columns if c.startswith(args.opt_col + "_")]].values

        # Process parameters: split into continuous and categorical parts.
        cont_df, cat_df = process_params_split(df, args.params_cols)
        print("Continuous param columns:", cont_df.columns.tolist())
        print("Categorical param columns:", cat_df.columns.tolist())
        if not cont_df.empty:
            scaler_cont = StandardScaler()
            cont_values = scaler_cont.fit_transform(cont_df.values)
        else:
            cont_values = np.empty((len(df), 0))
        cat_values = cat_df.values if not cat_df.empty else np.empty((len(df), 0))
        params_all = np.hstack([cont_values, cat_values])
        num_cont = cont_values.shape[1]
        num_cat = cat_values.shape[1]

        # Split the data
        Xinit_temp, Xinit_test, Xopt_temp, Xopt_test, params_temp, params_test, y_temp, y_test = train_test_split(
            X_init_all, X_opt_all, params_all, y_all,
            test_size=args.test_size, random_state=args.seed
        )
        Xinit_train, Xinit_val, Xopt_train, Xopt_val, params_train, params_val, y_train, y_val = train_test_split(
            Xinit_temp, Xopt_temp, params_temp, y_temp,
            test_size=args.val_size_of_train, random_state=args.seed
        )

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

        if args.scale_target:
            scaler_y = StandardScaler()
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()
            y_val_scaled   = scaler_y.transform(y_val.reshape(-1,1)).flatten()
            y_test_scaled  = scaler_y.transform(y_test.reshape(-1,1)).flatten()
        else:
            y_train_scaled, y_val_scaled, y_test_scaled = y_train, y_val, y_test

        train_dataset = Shape2ShapeWithParamsDataset(
            Xinit_train_scaled, Xopt_train_scaled, params_train_scaled, y_train_scaled
        )
        val_dataset   = Shape2ShapeWithParamsDataset(
            Xinit_val_scaled, Xopt_val_scaled, params_val_scaled, y_val_scaled
        )
        test_dataset  = Shape2ShapeWithParamsDataset(
            Xinit_test_scaled, Xopt_test_scaled, params_test_scaled, y_test_scaled
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        shape_dim = Xinit_train_scaled.shape[1]
        # Use cont_dim and cat_dim from our processed parameters.
        model = Shape2ShapeVAE(
            shape_dim=shape_dim,
            cont_dim=num_cont,
            cat_dim=num_cat,
            latent_dim=args.latent_dim,
            surrogate_hidden_layers=args.hidden_layers,
            surrogate_hidden_size=args.hidden_size,
            surrogate_activation=args.activation
        ).to(device)

        def train_step(x_init_batch, x_opt_batch, param_batch, cl_batch):
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

        custom_scalers = {
            "scaler_init": scaler_init,
            "scaler_opt": scaler_opt,
            "scaler_params": scaler_params,
            "scaler_cont": scaler_cont  # saved if needed
        }
        if args.scale_target:
            custom_scalers["scaler_y"] = scaler_y

    ################################################################
    # Unstructured MLP: process parameters similarly but simply concatenate.
    ################################################################
    else:
        # Process parameter columns with split processing.
        cont_df, cat_df = process_params_split(df, args.params_cols)
        print("Continuous param columns:", cont_df.columns.tolist())
        print("Categorical param columns:", cat_df.columns.tolist())
        if not cont_df.empty:
            scaler_cont = StandardScaler()
            cont_values = scaler_cont.fit_transform(cont_df.values)
        else:
            cont_values = np.empty((len(df), 0))
        cat_values = cat_df.values if not cat_df.empty else np.empty((len(df), 0))
        X_features_all = np.hstack([cont_values, cat_values])
        # Save scaler for continuous part if needed
        custom_scalers = {"scaler_X": None}
        if not cont_df.empty:
            custom_scalers["scaler_cont"] = scaler_cont

        X_temp, X_test, y_temp, y_test = train_test_split(
            X_features_all, y_all, test_size=args.test_size, random_state=args.seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=args.val_size_of_train, random_state=args.seed
        )

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

        train_dataset = PlainTabularDataset(X_train_scaled, y_train_scaled)
        val_dataset   = PlainTabularDataset(X_val_scaled, y_val_scaled)
        test_dataset  = PlainTabularDataset(X_test_scaled, y_test_scaled)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        input_dim = X_train_scaled.shape[1]
        model = make_mlp(
            input_dim=input_dim,
            hidden_layers=args.hidden_layers,
            hidden_size=args.hidden_size,
            activation=args.activation,
            output_dim=1
        ).to(device)

        def train_step(X_batch, y_batch):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch).squeeze(-1)
            loss = F.smooth_l1_loss(preds, y_batch)
            l2_pen = compute_l2_penalty(model)
            loss = loss + args.l2_lambda * l2_pen
            return loss, 0, 0, 0

        custom_scalers["scaler_X"] = scaler_X
        if args.scale_target:
            custom_scalers["scaler_y"] = scaler_y

    ################################################################
    # Training Loop
    ################################################################
    optimizer_ = None
    def make_optimizer(name: str, params, lr: float):
        if name == "sgd":
            return optim.SGD(params, lr=lr)
        elif name == "adam":
            return optim.Adam(params, lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {name}")
    optimizer_ = make_optimizer(args.optimizer, model.parameters(), args.learning_rate)

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
                loss, recon_loss, lv_loss, sur_loss = train_step(x_init_batch, x_opt_batch, param_batch, cl_batch)
            else:
                (X_batch, y_batch) = batch_data
                loss, _, _, _ = train_step(X_batch, y_batch)
            loss.backward()
            optimizer_.step()
            batch_size_ = len(batch_data[0])
            running_train_loss += loss.item() * batch_size_
        epoch_train_loss = running_train_loss / n_train
        train_losses.append(epoch_train_loss)
        model.eval()
        running_val_loss = 0.0
        n_val = len(val_loader.dataset)
        with torch.no_grad():
            for batch_data in val_loader:
                if have_shape_cols and args.structured:
                    (x_init_batch, x_opt_batch, param_batch, cl_batch) = batch_data
                    loss_val, _, _, _ = train_step(x_init_batch, x_opt_batch, param_batch, cl_batch)
                else:
                    (X_batch, y_batch) = batch_data
                    loss_val, _, _, _ = train_step(X_batch, y_batch)
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

    if args.save_model:
        os.makedirs(args.model_output_dir, exist_ok=True)
        model_path = os.path.join(args.model_output_dir, f"best_model_{run_name}_{args.target_col}.pth")
        save_dict = {"model_state_dict": model.state_dict()}
        save_dict.update(custom_scalers)
        torch.save(save_dict, model_path)
        print(f"Saved model to: {model_path}")

    if args.plot_loss:
        plt.figure(figsize=(6,4))
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Val")
        plt.legend()
        plt.title("Loss curves")
        plt.savefig(f"loss_{run_name}.png")
        plt.show()
    
    ###############################################################################
    # Visualization: plot initial vs. predicted/true optimized shapes (structured only)
    ###############################################################################
    if args.structured:
        model.eval()
        num_samples = 3
        sample_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        def invert_and_pair(scaler, tensor):
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
            cl_true_unscaled = cl_true
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

    ###############################################################################
    # Test evaluation: process entire test set.
    ###############################################################################
    if args.test_model:
        model.eval()
        test_preds, test_trues = [], []
        with torch.no_grad():
            for batch_data in test_loader:
                if have_shape_cols and args.structured:
                    (x_init_batch, x_opt_batch, param_batch, cl_batch) = batch_data
                    x_init_batch = x_init_batch.to(device)
                    param_batch  = param_batch.to(device)
                    cl_batch     = cl_batch.to(device)
                    _, _, _, _, cl_pred = model(x_init_batch, param_batch)
                    cl_pred = cl_pred.view(-1)
                    test_preds.append(cl_pred.cpu().numpy())
                    test_trues.append(cl_batch.cpu().numpy())
                else:
                    (X_batch, y_batch) = batch_data
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    preds = model(X_batch).squeeze(-1)
                    test_preds.append(preds.cpu().numpy())
                    test_trues.append(y_batch.cpu().numpy())
        test_preds = np.concatenate(test_preds)
        test_trues = np.concatenate(test_trues)
        if args.scale_target and "scaler_y" in custom_scalers:
            test_preds = custom_scalers["scaler_y"].inverse_transform(test_preds.reshape(-1, 1)).flatten()
            test_trues = custom_scalers["scaler_y"].inverse_transform(test_trues.reshape(-1, 1)).flatten()
        if args.log_target:
            test_preds = np.exp(test_preds) - 1e-8
            test_trues = np.exp(test_trues) - 1e-8
        print("Test samples:")
        for i in range(len(test_preds)):
            print(f"Sample {i:3d}: Pred={test_preds[i]:.4f}, True={test_trues[i]:.4f}")
        plt.figure(figsize=(8, 6))
        plt.scatter(test_trues, test_preds, alpha=0.7, label="Data points")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Predicted vs True Values: {run_name}")
        min_val = min(np.min(test_trues), np.min(test_preds))
        max_val = max(np.max(test_trues), np.max(test_preds))
        plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
        plt.legend()
        plot_filename = os.path.join(args.model_output_dir, f"pred_vs_true_{run_name}_{args.target_col}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved predicted vs true plot to {plot_filename}")

    if args.track:
        wandb.finish()
    
    return best_val_loss

###############################################################################
# 7) Entry Point
###############################################################################

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
