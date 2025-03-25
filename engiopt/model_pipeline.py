import os
import pickle
import torch
import numpy as np
import math
from typing import List, Optional, Dict, Any
import ast
import pandas as pd
from sklearn.preprocessing import RobustScaler

class DataPreprocessor:
    def __init__(self, args: dict):
        """
        args: typically vars(args) from your training script, 
              containing keys like 'structured', 'init_col', 'opt_col', etc.
        """
        self.args = args.copy()

        # We'll keep track of the final parameter columns after one-hot encoding, so we can 
        # replicate them exactly at inference time. 
        self.final_param_columns = None

        # Similarly, if you want to keep track of the shape columns, you can store them too.
        # For now we just store the prefixes: init_col + "_" etc.

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1) Strip spaces
        if self.args.get("strip_column_spaces", False):
            df.columns = [col.strip() for col in df.columns]

        # 2) Flatten columns
        if self.args.get("flatten_columns", []):
            df = self.flatten_list_columns(df, self.args["flatten_columns"])

        # 3) Subset condition
        subset_condition = self.args.get("subset_condition", None)
        if subset_condition is not None:
            df = df.query(subset_condition)

        # 4) Nondimensionalization
        nondim_map = self.args.get("nondim_map", None)
        if nondim_map is not None:
            import ast
            if isinstance(nondim_map, str):
                nondim_map = ast.literal_eval(nondim_map)
            for col, ref in nondim_map.items():
                if col in df.columns and ref in df.columns:
                    df[col] = df[col] / df[ref]

        # 5) log-transform
        if self.args.get("log_target", False):
            target_col = self.args.get("target_col", None)
            if target_col in df.columns:
                df[target_col] = np.log(df[target_col])
                print(f"[DataPreprocessor] Applied log-transform to {target_col}")

        return df

    def split_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Splits + one-hot encodes (if <5 unique) the param_cols from self.args.
        Returns a new DataFrame containing just those processed columns.
        Also sets self.final_param_columns to the final list of columns.
        """
        param_cols = self.args.get("params_cols", [])
        # If no param cols or if param_cols are empty, just return an empty DataFrame
        if not param_cols:
            self.final_param_columns = []
            return pd.DataFrame(index=df.index)

        cont_list = []
        cat_list = []
        for col in param_cols:
            if col not in df.columns:
                raise ValueError(f"[DataPreprocessor] param col '{col}' not found in DataFrame.")
            if df[col].nunique() < 5:
                # one-hot
                dummies = pd.get_dummies(df[col], prefix=col)
                cat_list.append(dummies)
            else:
                cont_list.append(df[[col]])

        cont_df = pd.concat(cont_list, axis=1) if cont_list else pd.DataFrame(index=df.index)
        cat_df = pd.concat(cat_list, axis=1) if cat_list else pd.DataFrame(index=df.index)

        # Combine them
        final_df = pd.concat([cont_df, cat_df], axis=1)
        # Record the final columns in the order they appear
        self.final_param_columns = list(final_df.columns)
        return final_df

    def transform_inputs(self, df: pd.DataFrame, fit_params: bool = False) -> Dict[str, np.ndarray]:
        """
        Called during training with fit_params=True: we do the splitting and store the final columns.
        Called during inference with fit_params=False: we reuse self.final_param_columns to preserve the same order.
        """
        # 1) Basic cleaning
        df_processed = self.preprocess_dataframe(df.copy())

        if self.args.get("structured", False):
            # shape data
            init_prefix = self.args.get("init_col", "")
            opt_prefix  = self.args.get("opt_col", "")
            init_cols = [c for c in df_processed.columns if c.startswith(init_prefix + "_")]
            opt_cols =  [c for c in df.columns if c.startswith(opt_prefix + "_")]

            x_init = df_processed[init_cols].values if init_cols else None
            x_opt  = df_processed[opt_cols].values if opt_cols else None

            # param data
            if fit_params:
                # during training, we do the actual splitting and record final columns
                param_df = self.split_parameters(df_processed)
            else:
                # inference: use the already known columns
                # (We assume self.final_param_columns is populated from training)
                if not self.final_param_columns:
                    raise ValueError("No final_param_columns known – did you call transform_inputs with fit_params=True at training?")
                param_df = df_processed[self.final_param_columns]

            params = param_df.values if len(param_df.columns) > 0 else None
            return {"x_init": x_init, "x_opt": x_opt, "params": params}
        else:
            # unstructured
            if fit_params:
                param_df = self.split_parameters(df_processed)
            else:
                if not self.final_param_columns:
                    raise ValueError("No final_param_columns known – did you call transform_inputs with fit_params=True at training?")
                param_df = df_processed[self.final_param_columns]

            X = param_df.values if len(param_df.columns) > 0 else None
            return {"X": X}


    def flatten_list_columns(self, df: pd.DataFrame, columns_to_flatten: list) -> pd.DataFrame:
        new_cols_list = []
        drop_cols = []
        for col in columns_to_flatten:
            if col not in df.columns:
                continue
            first_val = df[col].iloc[0]
            if isinstance(first_val, str):
                try:
                    _ = ast.literal_eval(first_val)
                    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                except:
                    continue
            if isinstance(df[col].iloc[0], (list, tuple)):
                flattened_rows = [self.recursive_flatten(x) for x in df[col]]
                lengths = [len(row) for row in flattened_rows]
                if len(set(lengths)) > 1:
                    raise ValueError(f"Column '{col}' has varying lengths.")
                n = lengths[0]
                new_col_names = [f"{col}_{i}" for i in range(n)]
                expanded_df = pd.DataFrame(flattened_rows, columns=new_col_names)
                new_cols_list.append(expanded_df)
                drop_cols.append(col)
        if new_cols_list:
            df = pd.concat([df.drop(columns=drop_cols).reset_index(drop=True)] + new_cols_list, axis=1)
        return df

    def recursive_flatten(self, val):
        if not isinstance(val, (list, tuple)):
            return [val]
        result = []
        for item in val:
            result.extend(self.recursive_flatten(item))
        return result


    def transform_inputs_old(self, raw_df: pd.DataFrame):
        df_processed = self.preprocess_dataframe(raw_df.copy())
        if self.args.get("structured", False):
            init_cols = [col for col in df_processed.columns if col.startswith(self.args["init_col"] + "_")]
            # Use stored parameter feature names if available
            param_feature_names = self.args.get("param_feature_names", self.args.get("params_cols", []))
            if param_feature_names and isinstance(param_feature_names, list):
                params = df_processed[param_feature_names].values
            else:
                params = df_processed[self.args.get("params_cols", [])].values
            x_init = df_processed[init_cols].values if init_cols else None
            return {"x_init": x_init, "params": params}
        else:
            feature_cols = self.args.get("params_cols", [])
            X = df_processed[feature_cols].values if feature_cols else None
            return {"X": X}

        """
        df_processed = self.preprocess_dataframe(raw_df.copy())
        if self.args.get("structured", False):
            init_cols = [col for col in df_processed.columns if col.startswith(self.args["init_col"] + "_")]
            x_init = df_processed[init_cols].values if init_cols else None

            params_cols = self.args.get("params_cols", [])
            if params_cols:
                cont_df, cat_df = self.process_params_split(df_processed, params_cols)
                if not cont_df.empty or not cat_df.empty:
                    params = pd.concat([cont_df, cat_df], axis=1).values
                else:
                    params = None
            else:
                params = None
            return {"x_init": x_init, "params": params}
        else:
            feature_cols = self.args.get("params_cols", [])
            X = df_processed[feature_cols].values if feature_cols else None
            return {"X": X}
        """




class ModelPipeline:
    """
    A pipeline that wraps:
      - One or more trained PyTorch models (e.g. MLP or VAE+surrogate).
      - All necessary scalers (for shapes, parameters, and/or targets).
      - Flags like `structured` (VAE+surrogate vs. MLP-only) and `log_target`.
    
    Provides:
      - .predict(...) to get predictions from raw data
      - .evaluate(...) to compute errors if ground truth is available
      - .save(...) and .load(...) for serialization
      - .to_device(...) to move all internal models to GPU/MPS/CPU
    """
    def __init__(
        self,
        models: List[torch.nn.Module],
        scalers: Dict[str, Any],
        structured: bool,
        log_target: bool,
        metadata: Optional[dict] = None,
        preprocessor: Optional[DataPreprocessor] = None  # new attribute
    ):
        """
        models:    list of trained PyTorch models. (ensemble or single-model)
        scalers:   dict of scalers, e.g. {
                     "scaler_init": ...,
                     "scaler_opt": ...,
                     "scaler_params": ...,
                     "scaler_y": ...,
                     ...
                  }
        structured: Whether it's the VAE+surrogate approach (True) or MLP-only (False).
        log_target: Whether the target was log-transformed during training.
        """
        self.models = models
        self.scalers = scalers
        self.structured = structured
        self.log_target = log_target
        self.metadata = metadata if metadata is not None else {}
        self.preprocessor = preprocessor  # save preprocessor

        for m in self.models:
            m.eval()
        
    def to_device(self, device: torch.device = torch.device("cpu")):
        """
        Move all models to the specified device ("cuda", "mps", or "cpu").
        """
        for i, m in enumerate(self.models):
            self.models[i] = m.to(device)

    def predict(
        self,
        raw_input,  # raw_input can be a DataFrame (raw data) or a dict/tuple of numpy arrays
        batch_size: int = 256,
        device: torch.device = torch.device("cpu")
    ) -> np.ndarray:
        """
        Predict target values given raw input data.

        For structured mode (VAE+surrogate):
        - raw_input should be a pandas DataFrame containing the raw data.
        - The preprocessor will extract and transform the columns corresponding to:
            * The initial shape (init_col)
            * The parameter columns (params_cols)
        For unstructured mode (MLP-only):
        - raw_input should be a pandas DataFrame containing the raw features.
        
        The preprocessor uses the metadata (stored during training) so that the same
        raw-to-model mapping is applied.

        batch_size: Number of samples per inference batch.
        device: Which device to use for the forward pass.
        """
        # 1) Move models to the chosen device
        self.to_device(device)
        
        # 2) Use the preprocessor to transform raw_input.
        # If raw_input is a DataFrame and a preprocessor exists, use it.
        # Otherwise, assume the caller passed already-processed arrays in a dict.
        if self.preprocessor is not None:
            # The preprocessor returns a dict:
            #   For structured mode: {"x_init": ..., "params": ...}
            #   For unstructured mode: {"X": ...}
            processed = self.preprocessor.transform_inputs(raw_input)
            if self.structured:
                x_init = processed["x_init"]
                params = processed["params"]
            else:
                params = processed["X"]
                x_init = None
        else:
            # If no preprocessor, assume raw_input is a dict with keys "params" and (optionally) "x_init"
            if self.structured:
                x_init = raw_input["x_init"]
                params = raw_input["params"]
            else:
                params = raw_input["X"]
                x_init = None

        # 3) Now apply scaling using the stored scalers
        if "scaler_params" in self.scalers:
            params_scaled = self.scalers["scaler_params"].transform(params)
        elif "scaler_X" in self.scalers:
            params_scaled = self.scalers["scaler_X"].transform(params)
        else:
            params_scaled = params

        params_tensor = torch.from_numpy(params_scaled).float()
        
        if self.structured:
            if x_init is None:
                raise ValueError("Structured mode requires x_init to be provided by the preprocessor.")
            if "scaler_init" not in self.scalers:
                raise ValueError("Structured mode requires 'scaler_init' in the pipeline.")
            x_init_scaled = self.scalers["scaler_init"].transform(x_init)
            x_init_tensor = torch.from_numpy(x_init_scaled).float()
        else:
            x_init_tensor = None

        # 4) Inference in batches
        all_preds = []
        num_samples = params_tensor.shape[0]
        n_batches = math.ceil(num_samples / batch_size)
        
        for b_idx in range(n_batches):
            start = b_idx * batch_size
            end = min(start + batch_size, num_samples)
            
            param_batch = params_tensor[start:end].to(device)
            if x_init_tensor is not None:
                x_init_batch = x_init_tensor[start:end].to(device)
            
            ensemble_batch_preds = []
            for model in self.models:
                with torch.no_grad():
                    if self.structured:
                        # For shape2shape: model(x_init_batch, param_batch) returns (opt_pred, mu, logvar, z, cl_pred)
                        _, _, _, _, cl_pred = model(x_init_batch, param_batch)
                        pred = cl_pred.view(-1)  # shape: [batch]
                    else:
                        pred = model(param_batch).view(-1)
                ensemble_batch_preds.append(pred)
            
            # Average predictions over the ensemble for this batch
            ensemble_batch_preds = torch.stack(ensemble_batch_preds, dim=0)
            batch_mean_preds = ensemble_batch_preds.mean(dim=0)
            all_preds.append(batch_mean_preds.cpu().numpy())
        
        predictions = np.concatenate(all_preds, axis=0)
        
        # 5) Invert target scaling if applicable
        if "scaler_y" in self.scalers:
            predictions = self.scalers["scaler_y"].inverse_transform(predictions.reshape(-1,1)).flatten()
        
        # 6) If log-transform was used during training, apply exponential
        if self.log_target:
            predictions = np.exp(predictions)
        
        return predictions

    def evaluate(
            self,
            raw_input,
            y_true: np.ndarray,
            batch_size: int = 256,
            device: torch.device = torch.device("cpu"),
            metrics: List[str] = ["mse", "rmse", "rel_err"]
        ) -> Dict[str, float]:
        """
        Evaluate the pipeline on a dataset with known ground-truth targets y_true.

        raw_input: Raw input data (e.g. a pandas DataFrame) that the preprocessor will convert.
        y_true: Ground truth target values as a numpy array.
        metrics: List of metrics to compute.
        Returns a dictionary of computed metric values.
        """
        y_pred = self.predict(raw_input, batch_size=batch_size, device=device)
        results = {}
        diff = y_pred - y_true
        epsilon = 1e-12  # for relative error computation
        
        for m in metrics:
            if m == "mse":
                results["mse"] = float(np.mean(diff**2))
            elif m == "rmse":
                results["rmse"] = float(np.sqrt(np.mean(diff**2)))
            elif m == "mae":
                results["mae"] = float(np.mean(np.abs(diff)))
            elif m == "rel_err":
                results["rel_err"] = float(np.mean(np.abs(diff) / (np.abs(y_true) + epsilon)))
            else:
                raise ValueError(f"Unknown metric: {m}")
        
        return results


    def save(self, filepath: str, device: torch.device = torch.device("cpu")):
        """
        Save the entire pipeline (models + scalers + flags) to disk
        using pickle. This lets you load it later and do .predict(...)
        immediately.
        """
        # Usually you'd want models on CPU to avoid device mismatch
        self.to_device(device)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Pipeline saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "ModelPipeline":
        """
        Load a pipeline from a pickle file. 
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"No such file: {filepath}")
        with open(filepath, "rb") as f:
            pipeline = pickle.load(f)
        # By default, pipeline's models are on CPU. 
        # You can .to_device("cuda") if you want
        return pipeline
