# model_pipeline.py
import os
import pickle
import torch
import numpy as np
import math
from typing import List, Optional, Dict, Any
import ast
import pandas as pd
from sklearn.preprocessing import RobustScaler

###############################################################################
# DataPreprocessor
###############################################################################
class DataPreprocessor:
    """
    A single place to handle raw data transformations, including:
      - flattening columns,
      - optional subset filtering,
      - optional nondimensionalizing,
      - optional log-transform of the target,
      - splitting parameters into continuous/categorical (with one-hot if <5 unique),
      - storing final parameter columns for consistent inference.

    Usage in training script:
      1) preprocessor = DataPreprocessor(vars(args))
      2) processed_dict = preprocessor.transform_inputs(df, fit_params=True)
         -> obtains x_init, x_opt, params or X (depending on 'structured')
         -> final param columns are recorded in self.final_param_columns
      3) do your train/val/test split, scaling, model training, etc.
      4) attach preprocessor + scalers to your pipeline for inference.

    Usage in inference:
      1) pipeline = ModelPipeline.load("...")
      2) pipeline.predict(raw_df)  # pipeline internally calls transform_inputs(..., fit_params=False)
    """

    def __init__(self, args: dict):
        """
        args: typically vars(args) from your training script,
              containing keys like 'structured', 'init_col', 'opt_col', etc.
        """
        self.args = args.copy()

        # We'll store the final param columns after one-hot, so that
        # at inference we re-use the same columns/order.
        self.final_param_columns: Optional[List[str]] = None

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies:
          1) strip column spaces (if requested),
          2) flatten list columns (if requested),
          3) subset condition (if any),
          4) nondimensionalization (if any),
          5) log-transform of the target column (if requested).

        Returns the cleaned DataFrame.
        """
        # 1) Strip spaces
        if self.args.get("strip_column_spaces", False):
            df.columns = [col.strip() for col in df.columns]

        # 2) Flatten columns
        flatten_cols = self.args.get("flatten_columns", [])
        if flatten_cols:
            df = self._flatten_list_columns(df, flatten_cols)

        # 3) Subset condition
        subset_condition = self.args.get("subset_condition", None)
        if subset_condition:
            df = df.query(subset_condition)

        # 4) Nondimensionalization
        nondim_map = self.args.get("nondim_map", None)
        if nondim_map:
            if isinstance(nondim_map, str):
                nondim_map = ast.literal_eval(nondim_map)
            for col, ref in nondim_map.items():
                if col in df.columns and ref in df.columns:
                    df[col] = df[col] / df[ref]

        # 5) Log-transform of target
        if self.args.get("log_target", False):
            target_col = self.args.get("target_col", None)
            if target_col in df.columns:
                df[target_col] = np.log(df[target_col])
                print(f"[DataPreprocessor] Applied log-transform to {target_col}")

        return df

    def transform_inputs(self, df: pd.DataFrame, fit_params: bool = False) -> Dict[str, np.ndarray]:
        """
        Called during training with fit_params=True:
          - We do param splitting + one-hot, record final columns in self.final_param_columns.

        Called during inference with fit_params=False:
          - We reuse self.final_param_columns so that the same columns are selected in the same order.

        Returns a dictionary:
          If structured=True => {"x_init": ..., "x_opt": ..., "params": ...}
          If structured=False => {"X": ...}
        """
        df_processed = self.preprocess_dataframe(df.copy())

        structured = self.args.get("structured", False)
        init_col = self.args.get("init_col", "")
        opt_col  = self.args.get("opt_col", "")
        param_cols = self.args.get("params_cols", [])

        if structured:
            # Gather shape columns
            init_cols = [c for c in df_processed.columns if c.startswith(init_col + "_")]
            opt_cols  = [c for c in df_processed.columns if c.startswith(opt_col + "_")]
            x_init = df_processed[init_cols].values if init_cols else None
            x_opt  = df_processed[opt_cols].values if opt_cols else None

            # Gather param columns
            if fit_params:
                param_df = self._split_params_with_onehot(df_processed, param_cols)
                self.final_param_columns = list(param_df.columns)
            else:
                if not self.final_param_columns:
                    raise ValueError("No final_param_columns known – "
                                     "did you run transform_inputs(..., fit_params=True) at training?")
                param_df = df_processed[self.final_param_columns] if len(self.final_param_columns) > 0 \
                           else pd.DataFrame(index=df_processed.index)

            params = param_df.values if len(param_df.columns) > 0 else np.empty((len(df_processed), 0))

            return {"x_init": x_init, "x_opt": x_opt, "params": params}

        else:
            # Unstructured MLP
            if fit_params:
                param_df = self._split_params_with_onehot(df_processed, param_cols)
                self.final_param_columns = list(param_df.columns)
            else:
                if not self.final_param_columns:
                    raise ValueError("No final_param_columns known – "
                                     "did you run transform_inputs(..., fit_params=True) at training?")
                param_df = df_processed[self.final_param_columns] if len(self.final_param_columns) > 0 \
                           else pd.DataFrame(index=df_processed.index)

            X = param_df.values if len(param_df.columns) > 0 else np.empty((len(df_processed), 0))
            return {"X": X}, df_processed

    def _split_params_with_onehot(self, df: pd.DataFrame, param_cols: List[str]) -> pd.DataFrame:
        """
        1) For each param col in param_cols:
           - If it has fewer than 5 unique values, one-hot encode
           - else treat it as continuous.
        2) Concatenate final columns (continuous first, then one-hot).
        3) Return the resulting DataFrame.
        """
        if not param_cols:
            return pd.DataFrame(index=df.index)

        cont_list = []
        cat_list = []
        for col in param_cols:
            if col not in df.columns:
                raise ValueError(f"[DataPreprocessor] param col '{col}' not found in DataFrame.")
            # one-hot if <5 unique
            if df[col].nunique() < 5:
                dummies = pd.get_dummies(df[col], prefix=col)
                cat_list.append(dummies)
            else:
                cont_list.append(df[[col]])

        cont_df = pd.concat(cont_list, axis=1) if cont_list else pd.DataFrame(index=df.index)
        cat_df  = pd.concat(cat_list, axis=1) if cat_list else pd.DataFrame(index=df.index)
        final_df = pd.concat([cont_df, cat_df], axis=1)
        return final_df

    def _flatten_list_columns(self, df: pd.DataFrame, columns_to_flatten: list) -> pd.DataFrame:
        new_cols_list = []
        drop_cols = []
        for col in columns_to_flatten:
            if col not in df.columns:
                continue
            first_val = df[col].iloc[0]
            if isinstance(first_val, str):
                try:
                    # attempt to parse string->list
                    _ = ast.literal_eval(first_val)
                    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                except:
                    continue
            if isinstance(df[col].iloc[0], (list, tuple)):
                flattened_rows = [self._recursive_flatten(x) for x in df[col]]
                lengths = [len(row) for row in flattened_rows]
                if len(set(lengths)) > 1:
                    raise ValueError(f"Column '{col}' has varying lengths among rows.")
                n = lengths[0]
                new_col_names = [f"{col}_{i}" for i in range(n)]
                expanded_df = pd.DataFrame(flattened_rows, columns=new_col_names)
                new_cols_list.append(expanded_df)
                drop_cols.append(col)
        if new_cols_list:
            df = pd.concat([df.drop(columns=drop_cols).reset_index(drop=True)] + new_cols_list, axis=1)
        return df

    def _recursive_flatten(self, val):
        if not isinstance(val, (list, tuple)):
            return [val]
        result = []
        for item in val:
            result.extend(self._recursive_flatten(item))
        return result


###############################################################################
# ModelPipeline
###############################################################################
class ModelPipeline:
    """
    Wraps:
      - One or more trained PyTorch models (ensemble or single-model).
      - All necessary scalers (for shapes, parameters, targets).
      - A DataPreprocessor that knows how to replicate the same transformations at inference.
      - A few convenience methods: predict, evaluate, save, load, to_device.
    """
    def __init__(
        self,
        models: List[torch.nn.Module],
        scalers: Dict[str, Any],
        structured: bool,
        log_target: bool,
        metadata: Optional[dict] = None,
        preprocessor: Optional[DataPreprocessor] = None
    ):
        """
        models:     list of trained PyTorch models (an ensemble or single model)
        scalers:    dict of fitted scalers, e.g.:
                     {
                       "scaler_init": ...,
                       "scaler_opt": ...,
                       "scaler_params": ... or "scaler_X": ...,
                       "scaler_y": ...,
                       ...
                     }
        structured: whether it's a VAE+Surrogate (True) or MLP-only (False)
        log_target: whether the target was log-transformed at training
        metadata:   optional dictionary with training info (args, param_feature_names, etc.)
        preprocessor: the same DataPreprocessor used during training
        """
        self.models = models
        self.scalers = scalers
        self.structured = structured
        self.log_target = log_target
        self.metadata = metadata if metadata is not None else {}
        self.preprocessor = preprocessor

        # Set models to eval mode by default
        for m in self.models:
            m.eval()

    def to_device(self, device: torch.device = torch.device("cpu")):
        """
        Move all internal models to the specified device, e.g. "cuda", "mps", or "cpu".
        """
        for i, m in enumerate(self.models):
            self.models[i] = m.to(device)

    def predict(
        self,
        raw_input: pd.DataFrame,
        batch_size: int = 256,
        device: torch.device = torch.device("cpu")
    ) -> np.ndarray:
        """
        Inference method. Takes raw data (a pandas DataFrame) -> calls
        self.preprocessor.transform_inputs(..., fit_params=False) -> scales -> forward pass
        on each model -> averages ensemble predictions -> returns final predicted targets.
        """
        # 1) Move models to chosen device
        self.to_device(device)

        # 2) Use the preprocessor to transform raw_input
        if not self.preprocessor:
            raise ValueError("No DataPreprocessor in pipeline; cannot transform raw data.")
        #processed_dict = self.preprocessor.transform_inputs(raw_input, fit_params=False)
        processed_dict, df_processed = self.preprocessor.transform_inputs(raw_input, fit_params=False)

        # 3) Distinguish structured vs. unstructured
        if self.structured:
            x_init = processed_dict["x_init"]
            # x_opt isn't used at inference time, only x_init + params
            params = processed_dict["params"]
        else:
            x_init = None
            params = processed_dict["X"]

        # 4) Apply the param scaler
        if "scaler_params" in self.scalers:
            params_scaled = self.scalers["scaler_params"].transform(params)
        elif "scaler_X" in self.scalers:
            params_scaled = self.scalers["scaler_X"].transform(params)
        else:
            params_scaled = params
        params_tensor = torch.from_numpy(params_scaled).float()

        # If structured, also scale x_init
        if self.structured:
            if "scaler_init" not in self.scalers:
                raise ValueError("Missing 'scaler_init' in pipeline for structured approach.")
            x_init_scaled = self.scalers["scaler_init"].transform(x_init)
            x_init_tensor = torch.from_numpy(x_init_scaled).float()
        else:
            x_init_tensor = None

        # 5) Batch inference
        all_preds = []
        num_samples = params_tensor.shape[0]
        n_batches = math.ceil(num_samples / batch_size)

        for b_idx in range(n_batches):
            start = b_idx * batch_size
            end = min(start + batch_size, num_samples)

            param_batch = params_tensor[start:end].to(device)
            if x_init_tensor is not None:
                x_init_batch = x_init_tensor[start:end].to(device)

            # Accumulate predictions from each ensemble member
            ensemble_batch_preds = []
            for model_e in self.models:
                with torch.no_grad():
                    if self.structured:
                        # shape2shape => model(x_init_batch, param_batch) => (x_opt_pred, mu, logvar, z, cl_pred)
                        _, _, _, _, cl_pred = model_e(x_init_batch, param_batch)
                        pred = cl_pred.view(-1)
                    else:
                        # MLP => model(param_batch) => shape [batch,1]
                        pred = model_e(param_batch).view(-1)
                ensemble_batch_preds.append(pred)

            # Average across ensemble
            ensemble_batch_preds = torch.stack(ensemble_batch_preds, dim=0)  # shape (#models, batch_size)
            batch_mean = ensemble_batch_preds.mean(dim=0)                    # shape (batch_size,)
            all_preds.append(batch_mean.cpu().numpy())

        predictions = np.concatenate(all_preds, axis=0)

        # 6) Invert target scaling if needed
        if "scaler_y" in self.scalers:
            predictions = self.scalers["scaler_y"].inverse_transform(
                predictions.reshape(-1, 1)
            ).flatten()

        # 7) If log_target was used, exponentiate
        if self.log_target:
            predictions = np.exp(predictions)

        return predictions

    def evaluate(
        self,
        raw_input: pd.DataFrame,
        y_true: np.ndarray,
        batch_size: int = 256,
        device: torch.device = torch.device("cpu"),
        metrics: List[str] = ("mse", "rmse", "rel_err")
    ) -> Dict[str, float]:
        """
        Compute predictions and compare to known y_true. Return dict of specified metrics.
        Allowed metrics: "mse", "rmse", "mae", "rel_err".
        """
        y_pred = self.predict(raw_input, batch_size=batch_size, device=device)
        diff = y_pred - y_true
        epsilon = 1e-12

        results = {}
        for m in metrics:
            if m == "mse":
                results["mse"] = float(np.mean(diff**2))
            elif m == "rmse":
                results["rmse"] = float(np.sqrt(np.mean(diff**2)))
            elif m == "mae":
                results["mae"] = float(np.mean(np.abs(diff)))
            elif m == "rel_err":
                # mean relative error
                val = np.mean(np.abs(diff) / (np.abs(y_true) + epsilon))
                results["rel_err"] = float(val)
            else:
                raise ValueError(f"Unknown metric: {m}")
        return results

    def save(self, filepath: str, device: torch.device = torch.device("cpu")):
        """
        Serialize entire pipeline (models + scalers + preprocessor + metadata) to a pickle.
        Typically move models to CPU first to avoid device mismatch on loading.
        """
        self.to_device(device)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Pipeline saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "ModelPipeline":
        """
        Load a pipeline from disk. By default, models come back on CPU.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"No such file: {filepath}")
        with open(filepath, "rb") as f:
            pipeline = pickle.load(f)
        return pipeline
