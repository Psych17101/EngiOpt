# model_pipeline.py
import ast
import math
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

MIN_UNIQUE_FOR_ONEHOT = 5

###############################################################################
# DataPreprocessor
###############################################################################
class DataPreprocessor:
    """Handles raw data transformations:
      - Strips spaces and flattens list columns.
      - Applies optional subset filtering, nondimensionalization, and log-transformation.
      - Splits parameter columns into continuous and categorical parts. For columns with
        fewer than 5 unique values, one-hot encoding is applied.
      - Stores the final parameter column names and, for categorical columns, the list of dummy columns.

    Usage in training:
      preprocessor = DataPreprocessor(vars(args))
      processed_dict = preprocessor.transform_inputs(df, fit_params=True)
      -> This returns a dict with keys:
           If structured: { "x_init": ..., "x_opt": ..., "params": ... }
           Else: { "X": ... }
      and it stores self.final_param_columns, self.continuous_columns,
      self.categorical_columns, and self.categorical_mapping.

    Usage in inference:
      (After loading the pipeline, the pipeline calls preprocessor.transform_inputs(raw_df, fit_params=False)
      so that the same dummy columns are created.)
    """
    def __init__(self, args: dict):
        self.args = args.copy()
        self.final_param_columns: Optional[List[str]] = None
        self.continuous_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.categorical_mapping: Dict[str, List[str]] = {}

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1) Strip column spaces if requested.
        if self.args.get("strip_column_spaces", False):
            df.columns = [col.strip() for col in df.columns]

        # 2) Flatten columns if requested.
        flatten_cols = self.args.get("flatten_columns", [])
        if flatten_cols:
            df = self._flatten_list_columns(df, flatten_cols)
            print(f"[DataPreprocessor] Flattened columns: {df.columns}")

        # 3) Subset condition.
        subset_condition = self.args.get("subset_condition", None)
        if subset_condition:
            df = df.query(subset_condition)

        # 4) Nondimensionalization.
        nondim_map = self.args.get("nondim_map", None)
        if nondim_map:
            if isinstance(nondim_map, str):
                nondim_map = ast.literal_eval(nondim_map)
            for col, ref in nondim_map.items():
                if col in df.columns and ref in df.columns:
                    df[col] = df[col] / df[ref]

        # 5) Log-transform target if requested.
        if self.args.get("log_target", False):
            target_col = self.args.get("target_col", None)
            if target_col in df.columns:
                df[target_col] = np.log(df[target_col])
                print(f"[DataPreprocessor] Applied log-transform to {target_col}")

        return df

    def transform_inputs(
        self, 
        df: pd.DataFrame, 
        fit_params: bool = False
    ) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
        """Transforms raw data into arrays for model input, returning both the arrays
        and the processed DataFrame.

        If 'structured' mode is True (VAE with shapes):
        - Returns ({"x_init": ..., "x_opt": ..., "params": ...}, df_processed).

        If 'structured' mode is False (plain MLP):
        - Returns ({"X": ...}, df_processed).

        When 'fit_params' is True, the function applies one-hot/dummy logic to
        parameter columns and updates internal mappings. When False, it uses
        stored mappings from a previous run.

        Args:
            df (pd.DataFrame): The raw DataFrame to be transformed.
            fit_params (bool, optional): If True, fits new parameter mappings
                (e.g., one-hot columns). If False, uses stored mappings.

        Returns:
            Tuple[Dict[str, np.ndarray], pd.DataFrame]:
            - A dictionary of NumPy arrays containing model inputs.
            * In structured mode: {"x_init", "x_opt", "params"}.
            * In unstructured mode: {"X"}.
            - The processed DataFrame (df_processed), which includes flattening,
            filtering, one-hot columns, and any log/nondimensional transformations.
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

            # Process parameter columns
            if fit_params:
                param_df = self._split_params_with_onehot(df_processed, param_cols)
            else:
                param_df = self._apply_params_inference(df_processed, param_cols)
            params = param_df.values if len(param_df.columns) > 0 else np.empty((len(df_processed), 0))
            return {"x_init": x_init, "x_opt": x_opt, "params": params}, df_processed
        else:
            # Unstructured mode
            if fit_params:
                param_df = self._split_params_with_onehot(df_processed, param_cols)
            else:
                param_df = self._apply_params_inference(df_processed, param_cols)
            X = param_df.values if len(param_df.columns) > 0 else np.empty((len(df_processed), 0))
            return {"X": X}, df_processed

    def _split_params_with_onehot(self, df: pd.DataFrame, param_cols: List[str]) -> pd.DataFrame:
        """Splits the specified parameter columns into continuous and categorical subsets,
        then applies one-hot encoding to categorical columns that have fewer than five unique values.

        This method is only called during the fitting phase (`fit_params=True`). It updates internal
        attributes such as `self.continuous_columns`, `self.categorical_columns`, and
        `self.categorical_mapping`.

        Args:
            df (pd.DataFrame): The preprocessed DataFrame containing parameter columns.
            param_cols (List[str]): The list of columns to be split into continuous or categorical.

        Raises:
            ValueError: If a specified parameter column does not exist in `df`.

        Returns:
            pd.DataFrame: A new DataFrame with continuous columns followed by one-hot-encoded columns.
        """
        if not param_cols:
            return pd.DataFrame(index=df.index)
        cont_list = []
        cat_list = []
        self.continuous_columns = []
        self.categorical_columns = []
        self.categorical_mapping = {}
        for col in param_cols:
            if col not in df.columns:
                raise ValueError(f"[DataPreprocessor] Parameter column '{col}' not found.")
            if df[col].nunique() < MIN_UNIQUE_FOR_ONEHOT:
                dummies = pd.get_dummies(df[col], prefix=col)
                cat_list.append(dummies)
                self.categorical_columns.append(col)
                self.categorical_mapping[col] = list(dummies.columns)
            else:
                cont_list.append(df[[col]])
                self.continuous_columns.append(col)
        cont_df = pd.concat(cont_list, axis=1) if cont_list else pd.DataFrame(index=df.index)
        cat_df = pd.concat(cat_list, axis=1) if cat_list else pd.DataFrame(index=df.index)
        final_df = pd.concat([cont_df, cat_df], axis=1)
        self.final_param_columns = list(final_df.columns)
        return final_df

    def _apply_params_inference(self, df: pd.DataFrame, param_cols: List[str]) -> pd.DataFrame:
        """Applies the parameter splitting logic in inference mode. Uses previously
        stored continuous/categorical mappings to reconstruct the same dummy columns
        as during training.

        For continuous columns, the method copies them directly if they exist, or
        creates columns of NaNs if missing. For categorical columns, the method
        re-creates dummies and reindexes them to match the training dummy columns.

        Args:
            df (pd.DataFrame): The DataFrame containing parameter columns for inference.
            param_cols (List[str]): The list of columns to be processed.

        Returns:
            pd.DataFrame: The transformed DataFrame with continuous and dummy columns
            in the same order as during training.
        """
        if not param_cols:
            return pd.DataFrame(index=df.index)
        # For continuous columns: simply get the column.
        cont_dfs = []
        for col in self.continuous_columns:
            if col in df.columns:
                cont_dfs.append(df[[col]])
            else:
                # If a continuous column is missing, create a column of NaNs.
                cont_dfs.append(pd.DataFrame(np.nan, index=df.index, columns=[col]))
        cont_df = pd.concat(cont_dfs, axis=1) if cont_dfs else pd.DataFrame(index=df.index)

        # For categorical columns: re-create dummies and reindex with stored dummy columns.
        cat_dfs = []
        for col in self.categorical_columns:
            if col not in df.columns:
                # Create a DataFrame with zeros for missing categorical column.
                dummy_cols = self.categorical_mapping.get(col, [])
                cat_dfs.append(pd.DataFrame(0, index=df.index, columns=dummy_cols))
            else:
                dummies = pd.get_dummies(df[col], prefix=col)
                # Reindex to ensure the same dummy columns as in training.
                expected_cols = self.categorical_mapping.get(col, [])
                dummies = dummies.reindex(columns=expected_cols, fill_value=0)
                cat_dfs.append(dummies)
        cat_df = pd.concat(cat_dfs, axis=1) if cat_dfs else pd.DataFrame(index=df.index)

        final_df = pd.concat([cont_df, cat_df], axis=1)
        # Reorder columns to match the training order.
        if self.final_param_columns:
            final_df = final_df.reindex(columns=self.final_param_columns, fill_value=0)
        return final_df

    def _flatten_list_columns(self, df: pd.DataFrame, columns_to_flatten: list) -> pd.DataFrame:
        """
        Flattens specified columns in which each entry can be a list, tuple, or numpy.ndarray.
        For each flattened column, creates new columns named '{original_col}_{index}'.
        """
        new_cols_list = []
        drop_cols = []

        for col in columns_to_flatten:
            if col not in df.columns:
                continue

            # Check the first row's value.
            first_val = df[col].iloc[0]

            # If it's a string that might represent a list, try literal_eval.
            if isinstance(first_val, str):
                try:
                    _ = ast.literal_eval(first_val)
                    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                except Exception as e:
                    print(f"[WARN] Could not parse '{col}' as string literal: {e}. Proceeding.")

            # If the value is not a list/tuple/ndarray, but is iterable, cast it.
            if not isinstance(df[col].iloc[0], (list, tuple, np.ndarray)):
                if hasattr(df[col].iloc[0], "__iter__") and not isinstance(df[col].iloc[0], str):
                    df[col] = df[col].apply(lambda x: list(x) if x is not None else x)

            # Also, if it is a numpy array, convert it to a list.
            if isinstance(df[col].iloc[0], np.ndarray):
                df[col] = df[col].apply(lambda x: x.tolist())

            # Now, if we have a list or tuple, flatten.
            if isinstance(df[col].iloc[0], (list, tuple)):
                # Use recursive flattening.
                flattened_rows = [self._recursive_flatten(x) for x in df[col]]
                lengths = [len(row) for row in flattened_rows]
                if len(set(lengths)) > 1:
                    raise ValueError(f"Column '{col}' has varying lengths among rows: {set(lengths)}")
                n = lengths[0]
                new_col_names = [f"{col}_{i}" for i in range(n)]
                expanded_df = pd.DataFrame(flattened_rows, columns=new_col_names)
                new_cols_list.append(expanded_df)
                drop_cols.append(col)
            else:
                print(f"Column '{col}' is not recognized as list-like; skipping flatten.")

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
      - One or more trained PyTorch models (ensemble or single-model),
      - Fitted scalers for shape/parameter arrays and target,
      - A DataPreprocessor to apply the same transformations at inference,
      - Convenience methods for predict, evaluate, save, and load.
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
        self.models = models
        self.scalers = scalers
        self.structured = structured
        self.log_target = log_target
        self.metadata = metadata if metadata is not None else {}
        self.preprocessor = preprocessor
        for m in self.models:
            m.eval()

    def to_device(self, device: torch.device = torch.device("cpu")):
        """Moves all models in the pipeline to the specified device (CPU or GPU).

        Args:
            device (torch.device, optional): The device to move the models onto.
                Defaults to CPU (torch.device("cpu")).
        """
        for i, m in enumerate(self.models):
            self.models[i] = m.to(device)

    def predict(
        self,
        raw_input: pd.DataFrame,
        batch_size: int = 256,
        device: torch.device = torch.device("cpu")
    ) -> np.ndarray:
        """Predicts the target values from raw input data using the stored ensemble of models.

        Steps:
            1) Applies the stored DataPreprocessor to transform raw_input into the model's 
            required format (structured or unstructured).
            2) Scales parameters and optionally shape arrays, then converts them to torch.Tensors.
            3) Batches the data to avoid memory issues, runs a forward pass for each model in the ensemble,
            and averages their predictions.
            4) If 'scaler_y' is found, inverses the scaling. If log_target is True, exponentiates the output.

        Args:
            raw_input (pd.DataFrame): The raw features to predict on.
            batch_size (int, optional): The batch size for inference. Defaults to 256.
            device (torch.device, optional): The device (CPU/GPU) to perform inference on. Defaults to CPU.

        Raises:
            ValueError: If no preprocessor is found (meaning the pipeline is incomplete), 
                        or if required scalers for structured mode are missing.

        Returns:
            np.ndarray: 1D array of predictions for each row in `raw_input`.
        """
        self.to_device(device)
        if not self.preprocessor:
            raise ValueError("No DataPreprocessor in pipeline; cannot transform raw data.")
        processed, _df = self.preprocessor.transform_inputs(raw_input, fit_params=False)
        if self.structured:
            x_init = processed["x_init"]
            params = processed["params"]
        else:
            x_init = None
            params = processed["X"]

        # Scale parameters
        if "scaler_params" in self.scalers:
            params_scaled = self.scalers["scaler_params"].transform(params)
        elif "scaler_X" in self.scalers:
            params_scaled = self.scalers["scaler_X"].transform(params)
        else:
            params_scaled = params
        params_tensor = torch.from_numpy(params_scaled).float()

        if self.structured:
            if "scaler_init" not in self.scalers:
                raise ValueError("Missing 'scaler_init' for structured mode.")
            x_init_scaled = self.scalers["scaler_init"].transform(x_init)
            x_init_tensor = torch.from_numpy(x_init_scaled).float()
        else:
            x_init_tensor = None

        # Batch inference
        all_preds = []
        num_samples = params_tensor.shape[0]
        n_batches = math.ceil(num_samples / batch_size)
        for b_idx in range(n_batches):
            start = b_idx * batch_size
            end = min(start + batch_size, num_samples)
            param_batch = params_tensor[start:end].to(device)
            if x_init_tensor is not None:
                x_init_batch = x_init_tensor[start:end].to(device)
            ensemble_preds = []
            for model in self.models:
                with torch.no_grad():
                    if self.structured:
                        # For structured mode, the model returns (x_opt_pred, mu, logvar, z, cl_pred)
                        _, _, _, _, cl_pred = model(x_init_batch, param_batch)
                        pred = cl_pred.view(-1)
                    else:
                        pred = model(param_batch).view(-1)
                ensemble_preds.append(pred)
            ensemble_preds = torch.stack(ensemble_preds, dim=0)
            batch_mean = ensemble_preds.mean(dim=0)
            all_preds.append(batch_mean.cpu().numpy())
        predictions = np.concatenate(all_preds, axis=0)
        if "scaler_y" in self.scalers:
            predictions = self.scalers["scaler_y"].inverse_transform(predictions.reshape(-1,1)).flatten()
        if self.log_target:
            predictions = np.exp(predictions)
        return predictions

    def evaluate(
        self,
        raw_input: pd.DataFrame,
        y_true: np.ndarray,
        batch_size: int = 256,
        device: torch.device = torch.device("cpu"),
        metrics: List[str] = ["mse", "rmse", "rel_err"]
    ) -> Dict[str, float]:
        """Evaluates the pipeline's predictions against ground-truth values using one or more metrics.

        Steps:
            1) Calls self.predict(...) to get predictions for the provided raw_input.
            2) Computes the error between predictions and the given y_true.
            3) Aggregates and returns the specified metrics in a dictionary.

        Args:
            raw_input (pd.DataFrame): The raw features for which to evaluate predictions.
            y_true (np.ndarray): The ground-truth target values, same length as raw_input.
            batch_size (int, optional): Batch size for prediction. Defaults to 256.
            device (torch.device, optional): The device for inference. Defaults to CPU.
            metrics (List[str], optional): The metrics to compute. 
                Supported options include "mse", "rmse", "mae", and "rel_err". Defaults to ["mse", "rmse", "rel_err"].

        Raises:
            ValueError: If an unknown metric is requested.

        Returns:
            Dict[str, float]: A dictionary of metric names mapped to their computed values (floats).
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
                results["rel_err"] = float(np.mean(np.abs(diff) / (np.abs(y_true) + epsilon)))
            else:
                raise ValueError(f"Unknown metric: {m}")
        return results

    def save(self, filepath: str, device: torch.device = torch.device("cpu")):
        """Saves the entire pipeline (models, scalers, preprocessor, etc.) to a file.

        1) Moves models to the specified device to ensure they are in a consistent state.
        2) Serializes the pipeline object using pickle.
        3) Writes the pickle file to the specified filepath.

        Args:
            filepath (str): The path where the pipeline should be saved (e.g., "pipeline.pkl").
            device (torch.device, optional): The device to move models onto before saving.
                Defaults to CPU.

        Raises:
            OSError: If there is any issue writing to the file.
        """
        self.to_device(device)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Pipeline saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "ModelPipeline":
        """Loads a serialized pipeline from a file.

        The pipeline includes:
        - One or more trained PyTorch models,
        - Any fitted scalers for features and/or targets,
        - A DataPreprocessor instance with its stored mappings.

        Args:
            filepath (str): The path to the pickle file containing the serialized pipeline.

        Raises:
            FileNotFoundError: If the specified filepath does not exist.

        Returns:
            ModelPipeline: An instance of ModelPipeline with models, scalers, and preprocessor.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"No such file: {filepath}")
        with open(filepath, "rb") as f:
            pipeline = pickle.load(f)
        return pipeline
