#!/usr/bin/env python3
"""
Single-file AutoGluon multi-modality training script.

This script demonstrates how to use AutoGluon for various modalities (tabular, text, image, object_detection).
It includes best practices like reproducibility (with seeds), experiment tracking with Weights & Biases (W&B), 
and single-file implementation for clarity.

Usage:
    python autoGluon_train.py --modality tabular --train_data_path data/train.csv --label target --track True
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
import time
from typing import Union

import numpy as np
import pandas as pd
import wandb
import tyro
from autogluon.tabular import TabularPredictor
from autogluon.vision import ImagePredictor, ObjectDetector
from autogluon.text import TextPredictor


@dataclass
class Args:
    """
    Command-line arguments for AutoGluon multi-modality modeling.
    
    Attributes:
        modality: Which modality to train on (tabular, text, image, or object_detection).
        train_data_path: Path to training data. 
            - For tabular/text, a CSV file is expected.
            - For image, a directory of images is expected.
            - For object_detection, a JSON or CSV annotation file is expected.
        label: Name of the target column for supervised tasks (tabular/text).
        problem_type: If specified, overrides AutoGluon's problem type (e.g., 'regression', 'classification').
        
        track: If True, initializes a Weights & Biases run for experiment tracking.
        wandb_project: W&B project name.
        wandb_entity: W&B entity (username or team).
        seed: Random seed for reproducibility.
        save_model: If True, saves the trained model (and logs it as a W&B artifact if tracking is enabled).

        time_limit: Time limit (in seconds) for training.
        presets: AutoGluon presets (e.g., 'high_quality', 'medium_quality').
        eval_metric: Evaluation metric for AutoGluon (e.g., 'accuracy', 'f1').
        enable_gpu: Whether to enable GPU training (if available).
    """

    modality: str = "tabular"
    train_data_path: str = "data/train.csv"
    label: str = "target"
    problem_type: str | None = None

    # Tracking
    track: bool = True
    wandb_project: str = "engiopt"
    wandb_entity: str | None = None
    seed: int = 1
    save_model: bool = False

    # AutoGluon-specific
    time_limit: int = 3600
    presets: str = "high_quality"
    eval_metric: str | None = None
    enable_gpu: bool = False


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across numpy and Python's built-in `random`.
    
    Args:
        seed: The seed to set for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)


def validate_inputs(args: Args) -> None:
    """
    Validate input paths and formats to ensure they align with the selected modality.
    
    Raises:
        FileNotFoundError: If the specified `train_data_path` does not exist.
        ValueError: If file format is invalid for the given modality, or if the label column is missing.
    """
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"Path '{args.train_data_path}' does not exist.")

    if args.modality in ["tabular", "text"]:
        if not args.train_data_path.endswith(".csv"):
            raise ValueError(
                f"Expected a CSV file for modality '{args.modality}', but got: {args.train_data_path}"
            )
        train_data = pd.read_csv(args.train_data_path)
        if args.label not in train_data.columns:
            raise ValueError(f"Label column '{args.label}' not found in the dataset.")
    elif args.modality == "image":
        if not os.path.isdir(args.train_data_path):
            raise ValueError(f"Expected a directory of images for modality 'image', but got: {args.train_data_path}")
    elif args.modality == "object_detection":
        if not args.train_data_path.endswith((".json", ".csv")):
            raise ValueError(
                f"Expected a JSON or CSV file for object detection, but got: {args.train_data_path}"
            )
    else:
        raise ValueError(f"Unsupported modality: {args.modality}")


def main(args: Args) -> None:
    """
    Main training entry point for the script. This function:
    
      1. Sets seeds for reproducibility.
      2. Validates input paths and formats.
      3. Optionally initializes a Weights & Biases (W&B) run.
      4. Trains an AutoGluon predictor for the specified modality:
         - 'tabular'
         - 'image'
         - 'object_detection'
         - 'text'
      5. Logs results (e.g., leaderboards or fit summaries) to W&B if requested.
      6. Optionally saves the trained model and logs it as a W&B artifact.
      7. Ends the W&B run if tracking was enabled.
    """
    # 1. Seeds
    set_seed(args.seed)

    # 2. Validate inputs
    validate_inputs(args)

    # Construct a run name for convenience
    run_name = f"{args.modality}__{os.path.basename(args.train_data_path)}__{args.seed}__{int(time.time())}"

    # 3. Initialize Weights & Biases if requested
    wandb_run = None
    if args.track:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            save_code=True,
            name=run_name,
        )

    # 4. Training by modality
    predictor = None
    if args.modality == "tabular":
        # Read CSV, train tabular predictor
        train_data = pd.read_csv(args.train_data_path)
        predictor = TabularPredictor(
            label=args.label,
            problem_type=args.problem_type,
            eval_metric=args.eval_metric
        ).fit(
            train_data=train_data,
            time_limit=args.time_limit,
            presets=args.presets,
            ag_args_fit={"num_gpus": 1 if args.enable_gpu else 0},
            random_seed=args.seed,
        )
        # Log results
        leaderboard = predictor.leaderboard(silent=True)
        print(leaderboard)
        if args.track and wandb_run is not None:
            wandb.log({"leaderboard": wandb.Table(dataframe=leaderboard)})

    elif args.modality == "image":
        # Train image predictor on images in a directory
        predictor = ImagePredictor(path=args.train_data_path)
        predictor.fit(time_limit=args.time_limit, presets=args.presets)
        # Log fit summary
        if args.track and wandb_run is not None:
            wandb.log(predictor.fit_summary())

    elif args.modality == "object_detection":
        # Train object detector
        predictor = ObjectDetector()
        predictor.fit(train_path=args.train_data_path, time_limit=args.time_limit)
        if args.track and wandb_run is not None:
            wandb.log(predictor.fit_summary())

    elif args.modality == "text":
        # Read CSV, train text predictor
        train_data = pd.read_csv(args.train_data_path)
        predictor = TextPredictor(
            label=args.label,
            problem_type=args.problem_type,
            eval_metric=args.eval_metric
        ).fit(
            train_data=train_data,
            time_limit=args.time_limit,
            presets=args.presets
        )

    # 5. & 6. Saving model if requested
    if predictor is not None and args.save_model:
        model_dir = f"{args.modality}_model"
        predictor.save(model_dir)
        # Log model artifact
        if args.track and wandb_run is not None:
            artifact = wandb.Artifact(f"{args.modality}_model", type="model")
            artifact.add_dir(model_dir)
            wandb.log_artifact(artifact)

    # 7. End the W&B run if tracking was enabled
    if args.track and wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    # Parse command-line args
    args = tyro.cli(Args)
    main(args)
