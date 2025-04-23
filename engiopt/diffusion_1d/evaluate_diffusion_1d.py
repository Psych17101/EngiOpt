"""Evaluation for the Diffusion 1d."""

from __future__ import annotations

import dataclasses
import os

from denoising_diffusion_pytorch import GaussianDiffusion1D
from denoising_diffusion_pytorch import Unet1D
from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import pandas as pd
import torch as th
import tyro

from engiopt import metrics
from engiopt.dataset_sample_conditions import sample_conditions
import wandb


@dataclasses.dataclass
class Args:
    """Command-line arguments."""

    problem_id: str = "airfoil"
    """Problem identifier."""
    seed_start: int = 1
    """Random starting seed."""
    seed_range: int = 1
    """Range of random seeds to run."""
    wandb_project: str = "engiopt"
    """Wandb project name."""
    wandb_entity: str | None = None
    """Wandb entity name."""
    n_samples: int = 5
    """Number of generated samples per seed."""
    sigma: float = 10.0
    """Kernel bandwidth for MMD and DPP metrics."""


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Initialize an empty DataFrame to collect results across seeds
    results_df = pd.DataFrame()

    for seed in range(args.seed_start, args.seed_start + args.seed_range):
        # Instantiate and reset problem
        problem = BUILTIN_PROBLEMS[args.problem_id]()
        problem.reset(seed=seed)

        # Seeding for reproducibility
        th.manual_seed(seed)
        np.random.seed(seed)
        th.backends.cudnn.deterministic = True

        # Select device
        if th.backends.mps.is_available():
            device = th.device("mps")
        elif th.cuda.is_available():
            device = th.device("cuda")
        else:
            device = th.device("cpu")

        ### Set up testing conditions ###
        conditions_tensor, sampled_conditions, sampled_designs_np, _ = sample_conditions(
            problem=problem,
            n_samples=args.n_samples,
            device=device,
            seed=seed,
        )

        # Restore the diffusion model from wandb
        if args.wandb_entity is not None:
            artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_diffusion_1d_model:seed_{seed}"
        else:
            artifact_path = f"{args.wandb_project}/{args.problem_id}_diffusion_1d_model:seed_{seed}"

        api = wandb.Api()
        artifact = api.artifact(artifact_path, type="model")

        class RunRetrievalError(ValueError):
            def __init__(self):
                super().__init__("Failed to retrieve the run")

        run = artifact.logged_by()
        if run is None or not hasattr(run, "config"):
            raise RunRetrievalError()

        artifact_dir = artifact.download()
        ckpt_path = os.path.join(artifact_dir, "model.pth")
        ckpt = th.load(ckpt_path, map_location=device)

        # Build model & diffusion
        model = Unet1D(
            dim=run.config["unet_dim"],  # sinusoidal positional embedding size
            channels=run.config["n_channels"],  # input channel count
        ).to(device)

        diffusion = GaussianDiffusion1D(
            model,
            seq_length=int(np.prod(problem.design_space.shape)),
            auto_normalize=run.config["auto_norm"],
        ).to(device)

        diffusion.load_state_dict(ckpt["model"])
        diffusion.eval()

        # Generate designs
        gen_designs = diffusion.sample(args.n_samples)
        gen_designs_np = gen_designs.detach().cpu().numpy()
        gen_designs_np = gen_designs_np.reshape(args.n_samples, *problem.design_space.shape)

        # Clip to valid range (problem dependent)
        gen_designs_np = np.clip(gen_designs_np, 1e-3, 1.0)

        # Compute metrics
        metrics_dict = metrics.metrics(
            problem,
            gen_designs_np,
            sampled_designs_np,
            sampled_conditions,
            sigma=args.sigma,
        )
        metrics_dict["seed"] = seed

        # Append to results DataFrame
        results_df = pd.concat(
            [results_df, pd.DataFrame([metrics_dict])],
            ignore_index=True,
        )

        # Save intermediate CSV after each seed
        csv_path = f"diffusion_1d_{args.problem_id}_metrics.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Seed {seed} done; results saved to {csv_path}")
