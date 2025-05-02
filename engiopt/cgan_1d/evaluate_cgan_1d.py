"""Evaluation for the CGAN 1D."""

from __future__ import annotations

import dataclasses
import os

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import pandas as pd
import torch as th
import tyro

from engiopt import metrics
from engiopt.cgan_1d.cgan_1d import Generator
from engiopt.cgan_1d.cgan_1d import prepare_data
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
    sigma: float = 1.0
    """Kernel bandwidth for MMD and DPP metrics."""


if __name__ == "__main__":
    args = tyro.cli(Args)

    # DataFrame to collect metrics across seeds
    results_df = pd.DataFrame()

    for seed in range(args.seed_start, args.seed_start + args.seed_range):
        # Initialize and reset problem
        problem = BUILTIN_PROBLEMS[args.problem_id]()
        problem.reset(seed=seed)

        # Seeding for reproducibility
        th.manual_seed(seed)
        rng = np.random.default_rng(seed)
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

        ### Set Up Generator ###
        if args.wandb_entity is not None:
            artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_cgan_1d_generator:seed_{seed}"
        else:
            artifact_path = f"{args.wandb_project}/{args.problem_id}_cgan_1d_generator:seed_{seed}"

        api = wandb.Api()
        artifact = api.artifact(artifact_path, type="model")

        class RunRetrievalError(ValueError):
            def __init__(self):
                super().__init__("Failed to retrieve the run")

        run = artifact.logged_by()
        if run is None or not hasattr(run, "config"):
            raise RunRetrievalError

        artifact_dir = artifact.download()
        ckpt_path = os.path.join(artifact_dir, "generator.pth")
        ckpt = th.load(ckpt_path, map_location=device)

        _, conds_normalizer, design_normalizer = prepare_data(problem, device)

        model = Generator(
            latent_dim=run.config["latent_dim"],
            n_conds=len(problem.conditions),
            design_shape=problem.design_space.shape,
            design_normalizer=design_normalizer,
            conds_normalizer=conds_normalizer,
        ).to(device)
        model.load_state_dict(ckpt["generator"])
        model.eval()

        # Sample noise and generate designs
        z = th.randn((args.n_samples, run.config["latent_dim"]), device=device)
        gen_designs = model(z, conditions_tensor)
        gen_designs_np = gen_designs.detach().cpu().numpy()

        # Compute metrics
        metrics_dict = metrics.metrics(
            problem,
            gen_designs_np,
            sampled_designs_np,
            sampled_conditions,
            sigma=args.sigma,
        )
        metrics_dict["seed"] = seed

        # Append and save
        results_df = pd.concat(
            [results_df, pd.DataFrame([metrics_dict])],
            ignore_index=True,
        )
        csv_path = f"cgan_1d_{args.problem_id}_metrics.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Seed {seed} done; results saved to {csv_path}")
