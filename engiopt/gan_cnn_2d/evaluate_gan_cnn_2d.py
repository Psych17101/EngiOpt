"""Evaluation for the CGAN 2D w/ CNN."""

from __future__ import annotations

import dataclasses
import os

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import pandas as pd
import torch as th
import tyro
import wandb

from engiopt import metrics
from engiopt.dataset_sample_conditions import sample_conditions
from engiopt.gan_cnn_2d.gan_cnn_2d import Generator


@dataclasses.dataclass
class Args:
    """Command-line arguments."""

    problem_id: str = "beams2d"
    """Problem identifier."""
    seed_start: int = 1
    """Random starting seed."""
    seed_range: int = 1
    """Range of random seeds."""
    wandb_project: str = "engiopt"
    """Wandb project name."""
    wandb_entity: str | None = None
    """Wandb entity name."""
    n_samples: int = 5
    """Number of generated samples."""
    sigma: float = 10
    """Kernel bandwidth for mmd and dpp"""


if __name__ == "__main__":
    args = tyro.cli(Args)

    for seed in range(args.seed_start, args.seed_start + args.seed_range):
        problem = BUILTIN_PROBLEMS[args.problem_id]()
        problem.reset(seed=seed)

        # Seeding
        th.manual_seed(seed)
        rng = np.random.default_rng(seed)
        th.backends.cudnn.deterministic = True

        if th.backends.mps.is_available():
            device = th.device("mps")
        elif th.cuda.is_available():
            device = th.device("cuda")
        else:
            device = th.device("cpu")

        ### Set up testing conditions ###
        conditions_tensor, sampled_conditions, sampled_designs_np, selected_indices = sample_conditions(
            problem=problem, n_samples=args.n_samples, device=device, seed=seed
        )

        # Reshape to match the expected input shape for the model
        conditions_tensor = conditions_tensor.unsqueeze(-1).unsqueeze(-1)

        ### Set Up Generator ###

        # Restores the pytorch model from wandb
        if args.wandb_entity is not None:
            artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_gan_cnn_2d_generator:seed_{seed}"
        else:
            artifact_path = f"{args.wandb_project}/{args.problem_id}_gan_cnn_2d_generator:seed_{seed}"

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
        ckpt = th.load(ckpt_path, map_location=th.device(device))
        model = Generator(latent_dim=run.config["latent_dim"], design_shape=problem.design_space.shape)
        model.load_state_dict(ckpt["generator"])
        model.eval()  # Set to evaluation mode
        model.to(device)

        # Sample noise as generator input
        z = th.randn((args.n_samples, run.config["latent_dim"], 1, 1), device=device, dtype=th.float)

        # Generate a batch of designs
        gen_designs = model(z)
        gen_designs_np = gen_designs.detach().cpu().numpy()
        gen_designs_np = gen_designs_np.reshape(args.n_samples, *problem.design_space.shape)

        # Clip to boundaries for running THIS IS PROBLEM DEPENDENT
        gen_designs_np = np.clip(gen_designs_np, 1e-3, 1)

        # Compute metrics
        # Initialize dataframe if first iteration
        if seed == args.seed_start:
            results_df = pd.DataFrame()

        # Get metrics and add seed number
        metrics_dict = metrics.metrics(problem, gen_designs_np, sampled_designs_np, sampled_conditions, sigma=args.sigma)
        metrics_dict["seed"] = seed

        # Append to dataframe
        results_df = pd.concat([results_df, pd.DataFrame([metrics_dict])], ignore_index=True)

        # Save after each iteration
        csv_path = f"gan_cnn_2d_{args.problem_id}_metrics.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Seed {seed} done; results saved to {csv_path}")
