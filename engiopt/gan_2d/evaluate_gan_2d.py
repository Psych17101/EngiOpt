"""Evaluation for the GAN 2D."""

from __future__ import annotations

import dataclasses
import os

from dataset_sample_conditions import sample_conditions
from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import torch as th
import tyro

from engiopt import metrics
from engiopt.gan_2d import Generator
import wandb


@dataclasses.dataclass
class Args:
    """Command-line arguments."""

    problem_id: str = "heatconduction2d"
    """Problem identifier."""
    seed: int = 1
    """Random seed."""
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
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    # Seeding
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    th.backends.cudnn.deterministic = True

    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    ### Set up testing conditions ###
    conditions_tensor, sampled_conditions, sampled_designs_np, selected_indices = sample_conditions(
        problem=problem, n_samples=args.n_samples, device=device, seed=args.seed
    )

    ### Set Up Generator ###

    # Restores the pytorch model from wandb
    if args.wandb_entity is not None:
        artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_gan_2d_generator:seed_{args.seed}"
    else:
        artifact_path = f"{args.wandb_project}/{args.problem_id}_gan_2d_generator:seed_{args.seed}"

    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")
    run = artifact.logged_by()
    artifact_dir = artifact.download()

    ckpt_path = os.path.join(artifact_dir, "generator.pth")
    ckpt = th.load(ckpt_path)
    model = Generator(latent_dim=run.config["latent_dim"], design_shape=problem.design_space.shape)
    model.load_state_dict(ckpt["generator"])
    model.eval()  # Set to evaluation mode
    model.to(device)

    # Sample noise as generator input
    z = th.randn((args.n_samples, run.config["latent_dim"]), device=device, dtype=th.float)

    # Generate a batch of designs
    gen_designs = model(z)
    gen_designs_np = gen_designs.detach().cpu().numpy()

    # Clip to boundaries for running THIS IS PROBLEM DEPENDENT
    gen_designs_np = np.clip(gen_designs_np, 1e-3, 1)

    # Compute metrics
    metrics = metrics.metrics(problem, gen_designs_np, sampled_designs_np, sampled_conditions, sigma=args.sigma)

    print(metrics)
