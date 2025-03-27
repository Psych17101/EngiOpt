"""Evaluation for the CGAN 1D."""

from __future__ import annotations

import dataclasses
import os

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import torch as th
import tyro

from engiopt.cgan_1d import Generator
import wandb


@dataclasses.dataclass
class Args:
    """Command-line arguments."""

    problem_id: str = "airfoil2d"
    """Problem identifier."""
    seed: int = 1
    """Random seed."""
    wandb_project: str = "engiopt"
    """Wandb project name."""
    wandb_entity: str | None = None
    """Wandb entity name."""
    wandb_artifact: str = "latest"


if __name__ == "__main__":
    args = tyro.cli(Args)
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)
    design_shape = problem.design_space.shape

    # Restores the pytorch model from wandb
    if args.wandb_entity is not None:
        artifact_path = f"{args.wandb_entity}/{args.wandb_project}/cgan_1d_generator:seed_{args.seed}"
    else:
        artifact_path = f"{args.wandb_project}/cgan_1d_generator:seed_{args.seed}"
    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")
    run = artifact.logged_by()
    artifact_dir = artifact.download()

    ckpt_path = os.path.join(artifact_dir, "generator.pth")
    ckpt = th.load(ckpt_path)
    model = Generator(latent_dim=run.config["latent_dim"], n_objs=run.config["n_objs"], design_shape=design_shape)
    model.load_state_dict(ckpt["generator"])
    model.eval()  # Set to evaluation mode
    print(model)
