"""This code is largely based on the excellent PyTorch GAN repo: https://github.com/eriklindernoren/PyTorch-GAN.

We essentially refreshed the Python style, use wandb for logging, and made a few little improvements.
There are also a couple of code parts that are problem dependent and need to be adjusted for the specific problem.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
import time

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch import nn
import tqdm
import tyro
import wandb


@dataclass
class Args:
    """Command-line arguments."""

    problem_id: str = "airfoil2d"
    """Problem identifier."""
    algo: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this algorithm."""

    # Tracking
    track: bool = True
    """Track the experiment with wandb."""
    wandb_project: str = "engiopt"
    """Wandb project name."""
    wandb_entity: str | None = None
    """Wandb entity name."""
    seed: int = 1
    """Random seed."""
    save_model: bool = False
    """Saves the model to disk."""

    # Algorithm specific
    n_epochs: int = 200
    """number of epochs of training"""
    batch_size: int = 64
    """size of the batches"""
    lr: float = 3e-4
    """learning rate"""
    b1: float = 0.5
    """decay of first order momentum of gradient"""
    b2: float = 0.999
    """decay of first order momentum of gradient"""
    n_cpu: int = 8
    """number of cpu threads to use during batch generation"""
    latent_dim: int = 100
    """dimensionality of the latent space"""
    n_objs: int = 2
    """number of objectives -- used as conditional input"""
    sample_interval: int = 400
    """interval between image samples"""


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_feat: int, out_feat: int, normalize: bool = True) -> list:  # noqa: FBT001, FBT002
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(args.latent_dim + args.n_objs, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(design_shape))),
            nn.Tanh(),
        )

    def forward(self, z: th.Tensor, objs: th.Tensor) -> th.Tensor:  # noqa: D102
        # Concatenate noise and objs to produce input
        gen_input = th.cat((z, objs), -1)
        design = self.model(gen_input)
        design = design.view(design.size(0), *design_shape)
        return design


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(design_shape)) + args.n_objs, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, design: th.Tensor, objs: th.Tensor) -> th.Tensor:  # noqa: D102
        design_flat = design.view(design.size(0), -1)
        d_in = th.cat((design_flat, objs), -1)
        validity = self.model(d_in)

        return validity


if __name__ == "__main__":
    args = tyro.cli(Args)

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    design_shape = problem.design_space.shape

    # Logging
    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), save_code=True, name=run_name)

    # Seeding
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    th.backends.cudnn.deterministic = True

    os.makedirs("images", exist_ok=True)

    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    # Loss function
    adversarial_loss = th.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)

    # Configure data loader
    training_ds = problem.dataset.with_format("torch", device=device)["train"]
    print(training_ds.shape)
    dataloader = th.utils.data.DataLoader(
        training_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Optimizers
    optimizer_generator = th.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_discriminator = th.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    @th.no_grad()
    def sample_designs(n_designs: int) -> th.Tensor:
        """Samples n_designs from the generator."""
        # Sample noise
        z = th.randn((n_designs, args.latent_dim), device=device, dtype=th.float)
        # THESE BOUNDS ARE PROBLEM DEPENDENT
        cls = th.linspace(0.3, 1.2, n_designs, device=device)
        cds = th.linspace(71, 600, n_designs, device=device)
        desired_objs = th.stack((cls, cds), dim=1)
        gen_imgs = generator(z, desired_objs)
        return desired_objs, gen_imgs

    # ----------
    #  Training
    # ----------
    for epoch in tqdm.trange(args.n_epochs):
        for i, data in enumerate(dataloader):
            # THIS IS PROBLEM DEPENDENT
            designs = data["optimized"]
            cl = data["cl_val"]
            cd = data["cd_val"]
            objs = th.stack((cl, cd), dim=1)

            # Adversarial ground truths
            valid = th.ones((designs.size(0), 1), requires_grad=False, device=device)
            fake = th.zeros((designs.size(0), 1), requires_grad=False, device=device)

            # -----------------
            #  Train Generator
            # min log(1 - D(G(z))) <==> max log(D(G(z)))
            # -----------------
            optimizer_generator.zero_grad()

            # Sample noise as generator input
            z = th.randn((designs.size(0), args.latent_dim), device=device, dtype=th.float)

            # Generate a batch of images
            gen_designs = generator(z, objs)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_designs, objs), valid)

            g_loss.backward()
            optimizer_generator.step()

            # ---------------------
            #  Train Discriminator
            # max log(D(real)) + log(1 - D(G(z)))
            # ---------------------
            optimizer_discriminator.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(designs, objs), valid)
            fake_loss = adversarial_loss(discriminator(gen_designs.detach(), objs), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_discriminator.step()

            # ----------
            #  Logging
            # ----------
            if args.track:
                batches_done = epoch * len(dataloader) + i
                wandb.log(
                    {
                        "d_loss": d_loss.item(),
                        "g_loss": g_loss.item(),
                        "epoch": epoch,
                        "batch": batches_done,
                    }
                )
                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
                )

                # This saves a grid image of 25 generated designs every sample_interval
                if batches_done % args.sample_interval == 0:
                    # Extract 25 designs
                    desired_objs, designs = sample_designs(25)
                    fig, axes = plt.subplots(5, 5, figsize=(12, 12))

                    # Flatten axes for easy indexing
                    axes = axes.flatten()

                    # Plot each tensor as a scatter plot
                    for j, tensor in enumerate(designs):
                        x, y = tensor.cpu()  # Extract x and y coordinates
                        cl, cd = desired_objs[j].cpu()
                        axes[j].scatter(x, y, s=10, alpha=0.7)  # Scatter plot
                        axes[j].title.set_text(f"CL: {cl:.2f}, CD: {cd:.2f}")
                        axes[j].set_xlim(-0.1, 1.1)  # Adjust x-axis limits
                        axes[j].set_ylim(-0.5, 0.5)  # Adjust y-axis limits
                        axes[j].set_xticks([])  # Hide x ticks
                        axes[j].set_yticks([])  # Hide y ticks

                    plt.tight_layout()
                    img_fname = f"images/{batches_done}.png"
                    plt.savefig(img_fname)
                    plt.close()
                    wandb.log({"designs": wandb.Image(img_fname)})

                    # --------------
                    #  Save models
                    # --------------
                    if args.save_model:
                        ckpt_gen = {
                            "epoch": epoch,
                            "batches_done": batches_done,
                            "generator": generator.state_dict(),
                            "optimizer_generator": optimizer_generator.state_dict(),
                            "loss": g_loss.item(),
                        }
                        ckpt_disc = {
                            "epoch": epoch,
                            "batches_done": batches_done,
                            "discriminator": discriminator.state_dict(),
                            "optimizer_discriminator": optimizer_discriminator.state_dict(),
                            "loss": d_loss.item(),
                        }

                        th.save(ckpt_gen, "generator.pth")
                        th.save(ckpt_disc, "discriminator.pth")
                        artifact_gen = wandb.Artifact(f"{args.algo}_generator", type="model")
                        artifact_gen.add_file("generator.pth")
                        artifact_disc = wandb.Artifact(f"{args.algo}_discriminator", type="model")
                        artifact_disc.add_file("discriminator.pth")

                        wandb.log_artifact(artifact_gen, aliases=[f"seed_{args.seed}"])
                        wandb.log_artifact(artifact_disc, aliases=[f"seed_{args.seed}"])

    wandb.finish()
