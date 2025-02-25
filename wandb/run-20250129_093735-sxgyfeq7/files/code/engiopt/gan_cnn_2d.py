"""This code is largely based on the excellent PyTorch GAN repo: https://github.com/eriklindernoren/PyTorch-GAN.

We essentially refreshed the Python style, use wandb for logging, and made a few little improvements.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
import time

from engibench.utils.all_problems import all_problems
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

    problem_id: str = "draw_circle_v0"
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
    n_epochs: int = 300
    """number of epochs of training"""
    batch_size: int = 256
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
    sample_interval: int = 400
    """interval between image samples"""

class Generator(nn.Module):
    def __init__(self, latent_dim=100, out_channels=1, g_features=64):
        """
        Generator for 100x100 images.

        Args:
            latent_dim (int): Dimensionality of the latent code (z).
            out_channels (int): Number of channels in the output image.
            g_features (int): Base number of generator feature maps. 
        """
        super().__init__()
        
        self.init_size = 25  # We'll create a 25x25 feature map after the first FC
        self.fc = nn.Linear(latent_dim, g_features * 4 * self.init_size * self.init_size)

        self.conv_blocks = nn.Sequential(
            # (g_features*4, 25, 25) -> (g_features*2, 50, 50)
            nn.ConvTranspose2d(
                in_channels=g_features * 4,
                out_channels=g_features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(g_features * 2),
            nn.ReLU(True),

            # (g_features*2, 50, 50) -> (g_features, 100, 100)
            nn.ConvTranspose2d(
                in_channels=g_features * 2,
                out_channels=g_features,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(g_features),
            nn.ReLU(True),

            # (g_features, 100, 100) -> (out_channels, 100, 100)
            nn.ConvTranspose2d(
                in_channels=g_features,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.Tanh()
        )

    def forward(self, z):
        # z: (batch, latent_dim)
        out = self.fc(z)  # (batch, g_features*4*25*25)
        out = out.view(out.size(0), -1, self.init_size, self.init_size) 
        img = self.conv_blocks(out)  # (batch, out_channels, 100, 100)
        return img


class Discriminator(nn.Module):

    def __init__(self, in_channels=1, d_features=64):
        super().__init__()

        """
        Discriminator for 100x100 images.

        Args:
            in_channels (int): Number of channels in the input image.
            d_features (int): Base number of discriminator feature maps.
        """
        self.conv_blocks = nn.Sequential(
            # 100 x 100 -> 50 x 50
            nn.Conv2d(
                in_channels,
                d_features,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # 50 x 50 -> 25 x 25
            nn.Conv2d(
                d_features,
                d_features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(d_features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 25 x 25 -> 13 x 13
            nn.Conv2d(
                d_features * 2,
                d_features * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(d_features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 13 x 13 -> 7 x 7
            nn.Conv2d(
                d_features * 4,
                d_features * 8,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(d_features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 7 x 7 -> 1 x 1
            nn.Conv2d(
                d_features * 8,
                1,
                kernel_size=7,
                stride=1,
                padding=0,
                bias=False
            )
        )

        # Final output activation
        self.output_act = nn.Sigmoid()

    def forward(self, img):
        # img: (batch, in_channels, 100, 100)
        out = self.conv_blocks(img)  
        # out: (batch, 1, 1, 1)
        out = out.view(out.size(0), -1)  # flatten to (batch, 1)
        validity = self.output_act(out)
        return validity

    def forward(self, img):
        # img: (batch, in_channels, 100, 100)
        out = self.conv_blocks(img)  
        # out: (batch, 1, 1, 1)
        out = out.view(out.size(0), -1)  # flatten to (batch, 1)
        validity = self.output_act(out)
        return validity

if __name__ == "__main__":
    args = tyro.cli(Args)

    problem = all_problems[args.problem_id].build()
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
    training_ds = th.tensor(np.load('/home/nathanielhoffman/Desktop/fake_dataset_2D/y.npy')).float().to(device).unsqueeze(1)
    print(training_ds.shape)
    dataloader = th.utils.data.DataLoader(
        training_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Optimizers
    optimizer_generator = th.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_discriminator = th.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # ----------
    #  Training
    # ----------
    for epoch in tqdm.trange(args.n_epochs):
        for i, designs in enumerate(dataloader):
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
            gen_designs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_designs), valid)

            g_loss.backward()
            optimizer_generator.step()

            # ---------------------
            #  Train Discriminator
            # max log(D(real)) + log(1 - D(G(z)))
            # ---------------------
            optimizer_discriminator.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(designs), valid)
            fake_loss = adversarial_loss(discriminator(gen_designs.detach()), fake)
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
                    tensors = gen_designs.data[:25]
                    fig, axes = plt.subplots(5, 5, figsize=(12, 12))

                    # Flatten axes for easy indexing
                    axes = axes.flatten()

                    # Plot each tensor as a iamge plot
                    for j, tensor in enumerate(tensors):
                        img = tensor.cpu()  # Extract x and y coordinates
                        axes[j].imshow(img)  # image plot
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
                        artifact_gen = wandb.Artifact("generator", type="model")
                        artifact_gen.add_file("generator.pth")
                        artifact_disc = wandb.Artifact("discriminator", type="model")
                        artifact_disc.add_file("discriminator.pth")

                        wandb.log_artifact(artifact_gen)
                        wandb.log_artifact(artifact_disc)

    wandb.finish()
