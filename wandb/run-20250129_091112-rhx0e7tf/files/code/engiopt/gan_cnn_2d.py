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
    def __init__(self, latent_dim=100, img_channels=1, feature_maps=64):
        """
        Args:
            latent_dim (int): Dimensionality of the latent vector z.
            img_channels (int): Number of image channels (1 for grayscale, 3 for RGB).
            feature_maps (int): Base number of feature maps (scales up/down in each layer).
        """
        super(Generator, self).__init__()
        
        # The idea is to start from (latent_dim, 1, 1) and scale up to the target image size.
        # For example, if your target image size is 1x28x28 (MNIST), you might do fewer layers.
        # If you have 3x64x64, you'll typically do more upsampling steps. 
        # The code below is a typical example for 1x28x28. 
        # Adjust kernel/stride/padding to match your exact desired output.
        
        self.net = nn.Sequential(
            # Input: (latent_dim) x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_maps * 4, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # State now: (feature_maps * 4) x 7 x 7
            
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # State now: (feature_maps * 2) x 14 x 14
            
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # State now: (feature_maps) x 28 x 28
            
            nn.ConvTranspose2d(feature_maps, img_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
            # Final output: (img_channels) x 28 x 28
        )

    def forward(self, z):
        # z is (batch_size, latent_dim). First reshape to (batch_size, latent_dim, 1, 1).
        z = z.view(z.size(0), -1, 1, 1)
        img = self.net(z)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_channels=1, feature_maps=64):
        """
        Args:
            img_channels (int): Number of image channels.
            feature_maps (int): Base number of feature maps (scaled up in intermediate layers).
        """
        super(Discriminator, self).__init__()
        
        # The idea here is to mirror the Generator layers but in the opposite (downsampling) direction.
        # Again, adjust kernel/stride/padding to match your image size. 
        # For 28x28 input, the layout below is a typical example.
        
        self.net = nn.Sequential(
            # Input shape: (img_channels) x 28 x 28
            nn.Conv2d(img_channels, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps) x 14 x 14
            
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps * 2) x 7 x 7

            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps * 4) x 4 x 4 (approximately, can vary with exact kernel/stride/padding)
            
            nn.Conv2d(feature_maps * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # Now this should be roughly 1x1 in spatial dimension:
            nn.Sigmoid()
        )

    def forward(self, img):
        # img shape: (batch_size, img_channels, H, W)
        validity = self.net(img)
        # Flatten the final output to get a scalar per image in the batch:
        return validity.view(-1, 1)

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
