"""3D Multiview VAE-GAN - Corrected implementation with proper VAE training.

Based on https://github.com/bryonkucharski/Multiview-3D-VAE-GAN
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
import torch.nn.functional as F
import tqdm
import tyro

import wandb

@dataclass
class Args:
    """Command-line arguments for 3D Multiview VAE-GAN."""

    problem_id: str = "beams3d"
    """Problem identifier for 3D engineering design."""
    algo: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this algorithm."""

    # Tracking
    track: bool = True
    """Track the experiment with wandb."""
    wandb_project: str = "engiopt_3d_vaegan"
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
    batch_size: int = 8
    """size of the batches"""
    lr_gen: float = 0.0025
    """learning rate for the generator"""
    lr_disc: float = 10e-5
    """learning rate for the discriminator"""
    lr_enc: float = 0.001
    """learning rate for the encoder"""
    b1: float = 0.5
    """decay of first order momentum of gradient"""
    b2: float = 0.999
    """decay of first order momentum of gradient"""
    n_cpu: int = 8
    """number of cpu threads to use during batch generation"""
    latent_dim: int = 64
    """dimensionality of the latent space"""
    sample_interval: int = 800
    """interval between volume samples"""
    
    # VAE specific parameters
    kl_weight: float = 0.01
    """Weight for KL divergence loss"""
    recon_weight: float = 1.0
    """Weight for reconstruction loss"""
    
    # Multiview parameters
    n_views: int = 3
    """Number of views for multiview rendering"""
    use_multiview: bool = True
    """Enable multiview consistency loss"""
    multiview_weight: float = 0.1
    """Weight for multiview consistency loss"""


def visualize_3d_designs(volumes: th.Tensor, conditions: th.Tensor, condition_names: list, 
                        save_path: str, max_designs: int = 9):
    """Visualize 3D volumes by showing cross-sectional slices."""
    n_designs = min(len(volumes), max_designs)
    volumes = volumes[:n_designs]
    conditions = conditions[:n_designs]
    
    rows = n_designs
    cols = 3  # XY, XZ, YZ slices
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_designs):
        vol = volumes[i, 0].cpu().numpy()
        D, H, W = vol.shape
        
        # XY slice (middle Z)
        axes[i, 0].imshow(vol[D//2, :, :], cmap='viridis', vmin=-1, vmax=1)
        axes[i, 0].set_title(f'Design {i+1} - XY slice (z={D//2})')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        
        # XZ slice (middle Y)
        axes[i, 1].imshow(vol[:, H//2, :], cmap='viridis', vmin=-1, vmax=1)
        axes[i, 1].set_title(f'Design {i+1} - XZ slice (y={H//2})')
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        
        # YZ slice (middle X)
        axes[i, 2].imshow(vol[:, :, W//2], cmap='viridis', vmin=-1, vmax=1)
        axes[i, 2].set_title(f'Design {i+1} - YZ slice (x={W//2})')
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        
        # Add condition information
        cond_text = []
        for j, name in enumerate(condition_names):
            cond_text.append(f"{name}: {conditions[i, j]:.2f}")
        axes[i, 0].text(0.02, 0.98, '\n'.join(cond_text), 
                       transform=axes[i, 0].transAxes, fontsize=8,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def reparameterize(mu: th.Tensor, logvar: th.Tensor) -> th.Tensor:
    """Reparameterization trick for VAE."""
    std = th.exp(0.5 * logvar)
    eps = th.randn_like(std)
    return mu + eps * std


def kl_divergence(mu: th.Tensor, logvar: th.Tensor) -> th.Tensor:
    """Calculate KL divergence for VAE."""
    return -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp())


def render_multiview(volume: th.Tensor, n_views: int = 3) -> th.Tensor:
    """Render multiple 2D views from 3D volume by projection."""
    B, C, D, H, W = volume.shape
    views = []
    
    # XY projection (sum along Z)
    xy_view = th.sum(volume, dim=2)  # (B, C, H, W)
    views.append(xy_view)
    
    # XZ projection (sum along Y)
    xz_view = th.sum(volume, dim=3)  # (B, C, D, W)
    views.append(xz_view)
    
    # YZ projection (sum along X)
    yz_view = th.sum(volume, dim=4)  # (B, C, D, H)
    views.append(yz_view)
    
    return views[:n_views]


class Encoder3D(nn.Module):
    """3D Encoder for VAE - produces latent mean and log-variance."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 64,
        num_filters: list[int] = [32, 64, 128, 256, 512],
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            # 64 → 32
            nn.Conv3d(in_channels, num_filters[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_filters[0]),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 → 16
            nn.Conv3d(num_filters[0], num_filters[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_filters[1]),
            nn.LeakyReLU(0.2, inplace=True),

            # 16 → 8
            nn.Conv3d(num_filters[1], num_filters[2], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_filters[2]),
            nn.LeakyReLU(0.2, inplace=True),

            # 8 → 4
            nn.Conv3d(num_filters[2], num_filters[3], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_filters[3]),
            nn.LeakyReLU(0.2, inplace=True),

            # 4 → 2
            nn.Conv3d(num_filters[3], num_filters[4], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_filters[4]),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Calculate flattened dimension
        self.flatten_dim = num_filters[4] * 2 * 2 * 2
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Encode input volume to latent distribution parameters."""
        # Ensure input is correct size
        if x.shape[2:] != (64, 64, 64):
            x = F.interpolate(x, size=(64, 64, 64), mode='trilinear', align_corners=False)
            
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# could possible just import class from cgan_cnn_3d.py
class Generator3D(nn.Module):
    """3D Generator - can work as both VAE decoder and GAN generator."""

    def __init__(
        self,
        latent_dim: int,
        n_conds: int,
        design_shape: tuple[int, int, int],
        num_filters: list[int] = [512, 256, 128, 64, 32],
        out_channels: int = 1,
    ):
        super().__init__()
        self.design_shape = design_shape
        self.latent_dim = latent_dim
        self.n_conds = n_conds
        
        # Combined input processing
        input_dim = latent_dim + n_conds
        
        # Initial projection to 4x4x4
        self.initial_proj = nn.Sequential(
            nn.Linear(input_dim, num_filters[0] * 4 * 4 * 4),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling blocks
        self.up_blocks = nn.Sequential(
            # 4x4x4 -> 8x8x8
            nn.ConvTranspose3d(num_filters[0], num_filters[1], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(num_filters[1]),
            nn.ReLU(inplace=True),
            
            # 8x8x8 -> 16x16x16
            nn.ConvTranspose3d(num_filters[1], num_filters[2], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(num_filters[2]),
            nn.ReLU(inplace=True),
            
            # 16x16x16 -> 32x32x32
            nn.ConvTranspose3d(num_filters[2], num_filters[3], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(num_filters[3]),
            nn.ReLU(inplace=True),
            
            # 32x32x32 -> 64x64x64
            nn.ConvTranspose3d(num_filters[3], num_filters[4], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(num_filters[4]),
            nn.ReLU(inplace=True),
            
            # Final output layer
            nn.ConvTranspose3d(num_filters[4], out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: th.Tensor, c: th.Tensor = None) -> th.Tensor:
        """Generate 3D volume from latent code and optional conditions."""
        # Handle different input formats
        if z.dim() == 2:  # (B, latent_dim) - from encoder
            batch_size = z.size(0)
            if c is not None:
                if c.dim() > 2:
                    c = c.view(batch_size, -1)  # Flatten conditions
                combined = th.cat([z, c], dim=1)
            else:
                combined = z
        else:  # (B, latent_dim, 1, 1, 1) - from noise
            batch_size = z.size(0)
            z_flat = z.view(batch_size, -1)
            if c is not None:
                c_flat = c.view(batch_size, -1)
                combined = th.cat([z_flat, c_flat], dim=1)
            else:
                combined = z_flat
        
        # Project to initial volume
        h = self.initial_proj(combined)
        h = h.view(batch_size, -1, 4, 4, 4)
        
        # Upsample to final volume
        out = self.up_blocks(h)
        
        # Resize to target shape if needed
        if out.shape[2:] != self.design_shape:
            out = F.interpolate(out, size=self.design_shape, mode='trilinear', align_corners=False)
        
        return out


class Discriminator3D(nn.Module):
    """3D Discriminator for both volumes and multiview consistency."""

    def __init__(
        self,
        n_conds: int,
        in_channels: int = 1,
        num_filters: list[int] = [32, 64, 128, 256, 512],
        out_channels: int = 1,
    ):
        super().__init__()
        self.n_conds = n_conds

        # Volume path
        self.vol_path = nn.Sequential(
            nn.Conv3d(in_channels, num_filters[0] // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Condition path
        self.cond_path = nn.Sequential(
            nn.Conv3d(n_conds, num_filters[0] // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Main discriminator blocks
        self.down_blocks = nn.Sequential(
            nn.Conv3d(num_filters[0], num_filters[1], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(num_filters[1]),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(num_filters[1], num_filters[2], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(num_filters[2]),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(num_filters[2], num_filters[3], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(num_filters[3]),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(num_filters[3], num_filters[4], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(num_filters[4]),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final classification
        self.final_conv = nn.Sequential(
            nn.Conv3d(num_filters[4], out_channels, kernel_size=2, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: th.Tensor, c: th.Tensor) -> th.Tensor:
        """Discriminate 3D volume with conditions."""
        # Ensure standard size
        if x.shape[2:] != (64, 64, 64):
            x = F.interpolate(x, size=(64, 64, 64), mode='trilinear', align_corners=False)

        # Expand conditions
        c_expanded = c.expand(-1, -1, *x.shape[2:])

        # Process through separate paths
        x_feat = self.vol_path(x)
        c_feat = self.cond_path(c_expanded)

        # Combine and discriminate
        h = th.cat([x_feat, c_feat], dim=1)
        h = self.down_blocks(h)
        return self.final_conv(h)


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Load problem
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    design_shape = problem.design_space.shape
    if len(design_shape) != 3:
        raise ValueError(f"Expected 3D design shape, got {design_shape}")
    
    conditions = problem.conditions
    n_conds = len(conditions)
    condition_names = [cond[0] for cond in conditions]

    # Setup logging
    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, 
                  config=vars(args), save_code=True, name=run_name)

    # Seeding
    th.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    th.backends.cudnn.deterministic = True

    os.makedirs("images_3d", exist_ok=True)

    # Device
    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    print(f"Using device: {device}")
    print(f"3D Design shape: {design_shape}")
    print(f"Number of conditions: {n_conds}")

    # Loss functions
    adversarial_loss = th.nn.BCELoss()
    reconstruction_loss = th.nn.MSELoss()

    # Initialize models
    encoder = Encoder3D(latent_dim=args.latent_dim)
    generator = Generator3D(
        latent_dim=args.latent_dim, 
        n_conds=n_conds, 
        design_shape=design_shape
    )
    discriminator = Discriminator3D(n_conds)

    # Move to device
    encoder.to(device)
    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)
    reconstruction_loss.to(device)

    # Print parameters
    enc_params = sum(p.numel() for p in encoder.parameters())
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Encoder parameters: {enc_params:,}")
    print(f"Generator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")

    # Setup data
    training_ds = problem.dataset.with_format("torch", device=device)["train"]
    designs_3d = training_ds["optimal_design"]
    condition_tensors = [training_ds[key] for key in problem.conditions_keys]
    
    training_ds = th.utils.data.TensorDataset(designs_3d, *condition_tensors)
    dataloader = th.utils.data.DataLoader(
        training_ds, batch_size=args.batch_size, shuffle=True
    )

    # Optimizers
    optimizer_encoder = th.optim.Adam(encoder.parameters(), lr=args.lr_enc, betas=(args.b1, args.b2))
    optimizer_generator = th.optim.Adam(generator.parameters(), lr=args.lr_gen, betas=(args.b1, args.b2))
    optimizer_discriminator = th.optim.Adam(discriminator.parameters(), lr=args.lr_disc, betas=(args.b1, args.b2))

    @th.no_grad()
    def sample_3d_designs(n_designs: int) -> tuple[th.Tensor, th.Tensor]:
        """Sample designs from trained generator."""
        encoder.eval()
        generator.eval()
        
        z = th.randn((n_designs, args.latent_dim), device=device)
        
        all_conditions = th.stack(condition_tensors, dim=1)
        linspaces = [
            th.linspace(all_conditions[:, i].min(), all_conditions[:, i].max(), n_designs, device=device) 
            for i in range(all_conditions.shape[1])
        ]
        desired_conds = th.stack(linspaces, dim=1)
        
        gen_volumes = generator(z, desired_conds)
        
        encoder.train()
        generator.train()
        return desired_conds, gen_volumes

    # Training loop
    print("Starting 3D VAE-GAN training...")
    
    for epoch in tqdm.trange(args.n_epochs):
        for i, data in enumerate(dataloader):
            designs_3d = data[0].unsqueeze(1)  # Add channel dim
            condition_data = data[1:]
            conds = th.stack(condition_data, dim=1)
            conds_expanded = conds.reshape(-1, n_conds, 1, 1, 1)
            
            batch_size = designs_3d.size(0)
            
            # Ground truth labels
            valid = th.ones((batch_size, 1, 1, 1, 1), device=device)
            fake = th.zeros((batch_size, 1, 1, 1, 1), device=device)

            # ==================
            # Train VAE (Encoder + Generator)
            # ==================
            optimizer_encoder.zero_grad()
            optimizer_generator.zero_grad()
            
            # Encode real data
            mu, logvar = encoder(designs_3d)
            z_encoded = reparameterize(mu, logvar)
            
            # Reconstruct
            reconstructed = generator(z_encoded, conds)
            
            # VAE losses
            recon_loss = reconstruction_loss(reconstructed, designs_3d)
            kl_loss = kl_divergence(mu, logvar) / batch_size
            
            # GAN loss for generator (fool discriminator)
            g_loss_adv = adversarial_loss(
                discriminator(reconstructed, conds_expanded), valid
            )
            
            # Multiview consistency loss
            multiview_loss = th.tensor(0.0, device=device)
            if args.use_multiview:
                real_views = render_multiview(designs_3d, args.n_views)
                recon_views = render_multiview(reconstructed, args.n_views)
                for real_view, recon_view in zip(real_views, recon_views):
                    multiview_loss += F.mse_loss(recon_view, real_view)
                multiview_loss /= len(real_views)
            
            # Combined VAE loss
            vae_loss = (args.recon_weight * recon_loss + 
                       args.kl_weight * kl_loss + 
                       g_loss_adv +
                       args.multiview_weight * multiview_loss)
            
            vae_loss.backward()
            optimizer_encoder.step()
            optimizer_generator.step()

            # ==================
            # Train Discriminator
            # ==================
            optimizer_discriminator.zero_grad()
            
            # Real loss
            real_loss = adversarial_loss(
                discriminator(designs_3d, conds_expanded), valid
            )
            
            # Fake loss (detached reconstructions)
            fake_loss = adversarial_loss(
                discriminator(reconstructed.detach(), conds_expanded), fake
            )
            
            # Random noise generation for additional adversarial training
            z_random = th.randn((batch_size, args.latent_dim), device=device)
            fake_designs = generator(z_random, conds)
            fake_loss_random = adversarial_loss(
                discriminator(fake_designs.detach(), conds_expanded), fake
            )
            
            d_loss = (real_loss + fake_loss + fake_loss_random) / 3
            d_loss.backward()
            optimizer_discriminator.step()

            # Logging
            batches_done = epoch * len(dataloader) + i
            
            if args.track:
                wandb.log({
                    "vae_loss": vae_loss.item(),
                    "recon_loss": recon_loss.item(),
                    "kl_loss": kl_loss.item(),
                    "g_loss_adv": g_loss_adv.item(),
                    "multiview_loss": multiview_loss.item(),
                    "d_loss": d_loss.item(),
                    "real_loss": real_loss.item(),
                    "fake_loss": fake_loss.item(),
                    "epoch": epoch,
                    "batch": batches_done,
                })
                
                if i % 10 == 0:
                    print(f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] "
                          f"[VAE loss: {vae_loss.item():.4f}] [D loss: {d_loss.item():.4f}] "
                          f"[Recon: {recon_loss.item():.4f}] [KL: {kl_loss.item():.4f}]")

                if batches_done % args.sample_interval == 0:
                    print("Generating 3D design samples...")
                    
                    desired_conds, volumes_3d = sample_3d_designs(9)
                    
                    img_fname = f"images_3d/{batches_done}.png"
                    visualize_3d_designs(
                        volumes_3d, desired_conds, condition_names, 
                        img_fname, max_designs=9
                    )
                    
                    wandb.log({
                        "3d_designs": wandb.Image(img_fname),
                        "sample_step": batches_done
                    })

            # Memory cleanup
            if i % 50 == 0:
                th.cuda.empty_cache() if device.type == 'cuda' else None

        # Save models
        if args.save_model and epoch == args.n_epochs - 1:
            print("Saving models...")
            
            # Save all three models
            th.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_encoder": optimizer_encoder.state_dict(),
                "optimizer_generator": optimizer_generator.state_dict(),
                "optimizer_discriminator": optimizer_discriminator.state_dict(),
                "args": vars(args),
                "design_shape": design_shape,
                "n_conds": n_conds,
            }, "multiview_3d_vaegan.pth")
            
            if args.track:
                artifact = wandb.Artifact(f"{args.problem_id}_{args.algo}_models", type="model")
                artifact.add_file("multiview_3d_vaegan.pth")
                wandb.log_artifact(artifact, aliases=[f"seed_{args.seed}"])

    if args.track:
        wandb.finish()
    
    print("Multiview 3D VAE-GAN training completed!")