"""3D SliceGAN Main Script - Generate 3D volumes by learning 2D slice distributions.

Based on SliceGAN: https://arxiv.org/abs/2102.07708
Extended from the original cDCGAN framework to handle 3D volumetric engineering designs
through 2D slice generation and consistency losses.
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
    """Command-line arguments for 3D SliceGAN."""

    problem_id: str = "heatconduction3d"  # Assume 3D problem
    """Problem identifier for 3D engineering design."""
    algo: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this algorithm."""

    # Tracking
    track: bool = True
    """Track the experiment with wandb."""
    wandb_project: str = "engiopt_slicegan"
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
    batch_size: int = 16  # Can use larger batch size than 3D conv
    """size of the batches"""
    lr_gen: float = 0.0002
    """learning rate for the generator"""
    lr_disc: float = 0.0002
    """learning rate for the discriminator"""
    b1: float = 0.5
    """decay of first order momentum of gradient"""
    b2: float = 0.999
    """decay of first order momentum of gradient"""
    n_cpu: int = 8
    """number of cpu threads to use during batch generation"""
    latent_dim: int = 100
    """dimensionality of the latent space"""
    sample_interval: int = 400
    """interval between volume samples"""
    
    # SliceGAN specific parameters
    consistency_loss_weight: float = 10.0
    """Weight for slice consistency loss"""
    slice_sampling_rate: float = 0.3
    """Fraction of slices to sample during training"""
    use_all_axes: bool = True
    """Use slices from all three axes (XY, XZ, YZ)"""


def extract_random_slices(volumes: th.Tensor, axis: int, n_slices: int = None) -> tuple[th.Tensor, th.Tensor]:
    """Extract random slices from 3D volumes along specified axis.
    
    Args:
        volumes: (B, C, D, H, W) tensor of 3D volumes
        axis: 0=D, 1=H, 2=W (which spatial dimension to slice along)
        n_slices: Number of slices to extract per volume (if None, extract all)
    
    Returns:
        slices: (B*n_slices, C, slice_H, slice_W) tensor of 2D slices
        positions: (B*n_slices,) tensor of normalized slice positions [0, 1]
    """
    B, C, D, H, W = volumes.shape
    
    if axis == 0:  # XY slices (slice along D dimension)
        max_pos = D
        if n_slices is None:
            positions = th.arange(D, device=volumes.device).float() / (D - 1)
            slices = volumes.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            positions = positions.repeat(B)
        else:
            pos_indices = th.randint(0, D, (B, n_slices), device=volumes.device)
            positions = pos_indices.float() / (D - 1)
            slices = []
            for b in range(B):
                for s in range(n_slices):
                    slices.append(volumes[b, :, pos_indices[b, s], :, :])
            slices = th.stack(slices)  # (B*n_slices, C, H, W)
            positions = positions.reshape(-1)
            
    elif axis == 1:  # XZ slices (slice along H dimension)
        max_pos = H
        if n_slices is None:
            positions = th.arange(H, device=volumes.device).float() / (H - 1)
            slices = volumes.permute(0, 3, 1, 2, 4).reshape(B * H, C, D, W)
            positions = positions.repeat(B)
        else:
            pos_indices = th.randint(0, H, (B, n_slices), device=volumes.device)
            positions = pos_indices.float() / (H - 1)
            slices = []
            for b in range(B):
                for s in range(n_slices):
                    slices.append(volumes[b, :, :, pos_indices[b, s], :])
            slices = th.stack(slices)  # (B*n_slices, C, D, W)
            positions = positions.reshape(-1)
            
    elif axis == 2:  # YZ slices (slice along W dimension)
        max_pos = W
        if n_slices is None:
            positions = th.arange(W, device=volumes.device).float() / (W - 1)
            slices = volumes.permute(0, 4, 1, 2, 3).reshape(B * W, C, D, H)
            positions = positions.repeat(B)
        else:
            pos_indices = th.randint(0, W, (B, n_slices), device=volumes.device)
            positions = pos_indices.float() / (W - 1)
            slices = []
            for b in range(B):
                for s in range(n_slices):
                    slices.append(volumes[b, :, :, :, pos_indices[b, s]])
            slices = th.stack(slices)  # (B*n_slices, C, D, H)
            positions = positions.reshape(-1)
    
    return slices, positions


def visualize_3d_designs(volumes: th.Tensor, conditions: th.Tensor, condition_names: list, 
                        save_path: str, max_designs: int = 9):
    """Visualize 3D volumes by showing cross-sectional slices."""
    n_designs = min(len(volumes), max_designs)
    volumes = volumes[:n_designs]
    conditions = conditions[:n_designs]
    
    # Create subplot grid (3 slices per design)
    rows = n_designs
    cols = 3  # XY, XZ, YZ slices
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_designs):
        vol = volumes[i, 0].cpu().numpy()  # Remove channel dimension
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
        
        # Add condition information as text
        cond_text = []
        for j, name in enumerate(condition_names):
            cond_text.append(f"{name}: {conditions[i, j]:.2f}")
        axes[i, 0].text(0.02, 0.98, '\n'.join(cond_text), 
                       transform=axes[i, 0].transAxes, fontsize=8,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


class SliceGenerator2D(nn.Module):
    """2D Generator for creating slices conditioned on position and design conditions.
    
    Takes noise + design conditions + slice position -> 2D slice
    """
    
    def __init__(
        self,
        latent_dim: int,
        n_design_conds: int,
        slice_size: tuple[int, int] = (64, 64),
        num_filters: list[int] = [512, 256, 128, 64],
        out_channels: int = 1,
    ):
        super().__init__()
        self.slice_size = slice_size
        
        # Add 1 for slice position conditioning
        total_conds = n_design_conds + 1
        
        # Initial dense layer to create 4x4 feature map
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + total_conds, num_filters[0], kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_filters[0]),
            nn.ReLU(inplace=True),
        )
        
        # Upsampling blocks: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.up_blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(num_filters[0], num_filters[1], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(inplace=True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(num_filters[1], num_filters[2], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(num_filters[2], num_filters[3], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[3]),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(num_filters[3], out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: th.Tensor, design_conds: th.Tensor, slice_pos: th.Tensor) -> th.Tensor:
        """
        Args:
            z: (B, latent_dim, 1, 1) noise
            design_conds: (B, n_design_conds, 1, 1) design conditions
            slice_pos: (B, 1, 1, 1) normalized slice position [0, 1]
        Returns:
            slices: (B, out_channels, H, W) generated 2D slices
        """
        # Combine all conditions
        all_conds = th.cat([design_conds, slice_pos], dim=1)  # (B, n_design_conds + 1, 1, 1)
        x = th.cat([z, all_conds], dim=1)  # (B, latent_dim + n_design_conds + 1, 1, 1)
        
        x = self.initial(x)  # (B, num_filters[0], 4, 4)
        x = self.up_blocks(x)  # (B, out_channels, 64, 64)
        
        # Resize if needed
        if x.shape[2:] != self.slice_size:
            x = F.interpolate(x, size=self.slice_size, mode='bilinear', align_corners=False)
        
        return x


class SliceDiscriminator2D(nn.Module):
    """2D Discriminator for slice classification.
    
    Takes 2D slice + design conditions + slice position -> real/fake score
    """
    
    def __init__(
        self,
        n_design_conds: int,
        in_channels: int = 1,
        num_filters: list[int] = [64, 128, 256, 512],
        out_channels: int = 1,
    ):
        super().__init__()
        
        # Add 1 for slice position conditioning
        total_conds = n_design_conds + 1
        
        # Initial conv for image
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_filters[0] // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Initial conv for conditions (broadcast to match image size)
        self.cond_conv = nn.Sequential(
            nn.Conv2d(total_conds, num_filters[0] // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Downsampling blocks
        self.down_blocks = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(num_filters[0], num_filters[1], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[1]),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(num_filters[1], num_filters[2], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[2]),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(num_filters[2], num_filters[3], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[3]),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Final classification
        self.final_conv = nn.Sequential(
            nn.Conv2d(num_filters[3], out_channels, kernel_size=4, stride=1, padding=0, bias=False),

        )

    def forward(self, x: th.Tensor, design_conds: th.Tensor, slice_pos: th.Tensor) -> th.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) input slices
            design_conds: (B, n_design_conds, 1, 1) design conditions  
            slice_pos: (B, 1, 1, 1) normalized slice position
        Returns:
            out: (B, out_channels, 1, 1) real/fake score
        """
        # Resize to standard size if needed
        if x.shape[2:] != (64, 64):
            x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        
        # Expand conditions to match image spatial dimensions
        all_conds = th.cat([design_conds, slice_pos], dim=1)  # (B, n_design_conds + 1, 1, 1)
        conds_expanded = all_conds.expand(-1, -1, *x.shape[2:])  # (B, n_design_conds + 1, H, W)
        
        # Process image and conditions
        x_feat = self.img_conv(x)  # (B, num_filters[0]//2, 32, 32)
        c_feat = self.cond_conv(conds_expanded)  # (B, num_filters[0]//2, 32, 32)
        
        # Combine features
        h = th.cat([x_feat, c_feat], dim=1)  # (B, num_filters[0], 32, 32)
        
        # Downsample
        h = self.down_blocks(h)  # (B, num_filters[3], 4, 4)
        
        # Final classification
        return self.final_conv(h)  # (B, out_channels, 1, 1)


class SliceGAN3D(nn.Module):
    """Complete SliceGAN system for 3D volume generation."""
    
    def __init__(
        self,
        latent_dim: int,
        n_design_conds: int,
        volume_shape: tuple[int, int, int],
        slice_generators: dict[str, SliceGenerator2D],
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_design_conds = n_design_conds
        self.volume_shape = volume_shape  # (D, H, W)
        self.slice_generators = nn.ModuleDict(slice_generators)
        
    def generate_volume(self, z: th.Tensor, design_conds: th.Tensor) -> th.Tensor:
        """Generate 3D volume by creating slices along all axes and blending them.
        
        Args:
            z: (B, latent_dim) noise vector
            design_conds: (B, n_design_conds) design conditions
            
        Returns:
            volume: (B, 1, D, H, W) generated 3D volume
        """
        batch_size = z.shape[0]
        D, H, W = self.volume_shape
        
        # Reshape inputs for 2D generators
        z_2d = z.unsqueeze(-1).unsqueeze(-1)  # (B, latent_dim, 1, 1)
        design_conds_2d = design_conds.unsqueeze(-1).unsqueeze(-1)  # (B, n_design_conds, 1, 1)
        
        # Initialize volume
        volume = th.zeros((batch_size, 1, D, H, W), device=z.device)
        weight_sum = th.zeros((batch_size, 1, D, H, W), device=z.device)
        
        # Generate XY slices (along Z axis)
        if 'xy' in self.slice_generators:
            for z_idx in range(D):
                # Fix: Create z_pos with correct batch size
                z_pos = th.full((batch_size, 1, 1, 1), z_idx / (D - 1), device=z.device)
                xy_slice = self.slice_generators['xy'](z_2d, design_conds_2d, z_pos)  # (B, 1, H, W)
                volume[:, :, z_idx, :, :] += xy_slice
                weight_sum[:, :, z_idx, :, :] += 1
        
        # Generate XZ slices (along Y axis)  
        if 'xz' in self.slice_generators:
            for y_idx in range(H):
                # Fix: Create y_pos with correct batch size
                y_pos = th.full((batch_size, 1, 1, 1), y_idx / (H - 1), device=z.device)
                xz_slice = self.slice_generators['xz'](z_2d, design_conds_2d, y_pos)  # (B, 1, D, W)
                volume[:, :, :, y_idx, :] += xz_slice
                weight_sum[:, :, :, y_idx, :] += 1
                
        # Generate YZ slices (along X axis)
        if 'yz' in self.slice_generators:
            for x_idx in range(W):
                # Fix: Create x_pos with correct batch size
                x_pos = th.full((batch_size, 1, 1, 1), x_idx / (W - 1), device=z.device)
                yz_slice = self.slice_generators['yz'](z_2d, design_conds_2d, x_pos)  # (B, 1, D, H)
                volume[:, :, :, :, x_idx] += yz_slice
                weight_sum[:, :, :, :, x_idx] += 1
        
        # Normalize by number of contributing slices
        volume = volume / (weight_sum + 1e-8)
        
        return volume

def compute_consistency_loss(volumes: th.Tensor, slice_generators: dict, design_conds: th.Tensor, 
                           latent_vectors: th.Tensor, device: th.device) -> th.Tensor:
    """Compute consistency loss between slices extracted from volumes and generated slices."""
    consistency_loss = 0.0
    n_comparisons = 0
    
    batch_size = volumes.shape[0]
    design_conds_2d = design_conds.unsqueeze(-1).unsqueeze(-1)  # (B, n_design_conds, 1, 1)
    z_2d = latent_vectors.unsqueeze(-1).unsqueeze(-1)  # (B, latent_dim, 1, 1)
    
    # Sample a few slice positions for consistency check
    n_test_slices = 3
    
    for axis_name, generator in slice_generators.items():
        if axis_name == 'xy':  # XY slices
            axis_idx = 0
            max_pos = volumes.shape[2]  # D dimension
        elif axis_name == 'xz':  # XZ slices  
            axis_idx = 1
            max_pos = volumes.shape[3]  # H dimension
        elif axis_name == 'yz':  # YZ slices
            axis_idx = 2
            max_pos = volumes.shape[4]  # W dimension
        else:
            continue
            
        # Sample random slice positions
        slice_indices = th.randint(0, max_pos, (n_test_slices,), device=device)
        
        for slice_idx in slice_indices:
            # Extract slice from volume
            slice_pos_norm = slice_idx.float() / (max_pos - 1)
            slice_pos_tensor = slice_pos_norm.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, 1, 1)
            
            if axis_name == 'xy':
                real_slice = volumes[:, :, slice_idx, :, :]  # (B, 1, H, W)
            elif axis_name == 'xz':
                real_slice = volumes[:, :, :, slice_idx, :]  # (B, 1, D, W)
            elif axis_name == 'yz':
                real_slice = volumes[:, :, :, :, slice_idx]  # (B, 1, D, H)
            
            # Generate corresponding slice
            generated_slice = generator(z_2d, design_conds_2d, slice_pos_tensor)
            
            # Compute L1 loss between real and generated slices
            consistency_loss += F.l1_loss(generated_slice, real_slice)
            n_comparisons += 1
    
    return consistency_loss / max(n_comparisons, 1)


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Load 3D problem
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    # Extract 3D design space information
    design_shape = problem.design_space.shape  # Should be (D, H, W) for 3D
    if len(design_shape) != 3:
        raise ValueError(f"Expected 3D design shape, got {design_shape}")
    
    conditions = problem.conditions
    n_conds = len(conditions)
    condition_names = [cond[0] for cond in conditions]

    # Logging
    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, 
                  config=vars(args), save_code=True, name=run_name)

    # Seeding
    th.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    th.backends.cudnn.deterministic = True

    os.makedirs("images_slicegan", exist_ok=True)

    # Device selection
    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    print(f"Using device: {device}")
    print(f"3D Design shape: {design_shape}")
    print(f"Number of conditions: {n_conds}")

    # Initialize SliceGAN components
    D, H, W = design_shape
    
    # Create slice generators for different axes
    slice_generators = {}
    slice_discriminators = {}
    
    if args.use_all_axes:
        # XY slices (vary along Z)
        slice_generators['xy'] = SliceGenerator2D(
            latent_dim=args.latent_dim,
            n_design_conds=n_conds,
            slice_size=(H, W),  # XY plane
        )
        slice_discriminators['xy'] = SliceDiscriminator2D(n_design_conds=n_conds)
        
        # XZ slices (vary along Y)
        slice_generators['xz'] = SliceGenerator2D(
            latent_dim=args.latent_dim,
            n_design_conds=n_conds,
            slice_size=(D, W),  # XZ plane
        )
        slice_discriminators['xz'] = SliceDiscriminator2D(n_design_conds=n_conds)
        
        # YZ slices (vary along X)
        slice_generators['yz'] = SliceGenerator2D(
            latent_dim=args.latent_dim,
            n_design_conds=n_conds,
            slice_size=(D, H),  # YZ plane
        )
        slice_discriminators['yz'] = SliceDiscriminator2D(n_design_conds=n_conds)
    else:
        # Use only XY slices for faster training
        slice_generators['xy'] = SliceGenerator2D(
            latent_dim=args.latent_dim,
            n_design_conds=n_conds,
            slice_size=(H, W),
        )
        slice_discriminators['xy'] = SliceDiscriminator2D(n_design_conds=n_conds)

    # Create complete SliceGAN system
    slicegan = SliceGAN3D(
        latent_dim=args.latent_dim,
        n_design_conds=n_conds,
        volume_shape=design_shape,
        slice_generators=slice_generators,
    )
    
    # Move to device
    slicegan.to(device)
    for disc in slice_discriminators.values():
        disc.to(device)

    # Print model parameters
    gen_params = sum(p.numel() for p in slicegan.parameters())
    disc_params = sum(sum(p.numel() for p in disc.parameters()) for disc in slice_discriminators.values())
    print(f"SliceGAN Generator parameters: {gen_params:,}")
    print(f"Total Discriminator parameters: {disc_params:,}")

    # Configure data loader
    training_ds = problem.dataset.with_format("torch", device=device)["train"]
    
    # Extract 3D designs and conditions
    designs_3d = training_ds["optimal_design"]  # Should be (N, D, H, W)
    condition_tensors = [training_ds[key] for key in problem.conditions_keys]
    
    training_ds = th.utils.data.TensorDataset(designs_3d, *condition_tensors)
    dataloader = th.utils.data.DataLoader(
        training_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Create separate optimizers for each generator and discriminator
    optimizer_generators = {}
    optimizer_discriminators = {}
    
    for axis_name in slice_generators.keys():
        optimizer_generators[axis_name] = th.optim.Adam(
            slice_generators[axis_name].parameters(), 
            lr=args.lr_gen, 
            betas=(args.b1, args.b2)
        )
        optimizer_discriminators[axis_name] = th.optim.Adam(
            slice_discriminators[axis_name].parameters(), 
            lr=args.lr_disc, 
            betas=(args.b1, args.b2)
        )

    @th.no_grad()
    def sample_3d_designs(n_designs: int) -> tuple[th.Tensor, th.Tensor]:
        """Sample n_designs^n_conditions 3D volumes from the SliceGAN."""
        slicegan.eval()

        # Create condition grid
        all_conditions = th.stack(condition_tensors, dim=1)
        linspaces = [
            th.linspace(all_conditions[:, i].min(), all_conditions[:, i].max(), n_designs, device=device) 
            for i in range(all_conditions.shape[1])
        ]
        desired_conds = th.stack(th.meshgrid(*linspaces, indexing='ij'), dim=-1).reshape(-1, all_conditions.shape[1])

        # Match z size to number of condition combinations
        total_samples = desired_conds.shape[0]
        z = th.randn((total_samples, args.latent_dim), device=device, dtype=th.float)

        gen_volumes = slicegan.generate_volume(z, desired_conds)
        slicegan.train()
        return gen_volumes, desired_conds

    
    
    # ------ Training Loop ------
    print("Starting SliceGAN training...")

    last_disc_acc = {axis: 0.0 for axis in slice_generators.keys()}  # Track discriminator accuracy per axis

    for epoch in tqdm.trange(args.n_epochs):
        for i, batch in enumerate(dataloader):
            # Extract 3D designs and conditions
            designs_3d = batch[0]  # (B, D, H, W)
            condition_data = batch[1:]  # List of condition tensors
            
            # Stack conditions
            conds = th.stack(condition_data, dim=1)  # (B, n_conds)
            
            # Move to device and add channel dimension to designs
            designs_3d = designs_3d.to(device).unsqueeze(1)  # (B, 1, D, H, W)
            conds = conds.to(device)
            batch_size = designs_3d.shape[0]
            
            # Calculate number of slices to sample per axis
            n_slices_xy = max(1, int(args.slice_sampling_rate * D))
            n_slices_xz = max(1, int(args.slice_sampling_rate * H))
            n_slices_yz = max(1, int(args.slice_sampling_rate * W))
            
            # Sample slices from the 3D designs
            real_slices = {}
            slice_positions = {}
            slice_conds = {}
            
            if 'xy' in slice_generators:
                xy_slices, xy_positions = extract_random_slices(designs_3d, axis=0, n_slices=n_slices_xy)
                real_slices['xy'] = xy_slices
                slice_positions['xy'] = xy_positions
                # Expand conditions to match number of slices
                slice_conds['xy'] = conds.repeat_interleave(n_slices_xy, dim=0)
            
            if 'xz' in slice_generators:
                xz_slices, xz_positions = extract_random_slices(designs_3d, axis=1, n_slices=n_slices_xz)
                real_slices['xz'] = xz_slices
                slice_positions['xz'] = xz_positions
                slice_conds['xz'] = conds.repeat_interleave(n_slices_xz, dim=0)
            
            if 'yz' in slice_generators:
                yz_slices, yz_positions = extract_random_slices(designs_3d, axis=2, n_slices=n_slices_yz)
                real_slices['yz'] = yz_slices
                slice_positions['yz'] = yz_positions
                slice_conds['yz'] = conds.repeat_interleave(n_slices_yz, dim=0)
            
            # -----------------
            #  Train Generators (with 2x batch size)
            # -----------------
            gen_losses = {}
            fake_slices = {}

            for axis_name, generator in slice_generators.items():
                optimizer_generators[axis_name].zero_grad()

                # Use 2x batch size for generator
                gen_batch_size = batch_size * 2 
                z = th.randn((gen_batch_size, args.latent_dim, 1, 1), device=device, dtype=th.float)
                gen_positions = th.rand((gen_batch_size, 1, 1, 1), device=device)
                # Repeat conditions to match generator batch size
                slice_conds_2d = conds.repeat(2, 1).unsqueeze(2).unsqueeze(3)  # (2B, n_conds, 1, 1)

                # Generate fake slices
                fake_slice = generator(z, slice_conds_2d, gen_positions)
                fake_slices[axis_name] = fake_slice[:batch_size]  # Only use first B for discriminator training

                # Generator loss - Wasserstein Loss
                g_loss = g_loss = -slice_discriminators[axis_name](
                    fake_slice[:batch_size], slice_conds_2d[:batch_size], gen_positions[:batch_size]
                ).mean()

                g_loss.backward()
                optimizer_generators[axis_name].step()
                gen_losses[axis_name] = g_loss.item()
            # ---------------------
            #  Train Discriminators (adaptive)
            # ---------------------
            disc_losses = {}
            real_losses = {}
            fake_losses = {}
            
            # Ensure at least 32 slices per axis for discriminator training
            min_slices = 32
            for axis_name in real_slices:
                n_real = real_slices[axis_name].shape[0]
                if n_real < min_slices:
                    # Repeat slices to reach min_slices
                    reps = (min_slices + n_real - 1) // n_real
                    real_slices[axis_name] = real_slices[axis_name].repeat((reps, 1, 1, 1))[:min_slices]
                    slice_positions[axis_name] = slice_positions[axis_name].repeat(reps)[:min_slices]
                    slice_conds[axis_name] = slice_conds[axis_name].repeat(reps, 1)[:min_slices]

            for axis_name, discriminator in slice_discriminators.items():
                # Get real slices and positions for this axis
                real_slice = real_slices[axis_name]
                real_pos = slice_positions[axis_name].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # (n_real_slices, 1, 1, 1)
                real_slice_conds = slice_conds[axis_name].unsqueeze(2).unsqueeze(3)  # (n_real_slices, n_conds, 1, 1)
                
                # Get fake slices for this axis - ensure they match real slice count
                n_real_slices = real_slice.shape[0]
                fake_slice = fake_slices[axis_name]
                
                # If fake_slice batch size doesn't match real_slice, repeat it
                if fake_slice.shape[0] != n_real_slices:
                    reps = (n_real_slices + fake_slice.shape[0] - 1) // fake_slice.shape[0]
                    fake_slice = fake_slice.repeat((reps, 1, 1, 1))[:n_real_slices]
                
                # Generate positions and conditions for fake slices to match real slice count
                fake_pos = th.rand((n_real_slices, 1, 1, 1), device=device)
                fake_slice_conds = slice_conds[axis_name].unsqueeze(2).unsqueeze(3)  # (n_real_slices, n_conds, 1, 1)
                
                # Compute discriminator predictions for real and fake
                with th.no_grad():
                    real_pred = discriminator(real_slice, real_slice_conds, real_pos)
                    fake_pred = discriminator(fake_slice.detach(), fake_slice_conds, fake_pos)
                    
                    # Flatten and threshold at 0.5 for accuracy
                    real_acc = (real_pred > 0.5).float().mean().item()
                    fake_acc = (fake_pred < 0.5).float().mean().item()
                    disc_acc = 0.5 * (real_acc + fake_acc)
                    last_disc_acc[axis_name] = disc_acc
                
                if last_disc_acc[axis_name] <= 0.8:
                    optimizer_discriminators[axis_name].zero_grad()
                    
                    # Real loss
                    real_score = discriminator(real_slice, real_slice_conds, real_pos).mean()
                    # Fake Loss
                    fake_score = discriminator(fake_slice.detach(), fake_slice_conds, fake_pos).mean()
                    # Discriminator Loss - Wasserstein
                    d_loss = -(real_score - fake_score)
                    
                    d_loss.backward()
                    optimizer_discriminators[axis_name].step()
                    
                    for p in discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)
                    
                    # For logging, set real_loss/fake_loss as Wasserstein scores
                    real_loss = -real_score
                    fake_loss = fake_score
                else:
                    # Skip update, but still log Wasserstein scores
                    real_score = discriminator(real_slice, real_slice_conds, real_pos).mean()
                    fake_score = discriminator(fake_slice.detach(), fake_slice_conds, fake_pos).mean()
                    d_loss = -(real_score - fake_score)
                    real_loss = -real_score
                    fake_loss = fake_score
                
                disc_losses[axis_name] = d_loss.item()
                real_losses[axis_name] = real_loss.item()
                fake_losses[axis_name] = fake_loss.item()
            
            # ----------
            #  Logging
            # ----------
            batches_done = epoch * len(dataloader) + i
            
            if args.track:
                # Log individual axis losses
                log_dict = {}
                for axis_name in slice_generators.keys():
                    log_dict[f"d_loss_{axis_name}"] = disc_losses[axis_name]
                    log_dict[f"g_loss_{axis_name}"] = gen_losses[axis_name]
                    log_dict[f"real_loss_{axis_name}"] = real_losses[axis_name]
                    log_dict[f"fake_loss_{axis_name}"] = fake_losses[axis_name]
                    log_dict[f"disc_acc_{axis_name}"] = last_disc_acc[axis_name]
                
                # Log averages
                log_dict["d_loss_avg"] = sum(disc_losses.values()) / len(disc_losses)
                log_dict["g_loss_avg"] = sum(gen_losses.values()) / len(gen_losses)
                log_dict["disc_acc_avg"] = sum(last_disc_acc.values()) / len(last_disc_acc)
                log_dict["epoch"] = epoch
                log_dict["batch"] = batches_done
                
                wandb.log(log_dict)
                
                if i % 10 == 0:  # Print less frequently
                    avg_d_loss = sum(disc_losses.values()) / len(disc_losses)
                    avg_g_loss = sum(gen_losses.values()) / len(gen_losses)
                    avg_disc_acc = sum(last_disc_acc.values()) / len(last_disc_acc)
                    
                    print(f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] "
                          f"[D loss: {avg_d_loss:.4f}] [G loss: {avg_g_loss:.4f}] "
                          f"[D acc: {avg_disc_acc:.3f}]")
                    
                    # Print per-axis details
                    for axis_name in slice_generators.keys():
                        print(f"  {axis_name.upper()}: D_loss={disc_losses[axis_name]:.4f}, "
                              f"G_loss={gen_losses[axis_name]:.4f}, "
                              f"D_acc={last_disc_acc[axis_name]:.3f}")
                
                # Sample and visualize 3D designs
                if batches_done % args.sample_interval == 0:
                    print("Generating 3D design samples...")
                    
                    gen_volumes, gen_conds = sample_3d_designs(9)
                    
                    img_fname = f"images_slicegan/{batches_done}.png"
                    visualize_3d_designs(
                        gen_volumes, gen_conds, condition_names,
                        img_fname, max_designs=9
                    )
                    
                    # Log to wandb
                    wandb.log({
                        "generated_designs": wandb.Image(img_fname),
                        "sample_step": batches_done
                    })
                    
                    print(f"3D design samples saved for step {batches_done}")
            
            # Clean up GPU memory periodically
            if i % 50 == 0:
                th.cuda.empty_cache() if device.type == 'cuda' else None
        
        # --------------
        #  Save models
        # --------------
        if args.save_model and epoch == args.n_epochs - 1:
            print("Saving SliceGAN models...")
            
            # Save generators
            for axis_name, generator in slice_generators.items():
                ckpt_gen = {
                    "epoch": epoch,
                    "batches_done": epoch * len(dataloader) + len(dataloader) - 1,
                    "generator": generator.state_dict(),
                    "optimizer_generator": optimizer_generators[axis_name].state_dict(),
                    "loss": gen_losses[axis_name],
                    "axis": axis_name,
                    "n_conds": n_conds,
                    "args": vars(args)
                }
                
                gen_filename = f"generator_slice_{axis_name}.pth"
                th.save(ckpt_gen, gen_filename)
                
                if args.track:
                    artifact_gen = wandb.Artifact(f"{args.problem_id}_{args.algo}_generator_slice_{axis_name}", type="model")
                    artifact_gen.add_file(gen_filename)
                    wandb.log_artifact(artifact_gen, aliases=[f"seed_{args.seed}"])
            
            # Save discriminators
            for axis_name, discriminator in slice_discriminators.items():
                ckpt_disc = {
                    "epoch": epoch,
                    "batches_done": epoch * len(dataloader) + len(dataloader) - 1,
                    "discriminator": discriminator.state_dict(),
                    "optimizer_discriminator": optimizer_discriminators[axis_name].state_dict(),
                    "loss": disc_losses[axis_name],
                    "axis": axis_name,
                    "n_conds": n_conds,
                    "args": vars(args)
                }
                
                disc_filename = f"discriminator_slice_{axis_name}.pth"
                th.save(ckpt_disc, disc_filename)
                
                if args.track:
                    artifact_disc = wandb.Artifact(f"{args.problem_id}_{args.algo}_discriminator_slice_{axis_name}", type="model")
                    artifact_disc.add_file(disc_filename)
                    wandb.log_artifact(artifact_disc, aliases=[f"seed_{args.seed}"])
            
            print("SliceGAN models saved successfully!")

    if args.track:
        wandb.finish()

    print("SliceGAN training completed!")