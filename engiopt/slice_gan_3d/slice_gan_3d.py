"""3D SliceGAN Main Script - Generate 3D volumes by learning 2D slice distributions.

Based on SliceGAN: https://arxiv.org/abs/2102.07708
Extended from the original cDCGAN framework to handle 3D volumetric engineering designs
through 2D slice generation.
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
from torch import autograd
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
    n_epochs: int = 300
    """number of epochs of training"""
    batch_size: int = 16  # Can use larger batch size than 3D conv
    """size of the batches"""
    lr_gen: float = 10e-5
    """learning rate for the generator"""
    lr_disc: float = 10e-5
    """learning rate for the discriminator"""
    b1: float = 0.9
    """decay of first order momentum of gradient"""
    b2: float = 0.99
    """decay of first order momentum of gradient"""
    n_cpu: int = 8
    """number of cpu threads to use during batch generation"""
    latent_dim: int = 64
    """dimensionality of the latent space"""
    sample_interval: int = 800
    """interval between volume samples"""
    
    # SliceGAN specific parameters
    discrim_iters: int = 5
    """Discriminator update"""
    slice_sampling_rate: float = 0.8
    """Fraction of slices to sample during training"""
    use_all_axes: bool = True
    """Use slices from all three axes (XY, XZ, YZ)"""


def extract_random_slices(volumes: th.Tensor, axis: int, n_slices: int = None, target_size: int = 64) -> tuple[th.Tensor, th.Tensor]:
    """Extract random slices from 3D volumes and resize to consistent dimensions.
    
    Args:
        volumes: (B, C, D, H, W) tensor of 3D volumes
        axis: 0=D, 1=H, 2=W (which spatial dimension to slice along)
        n_slices: Number of slices to extract per volume (if None, extract all)
        target_size: Target size for output slices (default 64 for 64x64)
    
    Returns:
        slices: (B*n_slices, C, target_size, target_size) tensor of 2D slices
        positions: (B*n_slices,) tensor of normalized slice positions [0, 1]
    """
    B, C, D, H, W = volumes.shape
    
    if axis == 0:  # XY slices (slice along D dimension)
        if n_slices is None or n_slices >= D:
            # Extract all slices
            positions = th.arange(D, device=volumes.device, dtype=th.float32)
            positions = positions / max(D - 1, 1)  # Avoid division by zero
            slices = volumes.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            positions = positions.repeat(B)
        else:
            # Sample random slices efficiently
            pos_indices = th.randint(0, D, (B, n_slices), device=volumes.device)
            positions = pos_indices.float() / max(D - 1, 1)
            
            # Efficient extraction using advanced indexing
            batch_indices = th.arange(B, device=volumes.device).unsqueeze(1).expand(-1, n_slices)
            slices = volumes[batch_indices.flatten(), :, pos_indices.flatten(), :, :]
            positions = positions.flatten()
            
    elif axis == 1:  # XZ slices (slice along H dimension)
        if n_slices is None or n_slices >= H:
            positions = th.arange(H, device=volumes.device, dtype=th.float32)
            positions = positions / max(H - 1, 1)
            slices = volumes.permute(0, 3, 1, 2, 4).reshape(B * H, C, D, W)
            positions = positions.repeat(B)
        else:
            pos_indices = th.randint(0, H, (B, n_slices), device=volumes.device)
            positions = pos_indices.float() / max(H - 1, 1)
            
            batch_indices = th.arange(B, device=volumes.device).unsqueeze(1).expand(-1, n_slices)
            slices = volumes[batch_indices.flatten(), :, :, pos_indices.flatten(), :]
            positions = positions.flatten()
            
    elif axis == 2:  # YZ slices (slice along W dimension)
        if n_slices is None or n_slices >= W:
            positions = th.arange(W, device=volumes.device, dtype=th.float32)
            positions = positions / max(W - 1, 1)
            slices = volumes.permute(0, 4, 1, 2, 3).reshape(B * W, C, D, H)
            positions = positions.repeat(B)
        else:
            pos_indices = th.randint(0, W, (B, n_slices), device=volumes.device)
            positions = pos_indices.float() / max(W - 1, 1)
            
            batch_indices = th.arange(B, device=volumes.device).unsqueeze(1).expand(-1, n_slices)
            slices = volumes[batch_indices.flatten(), :, :, :, pos_indices.flatten()]
            positions = positions.flatten()
    
    # CRITICAL FIX: Resize ALL slices to target_size x target_size
    if slices.shape[2:] != (target_size, target_size):
        slices = F.interpolate(slices, size=(target_size, target_size), mode='bilinear', align_corners=False)
    
    return slices, positions


def visualize_3d_designs(volumes: th.Tensor, conditions: th.Tensor, condition_names: list, 
                        save_path: str, max_designs: int = 9):
    """Visualize 3D volumes by showing cross-sectional slices with a red-white heatmap.
    
    Args:
        volumes: (N, 1, D, H, W) tensor of 3D designs
        conditions: (N, n_conds) tensor of conditions
        condition_names: List of condition names
        save_path: Path to save the visualization
        max_designs: Maximum number of designs to visualize
    """
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
        
        # Use 'Reds_r' colormap: low=red, high=white
        cmap = plt.get_cmap('Reds_r')
        
        # XY slice (middle Z)
        axes[i, 0].imshow(vol[D//2, :, :], cmap=cmap, vmin=-1, vmax=1)
        axes[i, 0].set_title(f'Design {i+1} - XY slice (z={D//2})')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        
        # XZ slice (middle Y)
        axes[i, 1].imshow(vol[:, H//2, :], cmap=cmap, vmin=-1, vmax=1)
        axes[i, 1].set_title(f'Design {i+1} - XZ slice (y={H//2})')
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        
        # YZ slice (middle X)
        axes[i, 2].imshow(vol[:, :, W//2], cmap=cmap, vmin=-1, vmax=1)
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
    
class SliceGenerator3D(nn.Module):
    """3D Conditional GAN generator that outputs volumetric designs.

    From noise + condition -> 3D volume (e.g., 64x64x64).

    Args:
        latent_dim (int): Dimensionality of the noise (latent) vector.
        n_conds (int): Number of conditional features.
        design_shape (tuple[int, int, int]): Target 3D design shape (D, H, W).
        num_filters (list of int): Number of filters in each upsampling stage.
        out_channels (int): Number of output channels (e.g., 1 for density).
    """

    def __init__(
        self,
        latent_dim: int,
        n_conds: int,
        design_shape: tuple[int, int, int],
        num_filters: list[int] = [512, 256, 128, 64, 32],  # Extra layer for 3D
        out_channels: int = 1,
    ):
        super().__init__()
        self.design_shape = design_shape
        
        # Path for noise z - start with 4x4x4 volume
        self.z_path = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, num_filters[0] // 2, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(num_filters[0] // 2),
            nn.ReLU(inplace=True),
        )
        
        # Path for condition c - start with 4x4x4 volume
        self.c_path = nn.Sequential(
            nn.ConvTranspose3d(n_conds, num_filters[0] // 2, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(num_filters[0] // 2),
            nn.ReLU(inplace=True),
        )

        # Upsampling blocks: 4x4x4 -> 8x8x8 -> 16x16x16 -> 32x32x32 -> 64x64x64
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
            
            # Final conv without changing spatial size
            nn.Conv3d(num_filters[4], out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),  # Output in [-1, 1] range
        )

    def forward(self, z: th.Tensor, c: th.Tensor) -> th.Tensor:
        """Forward pass for the 3D Generator.

        Inputs:
            z: (B, z_dim, 1, 1, 1) - noise vector
            c: (B, cond_features, 1, 1, 1) - condition vector
        Output:
            out: (B, out_channels, D, H, W) - 3D design
        """
        # Run noise & condition through separate stems
        z_feat = self.z_path(z)  # -> (B, num_filters[0]//2, 4, 4, 4)
        c_feat = self.c_path(c)  # -> (B, num_filters[0]//2, 4, 4, 4)

        # Concat along channel dimension
        x = th.cat([z_feat, c_feat], dim=1)  # (B, num_filters[0], 4, 4, 4)

        # Upsample through the main blocks
        out = self.up_blocks(x)  # -> (B, out_channels, 128, 128, 128)

        # Resize to target shape if needed
        #if out.shape[2:] != self.design_shape:
        #    out = F.interpolate(out, size=self.design_shape, mode='trilinear', align_corners=False)

        return out

class SliceDiscriminator2D(nn.Module):
    """2D Discriminator for slice classification with automatic resizing.
    
    Takes 2D slice + design conditions + slice position -> real/fake score
    Automatically resizes input to 64x64 regardless of input dimensions.
    """
    
    def __init__(
        self,
        n_design_conds: int,
        in_channels: int = 1,
        num_filters: list[int] = [64, 128, 256, 512],
        out_channels: int = 1,
        target_size: int = 64,  # Add target size parameter
    ):
        super().__init__()
        
        self.target_size = target_size
        
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
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(num_filters[1], num_filters[2], kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(num_filters[2], num_filters[3], kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Final classification
        self.final_conv = nn.Sequential(
            nn.Conv2d(num_filters[3], out_channels, kernel_size=4, stride=1, padding=0, bias=False),
        )

    def forward(self, x: th.Tensor, design_conds: th.Tensor, slice_pos: th.Tensor) -> th.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) input slices - can be any size
            design_conds: (B, n_design_conds, 1, 1) design conditions  
            slice_pos: (B, 1, 1, 1) normalized slice position
        Returns:
            out: (B, out_channels, 1, 1) real/fake score
        """
        # CRITICAL FIX: Always resize to target size regardless of input dimensions
        if x.shape[2:] != (self.target_size, self.target_size):
            x = F.interpolate(x, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
        
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

def compute_gradient_penalty(discriminator, real_samples, fake_samples, 
                           real_conds_axis, real_pos_axis, batch_size, device, lambda_gp=10.0):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = th.rand(real_samples.size(0), 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_samples)
    # Get interpolated samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # Interpolate conditions too
    alpha_cond = th.rand(real_conds_axis.size(0), 1, 1, 1, device=device)
    alpha_cond = alpha_cond.expand_as(real_conds_axis)
    interpolated_conds = alpha_cond * real_conds_axis + (1 - alpha_cond) * real_conds_axis
    
    alpha_pos = th.rand(real_conds_axis.size(0), 1, 1, 1, device=device) 
    alpha_pos = alpha_pos.expand_as(real_pos_axis)
    interpolated_pos = alpha_pos * real_pos_axis + (1 - alpha_pos) * real_pos_axis
    
    d_interpolates = discriminator(interpolates, interpolated_conds, interpolated_pos)
    
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=th.ones_like(d_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(real_samples.size(0), -1)
    gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

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
    
    slice_generator = SliceGenerator3D(
                latent_dim=args.latent_dim,
                n_conds=n_conds,
                design_shape=design_shape
            )
    
    # Create slice discriminators for different axes
    slice_discriminators = {}
    
    # XY slices (vary along Z)
    slice_discriminators['xy'] = SliceDiscriminator2D(n_design_conds=n_conds)
    
    # XZ slices (vary along Y)
    slice_discriminators['xz'] = SliceDiscriminator2D(n_design_conds=n_conds)
    
    # YZ slices (vary along X)
    slice_discriminators['yz'] = SliceDiscriminator2D(n_design_conds=n_conds)
    
    # Move to device
    slice_generator.to(device)
    for disc in slice_discriminators.values():
        disc.to(device)

    # Print model parameters
    gen_params = sum(p.numel() for p in slice_generator.parameters())
    disc_params = sum(sum(p.numel() for p in disc.parameters()) for disc in slice_discriminators.values())
    print(f"Slice Generator parameters: {gen_params:,}")
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
    optimizer_generator = th.optim.Adam(slice_generator.parameters(), lr=args.lr_gen, betas=(args.b1, args.b2))
    
    optimizer_discriminators = {}
    for axis_name in slice_discriminators.keys():
        optimizer_discriminators[axis_name] = th.optim.Adam(
            slice_discriminators[axis_name].parameters(), 
            lr=args.lr_disc, 
            betas=(args.b1, args.b2)
        )

    @th.no_grad()
    def sample_3d_designs(n_designs: int) -> tuple[th.Tensor, th.Tensor]:
        """Sample n_designs^n_conditions 3D volumes from the SliceGAN."""
        slice_generator.eval()

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

        z_3d = z.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (total_samples, latent_dim, 1, 1, 1)
        desired_conds_3d = desired_conds.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (total_samples, n_conds, 1, 1, 1)

        gen_volumes = slice_generator(z_3d, desired_conds_3d)  # (total_samples, out_channels, D, H, W)
        slice_generator.train()
        return gen_volumes, desired_conds

# ------ Training Loop ------
    print("Starting SliceGAN training...")

    last_disc_acc = {axis: 0.0 for axis in slice_discriminators.keys()}  # Track discriminator accuracy per axis

    for epoch in tqdm.trange(args.n_epochs):
        for i, batch in enumerate(dataloader):
            # Extract 3D designs and conditions
            designs_3d = batch[0]  # (B, D, H, W)
            condition_data = batch[1:]  # List of condition tensors
            
            # Convert condition data to float tensors and stack
            condition_tensors_batch = []
            for cond_tensor in condition_data:
                if not isinstance(cond_tensor, th.Tensor):
                    # Convert to float tensor if it's object type or not a tensor
                    if hasattr(cond_tensor, '__iter__') and not isinstance(cond_tensor, str):
                        cond_tensor = th.tensor([float(x) for x in cond_tensor], device=device)
                    else:
                        cond_tensor = th.tensor(float(cond_tensor), device=device)
                else:
                    cond_tensor = cond_tensor.float()
                condition_tensors_batch.append(cond_tensor)
            
            conds = th.stack(condition_tensors_batch, dim=1)  # (B, n_conds)
            
            # Move to device and add channel dimension to designs
            designs_3d = designs_3d.to(device).unsqueeze(1)  # (B, 1, D, H, W)
            conds = conds.to(device)
            
            # Pad to (B, 1, 64, 64, 64) if needed
            if designs_3d.shape[2:] == (51, 51, 51):
                designs_3d = F.pad(designs_3d, (6, 7, 6, 7, 6, 7), mode='constant', value=0)
            
            batch_size = designs_3d.shape[0]
            
            # Calculate number of slices to sample per axis
            n_slices_xy = max(1, int(args.slice_sampling_rate * D))
            n_slices_xz = max(1, int(args.slice_sampling_rate * H))
            n_slices_yz = max(1, int(args.slice_sampling_rate * W))
            
            # Sample slices from the 3D designs
            real_slices = {}
            slice_positions = {}
            slice_conds = {}

            if 'xy' in slice_discriminators:
                xy_slices, xy_positions = extract_random_slices(designs_3d, axis=0, n_slices=n_slices_xy)
                # CRITICAL FIX: Ensure 64x64 dimensions
                if xy_slices.shape[2:] != (64, 64):
                    xy_slices = F.interpolate(xy_slices, size=(64, 64), mode='bilinear', align_corners=False)
                real_slices['xy'] = xy_slices
                slice_positions['xy'] = xy_positions
                slice_conds['xy'] = conds.repeat_interleave(n_slices_xy, dim=0)

            if 'xz' in slice_discriminators:
                xz_slices, xz_positions = extract_random_slices(designs_3d, axis=1, n_slices=n_slices_xz)
                # CRITICAL FIX: Ensure 64x64 dimensions
                if xz_slices.shape[2:] != (64, 64):
                    xz_slices = F.interpolate(xz_slices, size=(64, 64), mode='bilinear', align_corners=False)
                real_slices['xz'] = xz_slices
                slice_positions['xz'] = xz_positions
                slice_conds['xz'] = conds.repeat_interleave(n_slices_xz, dim=0)

            if 'yz' in slice_discriminators:
                yz_slices, yz_positions = extract_random_slices(designs_3d, axis=2, n_slices=n_slices_yz)
                # CRITICAL FIX: Ensure 64x64 dimensions
                if yz_slices.shape[2:] != (64, 64):
                    yz_slices = F.interpolate(yz_slices, size=(64, 64), mode='bilinear', align_corners=False)
                real_slices['yz'] = yz_slices
                slice_positions['yz'] = yz_positions
                slice_conds['yz'] = conds.repeat_interleave(n_slices_yz, dim=0)
            
            # Generate fake volumes once (used for both discriminator and generator training)
            z = th.randn(batch_size, args.latent_dim, 1, 1, 1, device=device)
            conds_3d = conds.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, n_conds, 1, 1, 1)
            fake_volumes = slice_generator(z, conds_3d)
            
            # Pad to (B, 1, 64, 64, 64) if needed
            if fake_volumes.shape[2:] == (51, 51, 51):
                fake_volumes = F.pad(fake_volumes, (6, 7, 6, 7, 6, 7), mode='constant', value=0)
            
            # Update dimensions after padding
            _, _, D_padded, H_padded, W_padded = fake_volumes.shape
            
            # ---------------------
            #  Train Discriminators
            # ---------------------
            d_losses = {}
            real_losses = {}
            fake_losses = {}
            
            for discrim_iter in range(args.discrim_iters):
                # For each axis discriminator
                for axis_name, discriminator in slice_discriminators.items():
                    optimizer_discriminators[axis_name].zero_grad()
                    
                    # Train on real slices
                    real_slices_axis = real_slices[axis_name]
                    real_conds_axis = slice_conds[axis_name].unsqueeze(-1).unsqueeze(-1)
                    real_pos_axis = slice_positions[axis_name].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    
                    out_real = discriminator(real_slices_axis, real_conds_axis, real_pos_axis).mean()
                    
                    # Train on fake slices (from the generated volume)
                    with th.no_grad():
                        if axis_name == 'xy':
                            fake_slices_axis = fake_volumes.permute(0, 2, 1, 3, 4).reshape(-1, 1, H_padded, W_padded)
                            # CRITICAL FIX: Ensure 64x64 dimensions
                            if fake_slices_axis.shape[2:] != (64, 64):
                                fake_slices_axis = F.interpolate(fake_slices_axis, size=(64, 64), mode='bilinear', align_corners=False)
                        elif axis_name == 'xz':
                            fake_slices_axis = fake_volumes.permute(0, 3, 1, 2, 4).reshape(-1, 1, D_padded, W_padded)
                            # CRITICAL FIX: Ensure 64x64 dimensions
                            if fake_slices_axis.shape[2:] != (64, 64):
                                fake_slices_axis = F.interpolate(fake_slices_axis, size=(64, 64), mode='bilinear', align_corners=False)
                        elif axis_name == 'yz':
                            fake_slices_axis = fake_volumes.permute(0, 4, 1, 2, 3).reshape(-1, 1, D_padded, H_padded)
                            # CRITICAL FIX: Ensure 64x64 dimensions
                            if fake_slices_axis.shape[2:] != (64, 64):
                                fake_slices_axis = F.interpolate(fake_slices_axis, size=(64, 64), mode='bilinear', align_corners=False)

                    # Match batch size
                    fake_slices_axis = fake_slices_axis[:real_slices_axis.shape[0]]
                    out_fake = discriminator(fake_slices_axis, real_conds_axis, real_pos_axis).mean()
                    
                    # Gradient penalty
                    gradient_penalty = compute_gradient_penalty(discriminator, real_slices_axis, fake_slices_axis, 
                                                        real_conds_axis, real_pos_axis, batch_size, device)
                    
                    # Wasserstein loss with GP
                    d_loss = out_fake - out_real + gradient_penalty
                    d_loss.backward()
                    optimizer_discriminators[axis_name].step()
                    
                    # Store losses (only from the last critic iteration for logging)
                    if discrim_iter == args.discrim_iters - 1:
                        d_losses[axis_name] = d_loss.item()
                        real_losses[axis_name] = out_real.item()
                        fake_losses[axis_name] = out_fake.item()
                        
                        with th.no_grad():
                            real_acc = (out_real > 0).float().mean()
                            fake_acc = (out_fake < 0).float().mean()
                            last_disc_acc[axis_name] = (real_acc + fake_acc) / 2
            
            # -----------------
            #  Train Generator 
            # -----------------
            if i % args.discrim_iters == 0:
                optimizer_generator.zero_grad()
                
                # Generate new fake volumes for generator training
                z = th.randn(batch_size, args.latent_dim, 1, 1, 1, device=device)
                conds_3d = conds.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, n_conds, 1, 1, 1)
                fake_volumes = slice_generator(z, conds_3d)
                
                # Pad to (B, 1, 64, 64, 64) if needed
                if fake_volumes.shape[2:] == (51, 51, 51):
                    fake_volumes = F.pad(fake_volumes, (6, 7, 6, 7, 6, 7), mode='constant', value=0)
                
                # Update dimensions after padding
                _, _, D_padded, H_padded, W_padded = fake_volumes.shape
                
                g_loss = 0
                # For each axis discriminator
                for axis_name, discriminator in slice_discriminators.items():
                    # Extract slices from generated volume and feed to discriminator
                    if axis_name == 'xy':
                        # Permute and reshape: (B, 1, D, H, W) -> (B*D, 1, H, W)
                        fake_slices = fake_volumes.permute(0, 2, 1, 3, 4).reshape(-1, 1, H_padded, W_padded)
                    elif axis_name == 'xz':
                        fake_slices = fake_volumes.permute(0, 3, 1, 2, 4).reshape(-1, 1, D_padded, W_padded)
                    elif axis_name == 'yz':
                        fake_slices = fake_volumes.permute(0, 4, 1, 2, 3).reshape(-1, 1, D_padded, H_padded)
                    
                    # CRITICAL FIX: Ensure 64x64 dimensions for generator training too
                    if fake_slices.shape[2:] != (64, 64):
                        fake_slices = F.interpolate(fake_slices, size=(64, 64), mode='bilinear', align_corners=False)
                    
                    # Create corresponding conditions and positions for slices
                    n_slices_per_vol = fake_slices.shape[0] // batch_size
                    slice_conds_gen = conds.repeat_interleave(n_slices_per_vol, dim=0).unsqueeze(-1).unsqueeze(-1)
                    slice_positions_gen = th.linspace(0, 1, n_slices_per_vol, device=device).repeat(batch_size).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    
                    # Generator wants discriminator to think fake slices are real (maximize score)
                    output = discriminator(fake_slices, slice_conds_gen, slice_positions_gen)
                    g_loss -= output.mean()  # Negative because we want to maximize
                
                g_loss.backward()
                optimizer_generator.step()
            else:
                # Set g_loss to previous value for logging consistency
                g_loss = 0  # or keep track of last g_loss value
                    
            # Monitor and adjust discriminator learning rates based on accuracy
            batches_done = epoch * len(dataloader) + i
            if batches_done % 100 == 0 and batches_done > 0:
                avg_disc_acc = sum(last_disc_acc.values()) / len(last_disc_acc)
                if avg_disc_acc < 0.1 or avg_disc_acc > 0.9:  # Discriminator too weak or too strong
                    print(f"Adjusting learning rates due to discriminator accuracy: {avg_disc_acc:.3f}")
                    for axis_name, optimizer in optimizer_discriminators.items():
                        for param_group in optimizer.param_groups:
                            if avg_disc_acc < 0.1:
                                param_group['lr'] *= 1.1  # Increase discriminator LR
                                print(f"  Increased {axis_name} discriminator LR to {param_group['lr']:.6f}")
                            else:
                                param_group['lr'] *= 0.9  # Decrease discriminator LR
                                print(f"  Decreased {axis_name} discriminator LR to {param_group['lr']:.6f}")
                    
                    # Also adjust generator learning rate in opposite direction
                    for param_group in optimizer_generator.param_groups:
                        if avg_disc_acc < 0.1:
                            param_group['lr'] *= 0.9  # Decrease generator LR
                            print(f"  Decreased generator LR to {param_group['lr']:.6f}")
                        else:
                            param_group['lr'] *= 1.1  # Increase generator LR
                            print(f"  Increased generator LR to {param_group['lr']:.6f}")
                    
            # ----------
            #  Logging
            # ----------
            
            if args.track:
                # Log individual axis losses
                log_dict = {}
                for axis_name in slice_discriminators.keys():
                    log_dict[f"d_loss_{axis_name}"] = d_losses.get(axis_name, 0)
                    log_dict[f"real_loss_{axis_name}"] = real_losses.get(axis_name, 0)
                    log_dict[f"fake_loss_{axis_name}"] = fake_losses.get(axis_name, 0)
                    log_dict[f"disc_acc_{axis_name}"] = last_disc_acc[axis_name]
                
                # Log averages
                log_dict["d_loss_avg"] = sum(d_losses.values()) / len(d_losses) if d_losses else 0
                log_dict["g_loss"] = g_loss.item() if isinstance(g_loss, th.Tensor) else g_loss
                log_dict["disc_acc_avg"] = sum(last_disc_acc.values()) / len(last_disc_acc)
                log_dict["epoch"] = epoch
                log_dict["batch"] = batches_done
                
                wandb.log(log_dict)
                
                if i % 10 == 0:  # Print less frequently
                    avg_d_loss = sum(d_losses.values()) / len(d_losses) if d_losses else 0
                    avg_disc_acc = sum(last_disc_acc.values()) / len(last_disc_acc)
                    g_loss_val = g_loss.item() if isinstance(g_loss, th.Tensor) else g_loss
                    
                    print(f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] "
                        f"[D loss: {avg_d_loss:.4f}] [G loss: {g_loss_val:.4f}] "
                        f"[D acc: {avg_disc_acc:.3f}]")
                    
                    # Print per-axis details
                    for axis_name in slice_discriminators.keys():
                        if axis_name in d_losses:
                            print(f"  {axis_name.upper()}: D_loss={d_losses[axis_name]:.4f}, "
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
            for axis_name, generator in slice_discriminators.items():
                ckpt_gen = {
                    "epoch": epoch,
                    "batches_done": epoch * len(dataloader) + len(dataloader) - 1,
                    "generator": slice_generator.state_dict(),
                    "optimizer_generator": optimizer_generator.state_dict(),
                    "loss": g_loss,
                    "n_conds": n_conds,
                    "args": vars(args)
                }
                
                gen_filename = "generator_slicegan.pth"
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
                    "loss": d_losses[axis_name],
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
    
    # ---- Export final generated 3D designs for ParaView ----
    print("Exporting final generated 3D designs for ParaView...")
    slice_generator.eval()
    n_export = 8  # Number of designs to export
    z = th.randn((n_export, args.latent_dim, 1, 1, 1), device=device)
    all_conditions = th.stack(condition_tensors, dim=1)
    linspaces = [
        th.linspace(all_conditions[:, i].min(), all_conditions[:, i].max(), n_export, device=device)
        for i in range(all_conditions.shape[1])
    ]
    export_conds = th.stack(linspaces, dim=1)
    gen_volumes = slice_generator(z, export_conds.reshape(-1, n_conds, 1, 1, 1))
    gen_volumes_np = gen_volumes.squeeze(1).detach().cpu().numpy()

    os.makedirs("paraview_exports", exist_ok=True)
    for i, vol in enumerate(gen_volumes_np):
        np.save(f"paraview_exports/slice_gen3d_{i}.npy", vol)
        # Optionally: save as .vti using pyvista or pyevtk if you want native VTK format

    print(f"Saved {n_export} generated 3D designs to paraview_exports/ as .npy files.")