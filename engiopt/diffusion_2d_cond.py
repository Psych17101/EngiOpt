"""based on the diffuser example from huggingface: https://huggingface.co/learn/diffusion-course/unit1/2 ."""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
import time

from diffusers import DDPMScheduler
from diffusers import DDPMPipeline
from diffusers import UNet2DConditionModel
from diffusers.utils import make_image_grid
from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
import tyro
import wandb

@dataclass
class Args:
    """Command-line arguments."""

    problem_id: str = "beams2d"
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
    n_epochs: int = 1000
    """number of epochs of training"""
    batch_size: int = 32
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
    n_objs: int = 7
    """number of objectives -- used as conditional input"""
    sample_interval: int = 400
    """interval between image samples"""

def beta_schedule(T, start=1e-4, end=0.02, scale= 1.0, cosine=False, exp_biasing=False, exp_bias_factor=1):
    """Returns a beta schedule (default: linear) for the diffusion model.

    Args:
        T: Number of timesteps
        start: Starting value of beta
        end: Ending value of beta
        scale: Scaling factor for beta
        cosine: Whether to use a cosine beta schedule
        exp_biasing: Whether to use exponential biasing
        exp_bias_factor: Exponential biasing factor
    """
    beta = th.linspace(scale*start, scale*end, T)
    if cosine:
        beta = []
        a_func = lambda t_val: math.cos((t_val + 0.008) / 1.008 * np.pi / 2) ** 2
        for i in range(T):
            t1 = i / T
            t2 = (i + 1) / T
            beta.append(min(1 - a_func(t2) / a_func(t1), 0.999))

        beta = th.tensor(beta)

    if exp_biasing:
        beta = (th.flip(th.exp(-exp_bias_factor*th.linspace(0, 1, T)), dims=[0]))*beta

    return beta

def get_index_from_list(vals, t, x_shape):
    """ Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    Credit: 
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class DiffusionSampler():
    # Precompute the sqrt alphas and sqrt one minus alphas
    def __init__(self, T, betas):
        self.T = T
        self.betas = betas
        self.alphas = (1. - self.betas)
        self.alphas_cumprod = th.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = th.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = th.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = th.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def forward_diffusion_sample(self, x_0, t, device="cpu"):
        """Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noise = th.randn_like(x_0).to(device)
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        # mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device)\
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)
    
    def forward_diffusion_sample_partial(self, x_0, t_current, t_final, device="cpu"):
        """Takes an image at a timestep and
        adds noise to reach the desired timestep
        """
        for i in range(t_final[0]-t_current[0]):
            t = t_final - i

            noise = th.randn_like(x_0, ).to(device)
            x_0 = th.sqrt(get_index_from_list(self.alphas, t, x_0.shape)) * x_0.to(device)\
            + th.sqrt(get_index_from_list(1-self.alphas, t, x_0.shape)) * noise.to(device)

        # mean + variance
        return x_0, noise.to(device)

    def diffusion_step_sample(self, noise_pred, x_noisy,  t, device="cpu"):
        """Takes an image, noise and step; returns denoised image."""
        betas_t = get_index_from_list(self.betas, t, x_noisy.shape).to(device)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape
        ).to(device)
        sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, x_noisy.shape).to(device)
        model_mean = sqrt_recip_alphas_t * (
            x_noisy - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = get_index_from_list(self.posterior_variance, t, x_noisy.shape).to(device)

        # mean + variance
        return (model_mean + th.sqrt(posterior_variance_t) * noise_pred).to(device)

    def lossfn_builder(self):
        """Returns the loss function for the diffusion model."""
        def lossfn(noise_pred, noise):

            return F.mse_loss(noise_pred, noise)

        return lossfn

    def sample_timestep(self, model, x, t, encoder_hidden_states, c=None, t_mask=None):
        """Calls the model to predict the noise in the image and returns the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        """
        model.eval()
        with th.no_grad():
            betas_t = get_index_from_list(self.betas, t, x.shape)
            sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
                self.sqrt_one_minus_alphas_cumprod, t, x.shape
            )
            sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, x.shape)

            # Call model (current image - noise prediction)
            if c is not None:
                # with th.cuda.amp.autocast(dtype=th.float16):
                model_mean = sqrt_recip_alphas_t * (
                    x - betas_t * model(x, t, c) / sqrt_one_minus_alphas_cumprod_t
                )
            else:
                # with th.cuda.amp.autocast(dtype=th.float16):
                model_mean = sqrt_recip_alphas_t * (
                    x - betas_t * model(x, t, encoder_hidden_states).sample / sqrt_one_minus_alphas_cumprod_t
                )

            posterior_variance_t = get_index_from_list(self.posterior_variance, t, x.shape)
            if t_mask is None:
                device = x.device

                t_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))).to(device) 

            return model_mean + th.sqrt(posterior_variance_t) * th.randn_like(x) * t_mask

if __name__ == "__main__":
    args = tyro.cli(Args)

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    design_shape = (10000, )

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
    adversarial_loss = th.nn.MSELoss()

    # Initialize UNet from Huggingface
    model = UNet2DConditionModel(
        sample_size=(100, 100),
        in_channels=1,
        out_channels=1,
        cross_attention_dim=64,
        block_out_channels=(64, 128),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        layers_per_block=1,
        transformer_layers_per_block=0,
        only_cross_attention=True,
    )

    model.to(device)
    adversarial_loss.to(device)

    # Configure data loader
    training_ds = problem.dataset.with_format("torch", device=device)["train"]
    filtered_ds = th.zeros(len(training_ds), 100, 100, device=device)
    for i in range(len(training_ds)):
        filtered_ds[i] = transforms.Resize((100, 100))(training_ds[i]['optimal_design'].reshape(1, training_ds[i]['nelx'], training_ds[i]['nely']))
    filtered_ds_max = filtered_ds.max()
    filtered_ds_min = filtered_ds.min()
    filtered_ds *= 2
    filtered_ds -= 1
    filtered_ds_norm = (filtered_ds - filtered_ds_min) / (filtered_ds_max - filtered_ds_min)
    training_ds = th.utils.data.TensorDataset(filtered_ds_norm.flatten(1), training_ds['volfrac'])
    vf_min = training_ds.tensors[1].min()
    vf_max = training_ds.tensors[1].max()
    dataloader = th.utils.data.DataLoader(
        training_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )
    num_timesteps = 1000
    # Optimizer
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_timesteps, beta_schedule="linear")

    # Training loop

    optimizer = th.optim.AdamW(model.parameters(), lr=4e-4)

    ## Schedule Parameters
    T = num_timesteps # Number of timesteps
    start = 1e-4 # Starting variance
    end = 0.02 # Ending variance
    # Choose a schedule (if the following are False, then a linear schedule is used)
    cosine = False # Use cosine schedule
    exp_biasing = False # Use exponential schedule
    exp_biasing_factor = 1 # Exponential schedule factor (used if exp_biasing=True)
    ##

    # Choose a variance schedule

    betas = beta_schedule(T=T, start=start, end=end,
                        scale= 1.0, cosine=cosine,
                        exp_biasing=exp_biasing, exp_bias_factor=exp_biasing_factor
                        )

    ddm_sampler = DiffusionSampler(T, betas)

    # Loss function
    def ddm_loss_fn(noise_pred, noise):
        return F.l1_loss(noise_pred, noise)

    @th.no_grad()
    def sample_designs(model, n_designs=25):
        """Samples n_designs designs."""
        model.eval()
        with th.no_grad():

            dims = (n_designs, 1, 100, 100)
            image = th.randn(dims, device=device) # initial image
            encoder_hidden_states = th.linspace(vf_min, vf_max, n_designs, device=device)
            encoder_hidden_states = encoder_hidden_states.view(n_designs, 1, 1).expand(n_designs, 1, 32)
            for i in range(num_timesteps)[::-1]:
                t = th.full((n_designs,), i, device=device, dtype=th.long)

                image = ddm_sampler.sample_timestep(model, image, t, encoder_hidden_states)

        return image, encoder_hidden_states

    # ----------
    #  Training
    # ----------
    for epoch in tqdm.trange(args.n_epochs):
        for i, data in enumerate(dataloader):

            # Zero the parameter gradients
            optimizer.zero_grad()
            designs = data[0].reshape(-1,1,100,100)
            x = designs.to(device)
            objs = th.stack((data[1:]), dim=1).reshape(-1,1,1)
            objs_ex = objs.expand(-1,1,32)


            current_batch_size = x.shape[0]
            t = th.randint(0, T, (current_batch_size,), device=device).long()
            encoder_hidden_states = objs_ex.to(device)

            # Get the noise and the noisy input
            x_noisy, noise = ddm_sampler.forward_diffusion_sample(x, t, device)

            # Forward pass
            # if mp_mode:
            #     with torch.cuda.amp.autocast(dtype=torch.float16):
            #         noise_pred = model_diffuser(x_noisy, t, encoder_hidden_states).sample
            #         loss = ddm_loss_fn(noise_pred, noise)
            #     scaler.scale(loss).backward()
            #     scaler.step(optimizer)
            #     scaler.update()
            # else:
            noise_pred = model(x_noisy, t, encoder_hidden_states).sample
            loss = ddm_loss_fn(noise_pred, noise)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # ----------
            #  Logging
            # ----------
            if args.track:
                batches_done = epoch * len(dataloader) + i
                wandb.log(
                    {
                        "loss": loss.item(),
                        "epoch": epoch,
                        "batch": batches_done,
                    }
                )
                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] [loss: {loss.item()}]]"
                )

                # This saves a grid image of 25 generated designs every sample_interval
                if batches_done % args.sample_interval == 0:
                    # Extract 25 designs

                    designs, hidden_states = sample_designs(model, 25)
                    fig, axes = plt.subplots(5, 5, figsize=(12, 12))

                    # Flatten axes for easy indexing
                    axes = axes.flatten()

                    # Plot the iamge created by each output
                    for j, tensor in enumerate(designs):
                        img = tensor.cpu().numpy().reshape(100,100)  # Extract x and y coordinates
                        do = hidden_states[j,0,0].cpu()
                        axes[j].imshow(img.T)  # image plot
                        axes[j].title.set_text(f"volfrac: {do:.2f}")  # Set title
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
                        ckpt_model = {
                            "epoch": epoch,
                            "batches_done": batches_done,
                            "model": model.state_dict(),
                            "optimizer_generator": optimizer.state_dict(),
                            "loss": loss.item(),
                        }

                        th.save(ckpt_model, "diffusion_model.pth")
                        artifact_gen = wandb.Artifact(f"{args.problem_id}_{args.algo}_generator", type="model")
                        artifact_gen.add_file("generator.pth")
                        artifact_disc = wandb.Artifact(f"{args.problem_id}_{args.algo}_discriminator", type="model")
                        artifact_disc.add_file("discriminator.pth")

                        wandb.log_artifact(artifact_gen, aliases=[f"seed_{args.seed}"])
                        wandb.log_artifact(artifact_disc, aliases=[f"seed_{args.seed}"])

    wandb.finish()
