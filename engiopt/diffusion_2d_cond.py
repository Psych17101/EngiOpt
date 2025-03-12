'''based on the diffuser example from huggingface: https://huggingface.co/learn/diffusion-course/unit1/2
'''

from __future__ import annotations

from dataclasses import dataclass
import os
import random
import time

from diffusers import DDPMScheduler
from diffusers import UNet2DConditionModel
from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch import nn
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
    batch_size: int = 1
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

    # if th.backends.mps.is_available():
    #     device = th.device("mps")
    # elif th.cuda.is_available():
    #     device = th.device("cuda")
    # else:
    device = th.device("cpu")

    # Loss function
    adversarial_loss = th.nn.MSELoss()

    # Initialize generator and discriminator
    model = UNet2DConditionModel(
        sample_size=(100, 100),
        in_channels=1,
        out_channels=1,
        cross_attention_dim=64,
    )

    model.to(device)
    adversarial_loss.to(device)

    # Configure data loader
    training_ds = problem.dataset.with_format("torch", device=device)["train"]
    filtered_ds = th.zeros(len(training_ds), 100, 100, device=device)
    for i in range(len(training_ds)):
        filtered_ds[i] = transforms.Resize((100, 100))(training_ds[i]['optimal_design'].reshape(1, training_ds[i]['nelx'], training_ds[i]['nely']))
    training_ds = th.utils.data.TensorDataset(filtered_ds.flatten(1), training_ds['volfrac'])
    dataloader = th.utils.data.DataLoader(
        training_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Optimizer
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    # Training loop

    optimizer = th.optim.AdamW(model.parameters(), lr=4e-4)
    @th.no_grad()
    def sample_designs(n_designs: int) -> th.Tensor:
        """Samples n_designs from the generator."""
        # Sample noise
        z = th.randn((n_designs, args.latent_dim), device=device, dtype=th.float)
        # THESE BOUNDS ARE PROBLEM DEPENDENT

        linspaces = [th.linspace(objs[:, i].min(), objs[:, i].max(), n_designs, device=device) for i in range(objs.shape[1])]

        objs_small = th.stack(linspaces, dim=1)
        desired_objs = objs_small.reshape(-1,1,1)
        desired_objs = desired_objs.expand(-1,1,64)
        noise = th.randn((25,1,100,100)).to(device)
        timesteps = th.full((25,), (950))
        test_ds = th.utils.data.TensorDataset(noise, timesteps, desired_objs)
        dataloader = th.utils.data.DataLoader(test_ds, batch_size=1)
        gen_images = []
        for noise, timesteps, desired_objs in tqdm.tqdm(dataloader):
            gen_imgs = model(noise, timesteps, encoder_hidden_states=desired_objs)[0]
            gen_images.append(gen_imgs)
        
        return objs_small, th.cat(gen_images, dim=0)

    # ----------
    #  Training
    # ----------
    for epoch in tqdm.trange(args.n_epochs):
        for i, data in enumerate(dataloader):
            # THIS IS PROBLEM DEPENDENT
            designs = data[0].reshape(-1,1,100,100)

            objs = th.stack((data[1:]), dim=1).reshape(-1,1,1)
            objs_ex = objs.expand(-1,1,64)

            clean_images = designs

            # Sample noise to add to the images

            noise = th.randn(clean_images.shape).to(clean_images.device)

            bs = clean_images.shape[0]

            # Sample a random timestep for each image

            timesteps = th.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Get the model prediction

            noise_pred = model(noisy_images, timesteps, return_dict=False, encoder_hidden_states=objs_ex)[0]

            # Calculate the loss

            loss = th.nn.functional.mse_loss(noise_pred, noise)

            loss.backward(loss)

            # Update the model parameters with the optimizer

            optimizer.step()

            optimizer.zero_grad()

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
                    desired_objs, designs = sample_designs(25)
                    fig, axes = plt.subplots(5, 5, figsize=(12, 12))

                    # Flatten axes for easy indexing
                    axes = axes.flatten()

                    # Plot each tensor as a scatter plot
                    for j, tensor in enumerate(designs):
                        img = tensor.cpu().numpy().reshape(100,100)  # Extract x and y coordinates
                        do = desired_objs[j].cpu()
                        axes[j].imshow(img)  # Scatter plot
                        axes[j].title.set_text(f"volfrac: {do[0]:.2f}")
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
                        artifact_gen = wandb.Artifact(f"{args.algo}_generator", type="model")
                        artifact_gen.add_file("generator.pth")
                        artifact_disc = wandb.Artifact(f"{args.algo}_discriminator", type="model")
                        artifact_disc.add_file("discriminator.pth")

                        wandb.log_artifact(artifact_gen, aliases=[f"seed_{args.seed}"])
                        wandb.log_artifact(artifact_disc, aliases=[f"seed_{args.seed}"])

    wandb.finish()
