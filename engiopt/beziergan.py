"""This code is largely based on the excellent PyTorch GAN repo: https://github.com/eriklindernoren/PyTorch-GAN.

We essentially refreshed the Python style, use wandb for logging, and made a few little improvements.
There are also a couple of code parts that are problem dependent and need to be adjusted for the specific problem.
"""

from __future__ import annotations

from collections.abc import Callable
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
    sample_interval: int = 100
    """interval between image samples"""


_EPS = 1e-7


class GAN:
    def __init__(  # noqa: PLR0913
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        opt_g_lr: float = 1e-4,
        opt_g_betas: tuple = (0.5, 0.99),
        opt_g_eps: float = 1e-8,
        opt_d_lr: float = 1e-4,
        opt_d_betas: tuple = (0.5, 0.99),
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.name = name
        self.optimizer_G = th.optim.Adam(self.generator.parameters(), lr=opt_g_lr, betas=opt_g_betas, eps=opt_g_eps)
        self.optimizer_D = th.optim.Adam(self.discriminator.parameters(), lr=opt_d_lr, betas=opt_d_betas, eps=opt_g_eps)

    @classmethod
    def load_from_checkpoint(
        cls, generator: nn.Module, discriminator: nn.Module, checkpoint: str, *, train_mode: bool = True
    ) -> GAN:
        """Loads a saved model."""
        ckp = th.load(checkpoint)
        gan = cls(generator, discriminator)
        gan.discriminator.load_state_dict(ckp["discriminator"])
        gan.generator.load_state_dict(ckp["generator"])
        if train_mode:
            gan.discriminator.train()
            gan.generator.train()
        else:
            gan.discriminator.eval()
            gan.generator.eval()
        gan.optimizer_D.load_state_dict(ckp["optimizer_D"])
        gan.optimizer_G.load_state_dict(ckp["optimizer_G"])
        return gan

    def loss_g(self, batch: dict, noise_gen: Callable) -> th.Tensor:
        """Loss function for the generator."""
        fake = self.generator(noise_gen())
        return self.js_loss_g(batch, fake)

    def loss_d(self, batch: dict, noise_gen: Callable) -> th.Tensor:
        """Loss function for the discriminator."""
        fake = self.generator(noise_gen())
        return self.js_loss_d(batch, fake)

    def js_loss_d(self, real: dict, fake: tuple[th.Tensor, ...]) -> th.Tensor:
        """Entropy loss for the discriminator."""
        return nn.functional.binary_cross_entropy_with_logits(
            self.discriminator(real['optimal_design'], th.stack((real['cd_val'], real['cl_val']), 1)),
            th.ones(real['optimal_design'].shape[0], 1, device=real['optimal_design'].device),
        ) + nn.functional.binary_cross_entropy_with_logits(
            self.discriminator(fake[0], fake[1][:, :, 0]), th.zeros(len(fake[0]), 1, device=fake[0].device)
        )

    def js_loss_g(self, fake: tuple[th.Tensor, ...]) -> th.Tensor:
        """Entropy loss for the generator."""
        return nn.functional.binary_cross_entropy_with_logits(
            self.discriminator(fake[0], fake[1][:,:,0]), th.ones(len(fake[0]), 1, device=fake[0].device)
        )

    def _train_gen_criterion(self) -> bool:
        return True

    def _update_d(self, num_iter_d: int, batch: dict, noise_gen: Callable, **kwargs: object) -> th.Tensor:
        for _ in range(num_iter_d):
            self.optimizer_D.zero_grad()
            loss = self.loss_d(batch, noise_gen, **kwargs)
            self.loss_d(batch, noise_gen, **kwargs).backward()
            self.optimizer_D.step()
        return loss

    def _update_g(self, num_iter_g: int, batch:dict, noise_gen: Callable, **kwargs: object) -> th.Tensor:
        for _ in range(num_iter_g):
            self.optimizer_G.zero_grad()
            loss = self.loss_g(batch, noise_gen, **kwargs)
            self.loss_g(batch, noise_gen, **kwargs).backward()
            self.optimizer_G.step()
        return loss

    def prepare_batch_report(self, sample_interval: int, n_designs: int) -> Callable:
        """Creates a function for logging to wandb and plotting designs."""

        def batch_report(  # noqa: PLR0913
            batch_no: int, n_batches: int, epoch: int, n_epochs: int, d_loss: th.Tensor, g_loss: th.Tensor
        ) -> None:
            """Logs to wandb and plots designs."""
            batches_done = epoch * n_batches + batch_no
            wandb.log(
                {
                    "d_loss": d_loss.item(),
                    "g_loss": g_loss.item(),
                    "epoch": epoch,
                    "batch": batches_done,
                }
            )
            print(
                f"[Epoch {epoch}/{n_epochs}] [Batch {batch_no}/{n_batches}] [Discriminator loss: {d_loss.item()}] [Generator loss: {g_loss.item()}]"
            )

            # This saves a grid image of 25 generated designs every sample_interval
            if batches_done % sample_interval == 0:
                device = next(self.generator.parameters()).device
                # Extract 25 designs
                noise = th.randn((n_designs, Args.latent_dim), device=device, dtype=th.float)
                designs = self.generator(noise)

                fig, axes = plt.subplots(5, 5, figsize=(12, 12))

                # Flatten axes for easy indexing
                axes = axes.flatten()

                # Plot each tensor as a scatter plot
                for j, tensor in enumerate(designs[0]):
                    x, y = tensor.cpu().detach().numpy()  # Extract x and y coordinates
                    axes[j].scatter(x, y, s=10, alpha=0.7)  # Scatter plot
                    axes[j].set_xlim(-1.1, 1.1)  # Adjust x-axis limits
                    axes[j].set_ylim(-1.1, 0.5)  # Adjust y-axis limits
                    axes[j].set_xticks([])  # Hide x ticks
                    axes[j].set_yticks([])  # Hide y ticks

                plt.tight_layout()
                img_fname = f"images/{batches_done}.png"
                plt.savefig(img_fname)
                plt.close()
                wandb.log({"designs": wandb.Image(img_fname)})

        return batch_report

    def prepare_save(self, algo: str, seed: int) -> Callable:
        """Generates a function to save model checkpoints."""

        def save(batch_no: int, n_batches: int, epoch: int, gan: Callable, d_loss: th.Tensor, g_loss: th.Tensor) -> None:  # noqa: PLR0913
            batches_done = epoch * n_batches + batch_no
            ckpt_gen = {
                "epoch": epoch,
                "batches_done": batches_done,
                "generator": gan.generator.state_dict(),
                "optimizer_generator": gan.optimizer_G.state_dict(),
                "loss": g_loss.item(),
            }
            ckpt_disc = {
                "epoch": epoch,
                "batches_done": batches_done,
                "discriminator": gan.discriminator.state_dict(),
                "optimizer_discriminator": gan.optimizer_D.state_dict(),
                "loss": d_loss.item(),
            }

            th.save(ckpt_gen, "generator.pth")
            th.save(ckpt_disc, "discriminator.pth")
            artifact_gen = wandb.Artifact(f"{algo}_generator", type="model")
            artifact_gen.add_file("generator.pth")
            artifact_disc = wandb.Artifact(f"{algo}_discriminator", type="model")
            artifact_disc.add_file("discriminator.pth")

            wandb.log_artifact(artifact_gen, aliases=[f"seed_{seed}"])
            wandb.log_artifact(artifact_disc, aliases=[f"seed_{seed}"])

        return save

    def train(  # noqa: PLR0913
        self,
        dataloader: th.utils.data.DataLoader,
        noise_gen: Callable,
        n_designs: int,
        epochs: int,
        num_iter_d: int = 5,
        num_iter_g: int = 1,
        **kwargs: object,
    ) -> None:
        """Trains the model."""
        for epoch in tqdm.tqdm(range(epochs)):
            for i, batch in enumerate(dataloader):
                d_loss = self._update_d(num_iter_d, batch, noise_gen, **kwargs)
                if not self._train_gen_criterion():
                    continue
                g_loss = self._update_g(num_iter_g, batch, noise_gen, **kwargs)
                batch_report = self.prepare_batch_report(Args.sample_interval, n_designs)
                save_model = self.prepare_save(Args.algo, Args.seed)
                batch_report(i, len(dataloader), epoch, epochs, d_loss, g_loss)
                save_model(i, len(dataloader), epoch, self, d_loss, g_loss, **kwargs)


class InfoGAN(GAN):
    def loss_g(self, batch: dict, noise_gen: Callable, **_kwargs: object) -> th.Tensor:
        """Loss function for the generator."""
        noise = noise_gen()
        latent_code = noise[:, : noise_gen.sizes[0]]
        fake = self.generator(noise)
        js_loss = self.js_loss_g(batch, fake)
        info_loss = self.info_loss(fake, latent_code)
        return js_loss + info_loss

    def loss_d(self, batch: dict, noise_gen: Callable, **_kwargs: object) -> th.Tensor:
        """Loss function for the discriminator."""
        noise = noise_gen()
        latent_code = noise[:, : noise_gen().shape[0]]
        fake = self.generator(noise)
        js_loss = self.js_loss_d(batch, fake)
        info_loss = self.info_loss(fake, latent_code)
        return js_loss + info_loss

    def info_loss(self, fake: tuple[th.Tensor, ...], latent_code: float) -> th.Tensor:
        """Loss function for the InfoGAN."""
        q = self.discriminator(fake[0], fake[1][:, :, 0])
        q_mean = q.mean()
        q_logstd = q.std().log()
        epsilon = (latent_code - q_mean) / (th.exp(q_logstd) + _EPS)
        return th.mean(q_logstd + 0.5 * epsilon**2)


class BezierGAN(InfoGAN):
    def loss_g(self, batch: dict, noise_gen: Callable, **_kwargs: object) -> th.Tensor:
        """Loss function for the generator."""
        noise = noise_gen()
        latent_code = noise[:, : noise.shape[0]]
        fake, cp, w, pv, intvls = self.generator(noise)
        js_loss = self.js_loss_g(batch, (fake, cp))
        info_loss = self.info_loss((fake, cp), latent_code)
        reg_loss = self.regularizer(cp, w)
        return js_loss + info_loss + 10 * reg_loss

    def regularizer(self, cp: th.Tensor, w: th.Tensor) -> th.Tensor:
        """Returns a regularizaiton of the loss function."""
        w_loss = th.mean(w[:, :, 1:-1])
        cp_loss = th.norm(cp[:, :, 1:] - cp[:, :, :-1], dim=1).mean()
        end_loss = th.pairwise_distance(cp[:, :, 0], cp[:, :, -1]).mean()
        reg_loss = w_loss + cp_loss + end_loss
        return reg_loss

class BezierLayer(nn.Module):
    r"""Produces the data points on the Bezier curve, together with coefficients for regularization purposes.

    Args:
        in_features: size of each input sample.
        n_control_points: number of control points.
        n_data_points: number of data points to be sampled from the Bezier curve.

    Shape:
        - Input:
            - Input Features: `(N, H)` where H = in_features.
            - Control Points: `(N, D, CP)` where D stands for the dimension of Euclidean space,
            and CP is the number of control points. For 2D applications, D = 2.
            - Weights: `(N, 1, CP)` where CP is the number of control points.
        - Output:
            - Data Points: `(N, D, DP)` where D is the dimension and DP is the number of data points.
            - Parameter Variables: `(N, 1, DP)` where DP is the number of data points.
            - Intervals: `(N, DP)` where DP is the number of data points.
    """

    def __init__(self, in_features: int, n_control_points: int, n_data_points: int, eps: float) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points
        self.generate_intervals = nn.Sequential(
            nn.Linear(in_features, n_data_points - 1), nn.Softmax(dim=1), nn.ConstantPad1d([1, 0], 0)
        )
        self.eps = eps

    def forward(self, _input: th.Tensor, control_points: th.Tensor, weights: th.Tensor) -> tuple[th.Tensor, ...]:
        """Forward pass of the model.

        Args:
            _input (torch.Tensor): Input tensor.
            control_points (torch.Tensor): bezier curve control points.
            weights (torch.Tensor): `(N, 1, CP)` where CP is the number of control points.

        Returns:
            torch.Tensor: Validity score tensor.
        """
        cp, w = self._check_consistency(control_points, weights)  # [N, d, n_cp], [N, 1, n_cp]
        bs, pv, intvls = self.generate_bernstein_polynomial(_input)  # [N, n_cp, n_dp]
        dp = (cp * w) @ bs / (w @ bs)  # [N, d, n_dp]
        return dp, pv, intvls

    def _check_consistency(self, control_points: th.Tensor, weights: th.Tensor) -> tuple[th.Tensor, ...]:
        assert control_points.shape[-1] == self.n_control_points, "The number of control points is not consistent."
        assert weights.shape[-1] == self.n_control_points, "The number of weights is not consistent."
        assert weights.shape[1] == 1, "There should be only one weight corresponding to each control point."
        return control_points, weights

    def generate_bernstein_polynomial(self, input_: th.Tensor) -> tuple[th.Tensor, ...]:
        """Generates a Bernstein polynomial."""
        intvls = self.generate_intervals(input_)  # [N, n_dp]
        pv = th.cumsum(intvls, -1).clamp(0, 1).unsqueeze(1)  # [N, 1, n_dp]
        pw1 = th.arange(0.0, self.n_control_points, device=input_.device).view(1, -1, 1)  # [1, n_cp, 1]
        pw2 = th.flip(pw1, (1,))  # [1, n_cp, 1]
        lbs = (
            pw1 * th.log(pv + self.eps)
            + pw2 * th.log(1 - pv + self.eps)
            + th.lgamma(th.tensor(self.n_control_points, device=input_.device) + self.eps).view(1, -1, 1)
            - th.lgamma(pw1 + 1 + self.eps)
            - th.lgamma(pw2 + 1 + self.eps)
        )  # [N, n_cp, n_dp]
        bs = th.exp(lbs)  # [N, n_cp, n_dp]
        return bs, pv, intvls

    def extra_repr(self) -> str:
        """Returns the number of input features, control points, and data points."""
        return (
            f"in_features={self.in_features}, n_control_points={self.n_control_points}, n_data_points={self.n_data_points}"
        )


class _Combo(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None

    def forward(self, input_: th.Tensor) -> th.Tensor:
        return self.model(input_)


class LinearCombo(_Combo):
    r"""Regular fully connected layer combo."""

    def __init__(self, in_features: int, out_features: int, alpha: float = 0.2):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features), nn.LeakyReLU(alpha))


class Deconv1DCombo(_Combo):
    r"""Regular deconvolutional layer combo."""

    def __init__(  # noqa: PLR0913
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, alpha: float = 0.2
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(alpha),
        )


class MLP(nn.Module):
    """Regular fully connected network generating features.

    Args:
        in_features: The number of input features.
        out_feature: The number of output features.
        layer_width: The widths of the hidden layers.
        combo: The layer combination to be stacked up.

    Shape:
        - Input: `(N, H_in)` where H_in = in_features.
        - Output: `(N, H_out)` where H_out = out_features.
    """

    def __init__(self, in_features: int, out_features: int, layer_width: tuple[int, ...], combo: LinearCombo = LinearCombo):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.model = self._build_model(layer_width, combo)

    def forward(self, input_: th.Tensor) -> th.Tensor:
        """Forward pass for the model."""
        return self.model(input_)

    def _build_model(self, layer_width: tuple[int, ...], combo: LinearCombo) -> nn.Sequential:
        model = nn.Sequential()
        for idx, (in_ftr, out_ftr) in enumerate(zip((self.in_features, *layer_width), (*layer_width, self.out_features))):
            model.add_module(str(idx), combo(in_ftr, out_ftr))
        return model


class CPWGenerator(nn.Module):
    """Generate given number of control points and weights for Bezier Layer.

    Args:
        in_features: The number of input features.
        n_control_points: The number of control point and weights to be output.
            Should be even.
        dense_layers: The widths of the hidden layers of the MLP connecting
            input features and deconvolutional layers.
        deconv_channels: The number of channels deconvolutional layers have.

    Shape:
        - Input: `(N, H_in)` where H_in = in_features.
        - Output:
            - Control Points: `(N, 2, H_out)` where H_out = n_control_points.
            - Weights: `(N, 1, H_out)` where H_out = n_control_points.
    """

    def __init__(
        self,
        in_features: int,
        n_control_points: int,
        dense_layers: tuple[int, ...] = (1024,),
        deconv_channels: tuple[int, ...] = (96 * 8, 96 * 4, 96 * 2, 96),
    ):
        super().__init__()
        self.in_features = in_features
        self.n_control_points = n_control_points

        self.in_chnl, self.in_width = self._calculate_parameters(n_control_points, deconv_channels)

        self.dense = MLP(in_features, self.in_chnl * self.in_width, dense_layers)
        self.deconv = self._build_deconv(deconv_channels)
        self.cp_gen = nn.Sequential(nn.Conv1d(deconv_channels[-1], 2, 1), nn.Tanh())
        self.w_gen = nn.Sequential(nn.Conv1d(deconv_channels[-1], 1, 1), nn.Sigmoid())

    def forward(self, input_: th.Tensor) -> tuple[th.Tensor, ...]:
        """Forward pass for the model."""
        x = self.deconv(self.dense(input_).view(-1, self.in_chnl, self.in_width))
        cp = self.cp_gen(x)
        w = self.w_gen(x)
        return cp, w

    def _calculate_parameters(self, n_control_points: int, channels: tuple) -> tuple[int, ...]:
        n_l = len(channels) - 1
        in_chnl = channels[0]
        in_width = n_control_points // (2**n_l)
        assert in_width >= 4, f"Too many deconvolutional layers ({n_l}) for the {self.n_control_points} control points."  # noqa: PLR2004
        return in_chnl, in_width

    def _build_deconv(self, channels: tuple) -> nn.Module:
        deconv = nn.Sequential()
        for idx, (in_chnl, out_chnl) in enumerate(zip(channels[:-1], channels[1:])):
            deconv.add_module(str(idx), Deconv1DCombo(in_chnl, out_chnl, kernel_size=4, stride=2, padding=1))
        return deconv


class BezierGenerator(nn.Module):
    """Generator for BezierGAN alike projects.

    Args:
        in_features: The number of input features.
        n_control_points: The number of control point and weights to be output.
        n_data_points: The number of data points to output.
        m_features: The number of intermediate features for generating intervals.
        feature_gen_layer: The widths of hidden layers for generating intermediate features.
        dense_layers: The widths of the hidden layers of the MLP connecting
            input features and deconvolutional layers.
        deconv_channels: The number of channels deconvolutional layers have.

    Shape:
        - Input: `(N, H_in)` where H_in = in_features.
        - Output:
            - Data Points: `(N, D, DP)` where D is the dimension and DP is the number of data points.
            - Control Points: `(N, 2, CP)` where CP = n_control_points.
            - Weights: `(N, 1, CP)` where CP = n_control_points.
            - Parameter Variables: `(N, 1, DP)` where DP is the number of data points.
            - Intervals: `(N, DP)` where DP is the number of data points.
    """

    def __init__(  # noqa: PLR0913
        self,
        in_features: int,
        n_control_points: int,
        n_data_points: int,
        eps: float,
        m_features: int = 256,
        feature_gen_layers: tuple[int, ...] = (1024,),
        dense_layers: tuple[int, ...] = (1024,),
        deconv_channels: tuple[int, ...] = (96 * 8, 96 * 4, 96 * 2, 96),
    ):
        super().__init__()
        self.in_features = in_features
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points

        self.feature_generator = MLP(in_features, m_features, feature_gen_layers)
        self.cpw_generator = CPWGenerator(in_features, n_control_points, dense_layers, deconv_channels)
        self.bezier_layer = BezierLayer(m_features, n_control_points, n_data_points, eps)

    def forward(self, input_: th.Tensor) -> tuple[th.Tensor, ...]:
        """Forward pass for the model."""
        features = self.feature_generator(input_)
        cp, w = self.cpw_generator(input_)
        dp, pv, intvls = self.bezier_layer(features, cp, w)
        return dp, cp, w, pv, intvls

    def extra_repr(self) -> str:
        """Returns the number of input features, control points, and data points used in the model."""
        return (
            f"in_features={self.in_features}, n_control_points={self.n_control_points}, n_data_points={self.n_data_points}"
        )


class Discriminator(nn.Module):
    def __init__(self, n_objs: int, design_shape: tuple[int, ...]):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(design_shape)) + n_objs, 512),
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
    bezier_control_pts = 40
    generator = BezierGenerator(args.latent_dim, bezier_control_pts, problem.design_space.shape[1], eps=_EPS)
    discriminator = Discriminator(args.n_objs, problem.design_space.shape)

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
    gan = BezierGAN(
        generator,
        discriminator,
        opt_g_lr=args.lr,
        opt_g_betas=(args.b1, args.b2),
        opt_g_eps=_EPS,
        opt_d_lr=args.lr,
        opt_d_betas=(args.b1, args.b2),
    )

    @th.no_grad()
    def sample_designs(n_designs: int) -> tuple[th.Tensor, ...]:
        """Samples n_designs from the generator."""
        # Sample noise
        z = th.randn((n_designs, args.latent_dim), device=device, dtype=th.float)
        # THESE BOUNDS ARE PROBLEM DEPENDENT
        cls = th.linspace(0.3, 1.2, n_designs, device=device)
        cds = th.linspace(71, 600, n_designs, device=device)
        desired_objs = th.stack((cls, cds), dim=1)
        gen_imgs = generator(z, desired_objs)
        return desired_objs, gen_imgs

    n_designs = 25
    gan.train(
        dataloader,
        epochs=args.n_epochs,
        noise_gen=lambda: th.randn((Args.batch_size, Args.latent_dim), device=device, dtype=th.float),
        n_designs=n_designs,
    )
    wandb.finish()
