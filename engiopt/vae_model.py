# ruff: noqa: PLR0913
"""Variational Autoencoder model for shape-to-shape transformation."""

from __future__ import annotations

from surrogate_utils import HybridSurrogate
import torch
from torch import nn
import torch.nn.functional as f
from training_utils import make_activation


def least_volume_loss(z: torch.Tensor, eta: float = 1e-3) -> torch.Tensor:
    """Calculate least volume loss to encourage compact latent space.

    This loss encourages the latent space to be compact by approximating the volume
    from the standard deviations of each dimension.

    Implementation based on:
    Chen, Q., & Fuge, M. (2024). Compressing Latent Space via Least Volume.
    arXiv preprint arXiv:2404.17773.

    Args:
        z: Latent representation tensor.
        eta: Small constant to avoid numerical instability.

    Returns:
        Volume loss tensor.
    """
    std_z = torch.std(z, dim=0) + eta
    volume = torch.exp(torch.mean(torch.log(std_z)))
    return volume


def shape2shape_loss(
    x_opt_true: torch.Tensor,
    x_opt_pred: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    z: torch.Tensor,
    cl_pred: torch.Tensor,
    cl_true: torch.Tensor,
    lambda_lv: float = 1e-2,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate the combined loss for shape-to-shape transformation.

    Includes shape reconstruction, least-volume penalty, and surrogate MSE terms.

    Args:
        x_opt_true: True optimized shape tensor.
        x_opt_pred: Predicted optimized shape tensor.
        mu: Mean of the latent distribution.
        logvar: Log variance of the latent distribution.
        z: Sampled latent vector.
        cl_pred: Predicted constraint value.
        cl_true: True constraint value.
        lambda_lv: Weight for the least volume loss term.
        gamma: Weight for the surrogate loss term.

    Returns:
        A tuple of (total_loss, recon_loss, lv_loss, sur_loss).
    """
    # Mark these as intentionally unused.
    del mu, logvar
    recon_loss = f.smooth_l1_loss(x_opt_pred, x_opt_true, reduction="mean")
    lv_loss = least_volume_loss(z)
    sur_loss = f.smooth_l1_loss(cl_pred.view(-1), cl_true.view(-1), reduction="mean")
    total_loss = recon_loss + lambda_lv * lv_loss + gamma * sur_loss
    return total_loss, recon_loss, lv_loss, sur_loss


class Shape2ShapeVAE(nn.Module):
    """Structured mode VAE with optional HybridSurrogate.

    This model can use either a HybridSurrogate (if cat_dim > 0) or a simple MLP
    as the surrogate model.
    """

    def __init__(
        self,
        shape_dim: int,
        cont_dim: int,
        cat_dim: int,
        latent_dim: int = 32,
        surrogate_hidden_layers: int = 2,
        surrogate_hidden_size: int = 64,
        surrogate_activation: str = "relu",
    ) -> None:
        """Initialize the Shape2ShapeVAE model.

        Args:
            shape_dim: Dimension of the shape representation.
            cont_dim: Dimension of continuous parameters.
            cat_dim: Dimension of categorical parameters.
            latent_dim: Dimension of the latent space.
            surrogate_hidden_layers: Number of hidden layers in the surrogate model.
            surrogate_hidden_size: Size of hidden layers in the surrogate model.
            surrogate_activation: Activation function for the surrogate model.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.cont_dim = cont_dim
        self.cat_dim = cat_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(shape_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2),  # mu, logvar
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(latent_dim, 64)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(64, 128)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(128, shape_dim)),
        )

        # Surrogate
        if cat_dim > 0:
            self.use_hybrid = True
            self.surrogate = HybridSurrogate(
                latent_dim,
                cont_dim,
                cat_dim,
                hidden_layers=surrogate_hidden_layers,
                hidden_size=surrogate_hidden_size,
                activation=surrogate_activation,
            )
        else:
            self.use_hybrid = False
            act_fn = make_activation(surrogate_activation)
            layers = []
            in_dim = latent_dim + cont_dim
            for _ in range(surrogate_hidden_layers):
                layers.append(nn.Linear(in_dim, surrogate_hidden_size))
                layers.append(act_fn)
                in_dim = surrogate_hidden_size
            layers.append(nn.Linear(in_dim, 1))
            self.surrogate = nn.Sequential(*layers)

    def encode(self, x_init: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the input shape to latent space parameters.

        Args:
            x_init: Initial shape tensor.

        Returns:
            A tuple of (mu, logvar) representing the latent distribution.
        """
        h = self.encoder(x_init)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from the latent distribution using the reparameterization trick.

        Args:
            mu: Mean of the latent distribution.
            logvar: Log variance of the latent distribution.

        Returns:
            Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent vector to a shape.

        Args:
            z: Latent vector.

        Returns:
            Decoded shape tensor.
        """
        return self.decoder(z)

    def forward(
        self, x_init: torch.Tensor, param_vec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the model.

        Args:
            x_init: Initial shape tensor.
            param_vec: Parameter vector.

        Returns:
            A tuple of (x_opt_pred, mu, logvar, z, cl_pred).
        """
        mu, logvar = self.encode(x_init)
        z = self.reparameterize(mu, logvar)
        x_opt_pred = self.decode(z)

        if self.use_hybrid:
            cont_params = param_vec[:, : self.cont_dim]
            cat_params = param_vec[:, self.cont_dim :]
            cl_pred = self.surrogate(z, cont_params, cat_params)
        else:
            cl_pred = self.surrogate(torch.cat([z, param_vec], dim=1))

        return x_opt_pred, mu, logvar, z, cl_pred
