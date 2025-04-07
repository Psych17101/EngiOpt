# ruff: noqa: PLR0913
"""Utility functions and classes for surrogate models."""

from __future__ import annotations

import torch
from torch import nn
from training_utils import make_activation


class HybridSurrogate(nn.Module):
    """A hybrid surrogate model that processes latent and parameters separately.

    This model processes (latent + continuous parameters) and categorical parameters
    separately, then merges them for final prediction.
    """

    def __init__(
        self,
        latent_dim: int,
        cont_dim: int,
        cat_dim: int,
        hidden_layers: int = 2,
        hidden_size: int = 64,
        activation: str = "relu",
    ) -> None:
        """Initialize the hybrid surrogate model.

        Args:
            latent_dim: Dimension of the latent space.
            cont_dim: Dimension of continuous parameters.
            cat_dim: Dimension of categorical parameters.
            hidden_layers: Number of hidden layers in each sub-network.
            hidden_size: Size of hidden layers.
            activation: Activation function to use.
        """
        super().__init__()
        act_fn = make_activation(activation)
        in_dim_cont = latent_dim + cont_dim

        cont_layers = []
        for _ in range(hidden_layers):
            cont_layers.append(nn.Linear(in_dim_cont, hidden_size))
            cont_layers.append(act_fn)
            in_dim_cont = hidden_size
        self.cont_net = nn.Sequential(*cont_layers)

        if cat_dim > 0:
            cat_layers = []
            in_dim_cat = cat_dim
            for _ in range(hidden_layers):
                cat_layers.append(nn.Linear(in_dim_cat, hidden_size))
                cat_layers.append(act_fn)
                in_dim_cat = hidden_size
            self.cat_net = nn.Sequential(*cat_layers)
        else:
            self.cat_net = None

        combined_in = hidden_size * (2 if self.cat_net is not None else 1)
        self.combined = nn.Sequential(
            nn.Linear(combined_in, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1),
        )

    def forward(self, latent: torch.Tensor, cont_params: torch.Tensor, cat_params: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            latent: Latent representation tensor.
            cont_params: Continuous parameters tensor.
            cat_params: Categorical parameters tensor.

        Returns:
            Predicted output tensor.
        """
        cont_out = self.cont_net(torch.cat([latent, cont_params], dim=1))
        if self.cat_net is not None:
            cat_out = self.cat_net(cat_params)
            combined = torch.cat([cont_out, cat_out], dim=1)
        else:
            combined = cont_out
        return self.combined(combined)
