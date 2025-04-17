"""GANs taken from github.com:IDEALLab/CEBGAN_JMD_2021.git."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

_EPS = 1e-7


class GAN:
    def __init__(
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
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=opt_g_lr, betas=opt_g_betas, eps=opt_g_eps)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=opt_d_lr, betas=opt_d_betas, eps=opt_g_eps)

    @classmethod
    def load_from_checkpoint(
        cls, generator: nn.Module, discriminator: nn.Module, checkpoint: str, *, train_mode: bool = True
    ) -> GAN:
        ckp = torch.load(checkpoint)
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

    def loss_g(self, batch, noise_gen, **kwargs: object):
        fake = self.generator(noise_gen())
        return self.js_loss_g(batch, fake)

    def loss_d(self, batch, noise_gen, **kwargs: object):
        fake = self.generator(noise_gen())
        return self.js_loss_d(batch, fake)

    def js_loss_d(self, real, fake):
        return nn.functional.binary_cross_entropy_with_logits(
            self.discriminator(real), torch.ones(len(real), 1, device=real.device)
        ) + nn.functional.binary_cross_entropy_with_logits(
            self.discriminator(fake), torch.zeros(len(fake), 1, device=fake.device)
        )

    def js_loss_g(self, real, fake):
        return nn.functional.binary_cross_entropy_with_logits(
            self.discriminator(fake), torch.ones(len(fake), 1, device=fake.device)
        )

    def _train_gen_criterion(self, batch, noise_gen, epoch):
        return True

    def _update_d(self, num_iter_d: int, batch, noise_gen, **kwargs: object):
        for _ in range(num_iter_d):
            self.optimizer_D.zero_grad()
            loss = self.loss_d(batch, noise_gen, **kwargs).backward()
            self.optimizer_D.step()
        return loss

    def _update_g(self, num_iter_g: int, batch, noise_gen, **kwargs: object):
        for _ in range(num_iter_g):
            self.optimizer_G.zero_grad()
            loss = self.loss_g(batch, noise_gen, **kwargs).backward()
            self.optimizer_G.step()
        return loss

    def train(
        self,
        dataloader,
        noise_gen,
        epochs: int,
        num_iter_d: int = 5,
        num_iter_g: int = 1,
        batch_report: Callable[..., None] = lambda **_: None,
        save_model: Callable[..., None] = lambda **_: None,
        **kwargs: object,
    ):
        for epoch in range(epochs):
            for i, batch in enumerate(dataloader):
                d_loss = self._update_d(num_iter_d, batch, noise_gen, **kwargs)
                if not self._train_gen_criterion(batch, noise_gen, epoch):
                    continue
                g_loss = self._update_g(num_iter_g, batch, noise_gen, **kwargs)
                batch_report(i, len(dataloader), epoch, epochs, batch, noise_gen, d_loss, g_loss, **kwargs)
                save_model(i, len(dataloader), epoch, self, d_loss, g_loss, **kwargs)


class InfoGAN(GAN):
    def loss_g(self, batch, noise_gen, **_kwargs: object):
        noise = noise_gen()
        latent_code = noise[:, : noise_gen.sizes[0]]
        fake = self.generator(noise)
        js_loss = self.js_loss_g(batch, fake)
        info_loss = self.info_loss(fake, latent_code)
        return js_loss + info_loss

    def loss_d(self, batch, noise_gen, **_kwargs: object):
        noise = noise_gen()
        latent_code = noise[:, : noise_gen.sizes[0]]
        fake = self.generator(noise)
        js_loss = self.js_loss_d(batch, fake)
        info_loss = self.info_loss(fake, latent_code)
        return js_loss + info_loss

    def info_loss(self, fake, latent_code: float):
        q = self.discriminator(fake)[1]
        q_mean = q[:, :, 0]
        q_logstd = q[:, :, 1]
        epsilon = (latent_code - q_mean) / (torch.exp(q_logstd) + _EPS)
        return torch.mean(q_logstd + 0.5 * epsilon**2)


class BezierGAN(InfoGAN):
    def loss_g(self, batch, noise_gen, **_kwargs: object):
        noise = noise_gen()
        latent_code = noise[:, : noise_gen.sizes[0]]
        fake, cp, w, pv, intvls = self.generator(noise)
        js_loss = self.js_loss_g(batch, fake)
        info_loss = self.info_loss(fake, latent_code)
        reg_loss = self.regularizer(cp, w, pv, intvls)
        return js_loss + info_loss + 10 * reg_loss

    def regularizer(self, cp, w, pv, intvls):
        w_loss = torch.mean(w[:, :, 1:-1])
        cp_loss = torch.norm(cp[:, :, 1:] - cp[:, :, :-1], dim=1).mean()
        end_loss = torch.pairwise_distance(cp[:, :, 0], cp[:, :, -1]).mean()
        reg_loss = w_loss + cp_loss + end_loss
        return reg_loss
