"""Minimal end-to-end DM-GAN training step with checkpoint support."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn

from .config import DMGANConfig
from .data import build_word_mask
from .losses import discriminator_loss, generator_loss
from .models import DMGenerator, MultiscaleDiscriminator
from .part_aware import gaussian_part_heatmaps, part_aware_alignment_loss


class ExponentialMovingAverage:
    def __init__(self, module: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {name: parameter.detach().clone() for name, parameter in module.named_parameters()}

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        for name, parameter in module.named_parameters():
            self.shadow[name].lerp_(parameter.detach(), 1.0 - self.decay)

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "shadow": self.shadow}


class DMGANTrainer:
    def __init__(
        self,
        config: DMGANConfig,
        generator: DMGenerator,
        discriminators: Iterable[MultiscaleDiscriminator],
        text_encoder: nn.Module,
        image_encoder: nn.Module | None,
        device: torch.device,
    ) -> None:
        config.validate()
        self.config = config
        self.device = device
        self.generator = generator.to(device)
        self.discriminators = nn.ModuleList(discriminators).to(device)
        self.text_encoder = text_encoder.to(device).eval()
        self.image_encoder = image_encoder.to(device).eval() if image_encoder is not None else None
        for encoder in (self.text_encoder, self.image_encoder):
            if encoder is not None:
                for parameter in encoder.parameters():
                    parameter.requires_grad_(False)
        trainable_generator_parameters = [
            parameter for parameter in self.generator.parameters() if parameter.requires_grad
        ]
        if not trainable_generator_parameters:
            raise ValueError("Generator must have at least one trainable parameter")
        self.generator_optimizer = torch.optim.Adam(
            trainable_generator_parameters,
            lr=config.generator_lr,
            betas=(config.beta1, config.beta2),
        )
        self.discriminator_optimizers = [
            torch.optim.Adam(
                discriminator.parameters(),
                lr=config.discriminator_lr,
                betas=(config.beta1, config.beta2),
            )
            for discriminator in self.discriminators
        ]
        self.ema = ExponentialMovingAverage(self.generator)
        self.step = 0

    def _keep_frozen_generator_modules_in_eval(self) -> None:
        for module in self.generator.modules():
            parameters = list(module.parameters(recurse=True))
            if parameters and not any(parameter.requires_grad for parameter in parameters):
                module.eval()

    def _set_discriminator_grad(self, enabled: bool) -> None:
        for discriminator in self.discriminators:
            for parameter in discriminator.parameters():
                parameter.requires_grad_(enabled)

    def train_step(
        self,
        batch: dict[str, object],
        *,
        part_lambda: float = 0.0,
        part_sigma_fraction: float = 0.08,
    ) -> tuple[dict[str, float], list[torch.Tensor]]:
        if part_lambda < 0:
            raise ValueError("part_lambda must be non-negative")
        if part_sigma_fraction <= 0:
            raise ValueError("part_sigma_fraction must be positive")
        self.generator.train()
        self._keep_frozen_generator_modules_in_eval()
        real_images = [image.to(self.device, non_blocking=True) for image in batch["images"]]
        captions = batch["captions"].to(self.device, non_blocking=True)
        caption_lengths = batch["caption_lengths"].to(self.device, non_blocking=True)
        class_ids = batch["class_ids"].to(self.device, non_blocking=True)
        if captions.size(0) < 2:
            raise ValueError("DM-GAN training requires batch_size >= 2")

        with torch.no_grad():
            word_embeddings, sentence_embeddings = self.text_encoder(captions, caption_lengths)
        word_mask = build_word_mask(caption_lengths, captions.size(1))
        noise = torch.randn(captions.size(0), self.config.noise_dim, device=self.device)
        fake_images, diagnostics, mu, logvar = self.generator(
            noise, sentence_embeddings, word_embeddings, word_mask
        )

        metrics: dict[str, float] = {}
        self._set_discriminator_grad(True)
        for index, (discriminator, optimizer, real, fake) in enumerate(
            zip(self.discriminators, self.discriminator_optimizers, real_images, fake_images, strict=True)
        ):
            optimizer.zero_grad(set_to_none=True)
            loss, parts = discriminator_loss(discriminator, real, fake, sentence_embeddings)
            loss.backward()
            optimizer.step()
            metrics[f"d_{real.shape[-1]}"] = float(loss.detach())
            for name, value in parts.items():
                metrics[f"d_{index}_{name}"] = float(value)

        self._set_discriminator_grad(False)
        self.generator_optimizer.zero_grad(set_to_none=True)
        loss, parts = generator_loss(
            list(self.discriminators),
            fake_images,
            sentence_embeddings,
            mu,
            logvar,
            image_encoder=self.image_encoder,
            word_embeddings=word_embeddings,
            caption_lengths=caption_lengths,
            class_ids=class_ids,
            matching_lambda=self.config.matching_lambda,
            kl_lambda=self.config.kl_lambda,
            gamma1=self.config.gamma1,
            gamma2=self.config.gamma2,
            gamma3=self.config.gamma3,
        )
        if part_lambda > 0:
            required_fields = ("part_coordinates", "part_visible", "token_part_targets")
            missing = [field for field in required_fields if field not in batch]
            if missing:
                raise ValueError(f"Part-aware training requires batch fields: {', '.join(missing)}")
            coordinates = batch["part_coordinates"].to(self.device, non_blocking=True)
            visible = batch["part_visible"].to(self.device, non_blocking=True)
            token_targets = batch["token_part_targets"].to(self.device, non_blocking=True)
            scale_losses: list[torch.Tensor] = []
            for scale in (128, 256):
                attention = diagnostics[f"attention_{scale}"]
                sigma = max(1.0, attention.shape[-1] * part_sigma_fraction)
                heatmaps = gaussian_part_heatmaps(
                    coordinates,
                    visible,
                    attention.shape[-2],
                    attention.shape[-1],
                    sigma=sigma,
                )
                scale_loss = part_aware_alignment_loss(attention, heatmaps, token_targets, word_mask)
                scale_losses.append(scale_loss)
                parts[f"part_alignment_{scale}"] = scale_loss.detach()
            part_loss = torch.stack(scale_losses).mean()
            loss = loss + part_lambda * part_loss
            parts["part_alignment"] = part_loss.detach()
            parts["part_weighted"] = (part_lambda * part_loss).detach()
        loss.backward()
        self.generator_optimizer.step()
        self._set_discriminator_grad(True)
        self.ema.update(self.generator)
        self.step += 1
        metrics["g_total"] = float(loss.detach())
        metrics.update({f"g_{name}": float(value) for name, value in parts.items()})
        return metrics, [image.detach() for image in fake_images]

    def save_checkpoint(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config.to_dict(),
                "step": self.step,
                "generator": self.generator.state_dict(),
                "discriminators": self.discriminators.state_dict(),
                "generator_optimizer": self.generator_optimizer.state_dict(),
                "discriminator_optimizers": [
                    optimizer.state_dict() for optimizer in self.discriminator_optimizers
                ],
                "ema": self.ema.state_dict(),
            },
            destination,
        )
