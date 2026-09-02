"""Minimal end-to-end DM-GAN training step with checkpoint support."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn

from .config import DMGANConfig
from .contrastive import nt_xent_loss
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

    def _encode_text_preserving_order(
        self,
        captions: torch.Tensor,
        caption_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an arbitrary caption order and restore row correspondence."""
        order = torch.argsort(caption_lengths, descending=True, stable=True)
        inverse = torch.argsort(order)
        with torch.no_grad():
            words, sentence = self.text_encoder(captions[order], caption_lengths[order])
        return words[inverse], sentence[inverse]

    def train_step(
        self,
        batch: dict[str, object],
        *,
        part_lambda: float = 0.0,
        part_sigma_fraction: float = 0.08,
        contrastive_lambda: float | None = None,
        contrastive_temperature: float | None = None,
    ) -> tuple[dict[str, float], list[torch.Tensor]]:
        if contrastive_lambda is None:
            contrastive_lambda = self.config.contrastive_lambda
        if contrastive_temperature is None:
            contrastive_temperature = self.config.contrastive_temperature
        if part_lambda < 0:
            raise ValueError("part_lambda must be non-negative")
        if part_sigma_fraction <= 0:
            raise ValueError("part_sigma_fraction must be positive")
        if contrastive_lambda < 0:
            raise ValueError("contrastive_lambda must be non-negative")
        if contrastive_temperature <= 0:
            raise ValueError("contrastive_temperature must be positive")
        self.generator.train()
        self._keep_frozen_generator_modules_in_eval()
        real_images = [image.to(self.device, non_blocking=True) for image in batch["images"]]
        captions = batch["captions"].to(self.device, non_blocking=True)
        caption_lengths = batch["caption_lengths"].to(self.device, non_blocking=True)
        class_ids = batch["class_ids"].to(self.device, non_blocking=True)
        if captions.size(0) < 2:
            raise ValueError("DM-GAN training requires batch_size >= 2")

        has_paired_captions = "paired_captions" in batch
        has_paired_lengths = "paired_caption_lengths" in batch
        if has_paired_captions != has_paired_lengths:
            raise ValueError("Paired training requires both paired_captions and paired_caption_lengths")
        paired_training = has_paired_captions and has_paired_lengths
        if contrastive_lambda > 0 and not paired_training:
            raise ValueError("Contrastive training requires a second caption for every image")
        if contrastive_lambda > 0 and self.image_encoder is None:
            raise ValueError("Contrastive training requires a frozen DAMSM image encoder")
        if paired_training and part_lambda > 0:
            raise ValueError("Part-aware and dual-caption objectives must be run as separate ablations")

        with torch.no_grad():
            word_embeddings, sentence_embeddings = self.text_encoder(captions, caption_lengths)
        word_mask = build_word_mask(caption_lengths, captions.size(1))
        noise = torch.randn(captions.size(0), self.config.noise_dim, device=self.device)
        fake_images, diagnostics, mu, logvar = self.generator(
            noise, sentence_embeddings, word_embeddings, word_mask
        )
        paired_fake_images: list[torch.Tensor] | None = None
        paired_word_embeddings: torch.Tensor | None = None
        paired_sentence_embeddings: torch.Tensor | None = None
        paired_caption_lengths: torch.Tensor | None = None
        paired_mu: torch.Tensor | None = None
        paired_logvar: torch.Tensor | None = None
        if paired_training:
            paired_captions = batch["paired_captions"].to(self.device, non_blocking=True)
            paired_caption_lengths = batch["paired_caption_lengths"].to(self.device, non_blocking=True)
            paired_word_embeddings, paired_sentence_embeddings = self._encode_text_preserving_order(
                paired_captions, paired_caption_lengths
            )
            paired_word_mask = build_word_mask(paired_caption_lengths, paired_captions.size(1))
            paired_fake_images, _, paired_mu, paired_logvar = self.generator(
                noise,
                paired_sentence_embeddings,
                paired_word_embeddings,
                paired_word_mask,
            )

        metrics: dict[str, float] = {}
        self._set_discriminator_grad(True)
        for index, (discriminator, optimizer, real, fake) in enumerate(
            zip(self.discriminators, self.discriminator_optimizers, real_images, fake_images, strict=True)
        ):
            optimizer.zero_grad(set_to_none=True)
            discriminator_total, discriminator_parts = discriminator_loss(
                discriminator, real, fake, sentence_embeddings
            )
            if paired_fake_images is not None and paired_sentence_embeddings is not None:
                paired_discriminator_loss, paired_discriminator_parts = discriminator_loss(
                    discriminator,
                    real,
                    paired_fake_images[index],
                    paired_sentence_embeddings,
                )
                discriminator_total = discriminator_total + paired_discriminator_loss
                for name, value in discriminator_parts.items():
                    metrics[f"d_{index}_view1_{name}"] = float(value)
                for name, value in paired_discriminator_parts.items():
                    metrics[f"d_{index}_view2_{name}"] = float(value)
            else:
                for name, value in discriminator_parts.items():
                    metrics[f"d_{index}_{name}"] = float(value)
            discriminator_total.backward()
            optimizer.step()
            metrics[f"d_{real.shape[-1]}"] = float(discriminator_total.detach())

        self._set_discriminator_grad(False)
        self.generator_optimizer.zero_grad(set_to_none=True)
        image_features: tuple[torch.Tensor, torch.Tensor] | None = None
        paired_image_features: tuple[torch.Tensor, torch.Tensor] | None = None
        if paired_fake_images is not None and self.image_encoder is not None:
            image_features = self.image_encoder(fake_images[-1])
            paired_image_features = self.image_encoder(paired_fake_images[-1])
        loss, parts = generator_loss(
            list(self.discriminators),
            fake_images,
            sentence_embeddings,
            mu,
            logvar,
            image_encoder=self.image_encoder,
            image_features=image_features,
            word_embeddings=word_embeddings,
            caption_lengths=caption_lengths,
            class_ids=class_ids,
            matching_lambda=self.config.matching_lambda,
            kl_lambda=self.config.kl_lambda,
            gamma1=self.config.gamma1,
            gamma2=self.config.gamma2,
            gamma3=self.config.gamma3,
        )
        if paired_fake_images is not None:
            assert paired_sentence_embeddings is not None
            assert paired_word_embeddings is not None
            assert paired_caption_lengths is not None
            assert paired_mu is not None and paired_logvar is not None
            paired_loss, paired_parts = generator_loss(
                list(self.discriminators),
                paired_fake_images,
                paired_sentence_embeddings,
                paired_mu,
                paired_logvar,
                image_encoder=self.image_encoder,
                image_features=paired_image_features,
                word_embeddings=paired_word_embeddings,
                caption_lengths=paired_caption_lengths,
                class_ids=class_ids,
                matching_lambda=self.config.matching_lambda,
                kl_lambda=self.config.kl_lambda,
                gamma1=self.config.gamma1,
                gamma2=self.config.gamma2,
                gamma3=self.config.gamma3,
            )
            first_parts = parts
            parts = {f"view1_{name}": value for name, value in first_parts.items()}
            parts.update({f"view2_{name}": value for name, value in paired_parts.items()})
            loss = loss + paired_loss
            parts["dual_caption_standard"] = loss.detach()
            if contrastive_lambda > 0:
                assert image_features is not None and paired_image_features is not None
                contrastive = nt_xent_loss(
                    image_features[1],
                    paired_image_features[1],
                    temperature=contrastive_temperature,
                )
                loss = loss + contrastive_lambda * contrastive
                parts["contrastive"] = contrastive.detach()
                parts["contrastive_weighted"] = (contrastive_lambda * contrastive).detach()
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
