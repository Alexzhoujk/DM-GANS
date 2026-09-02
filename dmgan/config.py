"""Configuration objects shared by the model, losses, and training code."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DMGANConfig:
    noise_dim: int = 100
    text_dim: int = 256
    condition_dim: int = 100
    generator_channels: int = 64
    discriminator_channels: int = 32
    memory_dim: int = 128
    residual_blocks: int = 2
    branch_num: int = 3
    words_num: int = 18
    captions_per_image: int = 10
    batch_size: int = 10
    generator_lr: float = 2e-4
    discriminator_lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    gamma1: float = 4.0
    gamma2: float = 5.0
    gamma3: float = 10.0
    matching_lambda: float = 5.0
    kl_lambda: float = 1.0
    contrastive_lambda: float = 0.0
    contrastive_temperature: float = 0.5
    seed: int = 20260824

    def validate(self) -> None:
        if self.branch_num != 3:
            raise ValueError("This Session 6 baseline requires exactly three 64/128/256 branches")
        if self.memory_dim != self.generator_channels * 2:
            raise ValueError("Official DM-GAN uses memory_dim = 2 * generator_channels")
        if self.batch_size < 2:
            raise ValueError("Matching-aware discriminator loss requires batch_size >= 2")
        if self.contrastive_lambda < 0:
            raise ValueError("contrastive_lambda must be non-negative")
        if self.contrastive_temperature <= 0:
            raise ValueError("contrastive_temperature must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DMGANConfig:
        with Path(path).open("r", encoding="utf-8") as stream:
            values = yaml.safe_load(stream) or {}
        config = cls(**values)
        config.validate()
        return config
