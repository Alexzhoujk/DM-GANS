"""Faithful modern PyTorch implementation of the DM-GAN image model.

The architecture follows the official three-branch generator and dynamic-memory
write/read/reply computation. Modernizations are deliberately limited to safe API
changes such as BCE-with-logits compatibility and torch.nn.utils spectral norm.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


class GLU(nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.size(1) % 2:
            raise ValueError("GLU requires an even channel dimension")
        value, gate = tensor.chunk(2, dim=1)
        return value * torch.sigmoid(gate)


def up_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_channels, out_channels * 2, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_channels * 2),
        GLU(),
    )


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels * 2),
            GLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + self.block(tensor)


class ConditioningAugmentation(nn.Module):
    def __init__(self, text_dim: int = 256, condition_dim: int = 100) -> None:
        super().__init__()
        self.condition_dim = condition_dim
        self.projection = nn.Sequential(nn.Linear(text_dim, condition_dim * 4), GLU())

    def forward(self, sentence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        statistics = self.projection(sentence)
        mu = statistics[:, : self.condition_dim]
        logvar = statistics[:, self.condition_dim :]
        condition = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return condition, mu, logvar


class InitialImageStage(nn.Module):
    def __init__(self, noise_dim: int = 100, condition_dim: int = 100, channels: int = 64) -> None:
        super().__init__()
        self.channels = channels
        initial_channels = channels * 16
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + condition_dim, initial_channels * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(initial_channels * 4 * 4 * 2),
            GLU(),
        )
        self.upsample = nn.Sequential(
            up_block(initial_channels, channels * 8),
            up_block(channels * 8, channels * 4),
            up_block(channels * 4, channels * 2),
            up_block(channels * 2, channels),
        )

    def forward(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.fc(torch.cat([condition, noise], dim=1))
        hidden = hidden.view(noise.size(0), self.channels * 16, 4, 4)
        return self.upsample(hidden)


class DynamicMemory(nn.Module):
    """Official DM-GAN memory writing, key/value reading, and response gate."""

    def __init__(
        self,
        image_dim: int = 64,
        word_dim: int = 256,
        memory_dim: int | None = None,
        detach_image_summary: bool = True,
    ) -> None:
        super().__init__()
        memory_dim = memory_dim or image_dim * 2
        self.image_dim = image_dim
        self.detach_image_summary = detach_image_summary
        self.word_gate = nn.Linear(word_dim, 1, bias=False)
        self.image_gate = nn.Linear(image_dim, 1, bias=False)
        self.word_write = nn.Sequential(nn.Conv1d(word_dim, memory_dim, 1), nn.ReLU(inplace=True))
        self.image_write = nn.Sequential(nn.Conv1d(image_dim, memory_dim, 1), nn.ReLU(inplace=True))
        self.key = nn.Sequential(nn.Conv1d(memory_dim, image_dim, 1), nn.ReLU(inplace=True))
        self.value = nn.Sequential(nn.Conv1d(memory_dim, image_dim, 1), nn.ReLU(inplace=True))
        self.response = nn.Sequential(nn.Conv2d(image_dim * 2, 1, 1), nn.Sigmoid())

    def forward(
        self,
        image_features: torch.Tensor,
        word_features: torch.Tensor,
        word_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, image_dim, height, width = image_features.shape
        if image_dim != self.image_dim:
            raise ValueError(f"Expected image feature dimension {self.image_dim}, got {image_dim}")
        if word_features.ndim != 3 or word_features.size(0) != batch:
            raise ValueError("word_features must have shape [batch, word_dim, words]")
        words = word_features.size(2)

        image_summary = image_features.mean(dim=(2, 3))
        if self.detach_image_summary:
            image_summary = image_summary.detach()
        word_gate = self.word_gate(word_features.transpose(1, 2)).transpose(1, 2)
        image_gate = self.image_gate(image_summary).unsqueeze(-1).expand(-1, 1, words)
        writing_gate = torch.sigmoid(word_gate + image_gate)

        image_slots = image_summary.unsqueeze(-1).expand(-1, image_dim, words)
        memory = self.word_write(word_features) * writing_gate
        memory = memory + self.image_write(image_slots) * (1.0 - writing_gate)
        keys = self.key(memory)
        values = self.value(memory)

        queries = image_features.flatten(2).transpose(1, 2)
        logits = torch.bmm(queries, keys)
        if word_mask is not None:
            if word_mask.shape != (batch, words):
                raise ValueError(f"word_mask must have shape {(batch, words)}")
            word_mask = word_mask.bool()
            if word_mask.all(dim=1).any():
                raise ValueError("Every caption must contain at least one unmasked token")
            logits = logits.masked_fill(word_mask[:, None, :], torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)
        memory_read = torch.bmm(values, attention.transpose(1, 2)).view(batch, image_dim, height, width)

        response_gate = self.response(torch.cat([image_features, memory_read], dim=1))
        replied = image_features * (1.0 - response_gate) + memory_read * response_gate
        attention_map = attention.transpose(1, 2).reshape(batch, words, height, width)
        return replied, attention_map, writing_gate, response_gate


class RefinementStage(nn.Module):
    def __init__(
        self,
        image_dim: int = 64,
        word_dim: int = 256,
        memory_dim: int | None = None,
        residual_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.memory = DynamicMemory(image_dim, word_dim, memory_dim)
        self.residual = nn.Sequential(*[ResBlock(image_dim * 2) for _ in range(residual_blocks)])
        self.upsample = up_block(image_dim * 2, image_dim)

    def forward(
        self,
        image_features: torch.Tensor,
        word_features: torch.Tensor,
        word_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        replied, attention, writing_gate, response_gate = self.memory(
            image_features, word_features, word_mask
        )
        hidden = self.residual(torch.cat([replied, replied], dim=1))
        return self.upsample(hidden), {
            "attention": attention,
            "writing_gate": writing_gate,
            "response_gate": response_gate,
        }


class ImageHead(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(channels, 3, 3, 1, 1, bias=False), nn.Tanh())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden)


class DMGenerator(nn.Module):
    def __init__(
        self,
        noise_dim: int = 100,
        text_dim: int = 256,
        condition_dim: int = 100,
        channels: int = 64,
        memory_dim: int | None = None,
        residual_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.ca = ConditioningAugmentation(text_dim, condition_dim)
        self.initial = InitialImageStage(noise_dim, condition_dim, channels)
        self.refine_128 = RefinementStage(channels, text_dim, memory_dim, residual_blocks)
        self.refine_256 = RefinementStage(channels, text_dim, memory_dim, residual_blocks)
        self.to_image_64 = ImageHead(channels)
        self.to_image_128 = ImageHead(channels)
        self.to_image_256 = ImageHead(channels)

    def forward(
        self,
        noise: torch.Tensor,
        sentence_features: torch.Tensor,
        word_features: torch.Tensor,
        word_mask: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        condition, mu, logvar = self.ca(sentence_features)
        hidden_64 = self.initial(noise, condition)
        hidden_128, diagnostics_128 = self.refine_128(hidden_64, word_features, word_mask)
        hidden_256, diagnostics_256 = self.refine_256(hidden_128, word_features, word_mask)
        images = [
            self.to_image_64(hidden_64),
            self.to_image_128(hidden_128),
            self.to_image_256(hidden_256),
        ]
        diagnostics = {
            "attention_128": diagnostics_128["attention"],
            "attention_256": diagnostics_256["attention"],
            "writing_gate_128": diagnostics_128["writing_gate"],
            "writing_gate_256": diagnostics_256["writing_gate"],
            "response_gate_128": diagnostics_128["response_gate"],
            "response_gate_256": diagnostics_256["response_gate"],
        }
        return images, diagnostics, mu, logvar


def _sn_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> nn.Conv2d:
    return spectral_norm(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
    )


def _down_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(_sn_conv(in_channels, out_channels, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))


def _refine_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(_sn_conv(in_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))


class DiscriminatorFeatures(nn.Module):
    def __init__(self, image_size: int, channels: int = 32) -> None:
        super().__init__()
        if image_size not in (64, 128, 256):
            raise ValueError("image_size must be 64, 128, or 256")
        layers: list[nn.Module] = [
            _down_block(3, channels),
            _down_block(channels, channels * 2),
            _down_block(channels * 2, channels * 4),
            _down_block(channels * 4, channels * 8),
        ]
        if image_size >= 128:
            layers.extend([_down_block(channels * 8, channels * 16), _refine_block(channels * 16, channels * 8)])
        if image_size >= 256:
            layers[-1:-1] = [_down_block(channels * 16, channels * 32)]
            layers[-1:] = [
                _refine_block(channels * 32, channels * 16),
                _refine_block(channels * 16, channels * 8),
            ]
        self.encoder = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.encoder(image)
        if features.shape[-2:] != (4, 4):
            raise RuntimeError(f"Discriminator encoder must produce 4x4 features, got {features.shape[-2:]}")
        return features


class DiscriminatorHead(nn.Module):
    def __init__(self, image_channels: int, sentence_dim: int | None = None) -> None:
        super().__init__()
        self.sentence_dim = sentence_dim
        if sentence_dim is None:
            self.joint = nn.Identity()
        else:
            self.joint = _refine_block(image_channels + sentence_dim, image_channels)
        self.output = nn.Conv2d(image_channels, 1, 4, 4, 0)

    def forward(self, features: torch.Tensor, sentence: torch.Tensor | None = None) -> torch.Tensor:
        if self.sentence_dim is not None:
            if sentence is None:
                raise ValueError("Conditional discriminator head requires sentence features")
            sentence_map = sentence[:, :, None, None].expand(-1, -1, 4, 4)
            features = self.joint(torch.cat([features, sentence_map], dim=1))
        return self.output(features).flatten()


class MultiscaleDiscriminator(nn.Module):
    """One official-style discriminator for a particular image resolution."""

    def __init__(
        self,
        image_size: int,
        sentence_dim: int = 256,
        channels: int = 32,
        unconditional: bool = True,
    ) -> None:
        super().__init__()
        feature_channels = channels * 8
        self.image_size = image_size
        self.features = DiscriminatorFeatures(image_size, channels)
        self.conditional_head = DiscriminatorHead(feature_channels, sentence_dim)
        self.unconditional_head = DiscriminatorHead(feature_channels) if unconditional else None

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.features(image)

    def conditional_logits(self, features: torch.Tensor, sentence: torch.Tensor) -> torch.Tensor:
        return self.conditional_head(features, sentence)

    def unconditional_logits(self, features: torch.Tensor) -> torch.Tensor | None:
        if self.unconditional_head is None:
            return None
        return self.unconditional_head(features)


def build_discriminators(
    sentence_dim: int = 256, channels: int = 32, sizes: Iterable[int] = (64, 128, 256)
) -> nn.ModuleList:
    return nn.ModuleList([MultiscaleDiscriminator(size, sentence_dim, channels) for size in sizes])
