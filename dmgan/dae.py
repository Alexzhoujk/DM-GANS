"""Modern inference implementation of the released CUB DAE-GAN generator.

The module names intentionally match the official checkpoint.  DAE-GAN's
released attention code used ``mask.repeat(query_length, 1)`` after flattening
a batch-major tensor.  That associates many queries with another sample's
padding mask whenever a batch contains different caption lengths.  The
corrected broadcast is the default here; ``legacy_mask_repeat=True`` is kept
only for sensitivity checks against the released implementation.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class DAEGLU(nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.size(1) % 2:
            raise ValueError("DAEGLU requires an even channel dimension")
        value, gate = tensor.chunk(2, dim=1)
        return value * torch.sigmoid(gate)


def _conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)


def _up_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        _conv3x3(in_channels, out_channels * 2),
        nn.BatchNorm2d(out_channels * 2),
        DAEGLU(),
    )


class DAEResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            _conv3x3(channels, channels * 2),
            nn.BatchNorm2d(channels * 2),
            DAEGLU(),
            _conv3x3(channels, channels),
            nn.BatchNorm2d(channels),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + self.block(tensor)


class DAEConditioningAugmentation(nn.Module):
    """DAE-GAN conditioning augmentation with official checkpoint key names."""

    def __init__(self, text_dim: int = 256, condition_dim: int = 100) -> None:
        super().__init__()
        self.condition_dim = condition_dim
        self.fc = nn.Linear(text_dim, condition_dim * 4, bias=True)
        self.relu = DAEGLU()

    def forward(
        self,
        sentence: torch.Tensor,
        *,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        statistics = self.relu(self.fc(sentence))
        mu = statistics[:, : self.condition_dim]
        logvar = statistics[:, self.condition_dim :]
        condition = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        return condition, mu, logvar


class DAEGlobalAttention(nn.Module):
    """Aspect-conditioned word attention used by each DAE refinement stage."""

    def __init__(
        self,
        image_dim: int,
        context_dim: int,
        *,
        legacy_mask_repeat: bool = False,
    ) -> None:
        super().__init__()
        self.conv_context = nn.Conv2d(context_dim, image_dim, 1, bias=False)
        self.legacy_mask_repeat = legacy_mask_repeat

    @staticmethod
    def flattened_mask(
        word_mask: torch.Tensor,
        query_length: int,
        *,
        legacy: bool,
    ) -> torch.Tensor:
        if word_mask.ndim != 2:
            raise ValueError("word_mask must have shape [batch, words]")
        if legacy:
            return word_mask.repeat(query_length, 1)
        batch, words = word_mask.shape
        return (
            word_mask[:, None, :]
            .expand(batch, query_length, words)
            .reshape(batch * query_length, words)
        )

    def forward(
        self,
        image: torch.Tensor,
        words: torch.Tensor,
        aspect: torch.Tensor,
        word_mask: torch.Tensor | None,
        *,
        legacy_mask_repeat: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, image_dim, height, width = image.shape
        if words.ndim != 3 or words.size(0) != batch:
            raise ValueError("words must have shape [batch, context_dim, words]")
        if aspect.shape != (batch, words.size(1)):
            raise ValueError("aspect must have shape [batch, context_dim]")
        source_length = words.size(2)
        query_length = height * width

        target = image.view(batch, image_dim, query_length).transpose(1, 2).contiguous()
        projected_words = self.conv_context(words.unsqueeze(3)).squeeze(3)
        projected_aspect = self.conv_context(aspect[:, :, None, None]).squeeze(3)
        source = projected_words + projected_aspect
        attention = torch.bmm(target, source).view(batch * query_length, source_length)

        if word_mask is not None:
            if word_mask.shape != (batch, source_length):
                raise ValueError(f"word_mask must have shape {(batch, source_length)}")
            if word_mask.bool().all(dim=1).any():
                raise ValueError("Every caption must contain at least one unmasked token")
            legacy = self.legacy_mask_repeat if legacy_mask_repeat is None else legacy_mask_repeat
            flattened = self.flattened_mask(word_mask.bool(), query_length, legacy=legacy)
            attention = attention.masked_fill(flattened, torch.finfo(attention.dtype).min)

        attention = torch.softmax(attention, dim=1)
        attention = attention.view(batch, query_length, source_length).transpose(1, 2).contiguous()
        weighted_context = torch.bmm(source, attention).view(batch, image_dim, height, width)
        return weighted_context, attention.view(batch, source_length, height, width)


class DAEInitialStage(nn.Module):
    def __init__(
        self,
        channels: int = 1024,
        noise_dim: int = 100,
        condition_dim: int = 100,
    ) -> None:
        super().__init__()
        self.gf_dim = channels
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + condition_dim, channels * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(channels * 4 * 4 * 2),
            DAEGLU(),
        )
        self.upsample1 = _up_block(channels, channels // 2)
        self.upsample2 = _up_block(channels // 2, channels // 4)
        self.upsample3 = _up_block(channels // 4, channels // 8)
        self.upsample4 = _up_block(channels // 8, channels // 16)

    def forward(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.fc(torch.cat([condition, noise], dim=1))
        hidden = hidden.view(noise.size(0), self.gf_dim, 4, 4)
        hidden = self.upsample1(hidden)
        hidden = self.upsample2(hidden)
        hidden = self.upsample3(hidden)
        return self.upsample4(hidden)


class DAENextStage(nn.Module):
    def __init__(
        self,
        image_dim: int = 64,
        text_dim: int = 256,
        residual_blocks: int = 2,
        *,
        legacy_mask_repeat: bool = False,
    ) -> None:
        super().__init__()
        self.gf_dim = image_dim
        self.att = DAEGlobalAttention(
            image_dim,
            text_dim,
            legacy_mask_repeat=legacy_mask_repeat,
        )
        self.residual = nn.Sequential(
            *[DAEResBlock(image_dim * 2) for _ in range(residual_blocks)]
        )
        self.upsample = _up_block(image_dim * 2, image_dim)
        self.glu = DAEGLU()
        self.attr_trans = nn.Linear(text_dim, image_dim, bias=False)

    def forward(
        self,
        image: torch.Tensor,
        words: torch.Tensor,
        aspect: torch.Tensor,
        word_mask: torch.Tensor | None,
        *,
        upsample: bool,
        legacy_mask_repeat: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context, attention = self.att(
            image,
            words,
            aspect,
            word_mask,
            legacy_mask_repeat=legacy_mask_repeat,
        )
        hidden = self.residual(torch.cat([image, context], dim=1))
        hidden = self.upsample(hidden) if upsample else self.glu(hidden)
        height, width = hidden.shape[-2:]
        projected_aspect = self.attr_trans(aspect)
        # Preserve the released checkpoint's repeat/view layout.  Replacing it
        # with ordinary channel-wise broadcasting changes the trained function
        # substantially even though broadcasting looks more natural here.
        aspect_map = projected_aspect.repeat(1, height * width).view(
            -1, self.gf_dim, height, width
        )
        aspect_hidden = hidden + aspect_map
        return hidden, attention, aspect_hidden


class DAEImageHead(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.img = nn.Sequential(_conv3x3(channels, 3), nn.Tanh())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.img(hidden)


class DAEGenerator(nn.Module):
    """Four-output released DAE-GAN generator: 64, 128, 128, and 256 px."""

    def __init__(
        self,
        noise_dim: int = 100,
        text_dim: int = 256,
        condition_dim: int = 100,
        channels: int = 64,
        residual_blocks: int = 2,
        *,
        legacy_mask_repeat: bool = False,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.text_dim = text_dim
        self.legacy_mask_repeat = legacy_mask_repeat
        self.ca_net = DAEConditioningAugmentation(text_dim, condition_dim)
        self.h_net1 = DAEInitialStage(channels * 16, noise_dim, condition_dim)
        self.img_net1 = DAEImageHead(channels)
        self.h_net2 = DAENextStage(
            channels,
            text_dim,
            residual_blocks,
            legacy_mask_repeat=legacy_mask_repeat,
        )
        self.img_net2 = DAEImageHead(channels)
        self.h_net3 = DAENextStage(
            channels,
            text_dim,
            residual_blocks,
            legacy_mask_repeat=legacy_mask_repeat,
        )
        self.img_net3 = DAEImageHead(channels)
        self.h_net4 = DAENextStage(
            channels,
            text_dim,
            residual_blocks,
            legacy_mask_repeat=legacy_mask_repeat,
        )
        self.img_net4 = DAEImageHead(channels)

    def forward(
        self,
        noise: torch.Tensor,
        sentence_features: torch.Tensor,
        word_features: torch.Tensor,
        aspect_features: torch.Tensor,
        word_mask: torch.Tensor | None = None,
        *,
        sample_conditioning: bool = True,
        legacy_mask_repeat: bool | None = None,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        batch = noise.size(0)
        if noise.shape != (batch, self.noise_dim):
            raise ValueError(f"noise must have shape [batch, {self.noise_dim}]")
        if sentence_features.shape != (batch, self.text_dim):
            raise ValueError(f"sentence_features must have shape [batch, {self.text_dim}]")
        if aspect_features.shape != (batch, self.text_dim, 3):
            raise ValueError(f"aspect_features must have shape [batch, {self.text_dim}, 3]")
        legacy = self.legacy_mask_repeat if legacy_mask_repeat is None else legacy_mask_repeat

        condition, mu, logvar = self.ca_net(sentence_features, sample=sample_conditioning)
        aspects = aspect_features.unbind(dim=2)
        hidden_64 = self.h_net1(noise, condition)
        image_64 = self.img_net1(hidden_64)
        _, attention_1, aspect_hidden_128 = self.h_net2(
            hidden_64,
            word_features,
            aspects[0],
            word_mask,
            upsample=True,
            legacy_mask_repeat=legacy,
        )
        image_128_global = self.img_net2(aspect_hidden_128)
        _, attention_2, aspect_hidden_128_local = self.h_net3(
            aspect_hidden_128,
            word_features,
            aspects[1],
            word_mask,
            upsample=False,
            legacy_mask_repeat=legacy,
        )
        image_128_local = self.img_net3(aspect_hidden_128_local)
        _, attention_3, aspect_hidden_256 = self.h_net4(
            aspect_hidden_128_local,
            word_features,
            aspects[2],
            word_mask,
            upsample=True,
            legacy_mask_repeat=legacy,
        )
        image_256 = self.img_net4(aspect_hidden_256)
        return (
            [image_64, image_128_global, image_128_local, image_256],
            {
                "attention_128_global": attention_1,
                "attention_128_local": attention_2,
                "attention_256": attention_3,
            },
            mu,
            logvar,
        )


def load_dae_generator_checkpoint(
    generator: DAEGenerator,
    checkpoint: str | Path,
    *,
    strict: bool = True,
) -> DAEGenerator:
    """Load an official DAE-GAN state dictionary with auditable strictness."""

    state: Any = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    generator.load_state_dict(state, strict=strict)
    return generator


def load_dae_attribute_bank(
    metadata: str | Path,
    *,
    split: str = "test",
    max_aspects: int = 3,
    max_aspect_words: int = 5,
) -> tuple[np.ndarray, tuple[Any, Any, Any, Any]]:
    """Load DAE's preprocessed aspect phrases and its shared caption metadata.

    The official pickle appends train/test aspect arrays to the four standard
    AttnGAN/DM-GAN caption objects.  The returned integer bank is padded to
    ``[captions, max_aspects, max_aspect_words]``.
    """

    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    with Path(metadata).open("rb") as stream:
        payload = pickle.load(stream, encoding="latin1")
    if not isinstance(payload, (list, tuple)) or len(payload) != 6:
        raise ValueError("DAE metadata must contain four caption objects and two aspect arrays")
    caption_payload = tuple(payload[:4])
    raw_attributes = payload[4 if split == "train" else 5]
    bank = np.zeros((len(raw_attributes), max_aspects, max_aspect_words), dtype=np.int64)
    for caption_index, phrases in enumerate(raw_attributes):
        for aspect_index, phrase in enumerate(phrases[:max_aspects]):
            tokens = np.asarray(phrase[:max_aspect_words], dtype=np.int64)
            bank[caption_index, aspect_index, : tokens.size] = tokens
    return bank, caption_payload


@torch.inference_mode()
def encode_dae_aspects(
    text_encoder: nn.Module,
    aspect_tokens: torch.Tensor,
) -> torch.Tensor:
    """Encode ``[B,3,5]`` DAE aspects exactly as the released sampler does."""

    if aspect_tokens.ndim != 3 or aspect_tokens.shape[1:] != (3, 5):
        raise ValueError("aspect_tokens must have shape [batch, 3, 5]")
    lengths = torch.full(
        (aspect_tokens.size(0),),
        5,
        dtype=torch.long,
        device=aspect_tokens.device,
    )
    encoded: list[torch.Tensor] = []
    for aspect_index in range(3):
        _, sentence = text_encoder(aspect_tokens[:, aspect_index, :], lengths)
        encoded.append(sentence)
    return torch.stack(encoded, dim=2)
