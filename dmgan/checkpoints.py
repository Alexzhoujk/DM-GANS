"""Compatibility loaders for author-released DM-GAN checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

import torch

from .models import DMGenerator

GeneratorCheckpointFormat = Literal["auto", "official", "modern-raw", "modern-ema"]


def _generator_key(official_key: str) -> str:
    direct_prefixes = {
        "ca_net.fc.": "ca.projection.0.",
        "h_net1.fc.": "initial.fc.",
        "h_net1.upsample1.": "initial.upsample.0.",
        "h_net1.upsample2.": "initial.upsample.1.",
        "h_net1.upsample3.": "initial.upsample.2.",
        "h_net1.upsample4.": "initial.upsample.3.",
        "img_net1.img.": "to_image_64.head.",
        "img_net2.img.": "to_image_128.head.",
        "img_net3.img.": "to_image_256.head.",
    }
    for prefix, replacement in direct_prefixes.items():
        if official_key.startswith(prefix):
            return replacement + official_key[len(prefix) :]
    for official_stage, modern_stage in (("h_net2", "refine_128"), ("h_net3", "refine_256")):
        stage_prefixes = {
            f"{official_stage}.A.": f"{modern_stage}.memory.word_gate.",
            f"{official_stage}.B.": f"{modern_stage}.memory.image_gate.",
            f"{official_stage}.M_r.": f"{modern_stage}.memory.image_write.",
            f"{official_stage}.M_w.": f"{modern_stage}.memory.word_write.",
            f"{official_stage}.key.": f"{modern_stage}.memory.key.",
            f"{official_stage}.value.": f"{modern_stage}.memory.value.",
            f"{official_stage}.response_gate.": f"{modern_stage}.memory.response.",
            f"{official_stage}.residual.": f"{modern_stage}.residual.",
            f"{official_stage}.upsample.": f"{modern_stage}.upsample.",
        }
        for prefix, replacement in stage_prefixes.items():
            if official_key.startswith(prefix):
                return replacement + official_key[len(prefix) :]
    raise KeyError(f"Unrecognized official generator key: {official_key}")


def load_official_generator_checkpoint(
    generator: DMGenerator, path: str | Path, *, strict: bool = True
) -> DMGenerator:
    official = torch.load(Path(path), map_location="cpu", weights_only=True)
    converted = {_generator_key(key): value for key, value in official.items()}
    generator.load_state_dict(converted, strict=strict)
    return generator


def load_generator_checkpoint(
    generator: DMGenerator,
    path: str | Path,
    *,
    checkpoint_format: GeneratorCheckpointFormat = "auto",
    strict: bool = True,
) -> tuple[DMGenerator, str]:
    """Load an official or modern trainer generator checkpoint.

    Modern trainer checkpoints contain a complete raw generator state under
    ``generator`` and an EMA shadow under ``ema.shadow``.  The shadow only
    contains parameters, so EMA loading first takes the raw state (including
    BatchNorm buffers) and then overlays the averaged parameters.

    ``auto`` preserves official-checkpoint compatibility and selects
    ``modern-ema`` for nested modern checkpoints whenever a valid EMA shadow is
    present.  It falls back to ``modern-raw`` only when no EMA state is saved.
    The resolved format is returned for provenance reporting.
    """

    requested_formats = {"auto", "official", "modern-raw", "modern-ema"}
    if checkpoint_format not in requested_formats:
        choices = ", ".join(sorted(requested_formats))
        raise ValueError(f"checkpoint_format must be one of: {choices}")

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("Generator checkpoint must contain a mapping")

    resolved_format = checkpoint_format
    if resolved_format == "auto":
        if "generator" not in payload:
            resolved_format = "official"
        else:
            ema = payload.get("ema")
            has_ema = isinstance(ema, Mapping) and isinstance(ema.get("shadow"), Mapping)
            resolved_format = "modern-ema" if has_ema else "modern-raw"

    if resolved_format == "official":
        if "generator" in payload:
            raise ValueError(
                "Requested official format, but this is a nested modern trainer checkpoint"
            )
        converted = {_generator_key(str(key)): value for key, value in payload.items()}
        generator.load_state_dict(converted, strict=strict)
        return generator, resolved_format

    raw_state = payload.get("generator")
    if not isinstance(raw_state, Mapping):
        raise TypeError("Modern checkpoint is missing a mapping at 'generator'")
    state = dict(raw_state)

    if resolved_format == "modern-ema":
        ema = payload.get("ema")
        shadow = ema.get("shadow") if isinstance(ema, Mapping) else None
        if not isinstance(shadow, Mapping):
            raise ValueError("Modern EMA checkpoint is missing a mapping at 'ema.shadow'")
        shadow_keys = set(shadow)
        expected_parameter_keys = {name for name, _ in generator.named_parameters()}
        unknown_keys = sorted(shadow_keys - set(state))
        if unknown_keys:
            preview = ", ".join(str(key) for key in unknown_keys[:3])
            raise ValueError(f"EMA shadow contains keys absent from raw generator state: {preview}")
        missing_keys = sorted(expected_parameter_keys - shadow_keys)
        if missing_keys:
            preview = ", ".join(missing_keys[:3])
            raise ValueError(f"EMA shadow is missing generator parameters: {preview}")
        state.update(shadow)

    generator.load_state_dict(cast(dict[str, torch.Tensor], state), strict=strict)
    return generator, resolved_format
