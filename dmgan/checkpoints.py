"""Compatibility loaders for author-released DM-GAN checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from .models import DMGenerator


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
