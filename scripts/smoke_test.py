"""Session 6 acceptance test: forward, masks, losses, backward, and one optimizer step."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from dmgan.config import DMGANConfig
from dmgan.damsm import DAMSMTextEncoder, TinyMatchingImageEncoder
from dmgan.models import DMGenerator, build_discriminators
from dmgan.training import DMGANTrainer


def synthetic_batch(batch: int, words: int, vocabulary: int, device: torch.device) -> dict[str, object]:
    lengths = torch.tensor([words - index for index in range(batch)], dtype=torch.long)
    captions = torch.zeros(batch, words, dtype=torch.long)
    for index, length in enumerate(lengths):
        captions[index, : int(length)] = torch.randint(1, vocabulary, (int(length),))
    images = [torch.rand(batch, 3, size, size) * 2 - 1 for size in (64, 128, 256)]
    return {
        "images": images,
        "captions": captions,
        "caption_lengths": lengths,
        "class_ids": torch.arange(batch),
        "keys": [f"synthetic-{index}" for index in range(batch)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--full-channels", action="store_true", help="Use paper-sized 64/32 G/D channels")
    parser.add_argument("--output", type=Path, help="Optional JSON evidence path")
    parser.add_argument("--sample-dir", type=Path, help="Optional directory for clearly labeled synthetic samples")
    args = parser.parse_args()
    device = torch.device(args.device)
    seed = 20260824
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    generator_channels = 64 if args.full_channels else 8
    discriminator_channels = 32 if args.full_channels else 8
    config = DMGANConfig(
        generator_channels=generator_channels,
        discriminator_channels=discriminator_channels,
        memory_dim=generator_channels * 2,
        residual_blocks=1,
        batch_size=2,
        matching_lambda=1.0,
    )
    text_encoder = DAMSMTextEncoder(vocabulary_size=64, hidden_dim=config.text_dim, dropout=0.0)
    image_encoder = TinyMatchingImageEncoder(config.text_dim)
    generator = DMGenerator(
        noise_dim=config.noise_dim,
        text_dim=config.text_dim,
        condition_dim=config.condition_dim,
        channels=config.generator_channels,
        memory_dim=config.memory_dim,
        residual_blocks=config.residual_blocks,
    )
    discriminators = build_discriminators(config.text_dim, config.discriminator_channels)
    trainer = DMGANTrainer(config, generator, discriminators, text_encoder, image_encoder, device)
    batch = synthetic_batch(config.batch_size, 8, 64, device)
    metrics, images = trainer.train_step(batch)
    assert [tuple(image.shape) for image in images] == [
        (2, 3, 64, 64),
        (2, 3, 128, 128),
        (2, 3, 256, 256),
    ]
    assert all(np.isfinite(value) for value in metrics.values())
    report = {
        "status": "pass",
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "image_shapes": [list(image.shape) for image in images],
        "metrics": metrics,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.sample_dir:
        from torchvision.utils import save_image

        args.sample_dir.mkdir(parents=True, exist_ok=True)
        for image in images:
            size = image.shape[-1]
            save_image((image + 1.0) / 2.0, args.sample_dir / f"synthetic_untrained_{size}.png")


if __name__ == "__main__":
    main()
