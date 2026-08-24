"""Run one real CUB → DAMSM → DM-GAN → loss → backprop integration step."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dmgan.config import DMGANConfig
from dmgan.damsm import DAMSMImageEncoder, DAMSMTextEncoder, load_frozen_checkpoint
from dmgan.data import CUBCaptionDataset, collate_caption_samples
from dmgan.models import DMGenerator, build_discriminators
from dmgan.training import DMGANTrainer


def decode_caption(tokens: torch.Tensor, length: int, index_to_word: dict[int, str]) -> str:
    return " ".join(index_to_word[int(token)] for token in tokens[:length])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/birds"))
    parser.add_argument(
        "--text-checkpoint",
        type=Path,
        default=Path("checkpoints/DAMSMencoders/bird/text_encoder200.pth"),
    )
    parser.add_argument(
        "--image-checkpoint",
        type=Path,
        default=Path("checkpoints/DAMSMencoders/bird/image_encoder200.pth"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/session6/real_integration"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 2:
        raise SystemExit("batch-size must be at least 2")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    dataset = CUBCaptionDataset(args.data_root, "train", training=True)
    data_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=data_generator,
        num_workers=0,
        collate_fn=collate_caption_samples,
        drop_last=True,
    )
    batch = None
    for candidate in loader:
        if torch.unique(candidate["class_ids"]).numel() == args.batch_size:
            batch = candidate
            break
    if batch is None:
        raise RuntimeError("Could not construct a batch with distinct class IDs")

    config = DMGANConfig(batch_size=args.batch_size)
    text_encoder = load_frozen_checkpoint(
        DAMSMTextEncoder(len(dataset.ixtoword)), args.text_checkpoint
    )
    image_encoder = load_frozen_checkpoint(DAMSMImageEncoder(config.text_dim), args.image_checkpoint)
    generator = DMGenerator(
        noise_dim=config.noise_dim,
        text_dim=config.text_dim,
        condition_dim=config.condition_dim,
        channels=config.generator_channels,
        memory_dim=config.memory_dim,
        residual_blocks=config.residual_blocks,
    )
    trainer = DMGANTrainer(
        config,
        generator,
        build_discriminators(config.text_dim, config.discriminator_channels),
        text_encoder,
        image_encoder,
        device,
    )
    metrics, fake_images = trainer.train_step(batch)
    captions = [
        decode_caption(tokens, int(length), dataset.ixtoword)
        for tokens, length in zip(batch["captions"], batch["caption_lengths"], strict=True)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_image((batch["images"][-1] + 1.0) / 2.0, args.output_dir / "real_cub_batch_256.png", nrow=2)
    for image in fake_images:
        size = image.shape[-1]
        save_image(
            (image + 1.0) / 2.0,
            args.output_dir / f"local_random_init_after_one_step_{size}.png",
            nrow=2,
        )
    trainer.save_checkpoint(args.output_dir / "local_step_1.pt")
    report = {
        "status": "pass",
        "evidence": "One real CUB batch completed DAMSM encoding, G/D losses, backward, and optimizer steps",
        "warning": "Generated samples are from random initialization after one step; they are not quality results",
        "device": str(device),
        "torch": torch.__version__,
        "keys": batch["keys"],
        "class_ids": batch["class_ids"].tolist(),
        "captions": captions,
        "real_shapes": [list(image.shape) for image in batch["images"]],
        "fake_shapes": [list(image.shape) for image in fake_images],
        "metrics": metrics,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    (args.output_dir / "report.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
