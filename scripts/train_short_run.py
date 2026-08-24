"""Short real-CUB run for Session 6 progress evidence, not final evaluation."""

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
from dmgan.data import CUBCaptionDataset, build_word_mask, collate_caption_samples
from dmgan.models import DMGenerator, build_discriminators
from dmgan.training import DMGANTrainer


def save_fixed_sample(
    trainer: DMGANTrainer,
    batch: dict[str, object],
    path: Path,
    seed: int,
) -> None:
    trainer.generator.eval()
    captions = batch["captions"].to(trainer.device)
    lengths = batch["caption_lengths"].to(trainer.device)
    with torch.no_grad():
        words, sentence = trainer.text_encoder(captions, lengths)
        torch.manual_seed(seed)
        if trainer.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        noise = torch.randn(captions.size(0), trainer.config.noise_dim, device=trainer.device)
        images, _, _, _ = trainer.generator(
            noise, sentence, words, build_word_mask(lengths, captions.size(1))
        )
    save_image((images[-1] + 1.0) / 2.0, path, nrow=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/birds"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/session6/short_run"))
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 2:
        raise SystemExit("steps must be positive and batch-size must be at least 2")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    dataset = CUBCaptionDataset(args.data_root, "train", training=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
        collate_fn=collate_caption_samples,
        drop_last=True,
    )
    config = DMGANConfig(batch_size=args.batch_size)
    text_encoder = load_frozen_checkpoint(
        DAMSMTextEncoder(len(dataset.ixtoword)),
        "checkpoints/DAMSMencoders/bird/text_encoder200.pth",
    )
    image_encoder = load_frozen_checkpoint(
        DAMSMImageEncoder(config.text_dim),
        "checkpoints/DAMSMencoders/bird/image_encoder200.pth",
    )
    trainer = DMGANTrainer(
        config,
        DMGenerator(
            noise_dim=config.noise_dim,
            text_dim=config.text_dim,
            condition_dim=config.condition_dim,
            channels=config.generator_channels,
            memory_dim=config.memory_dim,
            residual_blocks=config.residual_blocks,
        ),
        build_discriminators(config.text_dim, config.discriminator_channels),
        text_encoder,
        image_encoder,
        device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    iterator = iter(loader)
    fixed_batch = next(iterator)
    save_fixed_sample(trainer, fixed_batch, args.output_dir / "local_untrained_fixed_256.png", args.seed)
    history: list[dict[str, float | int]] = []
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        metrics, _ = trainer.train_step(batch)
        record: dict[str, float | int] = {"step": step, **metrics}
        history.append(record)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps({key: record[key] for key in ("step", "g_total", "d_64", "d_128", "d_256")}))
    save_fixed_sample(trainer, fixed_batch, args.output_dir / "local_after_short_run_fixed_256.png", args.seed)
    trainer.save_checkpoint(args.output_dir / f"local_step_{args.steps}.pt")
    (args.output_dir / "loss_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    summary = {
        "status": "pass",
        "provenance": "Locally trained modern reimplementation",
        "warning": f"Only {args.steps} steps; samples are pipeline evidence, not quality or metric results",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "first": history[0],
        "last": history[-1],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
