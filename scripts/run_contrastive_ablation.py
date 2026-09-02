"""Run a paired equal-budget DM-GAN control versus CL2 fine-tuning ablation."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dmgan.checkpoints import load_official_generator_checkpoint
from dmgan.config import DMGANConfig
from dmgan.damsm import DAMSMImageEncoder, DAMSMTextEncoder, load_frozen_checkpoint
from dmgan.data import CUBCaptionDataset, build_word_mask, collate_caption_samples
from dmgan.losses import discriminator_loss
from dmgan.models import DMGenerator, MultiscaleDiscriminator, build_discriminators
from dmgan.training import DMGANTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def generator_from_official(checkpoint: Path) -> DMGenerator:
    return load_official_generator_checkpoint(
        DMGenerator(channels=64, memory_dim=128, residual_blocks=2), checkpoint
    )


def configure_refinement_finetuning(generator: DMGenerator) -> None:
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    prefixes = ("refine_128.", "refine_256.", "to_image_128.", "to_image_256.")
    for name, parameter in generator.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad_(True)


def next_batch(iterator: Any, loader: DataLoader) -> tuple[dict[str, object], Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


@torch.no_grad()
def encode_preserving_order(
    text_encoder: nn.Module,
    captions: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(lengths, descending=True, stable=True)
    inverse = torch.argsort(order)
    words, sentence = text_encoder(captions[order], lengths[order])
    return words[inverse], sentence[inverse]


def warmup_discriminators(
    generator: DMGenerator,
    discriminators: list[MultiscaleDiscriminator],
    optimizers: list[torch.optim.Optimizer],
    text_encoder: nn.Module,
    loader: DataLoader,
    iterator: Any,
    device: torch.device,
    steps: int,
    log_every: int,
) -> Any:
    """Warm up one shared D start using the same dual-caption batch structure."""
    generator.to(device).eval()
    module_list = nn.ModuleList(discriminators).to(device).train()
    for step in range(1, steps + 1):
        batch, iterator = next_batch(iterator, loader)
        real_images = [image.to(device, non_blocking=True) for image in batch["images"]]
        captions = batch["captions"].to(device, non_blocking=True)
        lengths = batch["caption_lengths"].to(device, non_blocking=True)
        paired_captions = batch["paired_captions"].to(device, non_blocking=True)
        paired_lengths = batch["paired_caption_lengths"].to(device, non_blocking=True)
        with torch.no_grad():
            words, sentence = text_encoder(captions, lengths)
            paired_words, paired_sentence = encode_preserving_order(
                text_encoder, paired_captions, paired_lengths
            )
            noise = torch.randn(captions.size(0), 100, device=device)
            fake_images, _, _, _ = generator(
                noise, sentence, words, build_word_mask(lengths, captions.size(1))
            )
            paired_fake_images, _, _, _ = generator(
                noise,
                paired_sentence,
                paired_words,
                build_word_mask(paired_lengths, paired_captions.size(1)),
            )
        losses: list[float] = []
        for index, (discriminator, optimizer, real, fake) in enumerate(
            zip(module_list, optimizers, real_images, fake_images, strict=True)
        ):
            optimizer.zero_grad(set_to_none=True)
            first_loss, _ = discriminator_loss(discriminator, real, fake, sentence)
            second_loss, _ = discriminator_loss(
                discriminator, real, paired_fake_images[index], paired_sentence
            )
            total = first_loss + second_loss
            total.backward()
            optimizer.step()
            losses.append(float(total.detach()))
        if step == 1 or step == steps or step % log_every == 0:
            print(
                json.dumps(
                    {
                        "phase": "dual_caption_discriminator_warmup",
                        "step": step,
                        "d_64": losses[0],
                        "d_128": losses[1],
                        "d_256": losses[2],
                    }
                ),
                flush=True,
            )
    return iterator


def build_trainer(
    config: DMGANConfig,
    generator_checkpoint: Path,
    discriminator_state: dict[str, Any],
    discriminator_optimizer_states: list[dict[str, Any]],
    text_encoder: nn.Module,
    image_encoder: nn.Module,
    device: torch.device,
) -> DMGANTrainer:
    generator = generator_from_official(generator_checkpoint)
    configure_refinement_finetuning(generator)
    discriminators = build_discriminators(config.text_dim, config.discriminator_channels)
    nn.ModuleList(discriminators).load_state_dict(discriminator_state)
    trainer = DMGANTrainer(
        config,
        generator,
        discriminators,
        text_encoder,
        image_encoder,
        device,
    )
    for optimizer, state in zip(
        trainer.discriminator_optimizers, discriminator_optimizer_states, strict=True
    ):
        optimizer.load_state_dict(state)
    return trainer


def save_generator_checkpoint(
    trainer: DMGANTrainer,
    path: Path,
    *,
    seed: int,
    variant: str,
    steps: int,
) -> None:
    torch.save(
        {
            "variant": variant,
            "seed": seed,
            "steps": steps,
            "config": trainer.config.to_dict(),
            "generator": cpu_clone(trainer.generator.state_dict()),
            "ema": cpu_clone(trainer.ema.state_dict()),
        },
        path,
    )


@torch.inference_mode()
def save_paired_preview(
    control: DMGANTrainer,
    contrastive: DMGANTrainer,
    batch: dict[str, object],
    output: Path,
    seed: int,
) -> None:
    captions = batch["captions"].to(control.device)
    lengths = batch["caption_lengths"].to(control.device)
    words, sentence = control.text_encoder(captions, lengths)
    word_mask = build_word_mask(lengths, captions.size(1))
    set_seed(seed)
    noise = torch.randn(captions.size(0), control.config.noise_dim, device=control.device)
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if control.device.type == "cuda" else None
    control.generator.eval()
    control_images, _, _, _ = control.generator(noise, sentence, words, word_mask)
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    contrastive.generator.eval()
    contrastive_images, _, _, _ = contrastive.generator(noise, sentence, words, word_mask)
    comparison = torch.cat([control_images[-1], contrastive_images[-1]], dim=0)
    save_image(((comparison + 1.0) / 2.0).clamp(0, 1), output, nrow=captions.size(0))


def decode_batch(batch: dict[str, object], index_to_word: dict[int, str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {"view1": [], "view2": []}
    for field, length_field, output_name in (
        ("captions", "caption_lengths", "view1"),
        ("paired_captions", "paired_caption_lengths", "view2"),
    ):
        for tokens, length in zip(batch[field], batch[length_field], strict=True):
            result[output_name].append(" ".join(index_to_word[int(token)] for token in tokens[: int(length)]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/birds"))
    parser.add_argument("--generator-checkpoint", type=Path, default=Path("checkpoints/bird_DMGAN.pth"))
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260902, 20260903, 20260904])
    parser.add_argument("--d-warmup-steps", type=int, default=200)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--generator-lr", type=float, default=2e-5)
    parser.add_argument("--discriminator-lr", type=float, default=2e-4)
    parser.add_argument("--contrastive-lambda", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/final/contrastive_finetune_ablation")
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 2 or args.steps < 1 or args.d_warmup_steps < 0:
        raise SystemExit("batch-size >= 2, steps >= 1, and d-warmup-steps >= 0 are required")
    if args.contrastive_lambda <= 0 or args.temperature <= 0:
        raise SystemExit("contrastive-lambda and temperature must be positive")

    device = torch.device(args.device)
    dataset = CUBCaptionDataset(
        args.data_root,
        "train",
        training=True,
        paired_captions=True,
    )
    text_encoder = load_frozen_checkpoint(DAMSMTextEncoder(len(dataset.ixtoword)), args.text_checkpoint).to(
        device
    )
    image_encoder = load_frozen_checkpoint(DAMSMImageEncoder(), args.image_checkpoint).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.monotonic()
    completed: list[dict[str, object]] = []

    for seed in args.seeds:
        seed_dir = args.output_dir / f"seed_{seed}"
        control_path = seed_dir / "dual_caption_control_final.pt"
        contrastive_path = seed_dir / "contrastive_final.pt"
        if control_path.exists() and contrastive_path.exists() and not args.overwrite:
            print(json.dumps({"seed": seed, "status": "skipped_existing"}), flush=True)
            completed.append({"seed": seed, "status": "skipped_existing"})
            continue
        seed_dir.mkdir(parents=True, exist_ok=True)
        set_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            num_workers=0,
            collate_fn=collate_caption_samples,
            drop_last=True,
        )
        iterator = iter(loader)
        base_config = DMGANConfig(
            batch_size=args.batch_size,
            generator_lr=args.generator_lr,
            discriminator_lr=args.discriminator_lr,
            contrastive_temperature=args.temperature,
            seed=seed,
        )
        warmup_generator = generator_from_official(args.generator_checkpoint)
        warmup_discriminators_list = list(
            build_discriminators(base_config.text_dim, base_config.discriminator_channels)
        )
        warmup_optimizers = [
            torch.optim.Adam(
                discriminator.parameters(),
                lr=base_config.discriminator_lr,
                betas=(base_config.beta1, base_config.beta2),
            )
            for discriminator in warmup_discriminators_list
        ]
        iterator = warmup_discriminators(
            warmup_generator,
            warmup_discriminators_list,
            warmup_optimizers,
            text_encoder,
            loader,
            iterator,
            device,
            args.d_warmup_steps,
            args.log_every,
        )
        discriminator_state = cpu_clone(nn.ModuleList(warmup_discriminators_list).state_dict())
        optimizer_states = [cpu_clone(optimizer.state_dict()) for optimizer in warmup_optimizers]
        torch.save(
            {
                "seed": seed,
                "warmup_steps": args.d_warmup_steps,
                "discriminators": discriminator_state,
                "discriminator_optimizers": optimizer_states,
            },
            seed_dir / "common_discriminator_start.pt",
        )
        del warmup_generator, warmup_discriminators_list, warmup_optimizers
        if device.type == "cuda":
            torch.cuda.empty_cache()

        control = build_trainer(
            replace(base_config, contrastive_lambda=0.0),
            args.generator_checkpoint,
            discriminator_state,
            optimizer_states,
            text_encoder,
            image_encoder,
            device,
        )
        contrastive = build_trainer(
            replace(base_config, contrastive_lambda=args.contrastive_lambda),
            args.generator_checkpoint,
            discriminator_state,
            optimizer_states,
            text_encoder,
            image_encoder,
            device,
        )
        fixed_batch, iterator = next_batch(iterator, loader)
        history: list[dict[str, object]] = []
        seed_started = time.monotonic()
        for step in range(1, args.steps + 1):
            batch = fixed_batch if step == 1 else None
            if batch is None:
                batch, iterator = next_batch(iterator, loader)
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
            control_metrics, _ = control.train_step(batch)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            contrastive_metrics, _ = contrastive.train_step(batch)
            if step == 1 or step == args.steps or step % args.log_every == 0:
                record = {
                    "step": step,
                    "control": control_metrics,
                    "contrastive": contrastive_metrics,
                }
                history.append(record)
                print(
                    json.dumps(
                        {
                            "phase": "paired_training",
                            "seed": seed,
                            "step": step,
                            "control_g": control_metrics["g_total"],
                            "contrastive_g": contrastive_metrics["g_total"],
                            "nt_xent": contrastive_metrics.get("g_contrastive"),
                        }
                    ),
                    flush=True,
                )

        save_generator_checkpoint(
            control,
            control_path,
            seed=seed,
            variant="dual_caption_control",
            steps=args.steps,
        )
        save_generator_checkpoint(
            contrastive,
            contrastive_path,
            seed=seed,
            variant="dmgan_cl2",
            steps=args.steps,
        )
        preview_batch = {key: value[: args.preview_count] for key, value in fixed_batch.items()}
        save_paired_preview(
            control,
            contrastive,
            preview_batch,
            seed_dir / "paired_preview_control_top_cl_bottom.png",
            seed + 9000,
        )
        seed_report = {
            "status": "complete",
            "seed": seed,
            "protocol": {
                "initial_generator": str(args.generator_checkpoint),
                "conditioning_and_matching_encoder": str(args.text_checkpoint),
                "discriminator_warmup_steps": args.d_warmup_steps,
                "paired_training_steps": args.steps,
                "batch_size": args.batch_size,
                "generator_lr": args.generator_lr,
                "discriminator_lr": args.discriminator_lr,
                "contrastive_lambda": args.contrastive_lambda,
                "temperature": args.temperature,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "control": "dual caption, two full DM-GAN losses, lambda_cl=0",
                "contrastive": "identical control plus lambda_cl * NT-Xent on 256px DAMSM codes",
                "trainable_generator_prefixes": [
                    "refine_128",
                    "refine_256",
                    "to_image_128",
                    "to_image_256",
                ],
            },
            "fixed_keys": fixed_batch["keys"],
            "fixed_captions": decode_batch(fixed_batch, dataset.ixtoword),
            "history": history,
            "elapsed_seconds": time.monotonic() - seed_started,
        }
        (seed_dir / "training_report.json").write_text(
            json.dumps(seed_report, indent=2) + "\n", encoding="utf-8"
        )
        completed.append(seed_report)
        del control, contrastive
        if device.type == "cuda":
            torch.cuda.empty_cache()

    run_report = {
        "status": "complete",
        "experiment": "paired equal-budget dual-caption control versus DM-GAN+CL2 fine-tuning",
        "seeds": args.seeds,
        "completed": [{"seed": item["seed"], "status": item["status"]} for item in completed],
        "elapsed_seconds": time.monotonic() - run_started,
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(run_report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
