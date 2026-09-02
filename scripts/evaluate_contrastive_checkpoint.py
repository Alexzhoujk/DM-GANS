"""Paired evaluation of official DM-GAN and DM-GAN+CL CUB checkpoints.

The two generators receive the same caption indices, latent noise, and
conditioning-augmentation random draws.  R-precision is deliberately measured
with a third, frozen DAMSM evaluator rather than either model's conditioning
encoder, which avoids letting DM-GAN+CL grade its own representation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.utils import save_image

try:
    from .evaluate_baseline import (
        encode_sentence_bank,
        file_sha256,
        load_caption_bank,
        sample_negative_indices,
        sorted_caption_batch,
    )
except ImportError:  # Direct execution: python scripts/evaluate_contrastive_checkpoint.py
    from evaluate_baseline import (
        encode_sentence_bank,
        file_sha256,
        load_caption_bank,
        sample_negative_indices,
        sorted_caption_batch,
    )

from dmgan.checkpoints import load_official_generator_checkpoint
from dmgan.damsm import DAMSMImageEncoder, DAMSMTextEncoder, load_frozen_checkpoint
from dmgan.data import build_word_mask
from dmgan.metrics import InceptionEvaluator, fid_from_features, inception_score, r_precision, split_mean_std
from dmgan.models import DMGenerator


def checkpoint_record(path: Path, *, required: bool = True) -> dict[str, str | None]:
    """Return an auditable absolute path and SHA256 for an input artifact."""

    if not path.exists():
        if required:
            raise SystemExit(f"Missing required file: {path}")
        return {"path": str(path.resolve()), "sha256": None}
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def reset_generation_rng(seed: int, device: torch.device) -> None:
    """Reset the global RNG used by conditioning augmentation."""

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def paired_generator_forward(
    baseline: DMGenerator,
    contrastive: DMGenerator,
    noise: torch.Tensor,
    baseline_sentence: torch.Tensor,
    baseline_words: torch.Tensor,
    contrastive_sentence: torch.Tensor,
    contrastive_words: torch.Tensor,
    word_mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a pair while also sharing the stochastic CA epsilon.

    Passing the same ``noise`` tensor controls z. Restoring the CPU/CUDA RNG
    state before the second forward controls the random draw made inside the
    conditioning-augmentation module.
    """

    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    baseline_images, _, _, _ = baseline(
        noise,
        baseline_sentence,
        baseline_words,
        word_mask,
    )
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)
    contrastive_images, _, _, _ = contrastive(
        noise,
        contrastive_sentence,
        contrastive_words,
        word_mask,
    )
    return baseline_images[-1], contrastive_images[-1]


def r_precision_summary(matches: np.ndarray, seed: int) -> dict[str, float]:
    mean, std = split_mean_std(matches, splits=10, seed=seed)
    return {
        "overall_percent": float(matches.mean() * 100.0),
        "split_mean_percent": mean * 100.0,
        "split_std_percent": std * 100.0,
    }


def paired_r_statistics(
    baseline_matches: np.ndarray,
    contrastive_matches: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, float | int | list[float]]:
    """Compute paired statistics with a CUB-image cluster bootstrap.

    CUB has ten captions per image and a 30,000-sample schedule repeats the
    first 670 captions. Resampling rows would therefore overstate the effective
    sample size. The bootstrap samples source images with replacement and keeps
    every evaluated caption belonging to each sampled image.
    """

    from scipy.stats import binomtest

    baseline = np.asarray(baseline_matches, dtype=np.int8)
    contrastive = np.asarray(contrastive_matches, dtype=np.int8)
    if baseline.shape != contrastive.shape or baseline.ndim != 1:
        raise ValueError("paired R-precision arrays must have the same one-dimensional shape")
    cluster_ids = np.asarray(cluster_ids, dtype=np.int64)
    if cluster_ids.shape != baseline.shape:
        raise ValueError("cluster_ids must have the same shape as paired R-precision arrays")

    both_correct = int(np.count_nonzero((baseline == 1) & (contrastive == 1)))
    baseline_only = int(np.count_nonzero((baseline == 1) & (contrastive == 0)))
    contrastive_only = int(np.count_nonzero((baseline == 0) & (contrastive == 1)))
    both_wrong = int(np.count_nonzero((baseline == 0) & (contrastive == 0)))
    discordant = baseline_only + contrastive_only
    p_value = (
        float(binomtest(contrastive_only, discordant, p=0.5, alternative="two-sided").pvalue)
        if discordant
        else 1.0
    )

    difference = contrastive - baseline
    _, cluster_inverse = np.unique(cluster_ids, return_inverse=True)
    cluster_count = int(cluster_inverse.max()) + 1
    cluster_sizes = np.bincount(cluster_inverse).astype(np.int64)
    cluster_differences = np.bincount(cluster_inverse, weights=difference).astype(np.float64)
    rng = np.random.default_rng(seed)
    delta_draws = np.empty(bootstrap_resamples, dtype=np.float64)
    bootstrap_chunk = 256
    for start in range(0, bootstrap_resamples, bootstrap_chunk):
        stop = min(start + bootstrap_chunk, bootstrap_resamples)
        sampled_clusters = rng.integers(
            0,
            cluster_count,
            size=(stop - start, cluster_count),
        )
        sampled_differences = cluster_differences[sampled_clusters].sum(axis=1)
        sampled_sizes = cluster_sizes[sampled_clusters].sum(axis=1)
        delta_draws[start:stop] = sampled_differences * (100.0 / sampled_sizes)
    lower, upper = np.percentile(delta_draws, [2.5, 97.5])
    return {
        "both_correct": both_correct,
        "baseline_only_correct": baseline_only,
        "contrastive_only_correct": contrastive_only,
        "both_wrong": both_wrong,
        "discordant_pairs": discordant,
        "mcnemar_exact_two_sided_p": p_value,
        "mcnemar_note": (
            "Descriptive only: caption rows from the same CUB image and repeated rows are correlated."
        ),
        "bootstrap_unit": "official CUB test image",
        "bootstrap_cluster_count": cluster_count,
        "bootstrap_resamples": bootstrap_resamples,
        "cluster_bootstrap_delta_percentage_points_95_ci": [float(lower), float(upper)],
    }


def metric_text(value: float | None, digits: int = 4) -> str:
    return "skipped" if value is None else f"{value:.{digits}f}"


def markdown_report(report: dict[str, Any]) -> str:
    baseline = report["results"]["baseline"]
    contrastive = report["results"]["contrastive"]
    delta = report["results"]["delta_contrastive_minus_baseline"]
    paired = report["paired_r_precision"]
    decision = report["decision"]
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    return f"""# Paired DM-GAN vs DM-GAN+CL Checkpoint Evaluation

## Conclusion

**{decision['verdict'].upper()}** — {decision['summary']}

This is a paired comparison: each row uses the same CUB test caption, latent
noise `z`, and conditioning-augmentation random draw. DM-GAN uses the original
DAMSM text encoder for conditioning; DM-GAN+CL uses its separate CL-trained text
encoder. Both outputs are judged by the same independently loaded original
DAMSM evaluator.

## Fixed protocol

- Samples: {report['sample_count']:,}
- Seed: {report['seed']}
- Resolution: 256 x 256
- Caption schedule: deterministic cycling over the official CUB test caption bank
- FID: one shared PyTorch Inception evaluator and one shared real-statistics file
- R-precision: one shared original DAMSM evaluator; correct caption versus the
  same 99 other-class negatives for both generators
- Preview layout: baseline on the left, DM-GAN+CL on the right

## Results

| Metric | DM-GAN | DM-GAN+CL | CL - baseline |
| --- | ---: | ---: | ---: |
| PyTorch FID ↓ | {metric_text(baseline['fid_pytorch'])} | {metric_text(contrastive['fid_pytorch'])} | {metric_text(delta['fid_pytorch'])} |
| R-precision ↑ | {baseline['r_precision']['split_mean_percent']:.2f}% ± {baseline['r_precision']['split_std_percent']:.2f}% | {contrastive['r_precision']['split_mean_percent']:.2f}% ± {contrastive['r_precision']['split_std_percent']:.2f}% | {delta['r_precision_percentage_points']:+.2f} pp |
| ImageNet IS ↑ | {metric_text(baseline['is_imagenet_mean'])} | {metric_text(contrastive['is_imagenet_mean'])} | {metric_text(delta['is_imagenet_mean'])} |

## Paired R-precision evidence

- Baseline-only correct: {paired['baseline_only_correct']:,}
- DM-GAN+CL-only correct: {paired['contrastive_only_correct']:,}
- Exact McNemar p-value: {paired['mcnemar_exact_two_sided_p']:.6g}
- McNemar caveat: {paired['mcnemar_note']}
- CUB-image cluster bootstrap ({paired['bootstrap_cluster_count']:,} image clusters)
  95% CI for the R-precision delta:
  [{paired['cluster_bootstrap_delta_percentage_points_95_ci'][0]:+.3f},
  {paired['cluster_bootstrap_delta_percentage_points_95_ci'][1]:+.3f}] percentage points

The per-sample binary outcomes and sample/caption indices are saved in
`paired_samples.npz`, so the paired test can be independently reproduced.

## Checkpoint provenance

Every checkpoint and the FID real-statistics file are recorded with absolute
paths and SHA256 digests in `report.json`.

## Limitations

{limitations}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fair paired evaluation of official DM-GAN and DM-GAN+CL checkpoints."
    )
    parser.add_argument("--metadata-root", type=Path, default=Path("data/birds"))
    parser.add_argument(
        "--baseline-generator-checkpoint",
        type=Path,
        default=Path("checkpoints/bird_DMGAN.pth"),
    )
    parser.add_argument(
        "--baseline-text-checkpoint",
        type=Path,
        default=Path("checkpoints/DAMSMencoders/bird/text_encoder200.pth"),
    )
    parser.add_argument("--cl-generator-checkpoint", type=Path, required=True)
    parser.add_argument("--cl-text-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--eval-text-checkpoint",
        type=Path,
        default=Path("checkpoints/DAMSMencoders/bird/text_encoder200.pth"),
        help="Independent frozen DAMSM text evaluator; do not pass the CL conditioner.",
    )
    parser.add_argument(
        "--eval-image-checkpoint",
        type=Path,
        default=Path("checkpoints/DAMSMencoders/bird/image_encoder200.pth"),
        help="Independent frozen DAMSM image evaluator (original DAMSM by default).",
    )
    parser.add_argument("--fid-stats", type=Path, default=Path("checkpoints/eval/bird_val.npz"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/final/contrastive_checkpoint_evaluation"),
    )
    parser.add_argument("--samples", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--caption-bank-batch-size", type=int, default=256)
    parser.add_argument("--preview-count", type=int, default=16, help="Number of image pairs.")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--skip-is", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 10:
        raise SystemExit("--samples must be at least 10")
    if args.batch_size < 1 or args.caption_bank_batch_size < 1:
        raise SystemExit("batch sizes must be positive")
    if args.preview_count < 0:
        raise SystemExit("--preview-count cannot be negative")
    if args.bootstrap_resamples < 100:
        raise SystemExit("--bootstrap-resamples must be at least 100")

    checkpoint_paths = {
        "baseline_generator": args.baseline_generator_checkpoint,
        "baseline_conditioning_text_encoder": args.baseline_text_checkpoint,
        "contrastive_generator": args.cl_generator_checkpoint,
        "contrastive_conditioning_text_encoder": args.cl_text_checkpoint,
        "evaluation_text_encoder": args.eval_text_checkpoint,
        "evaluation_image_encoder": args.eval_image_checkpoint,
    }
    checkpoints = {name: checkpoint_record(path) for name, path in checkpoint_paths.items()}
    checkpoints["fid_real_statistics"] = checkpoint_record(args.fid_stats, required=not args.skip_fid)
    if checkpoints["evaluation_text_encoder"]["sha256"] == checkpoints[
        "contrastive_conditioning_text_encoder"
    ]["sha256"]:
        raise SystemExit(
            "The evaluation text checkpoint is identical to the CL conditioning checkpoint. "
            "Use the original DAMSM via --eval-text-checkpoint to avoid self-evaluation."
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    bank = load_caption_bank(args.metadata_root, words_num=18, seed=args.seed)
    captions = bank["captions"]
    lengths = bank["lengths"]
    classes = bank["classes"]
    vocabulary_size = int(bank["vocabulary_size"])
    caption_indices = np.arange(args.samples, dtype=np.int64) % captions.shape[0]
    negative_indices = sample_negative_indices(
        classes,
        args.samples,
        negatives=99,
        seed=args.seed + 1,
    )

    baseline_text = DAMSMTextEncoder(vocabulary_size)
    load_frozen_checkpoint(baseline_text, args.baseline_text_checkpoint)
    baseline_text.to(device)
    contrastive_text = DAMSMTextEncoder(vocabulary_size)
    load_frozen_checkpoint(contrastive_text, args.cl_text_checkpoint)
    contrastive_text.to(device)

    # These are separate module instances even when the baseline conditioner and
    # evaluator intentionally point to the same original checkpoint file.
    evaluation_text = DAMSMTextEncoder(vocabulary_size)
    load_frozen_checkpoint(evaluation_text, args.eval_text_checkpoint)
    evaluation_text.to(device)
    evaluation_image = DAMSMImageEncoder()
    load_frozen_checkpoint(evaluation_image, args.eval_image_checkpoint)
    evaluation_image.to(device)

    baseline_generator = DMGenerator(channels=64, memory_dim=128, residual_blocks=2)
    load_official_generator_checkpoint(
        baseline_generator,
        args.baseline_generator_checkpoint,
        strict=True,
    )
    baseline_generator.to(device).eval()
    contrastive_generator = DMGenerator(channels=64, memory_dim=128, residual_blocks=2)
    load_official_generator_checkpoint(
        contrastive_generator,
        args.cl_generator_checkpoint,
        strict=True,
    )
    contrastive_generator.to(device).eval()

    inception = None
    if not args.skip_fid or not args.skip_is:
        inception = InceptionEvaluator().to(device).eval()

    print(
        f"Encoding {captions.shape[0]:,} captions with the independent evaluation text encoder...",
        flush=True,
    )
    sentence_bank = encode_sentence_bank(
        evaluation_text,
        captions,
        lengths,
        args.caption_bank_batch_size,
        device,
    )

    baseline_fid = (
        None if args.skip_fid else np.empty((args.samples, 2048), dtype=np.float32)
    )
    contrastive_fid = (
        None if args.skip_fid else np.empty((args.samples, 2048), dtype=np.float32)
    )
    baseline_is = None if args.skip_is else np.empty((args.samples, 1000), dtype=np.float32)
    contrastive_is = None if args.skip_is else np.empty((args.samples, 1000), dtype=np.float32)
    baseline_r = np.empty(args.samples, dtype=np.bool_)
    contrastive_r = np.empty(args.samples, dtype=np.bool_)
    preview_pairs: list[torch.Tensor] = []
    preview_manifest: list[dict[str, int | str]] = []

    latent_seed = args.seed + 4
    conditioning_seed = args.seed + 5
    latent_generator = torch.Generator(device=device)
    latent_generator.manual_seed(latent_seed)
    reset_generation_rng(conditioning_seed, device)
    started = time.monotonic()

    with torch.inference_mode():
        for start in range(0, args.samples, args.batch_size):
            stop = min(start + args.batch_size, args.samples)
            local_caption_indices = caption_indices[start:stop]
            caption_tensor, length_tensor, order = sorted_caption_batch(
                captions,
                lengths,
                local_caption_indices,
                device,
            )
            word_mask = build_word_mask(length_tensor, caption_tensor.size(1))
            baseline_words, baseline_sentence = baseline_text(caption_tensor, length_tensor)
            contrastive_words, contrastive_sentence = contrastive_text(caption_tensor, length_tensor)
            _, evaluation_sentence = evaluation_text(caption_tensor, length_tensor)
            noise = torch.randn(
                stop - start,
                100,
                device=device,
                generator=latent_generator,
            )
            baseline_image, contrastive_image = paired_generator_forward(
                baseline_generator,
                contrastive_generator,
                noise,
                baseline_sentence,
                baseline_words,
                contrastive_sentence,
                contrastive_words,
                word_mask,
                device,
            )
            baseline_01 = ((baseline_image + 1.0) / 2.0).clamp(0.0, 1.0)
            contrastive_01 = ((contrastive_image + 1.0) / 2.0).clamp(0.0, 1.0)

            inverse = torch.as_tensor(np.argsort(order), device=device)
            remaining_previews = args.preview_count - len(preview_manifest)
            if remaining_previews > 0:
                count = min(remaining_previews, stop - start)
                baseline_preview = baseline_01[inverse][:count].cpu()
                contrastive_preview = contrastive_01[inverse][:count].cpu()
                for local_index in range(count):
                    sample_index = start + local_index
                    caption_index = int(caption_indices[sample_index])
                    image_index = caption_index // int(bank["captions_per_image"])
                    preview_pairs.extend(
                        [baseline_preview[local_index], contrastive_preview[local_index]]
                    )
                    preview_manifest.append(
                        {
                            "row": len(preview_manifest),
                            "sample_index": sample_index,
                            "caption_index": caption_index,
                            "image_key": str(bank["keys"][image_index]),
                            "caption_slot": caption_index % int(bank["captions_per_image"]),
                            "left": "DM-GAN baseline",
                            "right": "DM-GAN+CL",
                        }
                    )

            if baseline_fid is not None and contrastive_fid is not None:
                assert inception is not None
                baseline_fid[start:stop] = inception.fid_features(baseline_01).cpu().numpy()
                contrastive_fid[start:stop] = inception.fid_features(contrastive_01).cpu().numpy()
            if baseline_is is not None and contrastive_is is not None:
                assert inception is not None
                baseline_is[start:stop] = inception.imagenet_probabilities(baseline_01).cpu().numpy()
                contrastive_is[start:stop] = inception.imagenet_probabilities(
                    contrastive_01
                ).cpu().numpy()

            _, baseline_image_code = evaluation_image(baseline_image)
            _, contrastive_image_code = evaluation_image(contrastive_image)
            sorted_negative_indices = negative_indices[start:stop][order]
            negative = sentence_bank[torch.from_numpy(sorted_negative_indices)].to(device)
            baseline_matches = r_precision(
                baseline_image_code,
                evaluation_sentence,
                negative,
            )
            contrastive_matches = r_precision(
                contrastive_image_code,
                evaluation_sentence,
                negative,
            )
            baseline_r[start:stop] = baseline_matches[inverse].cpu().numpy()
            contrastive_r[start:stop] = contrastive_matches[inverse].cpu().numpy()

            if stop == args.samples or stop % (args.batch_size * 25) == 0:
                elapsed = time.monotonic() - started
                rate = stop / max(elapsed, 1e-6)
                print(
                    f"Processed {stop:,}/{args.samples:,} paired samples ({rate:.1f} pairs/s)",
                    flush=True,
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = args.output_dir / "paired_preview_256.png"
    if preview_pairs:
        save_image(torch.stack(preview_pairs), preview_path, nrow=2)
    (args.output_dir / "paired_preview_manifest.json").write_text(
        json.dumps(preview_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output_dir / "paired_samples.npz",
        sample_index=np.arange(args.samples, dtype=np.int64),
        caption_index=caption_indices,
        image_cluster_id=caption_indices // int(bank["captions_per_image"]),
        caption_length=lengths[caption_indices],
        class_id=classes[caption_indices],
        baseline_r_match=baseline_r,
        contrastive_r_match=contrastive_r,
        r_match_difference=contrastive_r.astype(np.int8) - baseline_r.astype(np.int8),
    )

    baseline_fid_value: float | None = None
    contrastive_fid_value: float | None = None
    if baseline_fid is not None and contrastive_fid is not None:
        with np.load(args.fid_stats) as reference:
            reference_mean = reference["mu"]
            reference_covariance = reference["sigma"]
        print("Computing FID for both generators against the same real statistics...", flush=True)
        baseline_fid_value = fid_from_features(
            baseline_fid,
            reference_mean,
            reference_covariance,
        )
        contrastive_fid_value = fid_from_features(
            contrastive_fid,
            reference_mean,
            reference_covariance,
        )

    baseline_is_mean: float | None = None
    baseline_is_std: float | None = None
    contrastive_is_mean: float | None = None
    contrastive_is_std: float | None = None
    if baseline_is is not None and contrastive_is is not None:
        baseline_is_mean, baseline_is_std = inception_score(
            baseline_is,
            splits=10,
            seed=args.seed + 2,
        )
        contrastive_is_mean, contrastive_is_std = inception_score(
            contrastive_is,
            splits=10,
            seed=args.seed + 2,
        )

    baseline_r_summary = r_precision_summary(baseline_r, seed=args.seed + 3)
    contrastive_r_summary = r_precision_summary(contrastive_r, seed=args.seed + 3)
    paired_r = paired_r_statistics(
        baseline_r,
        contrastive_r,
        caption_indices // int(bank["captions_per_image"]),
        seed=args.seed + 6,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    fid_delta = (
        None
        if baseline_fid_value is None or contrastive_fid_value is None
        else contrastive_fid_value - baseline_fid_value
    )
    is_delta = (
        None
        if baseline_is_mean is None or contrastive_is_mean is None
        else contrastive_is_mean - baseline_is_mean
    )
    r_delta = (
        contrastive_r_summary["overall_percent"] - baseline_r_summary["overall_percent"]
    )

    r_noninferiority_margin = -1.0
    cluster_ci_lower = paired_r["cluster_bootstrap_delta_percentage_points_95_ci"][0]
    if args.samples < 30000:
        verdict = "smoke only"
        summary = "Fewer than 30,000 samples were requested; metrics validate plumbing, not quality."
    elif fid_delta is None:
        verdict = "incomplete"
        summary = "FID was skipped, so the primary image-quality claim cannot be tested."
    elif fid_delta < 0.0 and cluster_ci_lower > r_noninferiority_margin:
        verdict = "supported"
        summary = (
            "DM-GAN+CL lowers the FID point estimate and the image-cluster confidence interval supports "
            "R-precision non-inferiority within 1 percentage point."
        )
    elif fid_delta < 0.0:
        verdict = "mixed"
        summary = (
            "DM-GAN+CL lowers the FID point estimate, but the image-cluster confidence interval does not establish "
            "R-precision non-inferiority within 1 percentage point."
        )
    else:
        verdict = "not supported"
        summary = "DM-GAN+CL does not lower the FID point estimate under this fixed paired protocol."

    limitations = [
        "A single checkpoint pair and one random seed do not measure training-run variance.",
        "FID is a distribution-level statistic; the paired design controls inputs but does not make FID a paired per-image test.",
        "R-precision depends on the chosen frozen original DAMSM evaluator and its 99 sampled negatives.",
        "The sample-level McNemar p-value is descriptive because captions from one image and repeated rows are correlated; the image-cluster bootstrap is the uncertainty result to use.",
        "ImageNet IS is an internal health check and is not comparable with the paper's legacy CUB bird-classifier IS.",
        "Caption indices repeat after the finite official test-caption bank is exhausted.",
        "Checkpoint evaluation tests released models; it does not by itself verify that this repository can reproduce their training.",
    ]
    if args.samples < 30000:
        limitations.insert(
            0,
            "This run uses fewer than the predeclared 30,000 samples and must not support a final quality claim.",
        )

    report: dict[str, Any] = {
        "status": "complete",
        "sample_count": args.samples,
        "seed": args.seed,
        "device": str(device),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "protocol": {
            "split": "official CUB test metadata",
            "resolution": 256,
            "caption_words": 18,
            "generation_batch_size": args.batch_size,
            "caption_bank_batch_size": args.caption_bank_batch_size,
            "bootstrap_resamples": args.bootstrap_resamples,
            "pairing": "same caption index, latent z, CA epsilon, negatives, and evaluators",
            "caption_seed": args.seed,
            "negative_caption_seed": args.seed + 1,
            "inception_split_seed": args.seed + 2,
            "r_precision_split_seed": args.seed + 3,
            "latent_noise_seed": latent_seed,
            "conditioning_augmentation_seed": conditioning_seed,
            "paired_bootstrap_seed": args.seed + 6,
            "r_precision": "independent original DAMSM; correct caption vs 99 other-class captions",
            "fid": "shared repository PyTorch Inception path and shared author bird_val.npz",
            "is": "shared torchvision ImageNet Inception-v3; not paper-comparable",
        },
        "checkpoints": checkpoints,
        "results": {
            "baseline": {
                "fid_pytorch": baseline_fid_value,
                "r_precision": baseline_r_summary,
                "is_imagenet_mean": baseline_is_mean,
                "is_imagenet_std": baseline_is_std,
            },
            "contrastive": {
                "fid_pytorch": contrastive_fid_value,
                "r_precision": contrastive_r_summary,
                "is_imagenet_mean": contrastive_is_mean,
                "is_imagenet_std": contrastive_is_std,
            },
            "delta_contrastive_minus_baseline": {
                "fid_pytorch": fid_delta,
                "r_precision_percentage_points": r_delta,
                "is_imagenet_mean": is_delta,
            },
        },
        "paired_r_precision": paired_r,
        "decision": {
            "verdict": verdict,
            "summary": summary,
            "strict_rule": (
                "30k samples; CL FID < baseline FID and the 95% CUB-image cluster-bootstrap "
                "lower bound for CL-minus-baseline R-precision is > -1 percentage point"
            ),
            "r_precision_noninferiority_margin_percentage_points": r_noninferiority_margin,
            "r_precision_cluster_ci_lower_percentage_points": cluster_ci_lower,
        },
        "artifacts": {
            "paired_preview": str(preview_path.resolve()) if preview_pairs else None,
            "paired_preview_manifest": str(
                (args.output_dir / "paired_preview_manifest.json").resolve()
            ),
            "paired_samples": str((args.output_dir / "paired_samples.npz").resolve()),
        },
        "elapsed_seconds": time.monotonic() - started,
        "limitations": limitations,
    }
    report_path = args.output_dir / "report.json"
    markdown_path = args.output_dir / "CONTRASTIVE_CHECKPOINT_EVALUATION.md"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
