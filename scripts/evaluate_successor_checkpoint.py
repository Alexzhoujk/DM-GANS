"""Paired evaluation of official DM-GAN and a compatible successor checkpoint.

The two generators receive the same caption indices and latent noise, plus the
same conditioning-augmentation draws when both modes sample. R-precision is
measured with a separately loaded, frozen DAMSM evaluator rather than reusing
either live conditioning module. A candidate may have identical original-DAMSM
weights when that is part of the published method, as in DM-GAN-MDD.
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
except ImportError:  # Direct execution: python scripts/evaluate_successor_checkpoint.py
    from evaluate_baseline import (
        encode_sentence_bank,
        file_sha256,
        load_caption_bank,
        sample_negative_indices,
        sorted_caption_batch,
    )

from dmgan.checkpoints import load_generator_checkpoint
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
    candidate: DMGenerator,
    noise: torch.Tensor,
    baseline_sentence: torch.Tensor,
    baseline_words: torch.Tensor,
    candidate_sentence: torch.Tensor,
    candidate_words: torch.Tensor,
    word_mask: torch.Tensor,
    device: torch.device,
    baseline_sample_conditioning: bool,
    candidate_sample_conditioning: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a pair and share CA epsilon when both sides sample it.

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
        sample_conditioning=baseline_sample_conditioning,
    )
    # Keep the post-baseline state.  When only the baseline samples CA noise,
    # restoring the pre-forward state for a deterministic candidate would
    # otherwise make the baseline repeat the same epsilon in every batch.
    baseline_cpu_state = torch.random.get_rng_state()
    baseline_cuda_state = (
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    )
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)
    candidate_images, _, _, _ = candidate(
        noise,
        candidate_sentence,
        candidate_words,
        word_mask,
        sample_conditioning=candidate_sample_conditioning,
    )
    if baseline_sample_conditioning and not candidate_sample_conditioning:
        torch.random.set_rng_state(baseline_cpu_state)
        if baseline_cuda_state is not None:
            torch.cuda.set_rng_state(baseline_cuda_state, device)
    return baseline_images[-1], candidate_images[-1]


def conditioning_pairing_description(
    baseline_mode: str,
    candidate_mode: str,
) -> str:
    """Describe exactly which CA randomness is controlled in a comparison."""

    if baseline_mode == "sample" and candidate_mode == "sample":
        return "same caption index, latent z, CA epsilon, negatives, and evaluators"
    if baseline_mode == "mean" and candidate_mode == "mean":
        return "same caption index, latent z, deterministic CA means, negatives, and evaluators"
    return (
        "same caption index, latent z, negatives, and evaluators; the sample-mode side uses "
        "the recorded deterministic CA RNG stream while the mean-mode side uses no CA epsilon"
    )


def r_precision_summary(matches: np.ndarray, seed: int) -> dict[str, float]:
    mean, std = split_mean_std(matches, splits=10, seed=seed)
    return {
        "overall_percent": float(matches.mean() * 100.0),
        "split_mean_percent": mean * 100.0,
        "split_std_percent": std * 100.0,
    }


def paired_r_statistics(
    baseline_matches: np.ndarray,
    candidate_matches: np.ndarray,
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
    candidate = np.asarray(candidate_matches, dtype=np.int8)
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError("paired R-precision arrays must have the same one-dimensional shape")
    cluster_ids = np.asarray(cluster_ids, dtype=np.int64)
    if cluster_ids.shape != baseline.shape:
        raise ValueError("cluster_ids must have the same shape as paired R-precision arrays")

    both_correct = int(np.count_nonzero((baseline == 1) & (candidate == 1)))
    baseline_only = int(np.count_nonzero((baseline == 1) & (candidate == 0)))
    candidate_only = int(np.count_nonzero((baseline == 0) & (candidate == 1)))
    both_wrong = int(np.count_nonzero((baseline == 0) & (candidate == 0)))
    discordant = baseline_only + candidate_only
    p_value = (
        float(binomtest(candidate_only, discordant, p=0.5, alternative="two-sided").pvalue)
        if discordant
        else 1.0
    )

    difference = candidate - baseline
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
        "candidate_only_correct": candidate_only,
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
    candidate = report["results"]["candidate"]
    delta = report["results"]["delta_candidate_minus_baseline"]
    paired = report["paired_r_precision"]
    decision = report["decision"]
    baseline_name = report["baseline_name"]
    candidate_name = report["candidate_name"]
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    return f"""# Paired {baseline_name} vs {candidate_name} Checkpoint Evaluation

## Conclusion

**{decision["verdict"].upper()}** — {decision["summary"]}

This is a paired comparison: {report["protocol"]["pairing"]}. Each generator
uses the conditioning text encoder specified by its published method. Both
outputs are judged by the same frozen original DAMSM evaluator.

## Fixed protocol

- Samples: {report["sample_count"]:,}
- Seed: {report["seed"]}
- Resolution: 256 x 256
- Caption schedule: deterministic cycling over the official CUB test caption bank
- FID: one shared PyTorch Inception evaluator and one shared real-statistics file
- R-precision: one shared original DAMSM evaluator; correct caption versus the
  same 99 other-class negatives for both generators
- Preview layout: baseline on the left, {candidate_name} on the right

## Results

| Metric | {baseline_name} | {candidate_name} | candidate - baseline |
| --- | ---: | ---: | ---: |
| PyTorch FID ↓ | {metric_text(baseline["fid_pytorch"])} | {metric_text(candidate["fid_pytorch"])} | {metric_text(delta["fid_pytorch"])} |
| R-precision ↑ | {baseline["r_precision"]["split_mean_percent"]:.2f}% ± {baseline["r_precision"]["split_std_percent"]:.2f}% | {candidate["r_precision"]["split_mean_percent"]:.2f}% ± {candidate["r_precision"]["split_std_percent"]:.2f}% | {delta["r_precision_percentage_points"]:+.2f} pp |
| ImageNet IS ↑ | {metric_text(baseline["is_imagenet_mean"])} | {metric_text(candidate["is_imagenet_mean"])} | {metric_text(delta["is_imagenet_mean"])} |

## Paired R-precision evidence

- Baseline-only correct: {paired["baseline_only_correct"]:,}
- {candidate_name}-only correct: {paired["candidate_only_correct"]:,}
- Exact McNemar p-value: {paired["mcnemar_exact_two_sided_p"]:.6g}
- McNemar caveat: {paired["mcnemar_note"]}
- CUB-image cluster bootstrap ({paired["bootstrap_cluster_count"]:,} image clusters)
  95% CI for the R-precision delta:
  [{paired["cluster_bootstrap_delta_percentage_points_95_ci"][0]:+.3f},
  {paired["cluster_bootstrap_delta_percentage_points_95_ci"][1]:+.3f}] percentage points

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
        description="Fair paired evaluation of official DM-GAN and a compatible successor."
    )
    parser.add_argument("--baseline-name", default="DM-GAN")
    parser.add_argument("--candidate-name", default="DM-GAN successor")
    parser.add_argument("--metadata-root", type=Path, default=Path("data/birds"))
    parser.add_argument(
        "--baseline-generator-checkpoint",
        type=Path,
        default=Path("checkpoints/bird_DMGAN.pth"),
    )
    parser.add_argument(
        "--baseline-generator-format",
        choices=("auto", "official", "modern-raw", "modern-ema"),
        default="auto",
        help=(
            "Checkpoint representation. auto selects EMA for a nested modern trainer "
            "checkpoint and official format for an author checkpoint."
        ),
    )
    parser.add_argument(
        "--baseline-text-checkpoint",
        type=Path,
        default=Path("checkpoints/DAMSMencoders/bird/text_encoder200.pth"),
    )
    parser.add_argument(
        "--candidate-generator-checkpoint",
        "--cl-generator-checkpoint",
        dest="candidate_generator_checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--candidate-generator-format",
        choices=("auto", "official", "modern-raw", "modern-ema"),
        default="auto",
        help=(
            "Checkpoint representation. auto selects EMA for a nested modern trainer "
            "checkpoint and official format for an author checkpoint."
        ),
    )
    parser.add_argument(
        "--candidate-text-checkpoint",
        "--cl-text-checkpoint",
        dest="candidate_text_checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--eval-text-checkpoint",
        type=Path,
        default=Path("checkpoints/DAMSMencoders/bird/text_encoder200.pth"),
        help="Separately loaded frozen DAMSM text evaluator; do not pass the CL conditioner.",
    )
    parser.add_argument(
        "--eval-image-checkpoint",
        type=Path,
        default=Path("checkpoints/DAMSMencoders/bird/image_encoder200.pth"),
        help="Frozen DAMSM image evaluator (original DAMSM by default).",
    )
    parser.add_argument("--fid-stats", type=Path, default=Path("checkpoints/eval/bird_val.npz"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/final/candidate_checkpoint_evaluation"),
    )
    parser.add_argument("--samples", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--caption-bank-batch-size", type=int, default=256)
    parser.add_argument("--preview-count", type=int, default=16, help="Number of image pairs.")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--baseline-conditioning-mode", choices=("sample", "mean"), default="sample")
    parser.add_argument("--candidate-conditioning-mode", choices=("sample", "mean"), default="sample")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--skip-is", action="store_true")
    parser.add_argument(
        "--allow-shared-conditioning-evaluator",
        action="store_true",
        help="Allow the candidate conditioner to be the original DAMSM evaluator.",
    )
    parser.add_argument(
        "--extra-limitation",
        action="append",
        default=[],
        help="Append a method-specific limitation to report.json and the Markdown report.",
    )
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
        "candidate_generator": args.candidate_generator_checkpoint,
        "candidate_conditioning_text_encoder": args.candidate_text_checkpoint,
        "evaluation_text_encoder": args.eval_text_checkpoint,
        "evaluation_image_encoder": args.eval_image_checkpoint,
    }
    checkpoints = {name: checkpoint_record(path) for name, path in checkpoint_paths.items()}
    checkpoints["fid_real_statistics"] = checkpoint_record(args.fid_stats, required=not args.skip_fid)
    shared_candidate_evaluator = (
        checkpoints["evaluation_text_encoder"]["sha256"]
        == checkpoints["candidate_conditioning_text_encoder"]["sha256"]
    )
    if shared_candidate_evaluator and not args.allow_shared_conditioning_evaluator:
        raise SystemExit(
            "The candidate conditioning checkpoint is identical to the evaluation text checkpoint. "
            "Pass --allow-shared-conditioning-evaluator only when the published method uses the "
            "unchanged original DAMSM conditioner."
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
    candidate_text = DAMSMTextEncoder(vocabulary_size)
    load_frozen_checkpoint(candidate_text, args.candidate_text_checkpoint)
    candidate_text.to(device)

    # These are separate module instances even when the baseline conditioner and
    # evaluator intentionally point to the same original checkpoint file.
    evaluation_text = DAMSMTextEncoder(vocabulary_size)
    load_frozen_checkpoint(evaluation_text, args.eval_text_checkpoint)
    evaluation_text.to(device)
    evaluation_image = DAMSMImageEncoder()
    load_frozen_checkpoint(evaluation_image, args.eval_image_checkpoint)
    evaluation_image.to(device)

    baseline_generator, baseline_generator_format = load_generator_checkpoint(
        DMGenerator(channels=64, memory_dim=128, residual_blocks=2),
        args.baseline_generator_checkpoint,
        checkpoint_format=args.baseline_generator_format,
        strict=True,
    )
    baseline_generator.to(device).eval()
    candidate_generator, candidate_generator_format = load_generator_checkpoint(
        DMGenerator(channels=64, memory_dim=128, residual_blocks=2),
        args.candidate_generator_checkpoint,
        checkpoint_format=args.candidate_generator_format,
        strict=True,
    )
    candidate_generator.to(device).eval()
    checkpoints["baseline_generator"]["format"] = baseline_generator_format
    checkpoints["candidate_generator"]["format"] = candidate_generator_format

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

    baseline_fid = None if args.skip_fid else np.empty((args.samples, 2048), dtype=np.float32)
    candidate_fid = None if args.skip_fid else np.empty((args.samples, 2048), dtype=np.float32)
    baseline_is = None if args.skip_is else np.empty((args.samples, 1000), dtype=np.float32)
    candidate_is = None if args.skip_is else np.empty((args.samples, 1000), dtype=np.float32)
    baseline_r = np.empty(args.samples, dtype=np.bool_)
    candidate_r = np.empty(args.samples, dtype=np.bool_)
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
            candidate_words, candidate_sentence = candidate_text(caption_tensor, length_tensor)
            _, evaluation_sentence = evaluation_text(caption_tensor, length_tensor)
            noise = torch.randn(
                stop - start,
                100,
                device=device,
                generator=latent_generator,
            )
            baseline_image, candidate_image = paired_generator_forward(
                baseline_generator,
                candidate_generator,
                noise,
                baseline_sentence,
                baseline_words,
                candidate_sentence,
                candidate_words,
                word_mask,
                device,
                args.baseline_conditioning_mode == "sample",
                args.candidate_conditioning_mode == "sample",
            )
            baseline_01 = ((baseline_image + 1.0) / 2.0).clamp(0.0, 1.0)
            candidate_01 = ((candidate_image + 1.0) / 2.0).clamp(0.0, 1.0)

            inverse = torch.as_tensor(np.argsort(order), device=device)
            remaining_previews = args.preview_count - len(preview_manifest)
            if remaining_previews > 0:
                count = min(remaining_previews, stop - start)
                baseline_preview = baseline_01[inverse][:count].cpu()
                candidate_preview = candidate_01[inverse][:count].cpu()
                for local_index in range(count):
                    sample_index = start + local_index
                    caption_index = int(caption_indices[sample_index])
                    image_index = caption_index // int(bank["captions_per_image"])
                    preview_pairs.extend([baseline_preview[local_index], candidate_preview[local_index]])
                    preview_manifest.append(
                        {
                            "row": len(preview_manifest),
                            "sample_index": sample_index,
                            "caption_index": caption_index,
                            "image_key": str(bank["keys"][image_index]),
                            "caption_slot": caption_index % int(bank["captions_per_image"]),
                            "left": args.baseline_name,
                            "right": args.candidate_name,
                        }
                    )

            if baseline_fid is not None and candidate_fid is not None:
                assert inception is not None
                baseline_fid[start:stop] = inception.fid_features(baseline_01).cpu().numpy()
                candidate_fid[start:stop] = inception.fid_features(candidate_01).cpu().numpy()
            if baseline_is is not None and candidate_is is not None:
                assert inception is not None
                baseline_is[start:stop] = inception.imagenet_probabilities(baseline_01).cpu().numpy()
                candidate_is[start:stop] = inception.imagenet_probabilities(candidate_01).cpu().numpy()

            _, baseline_image_code = evaluation_image(baseline_image)
            _, candidate_image_code = evaluation_image(candidate_image)
            sorted_negative_indices = negative_indices[start:stop][order]
            negative = sentence_bank[torch.from_numpy(sorted_negative_indices)].to(device)
            baseline_matches = r_precision(
                baseline_image_code,
                evaluation_sentence,
                negative,
            )
            candidate_matches = r_precision(
                candidate_image_code,
                evaluation_sentence,
                negative,
            )
            baseline_r[start:stop] = baseline_matches[inverse].cpu().numpy()
            candidate_r[start:stop] = candidate_matches[inverse].cpu().numpy()

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
        candidate_r_match=candidate_r,
        r_match_difference=candidate_r.astype(np.int8) - baseline_r.astype(np.int8),
    )

    baseline_fid_value: float | None = None
    candidate_fid_value: float | None = None
    if baseline_fid is not None and candidate_fid is not None:
        with np.load(args.fid_stats) as reference:
            reference_mean = reference["mu"]
            reference_covariance = reference["sigma"]
        print("Computing FID for both generators against the same real statistics...", flush=True)
        baseline_fid_value = fid_from_features(
            baseline_fid,
            reference_mean,
            reference_covariance,
        )
        candidate_fid_value = fid_from_features(
            candidate_fid,
            reference_mean,
            reference_covariance,
        )

    baseline_is_mean: float | None = None
    baseline_is_std: float | None = None
    candidate_is_mean: float | None = None
    candidate_is_std: float | None = None
    if baseline_is is not None and candidate_is is not None:
        baseline_is_mean, baseline_is_std = inception_score(
            baseline_is,
            splits=10,
            seed=args.seed + 2,
        )
        candidate_is_mean, candidate_is_std = inception_score(
            candidate_is,
            splits=10,
            seed=args.seed + 2,
        )

    baseline_r_summary = r_precision_summary(baseline_r, seed=args.seed + 3)
    candidate_r_summary = r_precision_summary(candidate_r, seed=args.seed + 3)
    paired_r = paired_r_statistics(
        baseline_r,
        candidate_r,
        caption_indices // int(bank["captions_per_image"]),
        seed=args.seed + 6,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    fid_delta = (
        None
        if baseline_fid_value is None or candidate_fid_value is None
        else candidate_fid_value - baseline_fid_value
    )
    is_delta = (
        None
        if baseline_is_mean is None or candidate_is_mean is None
        else candidate_is_mean - baseline_is_mean
    )
    r_delta = candidate_r_summary["overall_percent"] - baseline_r_summary["overall_percent"]

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
            f"{args.candidate_name} lowers the FID point estimate and the image-cluster confidence interval supports "
            "R-precision non-inferiority within 1 percentage point."
        )
    elif fid_delta < 0.0:
        verdict = "mixed"
        summary = (
            f"{args.candidate_name} lowers the FID point estimate, but the image-cluster confidence interval does not establish "
            "R-precision non-inferiority within 1 percentage point."
        )
    else:
        verdict = "not supported"
        summary = (
            f"{args.candidate_name} does not lower the FID point estimate under this fixed paired protocol."
        )

    limitations = [
        "A single checkpoint pair and one random seed do not measure training-run variance.",
        "FID is a distribution-level statistic; the paired design controls inputs but does not make FID a paired per-image test.",
        "R-precision depends on the chosen frozen original DAMSM evaluator and its 99 sampled negatives.",
        "The sample-level McNemar p-value is descriptive because captions from one image and repeated rows are correlated; the image-cluster bootstrap is the uncertainty result to use.",
        "ImageNet IS is an internal health check and is not comparable with the paper's legacy CUB bird-classifier IS.",
        "Caption indices repeat after the finite official test-caption bank is exhausted.",
        "Checkpoint evaluation tests saved model states; it does not independently reproduce the full training trajectory.",
        "Both generators use the modern correctly broadcast padding mask. This is a shared and controlled implementation, but it is not bit-exact with the author repositories' batch>1 repeat-based legacy mask layout.",
    ]
    if shared_candidate_evaluator:
        limitations.append(
            "The candidate conditioner and R-precision text evaluator have identical original DAMSM weights. The evaluator is a separately loaded frozen module and is applied equally to both models, but it is not an encoder-family-independent judge of the candidate."
        )
    if args.baseline_conditioning_mode != args.candidate_conditioning_mode:
        limitations.append(
            "The native-mode comparison changes both checkpoint weights and the CA inference policy; use the matched-mean run for the controlled checkpoint comparison."
        )
    elif args.baseline_conditioning_mode == "mean":
        limitations.append(
            "Matched-mean conditioning removes CA sampling from both models; it is the controlled comparison for a mean-conditioned candidate, not the released DM-GAN baseline's default stochastic inference mode."
        )
    limitations.extend(args.extra_limitation)
    if args.samples < 30000:
        limitations.insert(
            0,
            "This run uses fewer than the predeclared 30,000 samples and must not support a final quality claim.",
        )

    report: dict[str, Any] = {
        "status": "complete",
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
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
            "pairing": conditioning_pairing_description(
                args.baseline_conditioning_mode,
                args.candidate_conditioning_mode,
            ),
            "baseline_conditioning_mode": args.baseline_conditioning_mode,
            "candidate_conditioning_mode": args.candidate_conditioning_mode,
            "baseline_generator_format": baseline_generator_format,
            "candidate_generator_format": candidate_generator_format,
            "caption_seed": args.seed,
            "negative_caption_seed": args.seed + 1,
            "inception_split_seed": args.seed + 2,
            "r_precision_split_seed": args.seed + 3,
            "latent_noise_seed": latent_seed,
            "conditioning_augmentation_seed": conditioning_seed,
            "paired_bootstrap_seed": args.seed + 6,
            "r_precision": (
                "shared frozen original DAMSM scorer; correct caption vs 99 other-class captions"
            ),
            "candidate_conditioner_is_evaluator": shared_candidate_evaluator,
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
            "candidate": {
                "fid_pytorch": candidate_fid_value,
                "r_precision": candidate_r_summary,
                "is_imagenet_mean": candidate_is_mean,
                "is_imagenet_std": candidate_is_std,
            },
            "delta_candidate_minus_baseline": {
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
                f"30k samples; {args.candidate_name} FID < {args.baseline_name} FID and the 95% "
                f"CUB-image cluster-bootstrap lower bound for {args.candidate_name}-minus-"
                f"{args.baseline_name} R-precision is > -1 percentage point"
            ),
            "r_precision_noninferiority_margin_percentage_points": r_noninferiority_margin,
            "r_precision_cluster_ci_lower_percentage_points": cluster_ci_lower,
        },
        "artifacts": {
            "paired_preview": str(preview_path.resolve()) if preview_pairs else None,
            "paired_preview_manifest": str((args.output_dir / "paired_preview_manifest.json").resolve()),
            "paired_samples": str((args.output_dir / "paired_samples.npz").resolve()),
        },
        "elapsed_seconds": time.monotonic() - started,
        "limitations": limitations,
    }
    report_path = args.output_dir / "report.json"
    markdown_path = args.output_dir / "SUCCESSOR_CHECKPOINT_EVALUATION.md"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
