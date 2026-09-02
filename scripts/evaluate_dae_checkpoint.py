"""Paired 30k evaluation of official DM-GAN and DAE-GAN CUB checkpoints."""

from __future__ import annotations

import argparse
import json
import pickle
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
except ImportError:  # Direct execution: python scripts/evaluate_dae_checkpoint.py
    from evaluate_baseline import (
        encode_sentence_bank,
        file_sha256,
        load_caption_bank,
        sample_negative_indices,
        sorted_caption_batch,
    )

from dmgan.checkpoints import load_official_generator_checkpoint
from dmgan.dae import (
    DAEGenerator,
    encode_dae_aspects,
    load_dae_attribute_bank,
    load_dae_generator_checkpoint,
)
from dmgan.damsm import DAMSMImageEncoder, DAMSMTextEncoder, load_frozen_checkpoint
from dmgan.data import build_word_mask
from dmgan.metrics import (
    InceptionEvaluator,
    fid_from_features,
    inception_score,
    r_precision,
    split_mean_std,
)
from dmgan.models import DMGenerator


def checkpoint_record(path: Path, *, required: bool = True) -> dict[str, str | None]:
    if not path.exists():
        if required:
            raise SystemExit(f"Missing required file: {path}")
        return {"path": str(path.resolve()), "sha256": None}
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def reset_generation_rng(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def paired_generator_forward(
    baseline: DMGenerator,
    dae: DAEGenerator,
    noise: torch.Tensor,
    sentence: torch.Tensor,
    words: torch.Tensor,
    aspects: torch.Tensor,
    word_mask: torch.Tensor,
    device: torch.device,
    *,
    legacy_mask_repeat: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Share z and the conditioning-augmentation epsilon across both models."""

    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    baseline_images, _, _, _ = baseline(noise, sentence, words, word_mask)
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)
    dae_images, _, _, _ = dae(
        noise,
        sentence,
        words,
        aspects,
        word_mask,
        legacy_mask_repeat=legacy_mask_repeat,
    )
    return baseline_images[-1], dae_images[-1]


def r_precision_summary(matches: np.ndarray, seed: int) -> dict[str, float]:
    mean, std = split_mean_std(matches, splits=10, seed=seed)
    return {
        "overall_percent": float(matches.mean() * 100.0),
        "split_mean_percent": mean * 100.0,
        "split_std_percent": std * 100.0,
    }


def paired_r_statistics(
    baseline_matches: np.ndarray,
    dae_matches: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, float | int | list[float]]:
    """Paired binary counts plus a CUB-image cluster-bootstrap interval."""

    from scipy.stats import binomtest

    baseline = np.asarray(baseline_matches, dtype=np.int8)
    dae = np.asarray(dae_matches, dtype=np.int8)
    clusters = np.asarray(cluster_ids, dtype=np.int64)
    if baseline.shape != dae.shape or baseline.ndim != 1:
        raise ValueError("paired R-precision arrays must be matching one-dimensional arrays")
    if clusters.shape != baseline.shape:
        raise ValueError("cluster_ids must match paired R-precision arrays")

    both_correct = int(np.count_nonzero((baseline == 1) & (dae == 1)))
    baseline_only = int(np.count_nonzero((baseline == 1) & (dae == 0)))
    dae_only = int(np.count_nonzero((baseline == 0) & (dae == 1)))
    both_wrong = int(np.count_nonzero((baseline == 0) & (dae == 0)))
    discordant = baseline_only + dae_only
    p_value = (
        float(binomtest(dae_only, discordant, p=0.5, alternative="two-sided").pvalue)
        if discordant
        else 1.0
    )

    difference = dae - baseline
    _, inverse = np.unique(clusters, return_inverse=True)
    cluster_count = int(inverse.max()) + 1
    cluster_sizes = np.bincount(inverse).astype(np.int64)
    cluster_differences = np.bincount(inverse, weights=difference).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(bootstrap_resamples, dtype=np.float64)
    for start in range(0, bootstrap_resamples, 256):
        stop = min(start + 256, bootstrap_resamples)
        sampled = rng.integers(0, cluster_count, size=(stop - start, cluster_count))
        draws[start:stop] = (
            cluster_differences[sampled].sum(axis=1)
            * 100.0
            / cluster_sizes[sampled].sum(axis=1)
        )
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return {
        "both_correct": both_correct,
        "baseline_only_correct": baseline_only,
        "dae_only_correct": dae_only,
        "both_wrong": both_wrong,
        "discordant_pairs": discordant,
        "mcnemar_exact_two_sided_p": p_value,
        "mcnemar_note": (
            "Descriptive only: captions from one CUB image and repeated rows are correlated."
        ),
        "bootstrap_unit": "official CUB test image",
        "bootstrap_cluster_count": cluster_count,
        "bootstrap_resamples": bootstrap_resamples,
        "cluster_bootstrap_delta_percentage_points_95_ci": [float(lower), float(upper)],
    }


def metric_text(value: float | None) -> str:
    return "skipped" if value is None else f"{value:.4f}"


def markdown_report(report: dict[str, Any]) -> str:
    baseline = report["results"]["baseline"]
    dae = report["results"]["dae_gan"]
    delta = report["results"]["delta_dae_minus_baseline"]
    paired = report["paired_r_precision"]
    decision = report["decision"]
    dae_name = report["dae_name"]
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    return f"""# Paired DM-GAN vs {dae_name} Checkpoint Evaluation

## Conclusion

**{decision['verdict'].upper()}** — {decision['summary']}

Both models receive the same deterministic CUB caption schedule, latent noise
and conditioning-augmentation draw. Both are judged by one separately loaded,
original frozen DAMSM evaluator. DAE-GAN additionally receives the official
preprocessed adjective/noun aspect phrases associated with each caption.

## Fixed protocol

- Samples: {report['sample_count']:,}
- Resolution: 256 x 256
- Attention mask: {report['protocol']['dae_attention_mask']}
- FID: shared PyTorch Inception path and shared `bird_val.npz`
- R-precision: correct caption versus the same 99 other-class negatives
- Preview: DM-GAN left, DAE-GAN right

## Results

| Metric | DM-GAN | {dae_name} | DAE - baseline |
| --- | ---: | ---: | ---: |
| PyTorch FID ↓ | {metric_text(baseline['fid_pytorch'])} | {metric_text(dae['fid_pytorch'])} | {metric_text(delta['fid_pytorch'])} |
| R-precision ↑ | {baseline['r_precision']['split_mean_percent']:.2f}% ± {baseline['r_precision']['split_std_percent']:.2f}% | {dae['r_precision']['split_mean_percent']:.2f}% ± {dae['r_precision']['split_std_percent']:.2f}% | {delta['r_precision_percentage_points']:+.2f} pp |
| ImageNet IS ↑ | {metric_text(baseline['is_imagenet_mean'])} | {metric_text(dae['is_imagenet_mean'])} | {metric_text(delta['is_imagenet_mean'])} |

## Paired R-precision evidence

- Baseline-only correct: {paired['baseline_only_correct']:,}
- DAE-GAN-only correct: {paired['dae_only_correct']:,}
- Exact McNemar p-value: {paired['mcnemar_exact_two_sided_p']:.6g}
- CUB-image cluster-bootstrap 95% CI for DAE-minus-baseline:
  [{paired['cluster_bootstrap_delta_percentage_points_95_ci'][0]:+.3f},
  {paired['cluster_bootstrap_delta_percentage_points_95_ci'][1]:+.3f}] percentage points

Per-sample outcomes and caption/image-cluster identifiers are stored in
`paired_samples.npz`. Exact input paths and SHA256 hashes are in `report.json`.

## Limitations

{limitations}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, default=Path("data/birds"))
    parser.add_argument("--dae-metadata", type=Path, required=True)
    parser.add_argument(
        "--baseline-generator-checkpoint",
        type=Path,
        default=Path("checkpoints/bird_DMGAN.pth"),
    )
    parser.add_argument("--dae-generator-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--conditioning-text-checkpoint",
        type=Path,
        default=Path("checkpoints/DAMSMencoders/bird/text_encoder200.pth"),
    )
    parser.add_argument(
        "--eval-text-checkpoint",
        type=Path,
        default=Path("checkpoints/DAMSMencoders/bird/text_encoder200.pth"),
    )
    parser.add_argument(
        "--eval-image-checkpoint",
        type=Path,
        default=Path("checkpoints/DAMSMencoders/bird/image_encoder200.pth"),
    )
    parser.add_argument("--fid-stats", type=Path, default=Path("checkpoints/eval/bird_val.npz"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/final/dae_checkpoint_evaluation"),
    )
    parser.add_argument("--samples", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--caption-bank-batch-size", type=int, default=256)
    parser.add_argument("--preview-count", type=int, default=16)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--legacy-attention-mask",
        action="store_true",
        help="Reproduce the released batch-order-dependent mask.repeat(Q,1) bug.",
    )
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

    paths = {
        "baseline_generator": args.baseline_generator_checkpoint,
        "dae_generator": args.dae_generator_checkpoint,
        "dae_metadata": args.dae_metadata,
        "conditioning_text_encoder": args.conditioning_text_checkpoint,
        "evaluation_text_encoder": args.eval_text_checkpoint,
        "evaluation_image_encoder": args.eval_image_checkpoint,
    }
    records = {name: checkpoint_record(path) for name, path in paths.items()}
    records["fid_real_statistics"] = checkpoint_record(
        args.fid_stats, required=not args.skip_fid
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    bank = load_caption_bank(args.metadata_root, words_num=18, seed=args.seed)
    captions = bank["captions"]
    lengths = bank["lengths"]
    classes = bank["classes"]
    captions_per_image = int(bank["captions_per_image"])
    vocabulary_size = int(bank["vocabulary_size"])
    attributes, dae_caption_payload = load_dae_attribute_bank(args.dae_metadata)
    with (args.metadata_root / "captions.pickle").open("rb") as stream:
        standard_payload = pickle.load(stream, encoding="latin1")
    if tuple(standard_payload) != dae_caption_payload:
        raise SystemExit("DAE metadata caption/vocabulary objects differ from the baseline metadata")
    if attributes.shape[0] != captions.shape[0]:
        raise SystemExit("DAE test-aspect rows do not align with the CUB test-caption bank")

    caption_indices = np.arange(args.samples, dtype=np.int64) % captions.shape[0]
    cluster_ids = caption_indices // captions_per_image
    negative_indices = sample_negative_indices(
        classes, args.samples, negatives=99, seed=args.seed + 1
    )

    conditioner = load_frozen_checkpoint(
        DAMSMTextEncoder(vocabulary_size), args.conditioning_text_checkpoint
    ).to(device)
    evaluation_text = load_frozen_checkpoint(
        DAMSMTextEncoder(vocabulary_size), args.eval_text_checkpoint
    ).to(device)
    evaluation_image = load_frozen_checkpoint(
        DAMSMImageEncoder(), args.eval_image_checkpoint
    ).to(device)
    baseline = DMGenerator(channels=64, memory_dim=128, residual_blocks=2)
    load_official_generator_checkpoint(baseline, args.baseline_generator_checkpoint, strict=True)
    baseline.to(device).eval()
    dae = DAEGenerator(legacy_mask_repeat=args.legacy_attention_mask)
    load_dae_generator_checkpoint(dae, args.dae_generator_checkpoint, strict=True)
    dae.to(device).eval()
    inception = (
        None
        if args.skip_fid and args.skip_is
        else InceptionEvaluator().to(device).eval()
    )

    print(f"Encoding {captions.shape[0]:,} frozen DAMSM evaluation captions...", flush=True)
    sentence_bank = encode_sentence_bank(
        evaluation_text,
        captions,
        lengths,
        args.caption_bank_batch_size,
        device,
    )
    baseline_fid = None if args.skip_fid else np.empty((args.samples, 2048), np.float32)
    dae_fid = None if args.skip_fid else np.empty((args.samples, 2048), np.float32)
    baseline_is = None if args.skip_is else np.empty((args.samples, 1000), np.float32)
    dae_is = None if args.skip_is else np.empty((args.samples, 1000), np.float32)
    baseline_r = np.empty(args.samples, dtype=np.bool_)
    dae_r = np.empty(args.samples, dtype=np.bool_)
    previews: list[torch.Tensor] = []
    preview_manifest: list[dict[str, int | str]] = []

    latent_seed = args.seed + 4
    conditioning_seed = args.seed + 5
    latent_generator = torch.Generator(device=device).manual_seed(latent_seed)
    reset_generation_rng(conditioning_seed, device)
    started = time.monotonic()

    with torch.inference_mode():
        for start in range(0, args.samples, args.batch_size):
            stop = min(start + args.batch_size, args.samples)
            local_indices = caption_indices[start:stop]
            caption_tensor, length_tensor, order = sorted_caption_batch(
                captions, lengths, local_indices, device
            )
            sorted_indices = local_indices[order]
            word_mask = build_word_mask(length_tensor, caption_tensor.size(1))
            words, sentence = conditioner(caption_tensor, length_tensor)
            _, evaluation_sentence = evaluation_text(caption_tensor, length_tensor)
            aspect_tokens = torch.from_numpy(attributes[sorted_indices]).to(
                device=device, dtype=torch.long
            )
            aspect_features = encode_dae_aspects(conditioner, aspect_tokens)
            noise = torch.randn(
                stop - start,
                100,
                device=device,
                generator=latent_generator,
            )
            baseline_image, dae_image = paired_generator_forward(
                baseline,
                dae,
                noise,
                sentence,
                words,
                aspect_features,
                word_mask,
                device,
                legacy_mask_repeat=args.legacy_attention_mask,
            )
            baseline_01 = ((baseline_image + 1.0) / 2.0).clamp(0.0, 1.0)
            dae_01 = ((dae_image + 1.0) / 2.0).clamp(0.0, 1.0)
            inverse = torch.as_tensor(np.argsort(order), device=device)

            remaining = args.preview_count - len(preview_manifest)
            if remaining > 0:
                count = min(remaining, stop - start)
                baseline_preview = baseline_01[inverse][:count].cpu()
                dae_preview = dae_01[inverse][:count].cpu()
                for local_index in range(count):
                    sample_index = start + local_index
                    caption_index = int(caption_indices[sample_index])
                    image_index = caption_index // captions_per_image
                    previews.extend([baseline_preview[local_index], dae_preview[local_index]])
                    preview_manifest.append(
                        {
                            "row": len(preview_manifest),
                            "sample_index": sample_index,
                            "caption_index": caption_index,
                            "image_key": str(bank["keys"][image_index]),
                            "caption_slot": caption_index % captions_per_image,
                            "left": "DM-GAN baseline",
                            "right": "DAE-GAN",
                        }
                    )

            if baseline_fid is not None and dae_fid is not None:
                assert inception is not None
                baseline_fid[start:stop] = inception.fid_features(baseline_01).cpu().numpy()
                dae_fid[start:stop] = inception.fid_features(dae_01).cpu().numpy()
            if baseline_is is not None and dae_is is not None:
                assert inception is not None
                baseline_is[start:stop] = inception.imagenet_probabilities(baseline_01).cpu().numpy()
                dae_is[start:stop] = inception.imagenet_probabilities(dae_01).cpu().numpy()

            _, baseline_code = evaluation_image(baseline_image)
            _, dae_code = evaluation_image(dae_image)
            sorted_negatives = negative_indices[start:stop][order]
            negative = sentence_bank[torch.from_numpy(sorted_negatives)].to(device)
            baseline_matches = r_precision(baseline_code, evaluation_sentence, negative)
            dae_matches = r_precision(dae_code, evaluation_sentence, negative)
            baseline_r[start:stop] = baseline_matches[inverse].cpu().numpy()
            dae_r[start:stop] = dae_matches[inverse].cpu().numpy()

            if stop == args.samples or stop % (args.batch_size * 25) == 0:
                elapsed = time.monotonic() - started
                print(
                    f"Processed {stop:,}/{args.samples:,} pairs "
                    f"({stop / max(elapsed, 1e-6):.1f} pairs/s)",
                    flush=True,
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = args.output_dir / "paired_preview_256.png"
    if previews:
        save_image(torch.stack(previews), preview_path, nrow=2)
    manifest_path = args.output_dir / "paired_preview_manifest.json"
    manifest_path.write_text(json.dumps(preview_manifest, indent=2) + "\n", encoding="utf-8")
    samples_path = args.output_dir / "paired_samples.npz"
    np.savez_compressed(
        samples_path,
        sample_index=np.arange(args.samples, dtype=np.int64),
        caption_index=caption_indices,
        image_cluster_id=cluster_ids,
        caption_length=lengths[caption_indices],
        class_id=classes[caption_indices],
        baseline_r_match=baseline_r,
        dae_r_match=dae_r,
        r_match_difference=dae_r.astype(np.int8) - baseline_r.astype(np.int8),
    )

    baseline_fid_value: float | None = None
    dae_fid_value: float | None = None
    if baseline_fid is not None and dae_fid is not None:
        with np.load(args.fid_stats) as reference:
            mean, covariance = reference["mu"], reference["sigma"]
        print("Computing both FIDs against the same real statistics...", flush=True)
        baseline_fid_value = fid_from_features(baseline_fid, mean, covariance)
        dae_fid_value = fid_from_features(dae_fid, mean, covariance)

    baseline_is_mean = baseline_is_std = dae_is_mean = dae_is_std = None
    if baseline_is is not None and dae_is is not None:
        baseline_is_mean, baseline_is_std = inception_score(
            baseline_is, splits=10, seed=args.seed + 2
        )
        dae_is_mean, dae_is_std = inception_score(dae_is, splits=10, seed=args.seed + 2)

    baseline_summary = r_precision_summary(baseline_r, args.seed + 3)
    dae_summary = r_precision_summary(dae_r, args.seed + 3)
    paired = paired_r_statistics(
        baseline_r,
        dae_r,
        cluster_ids,
        seed=args.seed + 6,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    fid_delta = (
        None
        if baseline_fid_value is None or dae_fid_value is None
        else dae_fid_value - baseline_fid_value
    )
    is_delta = (
        None
        if baseline_is_mean is None or dae_is_mean is None
        else dae_is_mean - baseline_is_mean
    )
    r_delta = dae_summary["overall_percent"] - baseline_summary["overall_percent"]
    ci_lower = paired["cluster_bootstrap_delta_percentage_points_95_ci"][0]
    dae_name = (
        "DAE-GAN (released legacy-mask execution)"
        if args.legacy_attention_mask
        else "DAE-GAN (corrected-mask checkpoint execution)"
    )

    if args.samples < 30000:
        verdict = "smoke only"
        summary = "Fewer than 30,000 samples were requested; this validates plumbing only."
    elif fid_delta is None:
        verdict = "incomplete"
        summary = "FID was skipped, so the primary image-quality claim cannot be tested."
    elif fid_delta < 0 and ci_lower > -1.0:
        verdict = "supported"
        summary = f"{dae_name} lowers the FID point estimate and supports R-precision non-inferiority."
    elif fid_delta < 0:
        verdict = "mixed"
        summary = (
            f"{dae_name} lowers the FID point estimate but R-precision non-inferiority "
            "is not established."
        )
    else:
        verdict = "not supported"
        summary = f"{dae_name} does not lower the FID point estimate under the fixed paired protocol."

    limitations = [
        "DAE-GAN is an independent aspect-aware multi-stage architecture, not a small DM-GAN plugin.",
        "The corrected attention mask implements the apparent batch-major intent but differs from released batch>1 execution; use --legacy-attention-mask only as a sensitivity check.",
        "Aspect preprocessing deterministically keeps the first three phrases and first five tokens, while the released data loader randomly subsamples phrases longer than five tokens; this removes data-loader RNG but is not bit-exact for those phrases.",
        "Checkpoint evaluation does not reproduce 600-epoch training or training-run variance.",
        "FID is distribution-level; pairing controls inputs but does not create a per-image FID test.",
        "R-precision depends on the frozen original DAMSM evaluator and sampled negatives.",
        "ImageNet IS is not comparable with the paper's legacy CUB bird-classifier IS.",
    ]
    if args.samples < 30000:
        limitations.insert(0, "This smoke run cannot support a final quality claim.")

    report: dict[str, Any] = {
        "status": "complete",
        "dae_name": dae_name,
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
            "dae_aspect_selection": (
                "first 3 phrases; deterministic first 5 tokens per phrase; zero padded"
            ),
            "pairing": "same caption index, latent z, CA epsilon, negatives, and evaluators",
            "dae_attention_mask": (
                "released legacy repeat(Q,1)"
                if args.legacy_attention_mask
                else "corrected batch-major broadcast (default)"
            ),
            "latent_noise_seed": latent_seed,
            "conditioning_augmentation_seed": conditioning_seed,
            "negative_caption_seed": args.seed + 1,
            "cluster_bootstrap_seed": args.seed + 6,
        },
        "inputs": records,
        "results": {
            "baseline": {
                "fid_pytorch": baseline_fid_value,
                "r_precision": baseline_summary,
                "is_imagenet_mean": baseline_is_mean,
                "is_imagenet_std": baseline_is_std,
            },
            "dae_gan": {
                "fid_pytorch": dae_fid_value,
                "r_precision": dae_summary,
                "is_imagenet_mean": dae_is_mean,
                "is_imagenet_std": dae_is_std,
            },
            "delta_dae_minus_baseline": {
                "fid_pytorch": fid_delta,
                "r_precision_percentage_points": r_delta,
                "is_imagenet_mean": is_delta,
            },
        },
        "paired_r_precision": paired,
        "decision": {
            "verdict": verdict,
            "summary": summary,
            "strict_rule": (
                "30k samples; DAE FID < baseline FID and the 95% CUB-image "
                "cluster-bootstrap lower bound for DAE-minus-baseline R is > -1 pp"
            ),
            "r_precision_cluster_ci_lower_percentage_points": ci_lower,
        },
        "artifacts": {
            "paired_preview": str(preview_path.resolve()) if previews else None,
            "paired_preview_manifest": str(manifest_path.resolve()),
            "paired_samples": str(samples_path.resolve()),
        },
        "elapsed_seconds": time.monotonic() - started,
        "limitations": limitations,
        "sources": {
            "paper": "https://openaccess.thecvf.com/content/ICCV2021/papers/Ruan_DAE-GAN_Dynamic_Aspect-Aware_GAN_for_Text-to-Image_Synthesis_ICCV_2021_paper.pdf",
            "official_code": "https://github.com/hiarsal/DAE-GAN",
        },
    }
    report_path = args.output_dir / "report.json"
    markdown_path = args.output_dir / "DAE_CHECKPOINT_EVALUATION.md"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
