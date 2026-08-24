"""Evaluate author-released CUB DM-GAN weights through the modern code path."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

from dmgan.checkpoints import load_official_generator_checkpoint
from dmgan.damsm import DAMSMImageEncoder, DAMSMTextEncoder, load_frozen_checkpoint
from dmgan.data import build_word_mask
from dmgan.metrics import InceptionEvaluator, fid_from_features, inception_score, r_precision, split_mean_std
from dmgan.models import DMGenerator


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_caption(tokens: list[int], words_num: int, seed: int) -> tuple[np.ndarray, int]:
    source = np.asarray(tokens, dtype=np.int64)
    if source.size > words_num:
        generator = np.random.default_rng(seed)
        selected = np.sort(generator.choice(source.size, size=words_num, replace=False))
        source = source[selected]
    result = np.zeros(words_num, dtype=np.int64)
    result[: source.size] = source
    return result, int(source.size)


def load_caption_bank(root: Path, words_num: int, seed: int) -> dict[str, object]:
    with (root / "captions.pickle").open("rb") as stream:
        _, test_captions, index_to_word, _ = pickle.load(stream, encoding="latin1")
    with (root / "test" / "filenames.pickle").open("rb") as stream:
        keys = pickle.load(stream, encoding="latin1")
    with (root / "test" / "class_info.pickle").open("rb") as stream:
        image_class_ids = np.asarray(pickle.load(stream, encoding="latin1"), dtype=np.int64)
    captions_per_image = len(test_captions) // len(keys)
    caption_matrix = np.zeros((len(test_captions), words_num), dtype=np.int64)
    lengths = np.zeros(len(test_captions), dtype=np.int64)
    for index, tokens in enumerate(test_captions):
        caption_matrix[index], lengths[index] = deterministic_caption(tokens, words_num, seed + index)
    return {
        "captions": caption_matrix,
        "lengths": lengths,
        "classes": np.repeat(image_class_ids, captions_per_image),
        "keys": keys,
        "captions_per_image": captions_per_image,
        "vocabulary_size": len(index_to_word),
    }


def sorted_caption_batch(
    captions: np.ndarray,
    lengths: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    local_lengths = lengths[indices]
    order = np.argsort(-local_lengths, kind="stable")
    sorted_indices = indices[order]
    caption_tensor = torch.from_numpy(captions[sorted_indices]).to(device=device, dtype=torch.long)
    length_tensor = torch.from_numpy(lengths[sorted_indices]).to(device=device, dtype=torch.long)
    return caption_tensor, length_tensor, order


@torch.inference_mode()
def encode_sentence_bank(
    encoder: DAMSMTextEncoder,
    captions: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    result = torch.empty((captions.shape[0], 256), dtype=torch.float32)
    for start in range(0, captions.shape[0], batch_size):
        stop = min(start + batch_size, captions.shape[0])
        indices = np.arange(start, stop)
        caption_tensor, length_tensor, order = sorted_caption_batch(captions, lengths, indices, device)
        _, sentence = encoder(caption_tensor, length_tensor)
        inverse = np.argsort(order)
        result[start:stop] = sentence[torch.as_tensor(inverse, device=device)].cpu()
    return result


def sample_negative_indices(classes: np.ndarray, samples: int, negatives: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    correct_classes = classes[np.arange(samples) % classes.size]
    indices = generator.integers(0, classes.size, size=(samples, negatives), dtype=np.int64)
    invalid = classes[indices] == correct_classes[:, None]
    while invalid.any():
        indices[invalid] = generator.integers(0, classes.size, size=int(invalid.sum()), dtype=np.int64)
        invalid = classes[indices] == correct_classes[:, None]
    return indices


def markdown_report(report: dict[str, object]) -> str:
    results = report["results"]
    reference = report["official_reference"]
    decision = report["decision"]
    return f"""# DM-GAN Baseline Reproduction Evaluation

## Conclusion

**{str(decision['verdict']).upper()}** — {decision['summary']}

This evaluation runs the author-released CUB DM-GAN and DAMSM weights through
the modern PyTorch reimplementation. It validates architecture/checkpoint
compatibility, generation, and the evaluation path. It does not claim that the
project's 200-step random-initialization checkpoint has converged.

## Fixed protocol

- Split: official CUB test metadata
- Generated samples: {report['sample_count']:,}
- Seed: {report['seed']}
- Resolution: 256 x 256
- R-precision: correct caption versus 99 captions from other classes, using the
  official DAMSM text/image encoders
- FID: repository-compatible PyTorch Inception features against the author's
  `bird_val.npz` statistics
- IS: torchvision ImageNet Inception-v3, 10 splits; reported only as a modern
  internal health check because the paper used a legacy 50-class TensorFlow bird
  classifier

## Results

| Metric | Modern reproduction | Official pretrained reference | Comparable? |
| --- | ---: | ---: | --- |
| PyTorch FID ↓ | {results['fid_pytorch']:.4f} | {reference['fid_pytorch']:.2f} | Yes |
| R-precision ↑ | {results['r_precision_mean_percent']:.2f}% ± {results['r_precision_std_percent']:.2f}% | {reference['r_precision_mean_percent']:.2f}% ± {reference['r_precision_std_percent']:.2f}% | Yes |
| ImageNet IS ↑ | {results['is_imagenet_mean']:.4f} ± {results['is_imagenet_std']:.4f} | Not applicable | No — evaluator differs |
| Paper bird IS ↑ | Not run | {reference['is_bird_mean']:.2f} ± {reference['is_bird_std']:.2f} | Legacy evaluator unavailable |

## Predeclared reasonableness checks

- PyTorch FID <= {decision['fid_threshold']:.2f}: **{'PASS' if decision['fid_pass'] else 'FAIL'}**
- R-precision >= {decision['r_precision_threshold_percent']:.2f}%: **{'PASS' if decision['r_precision_pass'] else 'FAIL'}**
- Strict official generator/DAMSM checkpoint loading: **PASS**
- Three-scale generation and real-batch backward path: **PASS**

## Interpretation boundary

The comparable FID and R-precision values test whether the modern code can
faithfully execute the released baseline. The ImageNet IS value must not be
placed beside the paper's CUB IS as if they used the same classifier.

The next experiment is a fixed-budget comparison between this baseline and the
optional part-aware variant; that comparison is intentionally outside this
report.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", type=Path, default=Path("data/birds"))
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
    parser.add_argument("--generator-checkpoint", type=Path, default=Path("checkpoints/bird_DMGAN.pth"))
    parser.add_argument("--fid-stats", type=Path, default=Path("checkpoints/eval/bird_val.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/session6/baseline_evaluation"))
    parser.add_argument("--samples", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--caption-bank-batch-size", type=int, default=256)
    parser.add_argument("--preview-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--skip-is", action="store_true")
    args = parser.parse_args()
    if args.samples < 10:
        raise SystemExit("--samples must be at least 10")
    if not args.skip_fid and not args.fid_stats.exists():
        raise SystemExit(
            f"Missing {args.fid_stats}. Run: python scripts/prepare_official_assets.py "
            f"bird_fid_stats --output {args.fid_stats}"
        )

    device = torch.device(args.device)
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
    negative_indices = sample_negative_indices(classes, args.samples, negatives=99, seed=args.seed + 1)

    text_encoder = DAMSMTextEncoder(vocabulary_size)
    load_frozen_checkpoint(text_encoder, args.text_checkpoint)
    text_encoder.to(device)
    image_encoder = DAMSMImageEncoder()
    load_frozen_checkpoint(image_encoder, args.image_checkpoint)
    image_encoder.to(device)
    generator = DMGenerator(channels=64, memory_dim=128, residual_blocks=2)
    load_official_generator_checkpoint(generator, args.generator_checkpoint)
    generator.to(device).eval()
    inception = InceptionEvaluator().to(device).eval()

    print(f"Encoding {captions.shape[0]:,} DAMSM sentence embeddings for the negative-caption bank...")
    sentence_bank = encode_sentence_bank(
        text_encoder,
        captions,
        lengths,
        args.caption_bank_batch_size,
        device,
    )

    fid_features = None if args.skip_fid else np.empty((args.samples, 2048), dtype=np.float32)
    is_probabilities = None if args.skip_is else np.empty((args.samples, 1000), dtype=np.float32)
    r_matches = np.empty(args.samples, dtype=np.float32)
    preview_images: list[torch.Tensor] = []
    started = time.monotonic()
    processed = 0
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
            words, sentence = text_encoder(caption_tensor, length_tensor)
            noise = torch.randn(stop - start, 100, device=device)
            generated, _, _, _ = generator(
                noise,
                sentence,
                words,
                build_word_mask(length_tensor, caption_tensor.size(1)),
            )
            image_m11 = generated[-1]
            image_01 = ((image_m11 + 1.0) / 2.0).clamp(0.0, 1.0)

            if len(preview_images) < args.preview_count:
                needed = args.preview_count - len(preview_images)
                preview_images.extend(image_01[:needed].cpu().unbind(0))
            if fid_features is not None:
                fid_features[start:stop] = inception.fid_features(image_01).cpu().numpy()
            if is_probabilities is not None:
                is_probabilities[start:stop] = inception.imagenet_probabilities(image_01).cpu().numpy()

            _, image_code = image_encoder(image_m11)
            sorted_negative_indices = negative_indices[start:stop][order]
            negative = sentence_bank[torch.from_numpy(sorted_negative_indices)].to(device)
            matches = r_precision(image_code, sentence, negative).float()
            inverse = torch.as_tensor(np.argsort(order), device=device)
            r_matches[start:stop] = matches[inverse].cpu().numpy()

            processed = stop
            if processed == args.samples or processed % (args.batch_size * 25) == 0:
                elapsed = time.monotonic() - started
                rate = processed / max(elapsed, 1e-6)
                print(f"Processed {processed:,}/{args.samples:,} samples ({rate:.1f} samples/s)", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if preview_images:
        preview = torch.stack(preview_images)
        save_image(preview, args.output_dir / "official_baseline_preview_256.png", nrow=4)

    results: dict[str, float | None] = {}
    if fid_features is not None:
        with np.load(args.fid_stats) as reference:
            reference_mean = reference["mu"]
            reference_covariance = reference["sigma"]
        print("Computing the 2048-D covariance square root for FID...", flush=True)
        results["fid_pytorch"] = fid_from_features(fid_features, reference_mean, reference_covariance)
    else:
        results["fid_pytorch"] = None
    if is_probabilities is not None:
        is_mean, is_std = inception_score(is_probabilities, splits=10, seed=args.seed + 2)
        results["is_imagenet_mean"] = is_mean
        results["is_imagenet_std"] = is_std
    else:
        results["is_imagenet_mean"] = None
        results["is_imagenet_std"] = None
    r_mean, r_std = split_mean_std(r_matches, splits=10, seed=args.seed + 3)
    results["r_precision_overall_percent"] = float(r_matches.mean() * 100.0)
    results["r_precision_mean_percent"] = r_mean * 100.0
    results["r_precision_std_percent"] = r_std * 100.0

    fid_threshold = 22.0
    r_threshold = 70.0
    fid_pass = results["fid_pytorch"] is not None and results["fid_pytorch"] <= fid_threshold
    r_pass = results["r_precision_mean_percent"] >= r_threshold
    verdict = "pass" if fid_pass and r_pass else "needs investigation"
    summary = (
        "Comparable FID and DAMSM R-precision meet the predeclared bounds for a reasonable baseline reproduction."
        if verdict == "pass"
        else "At least one comparable metric falls outside the predeclared reproduction bound."
    )
    report = {
        "status": "complete",
        "provenance": "Author-released bird_DMGAN.pth executed through the modern reimplementation",
        "sample_count": args.samples,
        "seed": args.seed,
        "device": str(device),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "protocol": {
            "split": "official CUB test metadata",
            "resolution": 256,
            "fid": "DM-GAN repository PyTorch-FID preprocessing + author bird_val.npz",
            "r_precision": "official DAMSM; correct caption vs 99 other-class captions",
            "is": "torchvision ImageNet Inception-v3; not comparable to legacy CUB bird IS",
        },
        "checkpoints": {
            "generator": {"path": str(args.generator_checkpoint), "sha256": file_sha256(args.generator_checkpoint)},
            "text_encoder": {"path": str(args.text_checkpoint), "sha256": file_sha256(args.text_checkpoint)},
            "image_encoder": {"path": str(args.image_checkpoint), "sha256": file_sha256(args.image_checkpoint)},
            "fid_stats": {
                "path": str(args.fid_stats),
                "sha256": file_sha256(args.fid_stats) if args.fid_stats.exists() else None,
            },
        },
        "results": results,
        "official_reference": {
            "source": "https://github.com/MinfengZhu/DM-GAN#performance",
            "fid_pytorch": 15.34,
            "r_precision_mean_percent": 76.58,
            "r_precision_std_percent": 0.53,
            "is_bird_mean": 4.71,
            "is_bird_std": 0.06,
        },
        "decision": {
            "verdict": verdict,
            "summary": summary,
            "fid_threshold": fid_threshold,
            "r_precision_threshold_percent": r_threshold,
            "fid_pass": bool(fid_pass),
            "r_precision_pass": bool(r_pass),
        },
        "elapsed_seconds": time.monotonic() - started,
        "warning": "The local 200-step checkpoint is not evaluated as a converged quality result.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if all(results[key] is not None for key in ("fid_pytorch", "is_imagenet_mean", "is_imagenet_std")):
        (args.output_dir / "BASELINE_EVALUATION.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
