"""Evaluate paired baseline-control and part-aware DM-GAN checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torchvision.utils import save_image

from dmgan.damsm import DAMSMImageEncoder, DAMSMTextEncoder, load_frozen_checkpoint
from dmgan.data import CUBCaptionDataset, build_word_mask
from dmgan.metrics import InceptionEvaluator, fid_from_features, r_precision
from dmgan.models import DMGenerator
from dmgan.part_aware import (
    ATTRIBUTE_WORDS,
    PART_ALIASES,
    gaussian_part_heatmaps,
    part_alignment_statistics,
    token_part_targets,
)

COLOR_WORDS = tuple(
    word
    for word in (
        "black",
        "blue",
        "brown",
        "buff",
        "cream",
        "gray",
        "green",
        "grey",
        "iridescent",
        "olive",
        "orange",
        "pink",
        "purple",
        "red",
        "rufous",
        "tan",
        "white",
        "yellow",
    )
    if word in ATTRIBUTE_WORDS
)


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
        _, test_captions, index_to_word, word_to_index = pickle.load(stream, encoding="latin1")
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
        "index_to_word": index_to_word,
        "word_to_index": word_to_index,
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


def caption_part_targets(captions: np.ndarray, index_to_word: dict[int, str]) -> torch.Tensor:
    return torch.stack([token_part_targets(torch.from_numpy(caption), index_to_word) for caption in captions])


def part_coordinate_bank(
    root: Path,
    keys: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = CUBCaptionDataset(root, "test", training=False, include_parts=True)
    coordinates: list[torch.Tensor] = []
    visible: list[torch.Tensor] = []
    for key in keys:
        local_coordinates, local_visible = dataset.part_coordinates_for_key(key)
        coordinates.append(local_coordinates)
        visible.append(local_visible)
    return torch.stack(coordinates), torch.stack(visible)


def find_part_colour_pair(words: list[str], length: int) -> tuple[int, int] | None:
    part_positions = [index for index, word in enumerate(words[:length]) if word in PART_ALIASES]
    colour_positions = [index for index, word in enumerate(words[:length]) if word in COLOR_WORDS]
    candidates = [
        (abs(colour_index - part_index), colour_index, part_index)
        for colour_index in colour_positions
        for part_index in part_positions
        if abs(colour_index - part_index) <= 4
    ]
    if not candidates:
        return None
    _, colour_index, part_index = min(candidates)
    return colour_index, part_index


def build_colour_swap_bank(
    captions: np.ndarray,
    lengths: np.ndarray,
    index_to_word: dict[int, str],
    word_to_index: dict[str, int],
    *,
    alternatives: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    available_colours = [word for word in COLOR_WORDS if word in word_to_index]
    caption_to_candidate = np.full(captions.shape[0], -1, dtype=np.int64)
    candidate_rows: list[np.ndarray] = []
    candidate_lengths: list[int] = []
    metadata: list[dict[str, object]] = []
    for caption_index, (caption, length_value) in enumerate(zip(captions, lengths, strict=True)):
        length = int(length_value)
        words = [str(index_to_word[int(token)]).lower() for token in caption[:length]]
        pair = find_part_colour_pair(words, length)
        if pair is None:
            continue
        colour_position, part_position = pair
        correct_colour = words[colour_position]
        negatives = [word for word in available_colours if word != correct_colour]
        if correct_colour in {"gray", "grey"}:
            negatives = [word for word in negatives if word not in {"gray", "grey"}]
        offset = caption_index % len(negatives)
        negatives = (negatives[offset:] + negatives[:offset])[:alternatives]
        if len(negatives) < alternatives:
            continue
        variants = np.repeat(caption[None, :], alternatives + 1, axis=0)
        for index, colour in enumerate(negatives, start=1):
            variants[index, colour_position] = int(word_to_index[colour])
        caption_to_candidate[caption_index] = len(candidate_rows)
        candidate_rows.append(variants)
        candidate_lengths.append(length)
        metadata.append(
            {
                "caption_index": caption_index,
                "colour": correct_colour,
                "part": words[part_position],
                "colour_position": colour_position,
                "part_position": part_position,
            }
        )
    return (
        np.stack(candidate_rows),
        np.asarray(candidate_lengths, dtype=np.int64),
        caption_to_candidate,
        metadata,
    )


@torch.inference_mode()
def encode_colour_swap_bank(
    encoder: DAMSMTextEncoder,
    candidates: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    rows, alternatives, words_num = candidates.shape
    flat = candidates.reshape(rows * alternatives, words_num)
    flat_lengths = np.repeat(lengths, alternatives)
    encoded = encode_sentence_bank(encoder, flat, flat_lengths, batch_size, device)
    return encoded.view(rows, alternatives, -1)


def load_generator(path: Path, device: torch.device, weights: str) -> DMGenerator:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = dict(checkpoint["generator"])
    if weights == "ema":
        state.update(checkpoint["ema"]["shadow"])
    generator = DMGenerator(channels=64, memory_dim=128, residual_blocks=2)
    generator.load_state_dict(state, strict=True)
    return generator.to(device).eval()


def cluster_bootstrap_ci(
    differences: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    seed: int,
    draws: int = 2000,
) -> tuple[float, float]:
    finite = np.isfinite(differences)
    differences = differences[finite]
    cluster_ids = cluster_ids[finite]
    unique = np.unique(cluster_ids)
    cluster_means = np.asarray([differences[cluster_ids == item].mean() for item in unique])
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(cluster_means), size=(draws, len(cluster_means)))
    boot = cluster_means[indices].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def mcnemar_exact(baseline: np.ndarray, part_aware: np.ndarray) -> dict[str, float | int]:
    finite = np.isfinite(baseline) & np.isfinite(part_aware)
    baseline = baseline[finite].astype(bool)
    part_aware = part_aware[finite].astype(bool)
    baseline_only = int(np.sum(baseline & ~part_aware))
    part_only = int(np.sum(~baseline & part_aware))
    discordant = baseline_only + part_only
    p_value = (
        float(stats.binomtest(min(baseline_only, part_only), discordant, 0.5).pvalue) if discordant else 1.0
    )
    return {
        "baseline_only": baseline_only,
        "part_aware_only": part_only,
        "discordant": discordant,
        "p_value_two_sided": p_value,
    }


@torch.inference_mode()
def evaluate_model(
    generator: DMGenerator,
    text_encoder: DAMSMTextEncoder,
    image_encoder: DAMSMImageEncoder,
    inception: InceptionEvaluator,
    captions: np.ndarray,
    lengths: np.ndarray,
    caption_indices: np.ndarray,
    negative_indices: np.ndarray,
    sentence_bank: torch.Tensor,
    candidate_bank: torch.Tensor,
    caption_to_candidate: np.ndarray,
    coordinates: torch.Tensor,
    visible: torch.Tensor,
    targets: torch.Tensor,
    captions_per_image: int,
    fid_stats: Path,
    batch_size: int,
    preview_count: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    samples = len(caption_indices)
    fid_features = np.empty((samples, 2048), dtype=np.float32)
    r_matches = np.empty(samples, dtype=np.float32)
    colour_matches = np.full(samples, np.nan, dtype=np.float32)
    part_ce = np.full(samples, np.nan, dtype=np.float32)
    part_support = np.full(samples, np.nan, dtype=np.float32)
    active_tokens = np.zeros(samples, dtype=np.int64)
    previews: list[torch.Tensor] = []
    started = time.monotonic()

    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        local_indices = caption_indices[start:stop]
        caption_tensor, length_tensor, order = sorted_caption_batch(captions, lengths, local_indices, device)
        sorted_indices = local_indices[order]
        words, sentence = text_encoder(caption_tensor, length_tensor)
        noise = torch.randn(stop - start, 100, device=device)
        generated, diagnostics, _, _ = generator(
            noise,
            sentence,
            words,
            build_word_mask(length_tensor, caption_tensor.size(1)),
        )
        image_m11 = generated[-1]
        image_01 = ((image_m11 + 1.0) / 2.0).clamp(0.0, 1.0)
        if len(previews) < preview_count:
            needed = preview_count - len(previews)
            previews.extend(image_01[:needed].cpu().unbind(0))
        fid_features[start:stop] = inception.fid_features(image_01).cpu().numpy()

        _, image_code = image_encoder(image_m11)
        sorted_negative = negative_indices[start:stop][order]
        negative = sentence_bank[torch.from_numpy(sorted_negative)].to(device)
        matches = r_precision(image_code, sentence, negative).float()

        candidate_indices = caption_to_candidate[sorted_indices]
        eligible = candidate_indices >= 0
        sorted_colour = np.full(stop - start, np.nan, dtype=np.float32)
        if eligible.any():
            local_candidates = candidate_bank[torch.from_numpy(candidate_indices[eligible]).long()].to(device)
            normalized_images = torch.nn.functional.normalize(image_code[eligible], dim=1)
            normalized_candidates = torch.nn.functional.normalize(local_candidates, dim=2)
            scores = torch.einsum("bd,bkd->bk", normalized_images, normalized_candidates)
            sorted_colour[eligible] = (scores.argmax(dim=1) == 0).float().cpu().numpy()

        image_indices = sorted_indices // captions_per_image
        local_coordinates = coordinates[torch.from_numpy(image_indices)].to(device)
        local_visible = visible[torch.from_numpy(image_indices)].to(device)
        local_targets = targets[torch.from_numpy(sorted_indices)].to(device)
        mask = build_word_mask(length_tensor, caption_tensor.size(1))
        scale_ce: list[torch.Tensor] = []
        scale_support: list[torch.Tensor] = []
        scale_counts: list[torch.Tensor] = []
        for scale in (128, 256):
            attention = diagnostics[f"attention_{scale}"]
            heatmaps = gaussian_part_heatmaps(
                local_coordinates,
                local_visible,
                attention.shape[-2],
                attention.shape[-1],
                sigma=max(1.0, attention.shape[-1] * 0.08),
            )
            ce, support, counts = part_alignment_statistics(attention, heatmaps, local_targets, mask)
            scale_ce.append(ce)
            scale_support.append(support)
            scale_counts.append(counts)
        local_ce = torch.nanmean(torch.stack(scale_ce), dim=0)
        local_support = torch.nanmean(torch.stack(scale_support), dim=0)
        local_counts = torch.stack(scale_counts).amax(dim=0)

        inverse = np.argsort(order)
        r_matches[start:stop] = matches[torch.as_tensor(inverse, device=device)].cpu().numpy()
        colour_matches[start:stop] = sorted_colour[inverse]
        part_ce[start:stop] = local_ce[torch.as_tensor(inverse, device=device)].cpu().numpy()
        part_support[start:stop] = local_support[torch.as_tensor(inverse, device=device)].cpu().numpy()
        active_tokens[start:stop] = local_counts[torch.as_tensor(inverse, device=device)].cpu().numpy()
        if stop == samples or stop % (batch_size * 25) == 0:
            elapsed = time.monotonic() - started
            print(
                f"Processed {stop:,}/{samples:,} samples ({stop / max(elapsed, 1e-6):.1f}/s)",
                flush=True,
            )

    with np.load(fid_stats) as reference:
        fid = fid_from_features(fid_features, reference["mu"], reference["sigma"])
    results = {
        "fid_pytorch": fid,
        "r_precision_percent": float(np.mean(r_matches) * 100.0),
        "part_colour_swap_accuracy_percent": float(np.nanmean(colour_matches) * 100.0),
        "part_colour_eligible_samples": int(np.isfinite(colour_matches).sum()),
        "part_alignment_ce": float(np.nanmean(part_ce)),
        "part_attention_support_mass_percent": float(np.nanmean(part_support) * 100.0),
        "part_alignment_samples": int(np.isfinite(part_ce).sum()),
        "mean_active_part_tokens": float(np.mean(active_tokens[active_tokens > 0])),
    }
    return {
        "results": results,
        "arrays": {
            "r_matches": r_matches,
            "colour_matches": colour_matches,
            "part_ce": part_ce,
            "part_support": part_support,
            "active_tokens": active_tokens,
        },
        "previews": torch.stack(previews),
        "elapsed_seconds": time.monotonic() - started,
    }


def markdown_report(report: dict[str, object]) -> str:
    decision = report["decision"]
    aggregate = report["aggregate"]
    seed_rows = []
    for row in report["per_seed"]:
        seed_rows.append(
            "| {seed} | {b_fid:.3f} | {p_fid:.3f} | {d_fid:+.3f} | "
            "{b_r:.2f}% | {p_r:.2f}% | {d_r:+.2f} pp | "
            "{b_c:.2f}% | {p_c:.2f}% | {d_c:+.2f} pp |".format(
                seed=row["seed"],
                b_fid=row["baseline"]["fid_pytorch"],
                p_fid=row["part_aware"]["fid_pytorch"],
                d_fid=row["delta"]["fid"],
                b_r=row["baseline"]["r_precision_percent"],
                p_r=row["part_aware"]["r_precision_percent"],
                d_r=row["delta"]["r_precision_pp"],
                b_c=row["baseline"]["part_colour_swap_accuracy_percent"],
                p_c=row["part_aware"]["part_colour_swap_accuracy_percent"],
                d_c=row["delta"]["part_colour_pp"],
            )
        )
    return f"""# Session 7 Part-Aware Ablation

## Answer

**{decision["verdict"].upper()}** - {decision["summary"]}

The formal comparison is the equal-budget baseline-control versus the
part-aware branch. Both start from the same author-released generator and the
same warmed-up discriminators; the only objective difference is
`lambda_part = {report["protocol"]["part_lambda"]}`.

## Fixed protocol

- Paired training seeds: {", ".join(str(seed) for seed in report["protocol"]["seeds"])}
- Fine-tuning steps per arm: {report["protocol"]["training_steps"]:,}
- Discriminator warm-up: {report["protocol"]["warmup_steps"]:,} steps per seed
- Evaluation: {report["protocol"]["evaluation_samples"]:,} fixed CUB test samples,
  identical captions, latent-noise sequence, and negative captions
- Trainable generator scope: both refinement stages and their 128/256 image heads
- Primary targeted diagnostic: correct colour versus nine colour-swapped captions
  for captions that explicitly connect a colour and a bird part
- Global safeguards: repository-compatible FID and official DAMSM R-precision

## Per-seed results

| Seed | Base FID | Part FID | Delta | Base R | Part R | Delta | Base part-colour | Part part-colour | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(seed_rows)}

## Aggregate paired effects

- Mean FID delta (part - baseline): {aggregate["fid_delta_mean"]:+.3f}
- R-precision delta: {aggregate["r_precision_delta_pp"]:+.2f} pp,
  cluster-bootstrap 95% CI [{aggregate["r_precision_delta_ci95_pp"][0]:+.2f},
  {aggregate["r_precision_delta_ci95_pp"][1]:+.2f}]
- Part-colour swap accuracy delta: {aggregate["part_colour_delta_pp"]:+.2f} pp,
  cluster-bootstrap 95% CI [{aggregate["part_colour_delta_ci95_pp"][0]:+.2f},
  {aggregate["part_colour_delta_ci95_pp"][1]:+.2f}]
- Part-attention CE delta: {aggregate["part_ce_delta"]:+.3f} (lower is better)
- Attention mass in the annotated 10% support: {aggregate["part_support_delta_pp"]:+.2f} pp

## Decision rule

We call the method better overall only if the targeted part-colour diagnostic
improves with a 95% CI above zero, mean FID does not regress by more than 1.0,
and R-precision does not fall by more than 1 percentage point. The attention
metric is mechanistic evidence only because it is directly optimized.

## Limits

- This is controlled fine-tuning from official weights, not a full 800-epoch
  from-scratch retraining.
- CUB part labels are extra supervision; the baseline does not receive them.
- A generated bird may use a different pose from the paired real image, so the
  landmark-attention score is a proxy, not direct generated-image keypoint accuracy.
- The colour-swap diagnostic uses the frozen DAMSM evaluator and does not replace
  a blinded human study or a separately trained bird-part detector.
- The official 30,000-sample baseline result remains a reproduction reference;
  only the paired controls in this table support the improvement conclusion.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=Path("artifacts/session7/ablation"))
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
    parser.add_argument("--fid-stats", type=Path, default=Path("checkpoints/eval/bird_val.npz"))
    parser.add_argument("--samples", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--caption-bank-batch-size", type=int, default=512)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.samples < 100:
        raise SystemExit("--samples must be at least 100")
    manifest = json.loads((args.experiment_dir / "training_manifest.json").read_text())
    seeds = [int(seed) for seed in manifest["seeds"]]
    training_reports = {
        seed: json.loads((args.experiment_dir / f"seed_{seed}" / "training_report.json").read_text())
        for seed in seeds
    }
    first_protocol = training_reports[seeds[0]]["protocol"]
    device = torch.device(args.device)
    bank = load_caption_bank(args.metadata_root, words_num=18, seed=args.seed)
    captions = bank["captions"]
    lengths = bank["lengths"]
    classes = bank["classes"]
    keys = bank["keys"]
    captions_per_image = int(bank["captions_per_image"])
    caption_indices = np.arange(args.samples, dtype=np.int64) % captions.shape[0]
    negative_indices = sample_negative_indices(classes, args.samples, negatives=99, seed=args.seed + 1)
    text_encoder = load_frozen_checkpoint(
        DAMSMTextEncoder(int(bank["vocabulary_size"])), args.text_checkpoint
    ).to(device)
    image_encoder = load_frozen_checkpoint(DAMSMImageEncoder(), args.image_checkpoint).to(device)
    inception = InceptionEvaluator().to(device).eval()

    print("Encoding the DAMSM negative-caption bank...", flush=True)
    sentence_bank = encode_sentence_bank(
        text_encoder, captions, lengths, args.caption_bank_batch_size, device
    )
    print("Preparing token targets and deterministic CUB part coordinates...", flush=True)
    targets = caption_part_targets(captions, bank["index_to_word"])
    coordinates, visible = part_coordinate_bank(args.metadata_root, keys)
    print("Encoding part-focused colour-swap captions...", flush=True)
    colour_candidates, colour_lengths, caption_to_candidate, colour_metadata = build_colour_swap_bank(
        captions,
        lengths,
        bank["index_to_word"],
        bank["word_to_index"],
    )
    candidate_bank = encode_colour_swap_bank(
        text_encoder,
        colour_candidates,
        colour_lengths,
        args.caption_bank_batch_size,
        device,
    )
    (args.experiment_dir / "colour_swap_protocol.json").write_text(
        json.dumps(
            {
                "eligible_caption_count": len(colour_metadata),
                "candidate_count": int(colour_candidates.shape[1]),
                "colours": COLOR_WORDS,
                "examples": colour_metadata[:20],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    per_seed: list[dict[str, object]] = []
    combined_arrays: dict[str, list[np.ndarray]] = {
        "baseline_r": [],
        "part_r": [],
        "baseline_colour": [],
        "part_colour": [],
        "baseline_ce": [],
        "part_ce": [],
        "baseline_support": [],
        "part_support": [],
        "clusters": [],
    }
    for seed_index, seed in enumerate(seeds):
        seed_dir = args.experiment_dir / f"seed_{seed}"
        evaluated: dict[str, dict[str, object]] = {}
        for variant, filename in (
            ("baseline", "baseline_final.pt"),
            ("part_aware", "part_aware_final.pt"),
        ):
            print(f"Evaluating seed {seed} / {variant}...", flush=True)
            generator = load_generator(seed_dir / filename, device, args.weights)
            evaluated[variant] = evaluate_model(
                generator,
                text_encoder,
                image_encoder,
                inception,
                captions,
                lengths,
                caption_indices,
                negative_indices,
                sentence_bank,
                candidate_bank,
                caption_to_candidate,
                coordinates,
                visible,
                targets,
                captions_per_image,
                args.fid_stats,
                args.batch_size,
                args.preview_count,
                args.seed,
                device,
            )
            del generator
            if device.type == "cuda":
                torch.cuda.empty_cache()

        baseline_results = evaluated["baseline"]["results"]
        part_results = evaluated["part_aware"]["results"]
        baseline_arrays = evaluated["baseline"]["arrays"]
        part_arrays = evaluated["part_aware"]["arrays"]
        delta = {
            "fid": part_results["fid_pytorch"] - baseline_results["fid_pytorch"],
            "r_precision_pp": part_results["r_precision_percent"] - baseline_results["r_precision_percent"],
            "part_colour_pp": part_results["part_colour_swap_accuracy_percent"]
            - baseline_results["part_colour_swap_accuracy_percent"],
            "part_ce": part_results["part_alignment_ce"] - baseline_results["part_alignment_ce"],
            "part_support_pp": part_results["part_attention_support_mass_percent"]
            - baseline_results["part_attention_support_mass_percent"],
        }
        row = {
            "seed": seed,
            "baseline": baseline_results,
            "part_aware": part_results,
            "delta": delta,
            "mcnemar_r": mcnemar_exact(baseline_arrays["r_matches"], part_arrays["r_matches"]),
            "mcnemar_part_colour": mcnemar_exact(
                baseline_arrays["colour_matches"], part_arrays["colour_matches"]
            ),
        }
        per_seed.append(row)
        np.savez_compressed(
            seed_dir / "paired_evaluation_arrays.npz",
            caption_indices=caption_indices,
            baseline_r=baseline_arrays["r_matches"],
            part_aware_r=part_arrays["r_matches"],
            baseline_part_colour=baseline_arrays["colour_matches"],
            part_aware_part_colour=part_arrays["colour_matches"],
            baseline_part_ce=baseline_arrays["part_ce"],
            part_aware_part_ce=part_arrays["part_ce"],
            baseline_part_support=baseline_arrays["part_support"],
            part_aware_part_support=part_arrays["part_support"],
        )
        comparison = torch.cat(
            [evaluated["baseline"]["previews"], evaluated["part_aware"]["previews"]], dim=0
        )
        save_image(
            comparison,
            seed_dir / "evaluation_preview_baseline_top_part_bottom.png",
            nrow=args.preview_count,
        )
        sample_clusters = (caption_indices // captions_per_image) + seed_index * len(keys)
        combined_arrays["baseline_r"].append(baseline_arrays["r_matches"])
        combined_arrays["part_r"].append(part_arrays["r_matches"])
        combined_arrays["baseline_colour"].append(baseline_arrays["colour_matches"])
        combined_arrays["part_colour"].append(part_arrays["colour_matches"])
        combined_arrays["baseline_ce"].append(baseline_arrays["part_ce"])
        combined_arrays["part_ce"].append(part_arrays["part_ce"])
        combined_arrays["baseline_support"].append(baseline_arrays["part_support"])
        combined_arrays["part_support"].append(part_arrays["part_support"])
        combined_arrays["clusters"].append(sample_clusters)

    joined = {key: np.concatenate(value) for key, value in combined_arrays.items()}
    r_difference = (joined["part_r"] - joined["baseline_r"]) * 100.0
    colour_difference = (joined["part_colour"] - joined["baseline_colour"]) * 100.0
    ce_difference = joined["part_ce"] - joined["baseline_ce"]
    support_difference = (joined["part_support"] - joined["baseline_support"]) * 100.0
    r_ci = cluster_bootstrap_ci(r_difference, joined["clusters"], seed=args.seed + 20)
    colour_ci = cluster_bootstrap_ci(colour_difference, joined["clusters"], seed=args.seed + 21)
    ce_ci = cluster_bootstrap_ci(ce_difference, joined["clusters"], seed=args.seed + 22)
    support_ci = cluster_bootstrap_ci(support_difference, joined["clusters"], seed=args.seed + 23)
    fid_deltas = np.asarray([row["delta"]["fid"] for row in per_seed])
    aggregate = {
        "fid_delta_mean": float(fid_deltas.mean()),
        "fid_delta_sd": float(fid_deltas.std(ddof=1)) if len(fid_deltas) > 1 else 0.0,
        "r_precision_delta_pp": float(np.nanmean(r_difference)),
        "r_precision_delta_ci95_pp": r_ci,
        "part_colour_delta_pp": float(np.nanmean(colour_difference)),
        "part_colour_delta_ci95_pp": colour_ci,
        "part_ce_delta": float(np.nanmean(ce_difference)),
        "part_ce_delta_ci95": ce_ci,
        "part_support_delta_pp": float(np.nanmean(support_difference)),
        "part_support_delta_ci95_pp": support_ci,
        "mcnemar_r": mcnemar_exact(joined["baseline_r"], joined["part_r"]),
        "mcnemar_part_colour": mcnemar_exact(joined["baseline_colour"], joined["part_colour"]),
        "positive_part_colour_seeds": int(sum(row["delta"]["part_colour_pp"] > 0 for row in per_seed)),
    }
    target_pass = aggregate["part_colour_delta_ci95_pp"][0] > 0.0
    fid_pass = aggregate["fid_delta_mean"] <= 1.0
    r_pass = aggregate["r_precision_delta_pp"] >= -1.0
    direction_pass = aggregate["positive_part_colour_seeds"] >= math.ceil(len(seeds) * 2 / 3)
    overall_pass = target_pass and fid_pass and r_pass and direction_pass
    if overall_pass:
        verdict = "supported"
        summary = (
            "Part-aware fine-tuning improves the targeted part-colour diagnostic "
            "without a material FID or R-precision regression under this fixed budget."
        )
    elif aggregate["part_ce_delta"] < 0 and not target_pass:
        verdict = "not demonstrated"
        summary = (
            "The auxiliary loss improves its attention proxy, but the output-level "
            "part-colour evidence is insufficient to claim that the method is better."
        )
    else:
        verdict = "not supported"
        summary = (
            "The paired experiment does not satisfy the predeclared targeted-improvement "
            "and global-quality safeguards."
        )
    report = {
        "status": "complete",
        "question": "Is part-aware DM-GAN better than the equal-budget baseline control?",
        "protocol": {
            "seeds": seeds,
            "warmup_steps": int(first_protocol["discriminator_warmup_steps"]),
            "training_steps": int(first_protocol["paired_training_steps"]),
            "part_lambda": float(first_protocol["part_lambda"]),
            "part_sigma_fraction": float(first_protocol["part_sigma_fraction"]),
            "evaluation_samples": args.samples,
            "evaluation_seed": args.seed,
            "weights": args.weights,
            "caption_bank_size": int(captions.shape[0]),
            "repeated_caption_samples": max(0, args.samples - int(captions.shape[0])),
            "colour_swap_eligible_caption_count": len(colour_metadata),
        },
        "checkpoints": {
            "text_encoder": {"path": str(args.text_checkpoint), "sha256": file_sha256(args.text_checkpoint)},
            "image_encoder": {
                "path": str(args.image_checkpoint),
                "sha256": file_sha256(args.image_checkpoint),
            },
            "fid_stats": {"path": str(args.fid_stats), "sha256": file_sha256(args.fid_stats)},
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
        "decision": {
            "verdict": verdict,
            "summary": summary,
            "criteria": {
                "part_colour_ci_lower_above_zero": bool(target_pass),
                "mean_fid_regression_at_most_1": bool(fid_pass),
                "r_precision_regression_at_most_1pp": bool(r_pass),
                "two_thirds_seeds_positive": bool(direction_pass),
            },
        },
        "limitations": [
            "Official-weight fine-tuning rather than full from-scratch convergence.",
            "CUB part annotations are extra supervision unavailable to the control.",
            "Real-image landmarks are a proxy because generated pose can differ.",
            "Colour-swap accuracy uses the same frozen DAMSM family as training/evaluation.",
            "A blinded human part-attribute study remains future validation.",
        ],
    }
    (args.experiment_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.experiment_dir / "PART_AWARE_ABLATION.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report["decision"], indent=2), flush=True)


if __name__ == "__main__":
    main()
