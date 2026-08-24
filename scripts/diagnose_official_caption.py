"""Export word attention and writing gates for one official-checkpoint caption."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import torch
from torchvision.utils import save_image

from dmgan.checkpoints import load_official_generator_checkpoint
from dmgan.damsm import DAMSMTextEncoder, load_frozen_checkpoint
from dmgan.data import build_word_mask
from dmgan.models import DMGenerator


def tokenize(caption: str, word_to_index: dict[str, int], words_num: int) -> list[int]:
    tokens = re.findall(r"[a-z0-9]+", caption.lower())
    indices = [word_to_index[token] for token in tokens if token in word_to_index]
    if not indices:
        raise ValueError(f"Caption has no known DAMSM tokens: {caption}")
    return indices[:words_num]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--caption",
        default="this bird has wings that are red and has a yellow belly",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/session6/diagnostics/red_wings"))
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    with Path("data/birds/captions.pickle").open("rb") as stream:
        _, _, index_to_word, word_to_index = pickle.load(stream, encoding="latin1")
    token_ids = tokenize(args.caption, word_to_index, 18)
    tokens = [index_to_word[index] for index in token_ids]
    captions = torch.zeros(1, 18, dtype=torch.long, device=device)
    captions[0, : len(token_ids)] = torch.tensor(token_ids, device=device)
    lengths = torch.tensor([len(token_ids)], device=device)
    text_encoder = load_frozen_checkpoint(
        DAMSMTextEncoder(len(index_to_word)),
        "checkpoints/DAMSMencoders/bird/text_encoder200.pth",
    ).to(device)
    generator = load_official_generator_checkpoint(
        DMGenerator(channels=64, memory_dim=128, residual_blocks=2),
        "checkpoints/bird_DMGAN.pth",
    ).to(device).eval()
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    with torch.no_grad():
        words, sentence = text_encoder(captions, lengths)
        noise = torch.randn(1, 100, device=device)
        images, diagnostics, _, _ = generator(
            noise, sentence, words, build_word_mask(lengths, captions.size(1))
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for image in images:
        save_image((image + 1.0) / 2.0, args.output_dir / f"official_{image.shape[-1]}.png")
    report: dict[str, object] = {
        "provenance": "Author-released checkpoint diagnostic",
        "caption": args.caption,
        "tokens": tokens,
        "seed": args.seed,
        "stages": {},
    }
    for stage in (128, 256):
        attention = diagnostics[f"attention_{stage}"][0, : len(tokens)].cpu()
        writing = diagnostics[f"writing_gate_{stage}"][0, 0, : len(tokens)].cpu()
        stage_stats = []
        for index, token in enumerate(tokens):
            word_map = attention[index]
            normalized = (word_map - word_map.min()) / (word_map.max() - word_map.min()).clamp_min(1e-8)
            save_image(normalized, args.output_dir / f"attention_{stage}_{index:02d}_{token}.png")
            stage_stats.append(
                {
                    "token": token,
                    "writing_gate": float(writing[index]),
                    "attention_mean": float(word_map.mean()),
                    "attention_max": float(word_map.max()),
                }
            )
        report["stages"][str(stage)] = stage_stats
    rendered = json.dumps(report, indent=2)
    (args.output_dir / "report.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
