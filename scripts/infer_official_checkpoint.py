"""Generate clearly labeled samples from the author-released CUB checkpoint."""

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
    parser.add_argument("--metadata", type=Path, default=Path("data/birds/captions.pickle"))
    parser.add_argument("--captions", type=Path, default=Path("data/birds/example_captions.txt"))
    parser.add_argument(
        "--text-checkpoint",
        type=Path,
        default=Path("checkpoints/DAMSMencoders/bird/text_encoder200.pth"),
    )
    parser.add_argument("--generator-checkpoint", type=Path, default=Path("checkpoints/bird_DMGAN.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/session6/official_pretrained"))
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    with args.metadata.open("rb") as stream:
        _, _, index_to_word, word_to_index = pickle.load(stream, encoding="latin1")
    captions = [line.strip() for line in args.captions.read_text(encoding="utf-8").splitlines() if line.strip()]
    captions = captions[: args.count]
    encoded = [tokenize(caption, word_to_index, 18) for caption in captions]
    order = sorted(range(len(encoded)), key=lambda index: len(encoded[index]), reverse=True)
    captions = [captions[index] for index in order]
    encoded = [encoded[index] for index in order]
    lengths = torch.tensor([len(tokens) for tokens in encoded], dtype=torch.long, device=device)
    caption_tensor = torch.zeros(len(encoded), 18, dtype=torch.long, device=device)
    for index, tokens in enumerate(encoded):
        caption_tensor[index, : len(tokens)] = torch.tensor(tokens, device=device)

    text_encoder = DAMSMTextEncoder(len(index_to_word))
    load_frozen_checkpoint(text_encoder, args.text_checkpoint)
    text_encoder.to(device)
    generator = DMGenerator(channels=64, memory_dim=128, residual_blocks=2)
    load_official_generator_checkpoint(generator, args.generator_checkpoint)
    generator.to(device).eval()
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    with torch.no_grad():
        words, sentence = text_encoder(caption_tensor, lengths)
        noise = torch.randn(len(encoded), 100, device=device)
        images, diagnostics, _, _ = generator(
            noise, sentence, words, build_word_mask(lengths, caption_tensor.size(1))
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for image in images:
        size = image.shape[-1]
        save_image((image + 1.0) / 2.0, args.output_dir / f"official_checkpoint_grid_{size}.png", nrow=2)
    report = {
        "provenance": "Author-released bird_DMGAN.pth; not trained by this project",
        "seed": args.seed,
        "captions": captions,
        "image_shapes": [list(image.shape) for image in images],
        "attention_shapes": {
            "128": list(diagnostics["attention_128"].shape),
            "256": list(diagnostics["attention_256"].shape),
        },
        "generator_checkpoint": str(args.generator_checkpoint),
        "text_checkpoint": str(args.text_checkpoint),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
