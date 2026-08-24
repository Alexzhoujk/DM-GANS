"""Download official DM-GAN metadata/checkpoints without downloading CUB images."""

from __future__ import annotations

import argparse
from pathlib import Path

ASSETS = {
    "bird_metadata": "1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ",
    "bird_damsm": "1GNUKjVeyWYBJ8hEU-yrfYQpDOkxEyP3V",
    "bird_dmgan": "1BmDKqIyNY_7XWhXpxa2gm6TYxB2DQHS3",
    "bird_fid_stats": "1747il5vnY2zNkmQ1x_8hySx537ZAJEtj",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", choices=sorted(ASSETS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        import gdown
    except ImportError as error:
        raise SystemExit("Install the download extra: python -m pip install '.[download]'") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={ASSETS[args.asset]}"
    result = gdown.download(url, str(args.output), quiet=False)
    if result is None:
        raise SystemExit(f"Download failed for {args.asset}")


if __name__ == "__main__":
    main()
