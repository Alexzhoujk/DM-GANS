# DM-GAN Session 6 Baseline

This repository is the runnable Session 6 baseline for **DM-GAN: Dynamic Memory
Generative Adversarial Networks for Text-to-Image Synthesis** on CUB-200-2011.
It separates three kinds of evidence:

1. official pretrained-model inference;
2. forward/loss/backward evidence from this modern reimplementation;
3. samples or metrics produced by checkpoints trained with this project.

These results must not be presented as interchangeable.

## Session 6 deliverables

- [Completion report](SESSION6_COMPLETION.md) — completed work mapped to the
  four-person A/B/C/D plan, verification evidence, and remaining work.
- [30,000-sample baseline evaluation](artifacts/session6/baseline_evaluation/BASELINE_EVALUATION.md)
  — protocol, comparable metrics, pass criteria, and interpretation boundary.
- [Updated 10-slide presentation](outputs/DM-GAN%20Baseline%20Implementation%20and%20Investigation%20-%20Session%206%20Updated.pptx).
- Selected evidence is committed under `artifacts/session6/`. Datasets,
  author-released weights, virtual environments, and local `.pt` checkpoints
  are intentionally excluded.

## Session 6 acceptance target

- Dataset → DAMSM → DM-GAN interfaces have explicit shapes.
- Generator produces 64×64, 128×128, and 256×256 images.
- Dynamic memory exposes attention, writing-gate, and response-gate tensors.
- Three spectral-normalized discriminators compute conditional, unconditional,
  fake, real, and wrong-caption losses.
- DAMSM word/sentence matching and conditioning-augmentation KL losses are present.
- A complete discriminator + generator optimizer step runs successfully.
- Fixed-caption local samples and a short loss log are saved before the presentation.
- Author-released weights pass a fixed 30,000-sample comparable FID and
  R-precision evaluation through the modern code path.

Full from-scratch convergence and the part-aware baseline/variant ablation are
later-stage targets.

## Baseline reproduction verdict

**PASS** — the author-released DM-GAN and DAMSM weights were evaluated through
this modern implementation on 30,000 fixed CUB test samples. The comparable
metrics are close to the official pretrained reference:

| Metric | Modern run | Official pretrained reference |
| --- | ---: | ---: |
| PyTorch FID ↓ | 15.7576 | 15.34 |
| DAMSM R-precision ↑ | 76.67% ± 0.83% | 76.58% ± 0.53% |

The ImageNet Inception Score is 5.7007 ± 0.0940, but it is an internal health
check only. It must not be compared directly with the paper's legacy
50-class bird-classifier IS.

See the [evaluation report](artifacts/session6/baseline_evaluation/BASELINE_EVALUATION.md),
[machine-readable results](artifacts/session6/baseline_evaluation/report.json),
and [30,000-sample preview](artifacts/session6/baseline_evaluation/official_baseline_preview_256.png).

## Environment

The project targets Python 3.11–3.13. RTX 50-series GPUs require a PyTorch build
with Blackwell support. On the current RTX 5080 workstation, use the CUDA 13.0
wheel:

```bash
uv venv --python 3.12 .venv
.venv/bin/python -m pip install torch==2.12.0 torchvision==0.27.0 \
  --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install -e '.[dev,download]' --no-deps
```

Check the runtime:

```bash
.venv/bin/python scripts/inspect_environment.py
```

## Local acceptance test

The default test uses smaller channel counts but exercises the complete model and
optimizer path:

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/smoke_test.py --device cuda
```

For paper-sized G/D channel widths:

```bash
.venv/bin/python scripts/smoke_test.py --device cuda --full-channels
```

## Module interfaces

```text
A Dataset
  images: [B,3,64,64], [B,3,128,128], [B,3,256,256]
  captions: [B,T] int64
  caption_lengths: [B] int64, sorted descending
  class_ids: [B] int64
  keys: list[str]

B Frozen DAMSM
  word_embeddings: [B,256,T]
  sentence_embeddings: [B,256]
  word_mask: [B,T] bool (True means padding)

C DM-GAN
  fake_images: three tensors at 64/128/256
  attention: [B,T,H,W]
  writing_gate: [B,1,T]
  response_gate: [B,1,H,W]
  mu/logvar: [B,100]

D Training
  discriminator losses at all scales
  generator adversarial + DAMSM word/sentence + KL losses
  backward, optimizer, EMA, checkpoint, fixed samples
```

## Official assets

The helper downloads author-provided Google Drive assets:

```bash
.venv/bin/python scripts/prepare_official_assets.py bird_metadata --output data/bird.zip
.venv/bin/python scripts/prepare_official_assets.py bird_damsm --output checkpoints/bird_damsm.zip
.venv/bin/python scripts/prepare_official_assets.py bird_dmgan --output checkpoints/bird_DMGAN.pth
.venv/bin/python scripts/prepare_official_assets.py bird_fid_stats --output checkpoints/eval/bird_val.npz
```

CUB images must be obtained from the official CUB-200-2011 dataset source and
placed under `data/birds/CUB_200_2011/`. Do not commit datasets or checkpoints.

Run the fixed formal evaluation after all four author assets are in place:

```bash
.venv/bin/python scripts/evaluate_baseline.py \
  --metadata-root data/birds \
  --generator-checkpoint checkpoints/bird_DMGAN.pth \
  --text-checkpoint checkpoints/DAMSMencoders/bird/text_encoder200.pth \
  --image-checkpoint checkpoints/DAMSMencoders/bird/image_encoder200.pth \
  --fid-stats checkpoints/eval/bird_val.npz \
  --output-dir artifacts/session6/baseline_evaluation \
  --samples 30000 --seed 20260824 --device cuda
```

## Fidelity decisions

Compared with the earlier core-only ZIP, this version restores the official
training behavior that materially affects the baseline:

- no Transformer-style `1/sqrt(d)` scaling in dynamic-memory addressing;
- detached current-image summary during memory writing;
- spectral normalization in discriminator feature/joint convolutions;
- both conditional and unconditional discriminator heads;
- DAMSM word-level and sentence-level matching losses;
- KL weight defaults to the official code path's effective value of 1.

The implementation uses logits plus `BCEWithLogitsLoss` rather than the old
Sigmoid-plus-BCE pair; this is numerically safer and mathematically equivalent.

## Session 6 investigation evidence

For every selected failure caption, save:

- the caption and fixed random seed;
- 64/128/256 outputs;
- 128/256 attention and writing gates;
- error label: part/color, object count, pose, or spatial relation;
- whether refinement corrected or preserved the initial error.

The proposed improvement is **part-aware memory alignment** using CUB part
annotations. An optional loss prototype and unit test are included, but it
remains an unvalidated experimental variant until a controlled baseline/variant
comparison is run. Its efficacy is explicitly scheduled as the next experiment,
not claimed by the baseline report.

## Primary sources

- Official repository: https://github.com/MinfengZhu/DM-GAN
- CVPR 2019 paper: https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_DM-GAN_Dynamic_Memory_Generative_Adversarial_Networks_for_Text-To-Image_Synthesis_CVPR_2019_paper.pdf
