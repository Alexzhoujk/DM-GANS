# DM-GAN Session 7 Part-Aware Ablation Completion Report

This document records the final-week controlled experiment that asked:

> **Is part-aware DM-GAN actually better than an equal-budget DM-GAN
> baseline?**

## Outcome

**NOT DEMONSTRATED.** Under the fixed protocol below, the part-aware auxiliary
loss improved the attention quantity that it directly optimizes, but it did not
produce a reliable improvement in the generated image's part-colour
relationship. Global text alignment was essentially unchanged and FID became
slightly worse.

This wording is deliberate. The experiment does **not** prove that every
part-aware method is ineffective. It shows that this implementation, loss
weight, training budget, and evaluation protocol do not provide enough evidence
to claim an output-level improvement.

## What was completed

The Session 7 implementation adds a runnable, paired baseline-versus-variant
experiment to the Session 6 DM-GAN reproduction:

- `dmgan/data.py` optionally loads all 15 CUB landmarks and applies the same
  bounding-box crop, resize, random crop, and horizontal flip to the image and
  landmark coordinates;
- `dmgan/part_aware.py` maps conservative part and attribute words to CUB
  landmarks, builds Gaussian target heatmaps, and computes the part-alignment
  loss and its diagnostic statistics;
- `dmgan/training.py` applies the optional auxiliary loss while preserving the
  baseline path when `part_lambda=0`;
- `scripts/run_part_aware_ablation.py` creates the equal-budget paired training
  runs;
- `scripts/evaluate_ablation.py` performs fixed 30,000-sample evaluation,
  paired significance tests, cluster-bootstrap confidence intervals, and the
  predeclared decision rule;
- `tests/test_part_aware.py` covers token-to-part targets, alignment behaviour,
  and integration with the existing test suite.

The current suite contains 13 passing tests, and Ruff reports no errors for
`dmgan`, `scripts`, and `tests`.

## Controlled experiment design

| Item | Fixed setting |
| --- | --- |
| Starting generator | Author-released DM-GAN generator |
| Paired training seeds | 20260824, 20260825, 20260826 |
| Discriminator warm-up | 200 steps per seed, with the generator frozen |
| Fine-tuning budget | 1,200 steps per arm and seed; batch size 10 |
| Generator learning rate | 2e-5 |
| Discriminator learning rate | 2e-4 |
| Trainable generator scope | `refine_128`, `refine_256`, `to_image_128`, `to_image_256` |
| Control arm | Standard objective, `part_lambda=0` |
| Part-aware arm | Same objective plus part loss, `part_lambda=0.05` |
| Heatmap width | `part_sigma_fraction=0.08` |
| Evaluated weights | Exponential moving average (EMA) |
| Evaluation | 30,000 fixed CUB test draws per model; seed 20260902 |

For each seed, the two arms start with the same official generator and cloned
discriminator/optimizer state after the shared warm-up. Every paired training
step receives the same batch, caption choice, augmentation, and restored random
number state. Evaluation likewise uses the same captions, latent noise, and
negative captions for both arms. Therefore, the intended objective difference
between the paired arms is the part-aware loss.

This is a controlled fine-tuning comparison, not a full 800-epoch reproduction
from random initialization. Also, the part-aware arm receives additional CUB
landmark supervision that the control arm does not receive; this is part of the
method's cost and must be disclosed.

## Metrics and decision rule

The primary output-level diagnostic is **part-colour swap accuracy**. For a
caption that links a colour word to a bird-part word, the generated image is
matched against the correct caption and nine otherwise identical captions with
the colour swapped. The correct caption should rank first. The 30,000-draw
evaluation contained 24,924 eligible rows per model, originating from 24,454
eligible captions in the 29,330-caption test bank; 670 caption draws wrap around
the bank.

Global safeguards are repository-compatible FID and the official frozen-DAMSM
R-precision. Part-attention cross-entropy (CE) and attention mass near the real
CUB landmark are mechanistic diagnostics, not independent proof of image
quality, because the auxiliary loss directly optimizes this attention target.

The method is called better only if all of the following hold:

1. the 95% confidence interval for the part-colour improvement is above zero;
2. at least two of the three seeds improve part-colour accuracy;
3. mean FID does not regress by more than 1.0;
4. R-precision does not regress by more than 1 percentage point.

Confidence intervals for paired sample-level metrics use a bootstrap clustered
by CUB image within each seed. Paired binary outcomes are also checked with an
exact McNemar test.

## Results

All deltas below are `part-aware - baseline`; lower FID and attention CE are
better, while higher R-precision and part-colour accuracy are better.

| Seed | Base FID | Part FID | FID delta | Base R | Part R | R delta | Base part-colour | Part part-colour | Part-colour delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260824 | 15.948 | 15.962 | +0.014 | 78.05% | 77.98% | -0.07 pp | 71.95% | 71.99% | +0.04 pp |
| 20260825 | 16.296 | 16.335 | +0.039 | 78.40% | 78.33% | -0.08 pp | 72.52% | 72.50% | -0.02 pp |
| 20260826 | 15.919 | 15.974 | +0.055 | 77.79% | 78.09% | +0.29 pp | 72.99% | 72.81% | -0.18 pp |

### Aggregate paired effects

| Metric | Part-aware minus baseline | Uncertainty / paired test | Interpretation |
| --- | ---: | --- | --- |
| FID | +0.0358 | seed-level SD 0.0208 | Tiny regression; global safeguard passes |
| R-precision | +0.0489 pp | 95% CI [-0.0495, +0.1551] pp; McNemar p=0.354 | No reliable change |
| Part-colour accuracy | -0.0522 pp | 95% CI [-0.1488, +0.0537] pp; McNemar p=0.327 | Primary improvement criterion fails |
| Part-attention CE | -0.00547 | 95% CI [-0.00557, -0.00544] | Directly optimized proxy improves |
| Attention mass in target support | +0.00088 pp | 95% CI [+0.00077, +0.00114] pp | Positive but practically negligible |

Part-colour accuracy improved in only one of three seeds, and its confidence
interval crosses zero. This fails both primary targeted criteria. The small FID
increase remains far inside the allowed +1.0 boundary, and R-precision remains
inside the -1 percentage-point boundary, so the variant does not materially
damage the global safeguards. However, passing safeguards is not evidence of a
targeted improvement.

The most defensible interpretation is that the auxiliary loss successfully
changes its own attention proxy without transferring that change into a
measurably better part-colour relationship in generated images.

## Relationship to the Session 6 baseline result

The Session 6 evaluation remains the baseline-reproduction anchor: the official
generator evaluated through this modern code path reached FID 15.7576 and
DAMSM R-precision 76.67% +/- 0.83% on 30,000 fixed samples, close to the
author-reported pretrained references of 15.34 and 76.58% +/- 0.53%.

That result supports compatibility of the modern inference/evaluation path. It
does not by itself show local from-scratch convergence. Likewise, only the
paired Session 7 controls above support the conclusion about the proposed
improvement; the official pretrained score and the paired fine-tuning scores
must not be treated as interchangeable experiments.

## Reproduce the experiment

Prepare the environment as described in `README.md`. The following local assets
are required and intentionally excluded from Git:

- CUB-200-2011 images, metadata, captions, bounding boxes, and
  `parts/part_locs.txt` under `data/birds/`;
- `checkpoints/bird_DMGAN.pth`;
- `checkpoints/DAMSMencoders/bird/text_encoder200.pth`;
- `checkpoints/DAMSMencoders/bird/image_encoder200.pth`;
- `checkpoints/eval/bird_val.npz`.

Run the paired fine-tuning:

```bash
.venv/bin/python scripts/run_part_aware_ablation.py \
  --data-root data/birds \
  --generator-checkpoint checkpoints/bird_DMGAN.pth \
  --text-checkpoint checkpoints/DAMSMencoders/bird/text_encoder200.pth \
  --image-checkpoint checkpoints/DAMSMencoders/bird/image_encoder200.pth \
  --seeds 20260824 20260825 20260826 \
  --d-warmup-steps 200 --steps 1200 --batch-size 10 \
  --generator-lr 2e-5 --discriminator-lr 2e-4 \
  --part-lambda 0.05 --part-sigma-fraction 0.08 \
  --output-dir artifacts/session7/ablation --device cuda
```

Then evaluate the EMA weights with the fixed protocol:

```bash
.venv/bin/python scripts/evaluate_ablation.py \
  --experiment-dir artifacts/session7/ablation \
  --metadata-root data/birds \
  --text-checkpoint checkpoints/DAMSMencoders/bird/text_encoder200.pth \
  --image-checkpoint checkpoints/DAMSMencoders/bird/image_encoder200.pth \
  --fid-stats checkpoints/eval/bird_val.npz \
  --samples 30000 --batch-size 32 --seed 20260902 \
  --weights ema --device cuda
```

Run the lightweight verification separately:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check dmgan scripts tests
```

## Artifact and repository policy

The complete local experiment directory, `artifacts/session7/ablation/`, is
approximately 1.9 GB and is ignored by Git. It contains, for every seed, two
roughly 172 MB generator checkpoints and one roughly 304 MB shared
discriminator-start checkpoint. These checkpoints, official weights, datasets,
and generated caches must **not** be uploaded to the course repository.

Selected lightweight evidence is force-tracked despite the broad `artifacts/`
ignore rule: `report.json`, `PART_AWARE_ABLATION.md`, `training_manifest.json`,
`colour_swap_protocol.json`, each seed's `training_report.json`, the compressed
paired metric arrays, and both preview PNGs per seed. These files provide a
machine-readable audit trail in a fresh clone without publishing model weights.
The exact headline results are also embedded in this document so that the
committed conclusion does not depend on unavailable checkpoint files.

## Limitations and next steps

- Only three seeds and 1,200 fine-tuning steps per arm were tested.
- Training starts from official weights instead of retraining DM-GAN fully from
  scratch.
- A real-image landmark is only a proxy for a generated bird whose pose may be
  different.
- The part-colour scorer uses the same frozen DAMSM model family involved in the
  wider training/evaluation pipeline.
- The token-to-part mapping is rule-based and can misinterpret unusual captions.
- FID measures distributional quality, not anatomical attribute localization.
- No blinded human comparison or independent generated-image part detector was
  included.

A stronger follow-up would use an independent bird-part detector or blinded
human evaluation, evaluate longer training schedules, and tune the auxiliary
loss on a separate validation protocol before a final held-out comparison.
