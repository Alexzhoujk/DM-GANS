# Session 7 Part-Aware Ablation

## Answer

**NOT DEMONSTRATED** - The auxiliary loss improves its attention proxy, but the output-level part-colour evidence is insufficient to claim that the method is better.

The formal comparison is the equal-budget baseline-control versus the
part-aware branch. Both start from the same author-released generator and the
same warmed-up discriminators; the only objective difference is
`lambda_part = 0.05`.

## Fixed protocol

- Paired training seeds: 20260824, 20260825, 20260826
- Fine-tuning steps per arm: 1,200
- Discriminator warm-up: 200 steps per seed
- Evaluation: 30,000 fixed CUB test samples,
  identical captions, latent-noise sequence, and negative captions
- Trainable generator scope: both refinement stages and their 128/256 image heads
- Primary targeted diagnostic: correct colour versus nine colour-swapped captions
  for captions that explicitly connect a colour and a bird part
- Global safeguards: repository-compatible FID and official DAMSM R-precision

## Per-seed results

| Seed | Base FID | Part FID | Delta | Base R | Part R | Delta | Base part-colour | Part part-colour | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260824 | 15.948 | 15.962 | +0.014 | 78.05% | 77.98% | -0.07 pp | 71.95% | 71.99% | +0.04 pp |
| 20260825 | 16.296 | 16.335 | +0.039 | 78.40% | 78.33% | -0.08 pp | 72.52% | 72.50% | -0.02 pp |
| 20260826 | 15.919 | 15.974 | +0.055 | 77.79% | 78.09% | +0.29 pp | 72.99% | 72.81% | -0.18 pp |

## Aggregate paired effects

- Mean FID delta (part - baseline): +0.036
- R-precision delta: +0.05 pp,
  cluster-bootstrap 95% CI [-0.05,
  +0.16]
- Part-colour swap accuracy delta: -0.05 pp,
  cluster-bootstrap 95% CI [-0.15,
  +0.05]
- Part-attention CE delta: -0.005 (lower is better)
- Attention mass in the annotated 10% support: +0.00 pp

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
