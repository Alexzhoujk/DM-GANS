# Paired DM-GAN vs DM-GAN-MDD Checkpoint Evaluation

## Conclusion

**SUPPORTED** — DM-GAN-MDD lowers the FID point estimate and the image-cluster confidence interval supports R-precision non-inferiority within 1 percentage point.

This is a paired comparison: same caption index, latent z, deterministic CA means, negatives, and evaluators. Each generator
uses the conditioning text encoder specified by its published method. Both
outputs are judged by the same frozen original DAMSM evaluator.

## Fixed protocol

- Samples: 30,000
- Seed: 20260902
- Resolution: 256 x 256
- Caption schedule: deterministic cycling over the official CUB test caption bank
- FID: one shared PyTorch Inception evaluator and one shared real-statistics file
- R-precision: one shared original DAMSM evaluator; correct caption versus the
  same 99 other-class negatives for both generators
- Preview layout: baseline on the left, DM-GAN-MDD on the right

## Results

| Metric | DM-GAN | DM-GAN-MDD | candidate - baseline |
| --- | ---: | ---: | ---: |
| PyTorch FID ↓ | 17.1708 | 16.1543 | -1.0164 |
| R-precision ↑ | 79.10% ± 0.72% | 79.77% ± 0.80% | +0.67 pp |
| ImageNet IS ↑ | 5.7822 | 5.7347 | -0.0476 |

## Paired R-precision evidence

- Baseline-only correct: 3,488
- DM-GAN-MDD-only correct: 3,690
- Exact McNemar p-value: 0.0176652
- McNemar caveat: Descriptive only: caption rows from the same CUB image and repeated rows are correlated.
- CUB-image cluster bootstrap (2,933 image clusters)
  95% CI for the R-precision delta:
  [+0.134,
  +1.225] percentage points

The per-sample binary outcomes and sample/caption indices are saved in
`paired_samples.npz`, so the paired test can be independently reproduced.

## Checkpoint provenance

Every checkpoint and the FID real-statistics file are recorded with absolute
paths and SHA256 digests in `report.json`.

## Limitations

- A single checkpoint pair and one random seed do not measure training-run variance.
- FID is a distribution-level statistic; the paired design controls inputs but does not make FID a paired per-image test.
- R-precision depends on the chosen frozen original DAMSM evaluator and its 99 sampled negatives.
- The sample-level McNemar p-value is descriptive because captions from one image and repeated rows are correlated; the image-cluster bootstrap is the uncertainty result to use.
- ImageNet IS is an internal health check and is not comparable with the paper's legacy CUB bird-classifier IS.
- Caption indices repeat after the finite official test-caption bank is exhausted.
- Checkpoint evaluation tests saved model states; it does not independently reproduce the full training trajectory.
- Both generators use the modern correctly broadcast padding mask. This is a shared and controlled implementation, but it is not bit-exact with the author repositories' batch>1 repeat-based legacy mask layout.
- The candidate conditioner and R-precision text evaluator have identical original DAMSM weights. The evaluator is a separately loaded frozen module and is applied equally to both models, but it is not an encoder-family-independent judge of the candidate.
- Matched-mean conditioning removes CA sampling from both models; it is the controlled comparison for a mean-conditioned candidate, not the released DM-GAN baseline's default stochastic inference mode.
- The author-released MDD and DM-GAN checkpoints came from different training configurations; this checkpoint comparison validates released models, not an isolated retraining ablation of the MDD loss.
- Both generators run through the corrected modern batch-mask implementation; the released legacy implementation misassigns padding masks when batch items have different caption lengths.
