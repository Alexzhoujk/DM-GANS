# Paired DM-GAN vs DM-GAN-MDD Checkpoint Evaluation

## Conclusion

**NOT SUPPORTED** — DM-GAN-MDD does not lower the FID point estimate under this fixed paired protocol.

This is a paired comparison: same caption index, latent z, negatives, and evaluators; the sample-mode side uses the recorded deterministic CA RNG stream while the mean-mode side uses no CA epsilon. Each generator
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
| PyTorch FID ↓ | 15.7047 | 16.1543 | 0.4496 |
| R-precision ↑ | 76.84% ± 0.69% | 79.77% ± 0.80% | +2.93 pp |
| ImageNet IS ↑ | 5.6606 | 5.7347 | 0.0741 |

## Paired R-precision evidence

- Baseline-only correct: 3,402
- DM-GAN-MDD-only correct: 4,280
- Exact McNemar p-value: 1.28528e-23
- McNemar caveat: Descriptive only: caption rows from the same CUB image and repeated rows are correlated.
- CUB-image cluster bootstrap (2,933 image clusters)
  95% CI for the R-precision delta:
  [+2.375,
  +3.485] percentage points

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
- The native-mode comparison changes both checkpoint weights and the CA inference policy; use the matched-mean run for the controlled checkpoint comparison.
- This native-behavior sensitivity comparison intentionally uses stochastic CA for released DM-GAN and deterministic mean CA for released MDD, so its difference includes the published inference-mode change.
- The author-released MDD and DM-GAN checkpoints came from different training configurations; this checkpoint comparison validates released models, not an isolated retraining ablation of the MDD loss.
