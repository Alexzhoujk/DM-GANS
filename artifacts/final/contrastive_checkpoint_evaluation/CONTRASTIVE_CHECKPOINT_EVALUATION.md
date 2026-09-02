# Paired DM-GAN vs DM-GAN+CL Checkpoint Evaluation

## Conclusion

**MIXED** — DM-GAN+CL lowers the FID point estimate, but the image-cluster confidence interval does not establish R-precision non-inferiority within 1 percentage point.

This is a paired comparison: each row uses the same CUB test caption, latent
noise `z`, and conditioning-augmentation random draw. DM-GAN uses the original
DAMSM text encoder for conditioning; DM-GAN+CL uses its separate CL-trained text
encoder. Both outputs are judged by the same independently loaded original
DAMSM evaluator.

## Fixed protocol

- Samples: 30,000
- Seed: 20260902
- Resolution: 256 x 256
- Caption schedule: deterministic cycling over the official CUB test caption bank
- FID: one shared PyTorch Inception evaluator and one shared real-statistics file
- R-precision: one shared original DAMSM evaluator; correct caption versus the
  same 99 other-class negatives for both generators
- Preview layout: baseline on the left, DM-GAN+CL on the right

## Results

| Metric | DM-GAN | DM-GAN+CL | CL - baseline |
| --- | ---: | ---: | ---: |
| PyTorch FID ↓ | 15.7047 | 14.6437 | -1.0611 |
| R-precision ↑ | 76.84% ± 0.69% | 71.29% ± 0.93% | -5.55 pp |
| ImageNet IS ↑ | 5.6606 | 5.5345 | -0.1261 |

## Paired R-precision evidence

- Baseline-only correct: 5,222
- DM-GAN+CL-only correct: 3,557
- Exact McNemar p-value: 5.58251e-71
- McNemar caveat: Descriptive only: caption rows from the same CUB image and repeated rows are correlated.
- CUB-image cluster bootstrap (2,933 image clusters)
  95% CI for the R-precision delta:
  [-6.164,
  -4.933] percentage points

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
- Checkpoint evaluation tests released models; it does not by itself verify that this repository can reproduce their training.
