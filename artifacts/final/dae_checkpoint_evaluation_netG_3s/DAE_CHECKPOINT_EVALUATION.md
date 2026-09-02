# Paired DM-GAN vs DAE-GAN (corrected-mask checkpoint execution) Evaluation

## Conclusion

**NOT SUPPORTED** — DAE-GAN (corrected-mask checkpoint execution) does not lower the FID point estimate under the fixed paired protocol.

Both models receive the same deterministic CUB caption schedule, latent noise
and conditioning-augmentation draw. Both are judged by one separately loaded,
original frozen DAMSM evaluator. DAE-GAN additionally receives the official
preprocessed adjective/noun aspect phrases associated with each caption.

## Fixed protocol

- Samples: 30,000
- Resolution: 256 x 256
- Attention mask: corrected batch-major broadcast (default)
- FID: shared PyTorch Inception path and shared `bird_val.npz`
- R-precision: correct caption versus the same 99 other-class negatives
- Preview: DM-GAN left, DAE-GAN right

## Results

| Metric | DM-GAN | DAE-GAN (corrected-mask checkpoint execution) | DAE - baseline |
| --- | ---: | ---: | ---: |
| PyTorch FID ↓ | 15.6693 | 21.9564 | 6.2871 |
| R-precision ↑ | 76.43% ± 0.47% | 86.13% ± 0.66% | +9.70 pp |
| ImageNet IS ↑ | 5.6769 | 5.4204 | -0.2564 |

## Paired R-precision evidence

- Baseline-only correct: 2,105
- DAE-GAN-only correct: 5,015
- Exact McNemar p-value: 3.73358e-268
- CUB-image cluster-bootstrap 95% CI for DAE-minus-baseline:
  [+9.135,
  +10.240] percentage points

Per-sample outcomes and caption/image-cluster identifiers are stored in
`paired_samples.npz`. Exact input paths and SHA256 hashes are in `report.json`.

## Limitations

- DAE-GAN is an independent aspect-aware multi-stage architecture, not a small DM-GAN plugin.
- The corrected attention mask implements the apparent batch-major intent but differs from released batch>1 execution; use --legacy-attention-mask only as a sensitivity check.
- Aspect preprocessing deterministically keeps the first three phrases and first five tokens, while the released data loader randomly subsamples phrases longer than five tokens; this removes data-loader RNG but is not bit-exact for those phrases.
- Checkpoint evaluation does not reproduce 600-epoch training or training-run variance.
- FID is distribution-level; pairing controls inputs but does not create a per-image FID test.
- R-precision depends on the frozen original DAMSM evaluator and sampled negatives.
- ImageNet IS is not comparable with the paper's legacy CUB bird-classifier IS.
