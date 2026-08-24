# DM-GAN Baseline Reproduction Evaluation

## Conclusion

**PASS** — Comparable FID and DAMSM R-precision meet the predeclared bounds for a reasonable baseline reproduction.

This evaluation runs the author-released CUB DM-GAN and DAMSM weights through
the modern PyTorch reimplementation. It validates architecture/checkpoint
compatibility, generation, and the evaluation path. It does not claim that the
project's 200-step random-initialization checkpoint has converged.

## Fixed protocol

- Split: official CUB test metadata
- Generated samples: 30,000
- Seed: 20260824
- Resolution: 256 x 256
- R-precision: correct caption versus 99 captions from other classes, using the
  official DAMSM text/image encoders
- FID: repository-compatible PyTorch Inception features against the author's
  `bird_val.npz` statistics
- IS: torchvision ImageNet Inception-v3, 10 splits; reported only as a modern
  internal health check because the paper used a legacy 50-class TensorFlow bird
  classifier

## Results

| Metric | Modern reproduction | Official pretrained reference | Comparable? |
| --- | ---: | ---: | --- |
| PyTorch FID ↓ | 15.7576 | 15.34 | Yes |
| R-precision ↑ | 76.67% ± 0.83% | 76.58% ± 0.53% | Yes |
| ImageNet IS ↑ | 5.7007 ± 0.0940 | Not applicable | No — evaluator differs |
| Paper bird IS ↑ | Not run | 4.71 ± 0.06 | Legacy evaluator unavailable |

## Predeclared reasonableness checks

- PyTorch FID <= 22.00: **PASS**
- R-precision >= 70.00%: **PASS**
- Strict official generator/DAMSM checkpoint loading: **PASS**
- Three-scale generation and real-batch backward path: **PASS**

## Interpretation boundary

The comparable FID and R-precision values test whether the modern code can
faithfully execute the released baseline. The ImageNet IS value must not be
placed beside the paper's CUB IS as if they used the same classifier.

The next experiment is a fixed-budget comparison between this baseline and the
optional part-aware variant; that comparison is intentionally outside this
report.
