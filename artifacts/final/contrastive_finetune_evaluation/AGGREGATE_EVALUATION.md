# Multiple seeded paired checkpoint evaluations

## Conclusion

**MIXED ACROSS SEEDS** — Only 2 of 3 supplied seeded evaluations is supported.

Comparison: **Dual-caption control** vs **DM-GAN + CL2**.

## Per-seed results

| Seed | N | Baseline FID | Candidate FID | ΔFID | Baseline R | Candidate R | ΔR | Verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20260902 | 30,000 | 16.1010 | 16.1541 | +0.0531 | 78.40% | 78.45% | +0.04 pp | not supported |
| 20260903 | 30,000 | 15.8674 | 15.7770 | -0.0904 | 77.57% | 77.55% | -0.03 pp | supported |
| 20260904 | 30,000 | 15.8876 | 15.8098 | -0.0778 | 78.00% | 78.16% | +0.16 pp | supported |

## Across-seed summary (mean ± sample SD)

- Baseline FID: 15.9520 ± 0.1294
- Candidate FID: 15.9136 ± 0.2089
- Candidate-minus-baseline FID: -0.0383 ± 0.0795
- Baseline R-precision: 77.99 ± 0.42%
- Candidate R-precision: 78.05 ± 0.46%
- Candidate-minus-baseline R-precision: 0.06 ± 0.09 pp

## Decision rule

Every seed must use at least 30,000 samples, lower candidate FID, and have a 95% image-cluster bootstrap lower bound above -1 percentage point for the candidate-minus-baseline R-precision delta.

## Limitations

- The report seed controls evaluation randomness. Unless training seed metadata is separately recorded and evaluation streams are held fixed, across-run SD conflates checkpoint/training variation with caption truncation, negatives, z, and CA sampling.
- With three supplied runs, across-run uncertainty remains imprecise; individual paired R-precision confidence intervals remain primary evidence.
- FID deltas are distribution-level differences and do not have a per-image paired confidence interval.
