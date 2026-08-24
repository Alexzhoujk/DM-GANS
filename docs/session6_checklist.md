# Session 6 evidence checklist

## A — Dataset

- [x] CUB image and caption load together.
- [x] Train/test split and class ID verified.
- [x] Bounding-box crop and `[-1, 1]` normalization verified.
- [x] One real batch and its caption report saved.

## B — DAMSM

- [x] Vocabulary and checkpoint sizes match (5,450 words).
- [x] Word embedding shape `[B,256,T]` recorded.
- [x] Sentence embedding shape `[B,256]` recorded.
- [x] Encoder is in eval mode with gradients disabled.

## C — DM-GAN

- [x] 64/128/256 generator shapes pass.
- [x] D64/D128/D256 shapes and spectral normalization pass.
- [x] Masked words receive zero attention.
- [x] Attention sums to one over words.
- [x] Writing/response gates stay in `[0,1]`.

## D — Integration

- [x] Official pretrained inference result labeled as official.
- [x] Local forward/loss/backward result labeled as reimplementation.
- [x] One optimizer step and checkpoint save pass.
- [x] Fixed-caption 200-step short-run samples saved.
- [x] Basic loss log saved.
- [x] Strict official generator/DAMSM checkpoint loading verified.
- [x] Fixed 30,000-sample CUB evaluation completed.
- [x] Comparable PyTorch FID recorded: 15.7576 vs 15.34 reference.
- [x] Comparable DAMSM R-precision recorded: 76.67% ± 0.83% vs 76.58% ± 0.53% reference.
- [x] ImageNet IS labeled as non-paper-comparable.

## Investigation and presentation

- [x] At least three locally executed official-checkpoint failure cases classified with provenance.
- [x] Attention/writing-gate evidence included for one caption.
- [x] Part-aware improvement has inputs, loss, hypothesis, implementation, and controlled test.
- [x] Baseline reasonableness verdict documented as PASS with predeclared bounds.
- [x] Improvement efficacy explicitly deferred to the next controlled ablation.
- [ ] Ten-minute rehearsal finishes within 9:30–10:00.
- [x] Paper/repository/local results use visibly different labels.
