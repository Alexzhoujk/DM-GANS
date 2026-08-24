# Session 6 qualitative baseline investigation

Evidence source: author-released `bird_DMGAN.pth`, fixed seed `20260824`, and the
16 captions listed in `artifacts/session6/official_pretrained_16/report.json`.
The grid is `official_checkpoint_grid_256.png`, read left-to-right and top-to-bottom.
These are **official checkpoint inference results**, not locally trained results.

## Observations

| Grid item | Caption focus | Observation | Error category |
|---|---|---|---|
| 4 | red/white, stubby beak, red eye rings | Strong red body, but the requested red eye-ring detail is not clearly localized. | fine part/attribute |
| 5 | yellow body, black head | Yellow/olive body is clear; black is weak and not confined to the head. | right attribute, wrong/weak region |
| 7 | yellow crown, black eye ring | Yellow is expressed globally; the crown/eye-ring distinction is weak. | part localization |
| 12 | red wings, yellow belly | Red/yellow color cues appear, but wing-vs-belly placement is ambiguous. | part-color placement |
| 16 | red/white, small curved beak | Color is plausible; beak curvature is not reliably represented. | local geometry |

## What this does and does not show

- It supports the hypothesis that the baseline often captures global colors more
  reliably than small part-specific relationships.
- It does not establish a metric improvement or a general failure rate; the set is
  small and hand-inspected.
- Count, multi-object, and spatial-relation claims cannot be tested on these bird-only
  captions and should not be presented as local CUB evidence.

## Proposed controlled improvement

Use CUB's 15 part coordinates to create differentiable Gaussian part maps. For a
caption token mapped to a part, normalize the token's dynamic-memory attention over
space and minimize cross-entropy to the corresponding part map:

```text
L_part = - mean_(active tokens) sum_(h,w) P_part(h,w) log A_word(h,w)
L_total = L_DM-GAN + lambda_part * L_part
```

The baseline remains unchanged by default. The experimental implementation is in
`dmgan/part_aware.py`; it should only be enabled in a separately labeled variant.

Controlled comparison:

- fixed CUB split, DAMSM encoders, seed set, optimizer, and training budget;
- baseline vs. baseline + part-aware alignment;
- report FID, R-precision, and part-attribute localization accuracy;
- reject the variant if localization improves but FID materially regresses.
