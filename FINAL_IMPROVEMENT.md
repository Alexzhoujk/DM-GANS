# Final Improvement Validation

## Conclusion

The selected published improvement is **DM-GAN-MDD** (Modality Disentangled
Discriminator). Under our controlled, matched-mean, 30,000-sample CUB checkpoint
protocol, its released generator outperformed the released DM-GAN generator on
both prespecified primary criteria:

- FID: **16.1543 vs 17.1708**, a reduction of 1.0164 points or 5.92%.
- R-precision: **79.7700% vs 79.0967%**, a gain of 0.6733 percentage points.
- CUB-image-cluster bootstrap 95% interval for the R-precision difference:
  **[+0.1339, +1.2252] percentage points**.

This supports a deliberately narrow claim: the released MDD checkpoint has an
advantage under a matched deterministic-mean conditioning policy. It does **not**
show that MDD is universally better, that the MDD loss alone caused the gain, or
that a part-aware generator is better.

The [original Session 7 part-aware report](SESSION7_COMPLETION.md) remains part
of the repository. That experiment did not demonstrate output-level superiority
and has not been replaced or re-labelled as a success.

## What DM-GAN-MDD changes

DM-GAN builds a low-resolution image and refines it with dynamic memory: writing
gates select useful words and response gates fuse retrieved word information
into image features. MDD retains a DM-GAN-compatible generator but changes the
training discriminator. Its modality-disentangled discriminator separates a
shared text-related content representation from image-specific style, with the
goal of improving both semantic correspondence and visual quality.

This is the closest evaluated published successor to the baseline, but it is a
**modality-disentangled discriminator**, not a part-aware or aspect-aware
generator.

## Fixed evaluation protocol

Every released-checkpoint comparison generated 30,000 images at 256 x 256 from
the official CUB test metadata with seed `20260902`. Within a paired comparison,
the models received the same caption schedule, latent noise, negative captions,
and frozen evaluators. When both models sampled conditioning augmentation (CA),
they also received the same CA random stream.

- FID uses the repository's shared PyTorch Inception feature path and the same
  author-provided `bird_val.npz` real statistics.
- R-precision uses one separately loaded, frozen original DAMSM evaluator and
  the same 99 other-class negative captions for both models.
- R-precision uncertainty is a 10,000-resample bootstrap clustered by the 2,933
  official CUB test images, because the 30,000-caption schedule cycles through
  a finite caption bank.
- ImageNet Inception Score is an internal health check only. It is not comparable
  with the papers' legacy CUB-specific bird-classifier IS and does not determine
  the verdict.

The prespecified success rule requires both:

1. candidate FID lower than the paired baseline FID; and
2. the lower endpoint of the 95% image-cluster bootstrap interval for
   candidate-minus-baseline R-precision greater than -1 percentage point.

Both DM-GAN and MDD use the same corrected batch-major padding-mask broadcast in
this repository. This removes a batch-indexing defect in the legacy repeat-based
implementation, but it is not bit-exact with released batch-greater-than-one
execution.

## Results

Do not compare baseline numbers across rows as if they came from one experiment.
Each row is a separate paired protocol with the controls stated in its label.

| Paired protocol, 30,000 CUB samples | Baseline FID ↓ | Candidate FID ↓ | Delta FID | Baseline R ↑ | Candidate R ↑ | Delta R (pp) | Cluster 95% CI for delta R (pp) | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DM-GAN vs MDD; **matched deterministic mean CA** | 17.1708 | **16.1543** | **-1.0164** | 79.0967% | **79.7700%** | **+0.6733** | **[+0.1339, +1.2252]** | **Supported** |
| DM-GAN vs MDD; each release's native policy | **15.7047** | 16.1543 | +0.4496 | 76.8433% | **79.7700%** | +2.9267 | [+2.3754, +3.4848] | Fails FID condition |
| DM-GAN vs official DM-GAN+CL; stochastic CA | 15.7047 | **14.6437** | -1.0611 | **76.8433%** | 71.2933% | -5.5500 | [-6.1644, -4.9332] | Fails R condition |
| DM-GAN vs DAE-GAN `netG_3s`; stochastic CA, corrected mask | **15.6693** | 21.9564 | +6.2871 | 76.4267% | **86.1267%** | +9.7000 | [+9.1348, +10.2398] | Fails FID condition |

Machine-readable evidence and paired samples:

- [MDD matched-mean report](artifacts/final/mdd_matched_mean_30k/report.json)
- [MDD native-policy report](artifacts/final/mdd_native_30k/report.json)
- [Official DM-GAN+CL report](artifacts/final/contrastive_checkpoint_evaluation/report.json)
- [DAE-GAN report](artifacts/final/dae_checkpoint_evaluation_netG_3s/report.json)
- [Local CL2 three-seed aggregate](artifacts/final/contrastive_finetune_evaluation/aggregate_report.json)

### Published context versus local evidence

The papers and official repositories report the following historical CUB
numbers. They motivate the candidates, but they must not be numerically merged
with our local table because checkpoints, conditioning policies, and metric
implementations differ.

| Published comparison | FID ↓ | R-precision ↑ | Legacy CUB IS ↑ |
| --- | ---: | ---: | ---: |
| Original DM-GAN paper | 16.09 | 72.31% ± 0.91% | 4.75 ± 0.07 |
| DM-GAN-MDD official repository | 15.76 | 79.73% ± 0.68% | 4.86 ± 0.06 |
| DM-GAN+CL paper, baseline → CL | 15.10 → 14.38 | 75.86% ± 0.83% → 78.99% ± 0.66% | 4.66 ± 0.06 → 4.77 ± 0.05 |
| DAE-GAN paper, DM-GAN → DAE-GAN | 16.09 → 15.19 | 72.31% ± 0.91% → 85.45% ± 0.57% | — |

The author-reported MDD direction agrees with our matched-mean result. Our CL
evaluation reproduced only the FID direction, and our DAE evaluation reproduced
only the R-precision direction. These discrepancies are why the final conclusion
is based on the fixed local protocol rather than on copied paper values.

### Why the native MDD result matters

The released DM-GAN sampler uses stochastic CA, while the released MDD sampler
uses the CA mean. Under those native policies, MDD improves R-precision by 2.93
points but worsens FID by 0.45. Therefore the final claim must always include the
phrase **matched deterministic-mean conditioning**. The sensitivity result shows
that the FID conclusion depends on inference policy.

### Other attempted improvements

**Official DM-GAN+CL.** Ye et al. add caption-caption contrastive learning (CL1)
and generated-image contrastive learning (CL2). The released checkpoint lowered
FID by 1.0611, but R-precision fell by 5.55 points under the independently loaded
original DAMSM judge. The FID gain alone is not an overall win.

**DAE-GAN.** This aspect-aware architecture adds sentence-, word-, and phrase-level
conditioning. It raised R-precision by 9.70 points, but FID worsened by 6.2871
(40.12%). This is direct evidence that the evaluated aspect-aware method improves
semantic retrieval but does not answer “is part-aware better overall?” in the
affirmative.

**Local CL2-only fine-tuning.** A dual-caption control and an otherwise identical
CL2 branch were fine-tuned for 1,200 steps at each of three seeds. Mean CL2-minus-
control changes were -0.0383 FID and +0.0578 R-precision points. Two seeds passed
the per-seed non-inferiority rule, but one reversed the FID direction; the effects
were tiny relative to across-seed dispersion. The correct verdict is **mixed
across seeds**, not robust improvement. This short CL2-only study is also not a
reproduction of the published 800-epoch CL1+CL2 training procedure.

| Local seed | Delta FID, CL2 - control ↓ | Delta R (pp) ↑ | Cluster 95% CI for delta R (pp) | Per-seed verdict |
| --- | ---: | ---: | ---: | --- |
| `20260902` | +0.0531 | +0.0433 | [-0.1167, +0.2008] | Not supported |
| `20260903` | -0.0904 | -0.0267 | [-0.2097, +0.1527] | Supported by FID plus R non-inferiority |
| `20260904` | -0.0778 | +0.1567 | [-0.0167, +0.3361] | Supported by FID plus R non-inferiority |

## Asset provenance and identity

Large datasets and checkpoints are intentionally excluded from Git. Download
them from the linked official project releases, place them at the paths below,
and verify their exact SHA-256 identities before evaluating.

| Local asset | SHA-256 | Source |
| --- | --- | --- |
| `checkpoints/bird_DMGAN.pth` | `444c9e43da1314fec6f6823eb995547312a9f7ecd2d50bf39d383fe0593bab55` | [Official DM-GAN repository](https://github.com/MinfengZhu/DM-GAN) / repository download helper |
| `checkpoints/DAMSMencoders/bird/text_encoder200.pth` | `d7278a9b4801633eb42d43de67cba51746daf5143be532660bf1c15608690c01` | [Official bird DAMSM release](https://drive.google.com/open?id=1GNUKjVeyWYBJ8hEU-yrfYQpDOkxEyP3V) |
| `checkpoints/DAMSMencoders/bird/image_encoder200.pth` | `459e25c9f9842c5754002cba15addaf66c7c618aa5bb87dea968b0ac3e3b2d5b` | [Official bird DAMSM release](https://drive.google.com/open?id=1GNUKjVeyWYBJ8hEU-yrfYQpDOkxEyP3V) |
| `checkpoints/eval/bird_val.npz` | `6ef8414fe0ad80ee87ff52e9d1142869b5a9952c34c4a5d518a060a7fa3a4c47` | [Official bird FID statistics](https://drive.google.com/file/d/1747il5vnY2zNkmQ1x_8hySx537ZAJEtj/view) |
| `checkpoints/mdd/bird/netG_DMGANMDD_bird.pth` | `e905baaffb54d2247f27146c7810aacd9bb707916d2163d90d62f3aa55bb98e1` | [Official DM-GAN-MDD bird checkpoint](https://drive.google.com/file/d/1TnKf-SKG06VxkvmpUe2anEIbOD5QsAva/view) |
| `checkpoints/contrastive/bird/netG_epoch_700.pth` | `8691d5e5061e0bc298b9cb8fd92431be67d6f3de6ac977b5d7842b6e7cfe4feb` | [Official DM-GAN+CL bird release](https://drive.google.com/file/d/1QIBMz3OSPGKe5W8_dlNTcaETivVPlUtf/view) |
| `checkpoints/contrastive/bird/text_encoder200.pth` | `2f2bc7d83754d499d3325d966806beb20f2a224acfbfbd836489f308a10dd2f6` | [Official DAMSM+CL bird release](https://drive.google.com/file/d/15w_mKV7UzmC3jMqplKyMawUEEJaJozTZ/view) |
| `checkpoints/dae/bird/netG_3s.pth` | `324251dcc350a9ec398487fceeb66a66bc01ee81e740fe6793aedd83c2b05286` | [Official DAE-GAN bird release](https://drive.google.com/drive/folders/1FzPOULU1Z5q3EcGm7m9w-Fl21QJmtcnm) |
| `data/dae/bird/captions_with_aspects.pickle` | `1005b0785ede13aa84b5faef645275c88896f717a9db7223b2c98ef81a8bd7c6` | [Official DAE-GAN bird metadata](https://drive.google.com/file/d/1KxbK71kgDKyDQKDeMOpPAgobIDNpt3-P/view) |

The local CL2 checkpoint files were produced by the training command below; the
evaluator loaded their EMA generator states. Their complete-file identities are:

| Seed and branch | SHA-256 |
| --- | --- |
| `20260902` dual-caption control | `07723de7bd321b2658325799ed96b543e9071f3865d7d442a727483fba247147` |
| `20260902` CL2 | `d96cc97a63ed2fdab0e0cde7d9bb9441d9b1a392bafada19b6b6061983986599` |
| `20260903` dual-caption control | `fe52bad09d941b13b5c97dae9faa40a6f7af5fe3eae2d5d2173eee03f3e132b5` |
| `20260903` CL2 | `b8b70022ded4cc59d480ee0011fddef62a7ba7bd06e44b931854b810f97eb54d` |
| `20260904` dual-caption control | `fbd1be581072f02691e39ee597fdfe5048dd2558b5c87a001943e98275a8aef6` |
| `20260904` CL2 | `33cb62c170c0c85c201daf4d315ef77a3303d6644fe2ab87344a10ef84f13965` |

Verify the files in one command:

```bash
sha256sum \
  checkpoints/bird_DMGAN.pth \
  checkpoints/DAMSMencoders/bird/text_encoder200.pth \
  checkpoints/DAMSMencoders/bird/image_encoder200.pth \
  checkpoints/eval/bird_val.npz \
  checkpoints/mdd/bird/netG_DMGANMDD_bird.pth \
  checkpoints/contrastive/bird/netG_epoch_700.pth \
  checkpoints/contrastive/bird/text_encoder200.pth \
  checkpoints/dae/bird/netG_3s.pth \
  data/dae/bird/captions_with_aspects.pickle
```

The baseline metadata and generator can also be downloaded with
`scripts/prepare_official_assets.py`, as documented in [README.md](README.md).
CUB images must be obtained from the official CUB-200-2011 source and placed at
`data/birds/CUB_200_2011/`.

## Reproduce the released-checkpoint comparisons

Prepare the Python/CUDA environment and baseline assets as described in
[README.md](README.md), then run the commands from the repository root.

### Primary MDD matched-mean comparison

```bash
.venv/bin/python scripts/evaluate_successor_checkpoint.py \
  --candidate-name DM-GAN-MDD \
  --candidate-generator-checkpoint checkpoints/mdd/bird/netG_DMGANMDD_bird.pth \
  --candidate-text-checkpoint checkpoints/DAMSMencoders/bird/text_encoder200.pth \
  --allow-shared-conditioning-evaluator \
  --baseline-conditioning-mode mean \
  --candidate-conditioning-mode mean \
  --samples 30000 --batch-size 32 --seed 20260902 --device cuda \
  --output-dir artifacts/final/mdd_matched_mean_30k \
  --extra-limitation "The author-released MDD and DM-GAN checkpoints came from different training configurations; this checkpoint comparison validates released models, not an isolated retraining ablation of the MDD loss."
```

### MDD native-policy sensitivity

```bash
.venv/bin/python scripts/evaluate_successor_checkpoint.py \
  --candidate-name DM-GAN-MDD \
  --candidate-generator-checkpoint checkpoints/mdd/bird/netG_DMGANMDD_bird.pth \
  --candidate-text-checkpoint checkpoints/DAMSMencoders/bird/text_encoder200.pth \
  --allow-shared-conditioning-evaluator \
  --baseline-conditioning-mode sample \
  --candidate-conditioning-mode mean \
  --samples 30000 --batch-size 32 --seed 20260902 --device cuda \
  --output-dir artifacts/final/mdd_native_30k \
  --extra-limitation "This native-behavior sensitivity comparison intentionally uses stochastic CA for released DM-GAN and deterministic mean CA for released MDD, so its difference includes the published inference-mode change."
```

### Official DM-GAN+CL checkpoint

```bash
.venv/bin/python scripts/evaluate_contrastive_checkpoint.py \
  --cl-generator-checkpoint checkpoints/contrastive/bird/netG_epoch_700.pth \
  --cl-text-checkpoint checkpoints/contrastive/bird/text_encoder200.pth \
  --samples 30000 --batch-size 32 --seed 20260902 --device cuda \
  --output-dir artifacts/final/contrastive_checkpoint_evaluation
```

The evaluator intentionally judges both outputs with the original DAMSM rather
than using the CL conditioning encoder as its own judge.

### DAE-GAN aspect-aware checkpoint

```bash
.venv/bin/python scripts/evaluate_dae_checkpoint.py \
  --dae-generator-checkpoint checkpoints/dae/bird/netG_3s.pth \
  --dae-metadata data/dae/bird/captions_with_aspects.pickle \
  --samples 30000 --batch-size 24 --seed 20260902 --device cuda \
  --output-dir artifacts/final/dae_checkpoint_evaluation_netG_3s
```

The default uses the corrected batch-major attention mask. The
`--legacy-attention-mask` flag exists only for sensitivity testing and should not
be mixed into the main comparison.

## Reproduce the local CL2 ablation

The two branches start from the same released DM-GAN generator, receive the same
dual-caption training budget, and train only the 128- and 256-pixel refinement
and output blocks. The candidate differs by adding `0.2 * NT-Xent` on frozen
DAMSM codes from the two generated 256-pixel images.

```bash
.venv/bin/python scripts/run_contrastive_ablation.py \
  --data-root data/birds \
  --generator-checkpoint checkpoints/bird_DMGAN.pth \
  --text-checkpoint checkpoints/DAMSMencoders/bird/text_encoder200.pth \
  --image-checkpoint checkpoints/DAMSMencoders/bird/image_encoder200.pth \
  --seeds 20260902 20260903 20260904 \
  --d-warmup-steps 200 --steps 1200 --batch-size 10 \
  --generator-lr 2e-5 --discriminator-lr 2e-4 \
  --contrastive-lambda 0.2 --temperature 0.5 --log-every 100 \
  --preview-count 8 --device cuda \
  --output-dir artifacts/final/contrastive_finetune_ablation
```

Evaluate each seed's EMA weights:

```bash
for seed in 20260902 20260903 20260904; do
  .venv/bin/python scripts/evaluate_successor_checkpoint.py \
    --baseline-name "Dual-caption control" \
    --candidate-name "DM-GAN + CL2" \
    --baseline-generator-checkpoint \
      "artifacts/final/contrastive_finetune_ablation/seed_${seed}/dual_caption_control_final.pt" \
    --candidate-generator-checkpoint \
      "artifacts/final/contrastive_finetune_ablation/seed_${seed}/contrastive_final.pt" \
    --baseline-generator-format modern-ema \
    --candidate-generator-format modern-ema \
    --candidate-text-checkpoint checkpoints/DAMSMencoders/bird/text_encoder200.pth \
    --allow-shared-conditioning-evaluator \
    --samples 30000 --batch-size 32 --seed "${seed}" --device cuda \
    --output-dir \
      "artifacts/final/contrastive_finetune_evaluation/seed_${seed}"
done

.venv/bin/python scripts/aggregate_paired_evaluations.py \
  artifacts/final/contrastive_finetune_evaluation/seed_20260902/report.json \
  artifacts/final/contrastive_finetune_evaluation/seed_20260903/report.json \
  artifacts/final/contrastive_finetune_evaluation/seed_20260904/report.json \
  --output-dir artifacts/final/contrastive_finetune_evaluation
```

Local checkpoint hashes are recorded in each per-seed `report.json`; the exact
set is also summarized in the aggregate evidence. The training checkpoints are
large and are not intended for Git.

## Verification

The final implementation is covered by **51 passing tests**, including official
checkpoint loading, deterministic mean conditioning, DAE checkpoint/mask
compatibility, paired evaluator logic, contrastive loss, paired-caption training,
and multi-seed aggregation. Ruff reports no errors.

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check dmgan scripts tests
```

## Limitations

- Each released-method row evaluates one checkpoint pair and one generation
  seed; it does not estimate retraining variance.
- The MDD and DM-GAN release checkpoints came from different author training
  configurations, so this validates saved models rather than isolating the MDD
  discriminator as a causal intervention.
- FID is distribution-level. Shared inputs improve comparability but do not make
  it a paired per-image statistic, and no FID confidence interval was computed.
- R-precision depends on the original DAMSM weight family and the sampled 99
  negative captions. The clustered interval handles repeated CUB source images,
  not every possible source of uncertainty.
- MDD's conclusion changes with CA inference policy: matched mean passes the
  joint rule, while native behavior fails the FID condition.
- DAE-GAN is architecturally distinct and receives preprocessed aspect phrases,
  so it is a method-level comparison rather than a one-component DM-GAN ablation.
- The three-seed CL2 run is short fine-tuning, not full published-method training.
- Internal ImageNet IS must not be compared with paper-reported CUB-specific IS.

## Primary sources

1. Zhu, M., Pan, P., Chen, W., and Yang, Y. (2019). “DM-GAN: Dynamic
   Memory Generative Adversarial Networks for Text-to-Image Synthesis.” CVPR.
   [Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_DM-GAN_Dynamic_Memory_Generative_Adversarial_Networks_for_Text-To-Image_Synthesis_CVPR_2019_paper.html);
   [official code](https://github.com/MinfengZhu/DM-GAN).
2. Feng, F., Niu, T., Li, R., and Wang, X. (2022). “Modality Disentangled
   Discriminator for Text-to-Image Synthesis.” *IEEE Transactions on Multimedia*,
   24, 2112-2124. [DOI](https://doi.org/10.1109/TMM.2021.3075997);
   [official code](https://github.com/FangxiangFeng/DM-GAN-MDD).
3. Ye, H., Yang, X., Takac, M., Sunderraman, R., and Ji, S. (2021).
   “Improving Text-to-Image Synthesis Using Contrastive Learning.”
   [Paper](https://arxiv.org/abs/2107.02423);
   [official code](https://github.com/huiyegit/T2I_CL).
4. Ruan, S., Zhang, Y., Zhang, K., Fan, Y., Tang, F., Liu, Q., and Chen, E.
   (2021). “DAE-GAN: Dynamic Aspect-Aware GAN for Text-to-Image Synthesis.”
   ICCV, 13960-13969. [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Ruan_DAE-GAN_Dynamic_Aspect-Aware_GAN_for_Text-to-Image_Synthesis_ICCV_2021_paper.html);
   [official code](https://github.com/hiarsal/DAE-GAN).
5. Xu, T. et al. (2018). “AttnGAN: Fine-Grained Text to Image Generation with
   Attentional Generative Adversarial Networks.” CVPR.
   [Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.html).
6. Heusel, M. et al. (2017). “GANs Trained by a Two Time-Scale Update Rule
   Converge to a Local Nash Equilibrium.” NeurIPS.
   [Paper](https://proceedings.neurips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html).
