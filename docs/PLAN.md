# FilaNet plan: smallest model, best judged result

## 1. What is actually being judged

| Item | Weight | Implication |
|---|---|---|
| Panoptic Quality (IoU>0.5 matching, pooled over every annotator set) | part of 70% | Instance-level: wrong *count* of filaments costs as much as bad outlines |
| Dice / IoU distributions of matched pairs | part of 70% | Outline quality matters (SQ) |
| One-to-many / many-to-one relations | part of 70% | Fragmentation and over-merging are penalised separately |
| End-to-end computational efficiency | named in the goal | **Small, fast models score here; ensembles lose** |
| Write-up, morphology of masks, code quality | 30% | Clean repo + notebook + 4-page report |

Public LB scores above ~0.45 are metric exploits (organisers' announcements of Aug 20 and Oct 23;
an empty 5-row file scored 1.00). Honest references:

| Entry | Params | OOF PQ | Public PQ |
|---|---|---|---|
| Two human experts vs each other | - | ~0.34 | - |
| ConvNeXt-Small U-Net (provided baseline, fold 0) | ~50 M | 0.450 | 0.38 |
| 10-model ResNet34 + EffNetV2-S ensemble (public notebook) | ~10 x 25 M | 0.450 | 0.39 |
| **FilaNet-Atto (this repo)** | **4.5 M** | target >= 0.45 | target >= 0.39 |

Goal: match or beat the honest frontier with ~10x fewer parameters and ~10x faster inference,
then spend the saved budget on what PQ actually rewards: deciding *which* components are filaments.

## 2. Architecture (FilaNet)

* **Tiny pretrained encoder** (timm): ConvNeXt-Atto (3.4 M) default; Femto / MobileNetV4-S / EfficientNet-B0
  as ablations. The baseline itself reports larger encoders did not help, which points to a
  label-noise ceiling, not a capacity ceiling.
* **Full-resolution DetailStem** (1/1 and 1/2) feeding the last decoder stages: patchify encoders
  start at stride 4, but barbs are 1-3 px wide.
* **Strip pooling** context at the bottleneck: row/column pooling = long, anisotropic receptive
  fields for long, thin filaments at negligible cost.
* **scSE-gated light decoder**, depthwise-separable at high resolution.
* **Two heads**: filament mask + filament *spine*. MAGFiLO ships hand-drawn spines; nearly every
  public entry ignores them. The spine head regularises topology and is a strong per-instance
  confidence feature.
* **Physics-aware inputs**: per-image disk fit; disk-normalised intensity; local contrast
  (removes limb darkening); radial position r/R. All D4-invariant, so D4 augmentation / TTA stay valid.
* **Single-pass full-disk inference** on GPU (no tiles, no seams).

## 3. Training

* Positive-centred 640 px tiles, D4 + photometric + seeing-blur + noise augmentation.
* Loss = BCE + soft Dice (mask) + 0.5 (BCE + Dice) on spine + 0.3 Skeleton-Recall
  (Kirchhoff et al., ECCV 2024; free here because spines are annotated).
* **Annotation-protocol-aware loss masking**: unannotated pixels beyond 70 deg longitude get zero
  weight (annotators were told to skip them; they are not negatives).
* Targets: medoid annotator (default), majority vote, random annotator, mixed.
* AdamW, warmup + cosine, EMA with warmup, bf16 on L4, channels-last.
* 4 day-grouped folds trained *in parallel* (one per L4) -> OOF maps for all 707 images in one run.

## 4. Post-processing (where PQ is won)

* Hysteresis components (extent from t_lo, existence from a t_hi seed), holes filled, masks disjoint.
* **PQ-optimal keep rule**: adding a component with TP-probability p and expected IoU q raises PQ
  iff p*q > PQ/2 (the denominator rises by 0.5 whether it hits or not). A LightGBM rescorer
  (~100 KB) regresses e = E[IoU * 1{IoU>0.5}] over annotators from 20 cheap features
  (confidence, spine evidence, shape, solar position, contrast) and keeps e > tau (tuned OOF).
* Fixed-rule grid (area / mean-prob / longitude) as the fallback; the better OOF option is shipped.

## 5. Experiment queue (GPU budget: 8 h until Nov 14, then 30 h)

| Run | GPUs | What | Decision it informs |
|---|---|---|---|
| A | 4x L4, ~2.5 h | FilaNet-Atto, 4 folds, medoid | CV PQ vs baseline; OOF maps for tuning |
| B | 4x L4, ~1.5 h | fold 0: Femto / mixed targets / no-spine / tile 896 | ablations for the report |
| C | 4x L4, ~1 h | full-data models (best config, 2 seeds) | final weights |
| D (opt.) | 4x L4 | distillation from a large teacher (ConvNeXt-Small, OOF maps as soft targets) | does KD close any gap? |

## 6. Ideas held in reserve

* Knowledge distillation from a stronger teacher's probability maps (`--teacher-dir --w-kd`).
* DINOv3-pretrained ConvNeXt-Tiny teacher (satellite/LVD pretraining) -> distil into Atto.
* Spine-guided merging of fragments (connect components whose spines align end-to-end).
* INT8 / fp16 export + `torch.compile` for the efficiency table.
