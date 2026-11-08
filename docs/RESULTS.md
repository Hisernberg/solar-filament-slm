# Results (living document)

All numbers use the competition metric (Panoptic Quality, IoU > 0.5 matching, pooled over every
annotator set of every image), reproduced exactly from the organisers' self-evaluation notebook
(`tests/test_metric.py`). Splits are day-grouped 5-fold, identical to the provided baseline's
(`filanet.data.day_folds`); fold 0 has exactly the baseline's 1,687 GT instances.

## Held-out fold 0 (post-processing fitted on folds 1-2 only, then applied unchanged)

| Method | Params | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| Classical dark-feature detector (training-free, tuned on folds 1-4) | 0 | 0.188 | 0.659 | 0.285 | 440 | 961 | 1247 |
| Human expert vs other experts (same images) | - | 0.334 | 0.639 | 0.522 | - | - | - |
| FilaNet, DINOv3 ConvNeXt-Tiny encoder (14 epochs, undertrained) | 30.2 M | 0.430 | 0.669 | 0.643 | 1047 | 522 | 640 |
| FilaNet-Atto v1 (31 epochs on T4) | 4.47 M | 0.437 | 0.670 | 0.653 | 1012 | 401 | 675 |
| FilaNet-Atto v2, limb-ring disk fit, medoid targets | 4.47 M | 0.434 | 0.670 | 0.648 | 1003 | 408 | 684 |
| **FilaNet-Atto v2, mixed multi-annotator targets** | **4.47 M** | **0.442** | 0.669 | 0.661 | 1021 | 380 | 666 |
| ConvNeXt-Small U-Net baseline (reported by its authors) | ~50 M | 0.450 | 0.684 | - | 1096 | 549 | 591 |

### Test-time augmentation (fold 0, 4 D4 views)

| Variant | fixed rule PQ | SQ | rescorer PQ (rescorer fitted on 1-view maps) |
|---|---|---|---|
| v2 medoid, 1 view | 0.401 | 0.665 | 0.434 |
| v2 medoid, 4 views | **0.416** | **0.668** | 0.425 (miscalibrated) |
| v2 mixed, 1 view | 0.408 | 0.663 | 0.442 |
| v2 mixed, 4 views | **0.418** | **0.666** | 0.430 (miscalibrated) |

TTA improves the probability maps (+0.010 to +0.015 PQ under fixed rules, +0.008 SQ after rescoring),
but a rescorer trained on single-view maps under-keeps on smoother TTA maps (TP 1003 -> 897).
A TTA-aware rescorer needs TTA OOF maps on the fitting folds: the final round dumps them for all folds.

## Out-of-fold, folds 0-2 (421 images, 4,898 GT instances), FilaNet-Atto v1

Post-processing: hysteresis components (t_lo 0.6, seed 0.95), 1 px erosion, learned rescorer
(tau 0.23). Rescorer scores are out-of-fold (refit per CV fold).

| Post-processing | PQ | SQ | RQ | FP | FN | 1:n | n:1 |
|---|---|---|---|---|---|---|---|
| Best fixed rule (area / mean-prob / longitude grid) | 0.402 | 0.662 | 0.607 | 2675 | 1596 | 190 | 191 |
| + learned rescorer (PQ-optimal keep rule) | 0.432 | 0.667 | 0.649 | 1418 | 1867 | 101 | 172 |
| + 1 px boundary erosion, t_lo 0.6 | **0.438** | 0.670 | 0.653 | 1313 | 1884 | 94 | 167 |
| + 1 px dilation instead (rejected) | 0.378 | 0.648 | 0.583 | | | | |
| + 2 px erosion instead (rejected) | 0.382 | 0.640 | 0.596 | | | | |

Matched pairs: IoU mean 0.670 / median 0.674, Dice mean 0.800 / median 0.805.
Fragmentation: 94 one-to-many (1.9% of GT), 167 many-to-one (3.4%).

Breakdowns (`docs/oof_eval_filanet_atto_v1.json`, figure `docs/figures/oof_filanet_atto_v1.png`):

* by observatory: Bh 0.441, Ch 0.427, Lh 0.409, Mh 0.432, Th 0.459, Uh 0.457
* by number of annotators: 1 -> 0.443, 2 -> 0.451, 3 -> 0.425
* recall by GT area: <500 px 0.38, 500-1k 0.58, 1-2k 0.68, 2-4k 0.70, >4k 0.69

## Findings so far

1. The learned rescorer is the largest single gain (+0.030 PQ) and halves false positives; its
   tuned threshold (0.23-0.26) matches the derived PQ-optimal value PQ/2 ~ 0.21.
2. FilaNet's masks are slightly too wide: 1 px erosion helps, any dilation hurts.
3. A 6.7x larger DINOv3 foundation encoder did not beat the 4.5 M model at this compute budget.
4. Small filaments (<500 px) are the main recall gap; annotators also disagree most there.
5. Training was compute-bound (T4, 31 epochs, val PQ still rising) - longer training is the most
   direct lever on the remaining 0.013 gap to the 50 M baseline.

6. The limb-ring disk fit (correct geometry) is PQ-neutral (-0.003, within noise).
7. Mixed multi-annotator targets (medoid half the time, a random annotator otherwise) give +0.009.

## Test-set inference (v1 weights, CPU, 4 cores)

180 test images -> 1,125 filaments on 177 images (provided baseline: 1,116 on 174); all masks decode,
are non-empty and pixel-disjoint; median predicted area 1,200 px (GT median 1,228 px);
13.4 s/image end to end on a 4-core CPU (12.0 s network, tiled 1024 px).

## Next (final round, `kaggle/launch_final.sh`, after the Nov 14 quota reset)

v2 + mixed targets, ~2x longer training, TTA OOF maps for every fold:
D = folds 1, 2; E = folds 3, 4; F = fold 0 + full-data final model.
