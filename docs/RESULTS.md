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
| **FilaNet-Atto v1** (31 epochs on T4) | **4.47 M** | **0.437** | 0.670 | 0.653 | 1012 | 401 | 675 |
| ConvNeXt-Small U-Net baseline (reported by its authors) | ~50 M | 0.450 | 0.684 | - | 1096 | 549 | 591 |

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

## Pending

* Run C (v2 limb-ring disk fit; medoid vs mixed multi-annotator targets; 4-view TTA), fold 0.
* Folds 3-4 and a full-data final model (GPU quota resets Nov 14).
