# FilaNet: a 4.5 M-parameter solar filament instance segmenter

Entry for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(GONG H-alpha images, MAGFiLO ground truth, Panoptic Quality metric).

**Goal:** match the honest state of the art (~0.45 out-of-fold PQ, ~0.39 public) with about 10x
fewer parameters and 10x less compute than the 50 M-parameter ConvNeXt-Small U-Net baseline, because the
judging rubric scores end-to-end efficiency, fragmentation / over-merging and mask morphology as
well as PQ.

> Status: the code and pipeline are complete and tested on synthetic data; competition training is pending
> (see `docs/PLAN.md` for the experiment queue). Results tables are filled in as runs finish.

## What is new compared with the baseline

| Component | Baseline | FilaNet |
|---|---|---|
| Encoder | ConvNeXt-Small (50 M) | ConvNeXt-Atto (3.4 M), timm, ImageNet-pretrained |
| Thin-structure path | none (stride-4 patchify) | full-resolution **DetailStem** into the last decoder stages |
| Context | none | **Strip pooling** (row / column), suited to long thin filaments |
| Outputs | mask | mask + **spine** (MAGFiLO's hand-drawn spines, unused by most entries) |
| Topology loss | none | **Skeleton Recall** on annotated spines |
| Input | grey repeated x3 | disk-normalised intensity, limb-free local contrast, radial position r/R |
| Annotation protocol | post-hoc cut | loss ignores unannotated pixels beyond 70 deg longitude |
| Inference | 768 px tiles, cosine blend | single full-disk pass on GPU (no seams), tiled on CPU |
| Instances | fixed thresholds | hysteresis components + learned rescorer with the PQ-optimal keep rule |
| Folds | 1 of 5 trained | 4 day-grouped folds trained in parallel (one per GPU) -> OOF for every image |

**PQ-optimal keep rule.** Adding one predicted filament raises the PQ denominator by exactly 0.5
whether it matches (TP +1, FN -1) or not (FP +1). With TP probability p and expected IoU q it
raises PQ iff `p*q > PQ/2`. The rescorer regresses `E[IoU * 1{IoU>0.5}]` over annotators from
20 component features and keeps components above a threshold tuned out-of-fold.

## Repository map

```
filanet/geometry.py   disk fit, physics-aware input channels, 70-degree longitude band
filanet/data.py       COCO parsing, per-annotator mask + spine rendering, tile dataset, augmentation
filanet/model.py      FilaNet (encoder, DetailStem, strip pooling, scSE decoder, mask + spine heads)
filanet/losses.py     BCE + Dice, spine loss, Skeleton Recall, optional distillation
filanet/infer.py      full-frame / tiled inference, D4 TTA
filanet/postproc.py   hysteresis components, hole filling, component features, keep rules
filanet/evaluate.py   cached PQ evaluator (any keep rule = column selection), grid search
filanet/rescore.py    LightGBM component rescorer
filanet/metric.py     host PQ (verified equal to the organisers' notebook, tests/test_metric.py)
filanet/train.py      one fold / full-data training (bf16, EMA, cosine)
scripts/train_folds.py   one process per GPU (4 folds on 4x L4 at once)
scripts/tune_oof.py      tune post-processing + rescorer on out-of-fold maps
scripts/predict.py       test images -> submission.csv
scripts/benchmark.py     params / GMACs / latency table
kaggle/build_kernel.py   packs the repo into a single Kaggle script kernel
notebooks/pipeline.ipynb end-to-end reproduction from the stored weights
weights/                 filanet.pt (fp16, ~9 MB), postproc.json, rescorer.txt
```

## Reproduce the submission

```bash
pip install -r requirements.txt
python scripts/predict.py --images <MAGFiLO_1.0_Kaggle_2026/test/test_images> \
    --weights weights/filanet.pt --post weights/postproc.json --out submission.csv
```

## Train

```bash
# 4 folds in parallel on a 4-GPU machine (one fold per GPU), then tune post-processing on OOF maps
python scripts/train_folds.py --data-dir <root> --out-root runs --common "--epochs 40"
python scripts/tune_oof.py --data-dir <root> --runs runs/fold_0 runs/fold_1 runs/fold_2 runs/fold_3 --out weights
# final full-data model
python -m filanet.train --data-dir <root> --out-dir runs/full --fold -1 --epochs 40
```

## Data and acknowledgments

Ground truth: MAGFiLO v1.0 (Ahmadzadeh et al., Scientific Data 2024, doi:10.1038/s41597-024-03876-y),
CC BY-NC 4.0, used only as provided by the competition (train split). No external labels are used.
Imagery: GONG H-alpha network (NSO/AURA/NSF). Starting point: the tiled ConvNeXt U-Net baseline
(team Vera). Code: MIT.
