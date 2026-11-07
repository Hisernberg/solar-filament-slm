"""Fast keep-rule search: components + IoU matrices are computed once per (image, extraction
setting); any keep rule is then just a column selection. Used for validation during training,
threshold tuning on out-of-fold maps, and training the component rescorer."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .geometry import Disk
from .metric import IOU_T, PQResult, encode, iou_matrix
from .postproc import component_features, label_components


@dataclass
class ImageComps:
    feats: dict          # feature name -> (n,)
    ious: list           # per annotator set: (n_gt, n) IoU matrix
    n: int
    lab: np.ndarray | None = None


def build_image(prob_u8, spine_u8, disk: Disk, gt_sets_img, t_lo, t_hi, close_px=0, local=None,
                keep_lab=False) -> ImageComps:
    lab, n = label_components(prob_u8, t_lo, t_hi, close_px)
    feats = component_features(lab, n, prob_u8, spine_u8, disk, local)
    rles = [encode(lab == i) for i in range(1, n + 1)]
    ious = [iou_matrix(gt, rles) for gt in gt_sets_img]
    return ImageComps(feats, ious, n, lab if keep_lab else None)


def pq_select(cache: dict[str, ImageComps], keep: dict[str, np.ndarray]) -> PQResult:
    tp_iou, fp, fn, o2m, m2o, n_gt = [], 0, 0, 0, 0, 0
    for f, ic in cache.items():
        k = keep.get(f, np.zeros(ic.n, bool))
        npred = int(k.sum())
        for iou in ic.ious:
            ng = iou.shape[0]
            n_gt += ng
            if ng == 0:
                fp += npred; continue
            if npred == 0:
                fn += ng; continue
            sub = iou[:, k]
            hit = sub > IOU_T
            tp_iou.extend(sub[hit].tolist())
            fp += int((hit.sum(0) == 0).sum())
            fn += int((hit.sum(1) == 0).sum())
            ov = sub > 0
            o2m += int((ov.sum(1) > 1).sum()); m2o += int((ov.sum(0) > 1).sum())
    tp = len(tp_iou)
    den = tp + 0.5 * fp + 0.5 * fn
    return PQResult(sum(tp_iou) / den if den else 0.0, tp, fp, fn, float(np.mean(tp_iou)) if tp else 0.0,
                    tp / den if den else 0.0, tp_iou, o2m, m2o, n_gt)


def component_targets(ic: ImageComps) -> tuple[np.ndarray, np.ndarray]:
    """Per component: e = mean over annotator sets of (IoU if IoU>0.5 else 0); hit-rate."""
    if ic.n == 0:
        return np.zeros(0), np.zeros(0)
    es, hs = [], []
    for iou in ic.ious:
        if iou.shape[0] == 0:
            es.append(np.zeros(ic.n)); hs.append(np.zeros(ic.n)); continue
        best = iou.max(0)
        es.append(np.where(best > IOU_T, best, 0.0)); hs.append((best > IOU_T).astype(float))
    return np.mean(es, 0), np.mean(hs, 0)


GRID_EXTRACT = [dict(t_lo=a, t_hi=b) for a, b in [(0.4, 0.85), (0.5, 0.9), (0.5, 0.95), (0.6, 0.95)]]
GRID_KEEP = [dict(min_area=a, min_meanp=m, max_sin_lon=s)
             for a, m, s in itertools.product([50, 100, 200, 400], [0.0, 0.6, 0.7, 0.8, 0.85, 0.9],
                                              [None, 0.96])]


def grid_search(maps: dict[str, tuple], gt_sets: dict, locals_: dict[str, np.ndarray] | None = None,
                extract_grid=GRID_EXTRACT, keep_grid=GRID_KEEP, verbose=False):
    """maps: file -> (prob_u8, spine_u8, disk); locals_: file -> local-contrast map (optional).
    Returns (best PQResult, best rule, all results)."""
    from .postproc import rule_keep
    results = []
    for ex in extract_grid:
        cache = {}
        for f, (pm, sm, disk) in maps.items():
            local = locals_.get(f) if locals_ else None
            cache[f] = build_image(pm, sm, disk, gt_sets[f], ex["t_lo"], ex["t_hi"], local=local)
        for kp in keep_grid:
            keep = {f: rule_keep(ic.feats, **kp) for f, ic in cache.items()}
            r = pq_select(cache, keep)
            results.append(({**ex, **kp}, r))
            if verbose:
                print({**ex, **kp}, r)
    best = max(results, key=lambda t: t[1].pq)
    return best[1], best[0], results
