"""From probability maps to filament instances.

1. Hysteresis components: 8-connected regions of p > t_lo that contain a seed p >= t_hi
   (extent from the low threshold, confidence from the high one).
2. Holes filled (MAGFiLO polygons are hole-free).
3. Per-component features (shape, confidence, spine evidence, solar position).
4. A keep rule: either fixed thresholds or a learned rescorer (see rescore.py).

PQ-optimal keep rule (derivation used by the rescorer): adding one prediction with
probability p of being a TP (IoU>0.5) at expected IoU q changes the numerator by p*q and the
denominator by +0.5 *whatever happens* (TP: +1 TP, -1 FN; miss: +1 FP). So it raises PQ iff
    p*q > PQ / 2.
The rescorer regresses e = E[IoU * 1{IoU>0.5}] (averaged over annotators) and keeps e > tau,
with tau tuned on out-of-fold data (theory: tau ~ PQ/2 ~ 0.2).
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy import ndimage as ndi

from .geometry import Disk, SIN70

S8 = np.ones((3, 3), bool)

FEATURES = ["log_area", "meanp", "maxp", "p50", "frac_hi", "spine_mean", "spine_max", "spine_frac",
            "elong", "major", "minor", "fill", "r", "abs_x", "abs_y", "sin_lon", "contrast",
            "n_comp_img", "rank_meanp", "dist_nn"]


def grow_labels(lab: np.ndarray, g: float) -> np.ndarray:
    """Boundary calibration: g > 0 grows every instance by g px into background (nearest-instance
    assignment, so masks stay disjoint); g < 0 erodes each instance by |g| px."""
    if g == 0 or lab.max() == 0:
        return lab
    if g > 0:
        dist, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
        out = lab[iy, ix]
        out[dist > g] = 0
        return out.astype(lab.dtype)
    inner = ndi.distance_transform_edt(lab > 0)
    out = lab.copy()
    out[inner <= -g] = 0
    return out


def label_components(prob_u8: np.ndarray, t_lo: float, t_hi: float, close_px: int = 0, grow_px: float = 0):
    lo = prob_u8 > int(round(t_lo * 255))
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
        lo = cv2.morphologyEx(lo.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    lab, n = ndi.label(lo, structure=S8)
    if n == 0:
        return lab, 0
    pmax = ndi.maximum(prob_u8, lab, index=np.arange(1, n + 1))
    keep = np.flatnonzero(np.asarray(pmax) >= int(round(t_hi * 255))) + 1
    remap = np.zeros(n + 1, np.int32)
    remap[keep] = np.arange(1, len(keep) + 1)
    lab = remap[lab]
    # fill holes component by component (only background pixels can be claimed)
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        sub = lab[sl]
        filled = ndi.binary_fill_holes(sub == i)
        sub[filled & (sub == 0)] = i
    if grow_px:
        lab = grow_labels(lab, grow_px)
    return lab, len(keep)


def component_features(lab: np.ndarray, n: int, prob_u8: np.ndarray, spine_u8: np.ndarray | None,
                       disk: Disk, local: np.ndarray | None = None) -> dict[str, np.ndarray]:
    if n == 0:
        return {k: np.zeros(0) for k in FEATURES}
    idx = np.arange(1, n + 1)
    flat = lab.ravel()
    nz = np.flatnonzero(flat)
    lv = flat[nz]
    area = np.bincount(lv, minlength=n + 1)[1:].astype(np.float64)
    p = prob_u8.ravel()[nz].astype(np.float64) / 255.0
    meanp = np.bincount(lv, p, n + 1)[1:] / area
    maxp = np.asarray(ndi.maximum(prob_u8, lab, idx), float) / 255.0
    p50 = np.asarray(ndi.median(prob_u8, lab, idx), float) / 255.0
    p90 = np.bincount(lv, (p >= 0.9).astype(float), n + 1)[1:] / area  # share of confident pixels
    if spine_u8 is not None:
        s = spine_u8.ravel()[nz].astype(np.float64) / 255.0
        spine_mean = np.bincount(lv, s, n + 1)[1:] / area
        spine_max = np.asarray(ndi.maximum(spine_u8, lab, idx), float) / 255.0
        spine_frac = np.bincount(lv, (s > 0.5).astype(float), n + 1)[1:] / area
    else:
        spine_mean = spine_max = spine_frac = np.zeros(n)
    yy, xx = np.divmod(nz, lab.shape[1])
    yy = yy.astype(np.float64); xx = xx.astype(np.float64)
    cy = np.bincount(lv, yy, n + 1)[1:] / area
    cx = np.bincount(lv, xx, n + 1)[1:] / area
    vyy = np.bincount(lv, yy * yy, n + 1)[1:] / area - cy ** 2
    vxx = np.bincount(lv, xx * xx, n + 1)[1:] / area - cx ** 2
    vxy = np.bincount(lv, xx * yy, n + 1)[1:] / area - cx * cy
    tr, det = vxx + vyy, vxx * vyy - vxy ** 2
    disc = np.sqrt(np.clip(tr ** 2 / 4 - det, 0, None))
    l1, l2 = tr / 2 + disc, np.clip(tr / 2 - disc, 1e-6, None)
    major, minor = 4 * np.sqrt(l1), 4 * np.sqrt(l2)
    fill = area / np.clip(np.pi * major * minor / 4, 1, None)
    dx, dy = (cx - disk.cx) / disk.r, (cy - disk.cy) / disk.r
    r = np.sqrt(dx ** 2 + dy ** 2)
    sin_lon = np.abs(dx) / np.sqrt(np.clip(1 - dy ** 2, 1e-6, None))
    if local is not None:
        contrast = np.bincount(lv, local.ravel()[nz].astype(np.float64), n + 1)[1:] / area
    else:
        contrast = np.zeros(n)
    order = np.argsort(np.argsort(-meanp))
    pts = np.stack([cx, cy], 1)
    if n > 1:
        from scipy.spatial import cKDTree
        dist_nn = cKDTree(pts).query(pts, k=2)[0][:, 1] / disk.r
    else:
        dist_nn = np.full(n, 2.0)
    return dict(log_area=np.log(area), meanp=meanp, maxp=maxp, p50=p50, frac_hi=p90, spine_mean=spine_mean,
                spine_max=spine_max, spine_frac=spine_frac, elong=np.sqrt(l1 / l2), major=major, minor=minor,
                fill=fill, r=r, abs_x=np.abs(dx), abs_y=np.abs(dy), sin_lon=sin_lon, contrast=contrast,
                n_comp_img=np.full(n, float(n)), rank_meanp=order.astype(float), dist_nn=dist_nn)


def rule_keep(feat: dict[str, np.ndarray], min_area: int = 100, min_meanp: float = 0.0,
              max_sin_lon: float | None = None) -> np.ndarray:
    k = (np.exp(feat["log_area"]) >= min_area) & (feat["meanp"] >= min_meanp)
    if max_sin_lon is not None:
        k &= feat["sin_lon"] <= max_sin_lon
    return k


def instance_masks(lab: np.ndarray, keep_ids) -> list[np.ndarray]:
    return [(lab == i).astype(np.uint8) for i in keep_ids]


DEFAULT_RULE = dict(t_lo=0.5, t_hi=0.9, min_area=100, min_meanp=0.7, max_sin_lon=SIN70 + 0.02)
