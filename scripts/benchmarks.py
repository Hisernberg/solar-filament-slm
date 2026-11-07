#!/usr/bin/env python3
"""Reference benchmarks on the competition training data (same day-grouped 5-fold split as training).

1. human     - inter-annotator agreement: each annotator's filaments scored as predictions against
               the other annotators of the same image (images with >= 2 annotators). This is the
               expert-vs-expert level under the competition PQ.
2. classical - training-free dark-feature detector: Gaussian-smoothed, limb-corrected local contrast
               -> soft score -> hysteresis components -> keep-rule grid. Tuned on folds 1-4, scored on fold 0.

    python scripts/benchmarks.py --data-dir <root> --out docs/benchmarks.json
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from filanet.data import day_folds, find_annotation_json, find_dir, fold_of, load_image, load_index, ann_rle  # noqa
from filanet.evaluate import GRID_KEEP, build_image, pq_select  # noqa
from filanet.geometry import make_inputs, radial_map  # noqa
from filanet.metric import build_gt_sets, iou_matrix, PQResult  # noqa
from filanet.postproc import rule_keep  # noqa

G = {}


def human(ann, files):
    img, anns, ids = load_index(ann)
    tp_iou, fp, fn, per_img = [], 0, 0, []
    for f in files:
        sets = ids[f]
        if len(sets) < 2:
            continue
        rles = {i: [ann_rle(a) for a in anns.get(i, [])] for i in sets}
        for pi in sets:
            for gi in sets:
                if gi == pi:
                    continue
                gt, pr = rles[gi], rles[pi]
                if not gt:
                    fp += len(pr); continue
                if not pr:
                    fn += len(gt); continue
                iou = iou_matrix(gt, pr)
                hit = iou > 0.5
                tp_iou += iou[hit].tolist()
                fp += int((hit.sum(0) == 0).sum()); fn += int((hit.sum(1) == 0).sum())
    tp = len(tp_iou)
    den = tp + 0.5 * fp + 0.5 * fn
    return dict(pq=sum(tp_iou) / den, sq=float(np.mean(tp_iou)), rq=tp / den, tp=tp, fp=fp, fn=fn,
                pairs="each annotator vs each other annotator of the same image")


def classical_map(f, k0, sigma):
    img = load_image(os.path.join(G["image_dir"], f))
    x, disk = make_inputs(img)
    loc = cv2.GaussianBlur(x[1], (0, 0), sigma)
    s = 1.0 / (1.0 + np.exp(-(-loc - k0) * 4.0))       # dark -> high score
    s[radial_map(img.shape, disk) > 0.97] = 0.0           # drop limb ring / sky
    return (s * 255).round().astype(np.uint8), disk, x[1]


def _job(args):
    f, k0, sigma, t_lo, t_hi = args
    pm, disk, loc = classical_map(f, k0, sigma)
    return f, build_image(pm, None, disk, G["gt"][f], t_lo, t_hi, 0, loc)


def classical(ann, files, folds, workers):
    G["gt"] = build_gt_sets(ann, files)
    tr = [f for f in files if fold_of(f, folds) != 0]
    va = [f for f in files if fold_of(f, folds) == 0]
    rng = np.random.default_rng(0)
    tr_s = list(rng.choice(tr, min(160, len(tr)), replace=False))
    best = None
    for k0 in [0.6, 0.9, 1.2]:
        for sigma in [1.5, 3.0]:
            for t_lo, t_hi in [(0.5, 0.8), (0.5, 0.95)]:
                with ProcessPoolExecutor(workers) as ex:
                    cache = dict(ex.map(_job, [(f, k0, sigma, t_lo, t_hi) for f in tr_s], chunksize=4))
                for kp in GRID_KEEP:
                    r = pq_select(cache, {f: rule_keep(ic.feats, **kp) for f, ic in cache.items()})
                    if best is None or r.pq > best[0].pq:
                        best = (r, dict(k0=k0, sigma=sigma, t_lo=t_lo, t_hi=t_hi, **kp))
                print("k0", k0, "sigma", sigma, t_lo, t_hi, "best so far", round(best[0].pq, 4), best[1], flush=True)
    p = best[1]
    with ProcessPoolExecutor(workers) as ex:
        cache = dict(ex.map(_job, [(f, p["k0"], p["sigma"], p["t_lo"], p["t_hi"]) for f in va], chunksize=4))
    keep = {f: rule_keep(ic.feats, p["min_area"], p["min_meanp"], p["max_sin_lon"]) for f, ic in cache.items()}
    r = pq_select(cache, keep)
    return dict(fold0_pq=r.pq, sq=r.sq, rq=r.rq, tp=r.tp, fp=r.fp, fn=r.fn, o2m=r.one_to_many, m2o=r.many_to_one,
                params=p, tuned_on=f"{len(tr_s)} images from folds 1-4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="docs/benchmarks.json")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    ann = find_annotation_json(a.data_dir)
    G["image_dir"] = find_dir(a.data_dir, "train_images")
    _, _, ids = load_index(ann)
    files = sorted(ids)
    folds = day_folds(files, 5)
    out = {}
    out["human_all"] = human(ann, files)
    out["human_fold0"] = human(ann, [f for f in files if fold_of(f, folds) == 0])
    print("human", out["human_all"], "\nfold0", out["human_fold0"], flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    out["classical"] = classical(ann, files, folds, a.workers)
    print("classical", out["classical"], flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
