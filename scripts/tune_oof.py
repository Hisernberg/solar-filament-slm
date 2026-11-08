#!/usr/bin/env python3
"""Tune post-processing on out-of-fold maps (CPU): extraction grid, fixed keep-rule grid,
and the learned rescorer. Writes weights/postproc.json (+ weights/rescorer.txt).

    python scripts/tune_oof.py --data-dir <root> --runs runs/fold_0 runs/fold_1 ... --out weights
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from filanet.data import day_folds, find_annotation_json, find_dir, fold_of, load_image, stem_of  # noqa: E402
from filanet.evaluate import GRID_EXTRACT, GRID_KEEP, build_image, pq_select  # noqa: E402
from filanet.geometry import Disk, make_inputs  # noqa: E402
from filanet.metric import build_gt_sets  # noqa: E402
from filanet.postproc import rule_keep  # noqa: E402
from filanet import rescore  # noqa: E402

G = {}


def _build(args):
    f, ex = args
    pm = cv2.imread(G["maps"][f] + "_m.png", cv2.IMREAD_GRAYSCALE)
    sm = cv2.imread(G["maps"][f] + "_s.png", cv2.IMREAD_GRAYSCALE)
    disk = Disk(*G["disks"][f])
    local = make_inputs(load_image(os.path.join(G["image_dir"], f)), disk)[0][1]
    return f, build_image(pm, sm, disk, G["gt"][f], ex["t_lo"], ex["t_hi"], ex.get("close_px", 0), local,
                          grow_px=ex.get("grow_px", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="weights")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--oof-subdir", default="oof", help="oof or oof_tta4")
    ap.add_argument("--extract", default=None, help='JSON list of extraction settings to try instead of the default grid')
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    ann = find_annotation_json(a.data_dir)
    G["image_dir"] = find_dir(a.data_dir, "train_images")
    maps, disks = {}, {}
    for r in a.runs:
        od = os.path.join(r, a.oof_subdir)
        d = json.load(open(os.path.join(od, "disks.json")))
        for f, v in d.items():
            maps[f] = os.path.join(od, stem_of(f)); disks[f] = v
    files = sorted(maps)
    G["maps"], G["disks"], G["gt"] = maps, disks, build_gt_sets(ann, files)
    folds = day_folds(sorted({*files}), a.n_folds)
    fold_of_file = {f: fold_of(f, folds) for f in files}
    print(f"{len(files)} OOF images", flush=True)

    report, best = {"extract": []}, None
    caches = {}
    extra = [dict(t_lo=0.5, t_hi=0.9, close_px=2), dict(t_lo=0.5, t_hi=0.9, grow_px=1),
             dict(t_lo=0.5, t_hi=0.9, grow_px=2), dict(t_lo=0.5, t_hi=0.9, grow_px=-1)]
    grid = json.loads(a.extract) if a.extract else GRID_EXTRACT + extra
    for ex in grid:
        with ProcessPoolExecutor(a.workers) as pool:
            cache = dict(pool.map(_build, [(f, ex) for f in files], chunksize=4))
        key = json.dumps(ex, sort_keys=True)
        caches[key] = cache
        rows = []
        for kp in GRID_KEEP:
            r = pq_select(cache, {f: rule_keep(ic.feats, **kp) for f, ic in cache.items()})
            rows.append(({**ex, **kp}, r))
        b = max(rows, key=lambda t: t[1].pq)
        print(f"extract {ex}: best rule PQ {b[1]} with {b[0]}", flush=True)
        report["extract"].append(dict(rule=b[0], pq=b[1].pq, tp=b[1].tp, fp=b[1].fp, fn=b[1].fn, sq=b[1].sq))
        # learned rescorer on the same components
        scores, pred, y = rescore.fit_oof(cache, fold_of_file)
        (tau, rr), _ = rescore.sweep_tau(cache, scores)
        print(f"   rescorer: PQ {rr} at tau {tau:.3f}", flush=True)
        report["extract"][-1].update(rescorer_pq=rr.pq, rescorer_tau=tau)
        for kind, pq, extra in [("rule", b[1].pq, b[0]), ("rescorer", rr.pq, dict(**ex, tau=tau))]:
            if best is None or pq > best[1]:
                best = (kind, pq, extra, key)
    print("BEST", best, flush=True)
    os.makedirs(a.out, exist_ok=True)
    cfg = dict(kind=best[0], oof_pq=best[1], **best[2])
    if best[0] == "rescorer":
        rescore.fit_final(caches[best[3]], os.path.join(a.out, "rescorer.txt"), best[2]["tau"],
                          meta=dict(oof_pq=best[1]))
    json.dump(cfg, open(os.path.join(a.out, "postproc.json"), "w"), indent=1)
    json.dump(report, open(os.path.join(a.out, "oof_report.json"), "w"), indent=1, default=float)
    print("wrote", os.path.join(a.out, "postproc.json"), cfg)


if __name__ == "__main__":
    main()
