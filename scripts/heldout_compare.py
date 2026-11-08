#!/usr/bin/env python3
"""Honest held-out comparison on one fold: post-processing (fixed-rule grid winner and rescorer)
is fitted on the OTHER folds' OOF maps only, then applied unchanged to the held-out fold for each
candidate model.

    python scripts/heldout_compare.py --data-dir <root> --fit-runs runs/fold_1 runs/fold_2 \
        --eval name1=runs/fold_0 name2=runs/teacher_fold_0 --post '{"t_lo":0.6,"t_hi":0.95,"grow_px":-1}'
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from filanet.data import find_annotation_json, find_dir, load_image, stem_of  # noqa: E402
from filanet.evaluate import GRID_KEEP, build_image, pq_select  # noqa: E402
from filanet.geometry import Disk, make_inputs  # noqa: E402
from filanet.metric import build_gt_sets  # noqa: E402
from filanet.postproc import FEATURES, rule_keep  # noqa: E402
from filanet import rescore  # noqa: E402

G = {}


def _build(args):
    f, base = args
    ex = G["ex"]
    pm = cv2.imread(base + "_m.png", cv2.IMREAD_GRAYSCALE)
    sm = cv2.imread(base + "_s.png", cv2.IMREAD_GRAYSCALE)
    disk = Disk(*G["disks"][base])
    local = make_inputs(load_image(os.path.join(G["image_dir"], f)), disk)[0][1]
    return f, build_image(pm, sm, disk, G["gt"][f], ex["t_lo"], ex["t_hi"], ex.get("close_px", 0), local,
                          grow_px=ex.get("grow_px", 0))


def load_runs(runs, sub):
    items = []
    for r in runs:
        od = os.path.join(r, sub)
        for f, v in json.load(open(os.path.join(od, "disks.json"))).items():
            base = os.path.join(od, stem_of(f))
            G["disks"][base] = v
            items.append((f, base))
    return items


def build(items, workers):
    with ProcessPoolExecutor(workers) as ex:
        return dict(ex.map(_build, items, chunksize=4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--fit-runs", nargs="+", required=True)
    ap.add_argument("--eval", nargs="+", required=True, help="name=run_dir[:oof_subdir]")
    ap.add_argument("--post", default='{"t_lo":0.6,"t_hi":0.95,"grow_px":-1}')
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    ann = find_annotation_json(a.data_dir)
    G.update(image_dir=find_dir(a.data_dir, "train_images"), disks={}, ex=json.loads(a.post))
    fit_items = load_runs(a.fit_runs, "oof")
    evals = []
    for e in a.eval:
        name, spec = e.split("=", 1)
        run, sub = (spec.split(":") + ["oof"])[:2]
        evals.append((name, load_runs([run], sub)))
    G["gt"] = build_gt_sets(ann, sorted({f for f, _ in fit_items} | {f for _, it in evals for f, _ in it}))
    fit = build(fit_items, a.workers)
    # fixed rule chosen on the fit folds
    best_rule = max(GRID_KEEP, key=lambda kp: pq_select(fit, {f: rule_keep(ic.feats, **kp) for f, ic in fit.items()}).pq)
    # rescorer trained on the fit folds, tau tuned there
    import lightgbm as lgb
    X, y, _, _ = rescore.table(fit)
    booster = lgb.train(rescore.lgb_params(), lgb.Dataset(X, y), 400)
    in_scores = {f: booster.predict(np.stack([ic.feats[k] for k in FEATURES], 1)) if ic.n else np.zeros(0)
                 for f, ic in fit.items()}
    (tau, _), _ = rescore.sweep_tau(fit, in_scores)
    # in-sample scores are optimistic -> also report the theory value tau = PQ/2 of the rule
    out = dict(post=G["ex"], rule=best_rule, tau=tau, models={})
    for name, items in evals:
        c = build(items, a.workers)
        r_rule = pq_select(c, {f: rule_keep(ic.feats, **best_rule) for f, ic in c.items()})
        sc = {f: booster.predict(np.stack([ic.feats[k] for k in FEATURES], 1)) if ic.n else np.zeros(0)
              for f, ic in c.items()}
        r_res = pq_select(c, {f: sc[f] > tau for f in c})
        out["models"][name] = dict(images=len(c), rule=str(r_rule), rescorer=str(r_res),
                                   rule_pq=r_rule.pq, rescorer_pq=r_res.pq, sq=r_res.sq, rq=r_res.rq,
                                   tp=r_res.tp, fp=r_res.fp, fn=r_res.fn)
        print(f"{name:28s} rule  {r_rule}\n{'':28s} rescr {r_res}", flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
