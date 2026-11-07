#!/usr/bin/env python3
"""Rubric-level evaluation of out-of-fold predictions with the shipped post-processing.

Reports PQ / SQ / RQ, matched IoU and Dice distributions, one-to-many and many-to-one relations,
and breakdowns by observatory, year, number of annotators and GT filament size. Writes
docs/oof_eval.json and docs/figures/*.png.

    python scripts/analyze_oof.py --data-dir <root> --runs runs/fold_0 ... --post weights/postproc.json
"""
import argparse
import collections
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from filanet.data import find_annotation_json, find_dir, load_image, stem_of  # noqa: E402
from filanet.evaluate import build_image, pq_select  # noqa: E402
from filanet.geometry import Disk, make_inputs  # noqa: E402
from filanet.metric import build_gt_sets  # noqa: E402
from filanet.postproc import rule_keep  # noqa: E402

G = {}


def _build(f):
    p = G["post"]
    pm = cv2.imread(G["maps"][f] + "_m.png", cv2.IMREAD_GRAYSCALE)
    sm = cv2.imread(G["maps"][f] + "_s.png", cv2.IMREAD_GRAYSCALE)
    disk = Disk(*G["disks"][f])
    local = make_inputs(load_image(os.path.join(G["image_dir"], f)), disk)[0][1]
    ic = build_image(pm, sm, disk, G["gt"][f], p["t_lo"], p["t_hi"], p.get("close_px", 0), local,
                     grow_px=p.get("grow_px", 0))
    return f, ic


def keep_of(ic, post, rescorer):
    if rescorer is not None:
        return rescorer.keep(ic.feats)
    return rule_keep(ic.feats, post.get("min_area", 100), post.get("min_meanp", 0.0), post.get("max_sin_lon"))


def summarize(r):
    iou = np.array(r.tp_iou)
    dice = 2 * iou / (1 + iou) if len(iou) else iou
    return dict(pq=r.pq, sq=r.sq, rq=r.rq, tp=r.tp, fp=r.fp, fn=r.fn, n_gt=r.n_gt,
                one_to_many=r.one_to_many, many_to_one=r.many_to_one,
                iou_mean=float(iou.mean()) if len(iou) else 0, iou_median=float(np.median(iou)) if len(iou) else 0,
                dice_mean=float(dice.mean()) if len(dice) else 0,
                dice_median=float(np.median(dice)) if len(dice) else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--post", default="weights/postproc.json")
    ap.add_argument("--oof-subdir", default="oof")
    ap.add_argument("--out", default="docs")
    ap.add_argument("--tag", default="filanet")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    ann = find_annotation_json(a.data_dir)
    G["image_dir"] = find_dir(a.data_dir, "train_images")
    maps, disks = {}, {}
    for r in a.runs:
        od = os.path.join(r, a.oof_subdir)
        for f, v in json.load(open(os.path.join(od, "disks.json"))).items():
            maps[f] = os.path.join(od, stem_of(f)); disks[f] = v
    files = sorted(maps)
    post = json.load(open(a.post))
    rescorer = None
    if post.get("kind") == "rescorer":
        from filanet.rescore import Rescorer
        rescorer = Rescorer(os.path.join(os.path.dirname(a.post), "rescorer.txt"))
    G.update(maps=maps, disks=disks, gt=build_gt_sets(ann, files), post=post)
    with ProcessPoolExecutor(a.workers) as ex:
        cache = dict(ex.map(_build, files, chunksize=4))
    keep = {f: keep_of(ic, post, rescorer) for f, ic in cache.items()}
    out = {"n_images": len(files), "post": post, "overall": summarize(pq_select(cache, keep))}
    print("overall", out["overall"], flush=True)

    groups = collections.defaultdict(dict)
    for f in files:
        s = stem_of(f)
        groups["site"].setdefault(s[14:16], []).append(f)
        groups["year"].setdefault(s[:4], []).append(f)
        groups["n_annotators"].setdefault(str(len(G["gt"][f])), []).append(f)
    for g, d in groups.items():
        out[g] = {k: summarize(pq_select({f: cache[f] for f in v}, {f: keep[f] for f in v}))
                  for k, v in sorted(d.items())}

    # recall by GT filament size (a GT counts as found if any kept component has IoU > 0.5)
    bins = [0, 500, 1000, 2000, 4000, 1e9]
    found, total = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
    for f, ic in cache.items():
        for gi, gt in enumerate(G["gt"][f]):
            if not gt:
                continue
            areas = mask_util.area(gt)
            hit = (ic.ious[gi][:, keep[f]] > 0.5).any(1) if keep[f].any() else np.zeros(len(gt), bool)
            b = np.digitize(areas, bins) - 1
            np.add.at(total, b, 1); np.add.at(found, b, hit.astype(float))
    out["recall_by_gt_area"] = {f"{int(bins[i])}-{int(bins[i + 1]) if bins[i + 1] < 1e9 else 'inf'}":
                                dict(recall=float(found[i] / max(total[i], 1)), n=int(total[i])) for i in range(len(total))}
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    json.dump(out, open(os.path.join(a.out, f"oof_eval_{a.tag}.json"), "w"), indent=1, default=float)

    r = pq_select(cache, keep)
    iou = np.array(r.tp_iou)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(iou, bins=np.linspace(0.5, 1, 26), color="#3b6ea5", edgecolor="white")
    ax[0].axvline(iou.mean(), color="#c0392b", ls="--", label=f"mean {iou.mean():.3f}")
    ax[0].set(title="Matched-pair IoU", xlabel="IoU", ylabel="pairs"); ax[0].legend()
    d = 2 * iou / (1 + iou)
    ax[1].hist(d, bins=np.linspace(0.66, 1, 26), color="#3b6ea5", edgecolor="white")
    ax[1].axvline(d.mean(), color="#c0392b", ls="--", label=f"mean {d.mean():.3f}")
    ax[1].set(title="Matched-pair Dice", xlabel="Dice"); ax[1].legend()
    keys = list(out["recall_by_gt_area"])
    ax[2].bar(keys, [out["recall_by_gt_area"][k]["recall"] for k in keys], color="#3b6ea5")
    ax[2].set(title="Recall by GT filament area (px)", ylim=(0, 1)); ax[2].tick_params(axis="x", rotation=30)
    fig.suptitle(f"{a.tag}: OOF PQ {r.pq:.4f} (SQ {r.sq:.3f}, RQ {r.rq:.3f}) on {len(files)} images")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "figures", f"oof_{a.tag}.png"), dpi=130)
    print("wrote", os.path.join(a.out, f"oof_eval_{a.tag}.json"))


if __name__ == "__main__":
    main()
