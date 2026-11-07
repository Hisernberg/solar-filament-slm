#!/usr/bin/env python3
"""Test images -> submission.csv (the exact leaderboard pipeline).

    python scripts/predict.py --images <test_images> --weights weights/filanet.pt --post weights/postproc.json
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from filanet.data import load_image  # noqa: E402
from filanet.geometry import make_inputs  # noqa: E402
from filanet.infer import predict_maps  # noqa: E402
from filanet.metric import encode  # noqa: E402
from filanet.model import count_params, load_checkpoint  # noqa: E402
from filanet.postproc import component_features, label_components, rule_keep  # noqa: E402


def extract(pm, sm, disk, img, post, rescorer=None):
    lab, n = label_components(pm, post["t_lo"], post["t_hi"], post.get("close_px", 0))
    feats = component_features(lab, n, pm, sm, disk, make_inputs(img, disk)[0][1])
    if rescorer is not None:
        keep = rescorer.keep(feats)
    else:
        keep = rule_keep(feats, post.get("min_area", 100), post.get("min_meanp", 0.0), post.get("max_sin_lon"))
    return [(lab == i + 1) for i in np.flatnonzero(keep)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--weights", nargs="+", default=["weights/filanet.pt"])
    ap.add_argument("--post", default="weights/postproc.json")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--tile", type=int, default=0, help="0 = full image on GPU, 1024 tiles on CPU")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load_checkpoint(w, device)[0] for w in a.weights]
    print(f"device {device} | {len(models)} model(s) | {count_params(models[0]) / 1e6:.2f}M params each", flush=True)
    post = json.load(open(a.post))
    rescorer = None
    if post.get("kind") == "rescorer":
        from filanet.rescore import Rescorer
        rescorer = Rescorer(os.path.join(os.path.dirname(a.post), "rescorer.txt"))
    tile = a.tile or (None if device == "cuda" else 1024)
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(a.images) if f.lower().endswith((".jpeg", ".jpg", ".png")))
    exts = {os.path.splitext(f)[0]: f for f in os.listdir(a.images)}
    rows, t0, t_net = [], time.time(), 0.0
    for n, stem in enumerate(stems):
        img = load_image(os.path.join(a.images, exts[stem]))
        t1 = time.time()
        outs = [predict_maps(m, img, device=device, tta=a.tta, tile=tile) for m in models]
        if device == "cuda":
            torch.cuda.synchronize()
        t_net += time.time() - t1
        pm = np.round(np.mean([o[0] for o in outs], 0)).astype(np.uint8)
        sm = np.round(np.mean([o[1] for o in outs], 0)).astype(np.uint8)
        masks = extract(pm, sm, outs[0][2], img, post, rescorer)
        for k, m in enumerate(masks, 1):
            rows.append((f"{stem}_{k}", encode(m)["counts"].decode("ascii")))
        if (n + 1) % 20 == 0:
            print(f"  {n + 1}/{len(stems)} images ({time.time() - t0:.0f}s)", flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filament_id", "segmentation_rle"])
        w.writerows(rows)
    dt = time.time() - t0
    covered = len({r[0].rsplit("_", 1)[0] for r in rows})
    print(f"wrote {a.out}: {len(rows)} instances on {covered}/{len(stems)} images | total {dt:.0f}s "
          f"({dt / len(stems):.2f}s/img, network {t_net / len(stems):.3f}s/img)")
    print("md5", hashlib.md5(open(a.out, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
