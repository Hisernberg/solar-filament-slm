"""Generate a tiny synthetic MAGFiLO-format dataset (for CI / pipeline smoke tests only)."""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np


def poly_from_mask(m):
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    c = np.vstack([c, c[:1]])
    return c.ravel().tolist()


def main(out, n_train=8, n_test=2, size=2048, seed=0):
    rng = np.random.default_rng(seed)
    root = os.path.join(out, "MAGFiLO_1.0_Kaggle_2026")
    tr_dir, te_dir = os.path.join(root, "train", "train_images"), os.path.join(root, "test", "test_images")
    os.makedirs(tr_dir, exist_ok=True); os.makedirs(te_dir, exist_ok=True)
    images, anns = [], []
    for i in range(n_train + n_test):
        name = f"2015{1 + i % 12:02d}{1 + i:02d}120000Bh.jpeg"
        img = np.zeros((size, size), np.float32)
        cx, cy, r = size / 2 + rng.normal(0, 10), size / 2 + rng.normal(0, 10), 900 + rng.normal(0, 20)
        yy, xx = np.mgrid[0:size, 0:size]
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
        disk = rr <= 1
        img[disk] = 150 * np.sqrt(np.clip(1 - 0.6 * rr[disk] ** 2, 0, 1))
        fil_masks, spines = [], []
        for _ in range(rng.integers(2, 6)):
            ang = rng.uniform(0, 2 * np.pi); rad = rng.uniform(0, 0.7) * r
            p0 = np.array([cx + rad * np.cos(ang), cy + rad * np.sin(ang)])
            pts = [p0]
            d = rng.uniform(0, 2 * np.pi)
            for _ in range(rng.integers(4, 10)):
                d += rng.normal(0, 0.4)
                pts.append(pts[-1] + 25 * np.array([np.cos(d), np.sin(d)]))
            pts = np.round(np.array(pts)).astype(np.int32)
            m = np.zeros((size, size), np.uint8)
            cv2.polylines(m, [pts], False, 1, thickness=int(rng.integers(6, 14)))
            fil_masks.append(m); spines.append(pts)
            img[m > 0] *= 0.55
        img += rng.normal(0, 4, img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        split_dir = tr_dir if i < n_train else te_dir
        cv2.imwrite(os.path.join(split_dir, name), img)
        if i >= n_train:
            continue
        for a in range(2):
            iid = f"0101{a:02d}-{name[:-5]}"
            images.append(dict(id=iid, width=size, height=size, file_name=name, license=1))
            for k, (m, sp) in enumerate(zip(fil_masks, spines)):
                if a == 1 and rng.random() < 0.25:
                    continue  # second annotator skips some filaments
                mm = cv2.dilate(m, np.ones((3, 3), np.uint8)) if a == 1 else m
                anns.append(dict(id=f"{iid}-{k}", image_id=iid, category_id=1, iscrowd=0,
                                 segmentation=[poly_from_mask(mm)], spine=sp.ravel().astype(float).tolist(),
                                 area=float(mm.sum()), bbox=list(cv2.boundingRect(mm))))
    json.dump(dict(images=images, annotations=anns, categories=[dict(id=1, name="Left")]),
              open(os.path.join(root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"), "w"))
    print("synthetic data at", root)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/synth")
