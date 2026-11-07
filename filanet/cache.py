"""One-off preprocessing into memory-mapped uint8 arrays, so data loading is a slice + augment
(Kaggle GPU boxes have 4 CPU cores; per-image decoding/feature extraction would starve the GPUs).

<cache>/inputs.u8   [N, 2, H, W]  quantised channels 0/1 of make_inputs (q = clip(v*32 + 128))
<cache>/targets.u8  [M, H, W]     one map per (image, annotator): bit0 mask, bit1 spine, bit2 limb band
<cache>/meta.json   files, disks, annotator ranges, medoid index
<cache>/pos.npz     per image: up to 4096 sampled filament pixel coordinates (union of annotators)
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from .data import H, W, load_image, load_index, medoid_index, render_annotator
from .geometry import longitude_ignore_mask, make_inputs

QS, QZ = 32.0, 128.0

_G = {}


def quantise(v):
    return np.clip(np.round(v * QS + QZ), 0, 255).astype(np.uint8)


def dequantise(q):
    return (q.astype(np.float32) - QZ) / QS


def _job(args):
    i, f, a0, ann_lists = args
    img = load_image(os.path.join(_G["image_dir"], f))
    x, disk = make_inputs(img)
    inp = np.memmap(_G["inp"], np.uint8, "r+", shape=(_G["N"], 2, H, W))
    inp[i] = quantise(x[:2])
    inp.flush(); del inp
    band = longitude_ignore_mask((H, W), disk) << 2
    maps = [render_annotator(al, H, W) | band for al in ann_lists]
    tg = np.memmap(_G["tgt"], np.uint8, "r+", shape=(_G["M"], H, W))
    for k, m in enumerate(maps):
        tg[a0 + k] = m
    tg.flush(); del tg
    union = np.zeros((H, W), bool)
    for m in maps:
        union |= (m & 1).astype(bool)
    ys, xs = np.nonzero(union)
    rng = np.random.default_rng(i)
    sel = rng.choice(len(ys), min(4096, len(ys)), replace=False) if len(ys) else np.zeros(0, int)
    return i, [disk.cx, disk.cy, disk.r], medoid_index(maps), np.stack([ys[sel], xs[sel]], 1).astype(np.int16)


def build_cache(ann_path: str, image_dir: str, out_dir: str, workers: int = 4) -> str:
    meta_p = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_p):
        return out_dir
    os.makedirs(out_dir, exist_ok=True)
    img, anns, ids_by_file = load_index(ann_path)
    files = sorted(f for f in ids_by_file if os.path.exists(os.path.join(image_dir, f)))
    starts, M = [], 0
    for f in files:
        starts.append(M); M += len(ids_by_file[f])
    N = len(files)
    _G.update(image_dir=image_dir, inp=os.path.join(out_dir, "inputs.u8"), tgt=os.path.join(out_dir, "targets.u8"),
              N=N, M=M)
    np.memmap(_G["inp"], np.uint8, "w+", shape=(N, 2, H, W)).flush()
    np.memmap(_G["tgt"], np.uint8, "w+", shape=(M, H, W)).flush()
    jobs = [(i, f, starts[i], [anns.get(iid, []) for iid in ids_by_file[f]]) for i, f in enumerate(files)]
    disks, medoids, pos = [None] * N, [0] * N, [None] * N
    with ProcessPoolExecutor(workers) as ex:
        for i, d, med, p in ex.map(_job, jobs, chunksize=2):
            disks[i], medoids[i], pos[i] = d, med, p
    np.savez(os.path.join(out_dir, "pos.npz"), *pos)
    json.dump(dict(files=files, disks=disks, starts=starts, counts=[len(ids_by_file[f]) for f in files],
                   medoid=medoids, N=N, M=M, H=H, W=W), open(meta_p, "w"))
    return out_dir


class Cache:
    def __init__(self, out_dir: str):
        m = json.load(open(os.path.join(out_dir, "meta.json")))
        self.meta = m
        self.files = m["files"]
        self.index = {f: i for i, f in enumerate(self.files)}
        self.dir = out_dir
        z = np.load(os.path.join(out_dir, "pos.npz"))
        self.pos = [z[f"arr_{i}"] for i in range(m["N"])]
        self._inp = self._tgt = None

    @property
    def inp(self):
        if self._inp is None:
            self._inp = np.memmap(os.path.join(self.dir, "inputs.u8"), np.uint8, "r",
                                  shape=(self.meta["N"], 2, self.meta["H"], self.meta["W"]))
        return self._inp

    @property
    def tgt(self):
        if self._tgt is None:
            self._tgt = np.memmap(os.path.join(self.dir, "targets.u8"), np.uint8, "r",
                                  shape=(self.meta["M"], self.meta["H"], self.meta["W"]))
        return self._tgt

    def __getstate__(self):  # memmaps re-open lazily in each DataLoader worker
        s = dict(self.__dict__); s["_inp"] = s["_tgt"] = None
        return s
