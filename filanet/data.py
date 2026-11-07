"""MAGFiLO COCO parsing, per-annotator target rendering and the tile dataset.

Targets are rendered once per (image, annotator) to small PNGs:
    <cache>/<stem>__<k>.png   uint8, bit0 = filament mask, bit1 = spine (drawn polyline, 5 px)
so every target policy (medoid / majority vote / random annotator) is available at train time
without re-rasterising polygons.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_util
from torch.utils.data import Dataset

from .geometry import fit_disk, longitude_ignore_mask, make_inputs

H = W = 2048
SPINE_THICKNESS = 5


# ----------------------------------------------------------------------------- COCO index
def find_annotation_json(data_dir: str) -> str:
    hits = [p for p in glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True)
            if "Annotations" in os.path.basename(p)]
    assert hits, f"MAGFiLO annotations json not found under {data_dir}"
    return sorted(hits)[0]


def find_dir(data_dir: str, name: str) -> str:
    hits = [d for d in glob.glob(os.path.join(data_dir, "**", name), recursive=True) if os.path.isdir(d)]
    assert hits, f"{name}/ not found under {data_dir}"
    return sorted(hits)[0]


def load_index(ann_path: str):
    """-> (images by id, annotations by image id, annotator image-ids by file name)."""
    d = json.load(open(ann_path))
    img = {im["id"]: im for im in d["images"]}
    anns = collections.defaultdict(list)
    for a in d["annotations"]:
        if not a.get("iscrowd", 0):
            anns[a["image_id"]].append(a)
    ids_by_file = collections.defaultdict(list)
    for iid, im in img.items():
        ids_by_file[im["file_name"]].append(iid)
    for f in ids_by_file:
        ids_by_file[f].sort()
    return img, anns, dict(ids_by_file)


def day_folds(files: list[str], n_folds: int = 4) -> dict[str, int]:
    """Group by observation day (no same-day leakage between folds), round-robin over days."""
    days = sorted({os.path.basename(f)[:8] for f in files})
    return {day: i % n_folds for i, day in enumerate(days)}


def fold_of(f: str, folds: dict[str, int]) -> int:
    return folds[os.path.basename(f)[:8]]


def stem_of(f: str) -> str:
    return os.path.splitext(os.path.basename(f))[0]


def ann_rle(a: dict, h: int = H, w: int = W):
    s = a["segmentation"]
    if isinstance(s, list):
        return mask_util.merge(mask_util.frPyObjects(s, h, w))
    return mask_util.frPyObjects(s, h, w) if isinstance(s.get("counts"), list) else s


def _spine_points(a: dict) -> list[np.ndarray]:
    sp = a.get("spine") or []
    if not sp:
        return []
    paths = sp if isinstance(sp[0], (list, tuple)) else [sp]
    out = []
    for p in paths:
        arr = np.asarray(p, dtype=np.float32).reshape(-1, 2)
        if len(arr) >= 2:
            out.append(np.round(arr).astype(np.int32))
    return out


def render_annotator(anns: list[dict], h: int = H, w: int = W) -> np.ndarray:
    """uint8 map: bit0 = union of filament masks, bit1 = spines."""
    m = np.zeros((h, w), np.uint8)
    sp = np.zeros((h, w), np.uint8)
    for a in anns:
        m |= mask_util.decode(ann_rle(a, h, w)).astype(np.uint8)
        for pts in _spine_points(a):
            cv2.polylines(sp, [pts], False, 1, thickness=SPINE_THICKNESS)
    return m | (sp << 1)


def _render_job(args):
    out_path, anns, h, w = args
    if not os.path.exists(out_path):
        cv2.imwrite(out_path, render_annotator(anns, h, w))
    return out_path


def prepare_targets(ann_path: str, cache_dir: str, workers: int = 8) -> dict[str, list[str]]:
    """Render every annotator of every image. Returns {file_name: [png paths, one per annotator]}."""
    os.makedirs(cache_dir, exist_ok=True)
    img, anns, ids_by_file = load_index(ann_path)
    jobs, out = [], {}
    for f, ids in ids_by_file.items():
        paths = []
        for k, iid in enumerate(ids):
            p = os.path.join(cache_dir, f"{stem_of(f)}__{k}.png")
            paths.append(p)
            jobs.append((p, anns.get(iid, []), int(img[iid]["height"]), int(img[iid]["width"])))
        out[f] = paths
    todo = [j for j in jobs if not os.path.exists(j[0])]
    if todo:
        with ProcessPoolExecutor(workers) as ex:
            list(ex.map(_render_job, todo, chunksize=4))
    return out


def medoid_index(masks: list[np.ndarray]) -> int:
    """Annotator whose semantic mask agrees most (mean IoU) with the other annotators."""
    if len(masks) == 1:
        return 0
    rles = [mask_util.encode(np.asfortranarray((m & 1).astype(np.uint8))) for m in masks]
    iou = np.asarray(mask_util.iou(rles, rles, [0] * len(rles)), dtype=float)
    return int(np.argmax((iou.sum(1) - 1.0) / (len(masks) - 1)))


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, path
    return img


# ----------------------------------------------------------------------------- dataset
class TileDataset(Dataset):
    """Each item = one image -> `tiles_per_image` random tiles, read from the memory-mapped cache.

    target_mode:
      medoid   - the most consensual annotator (baseline policy)
      majority - pixel is filament if >= half of the annotators marked it
      random   - a different annotator each time (stochastic multi-annotator training)
      mixed    - medoid 50%, random 50%
    """

    def __init__(self, cache, files: list[str], tile: int = 640, tiles_per_image: int = 4, repeats: int = 1,
                 pos_frac: float = 0.75, target_mode: str = "medoid", ignore_limb_band: bool = True,
                 teacher_dir: str | None = None):
        self.c = cache
        self.idx = [cache.index[f] for f in files]
        self.files = list(files)
        self.tile, self.k, self.repeats, self.pos_frac = tile, tiles_per_image, repeats, pos_frac
        self.target_mode, self.ignore_limb_band, self.teacher_dir = target_mode, ignore_limb_band, teacher_dir

    def __len__(self):
        return len(self.idx) * self.repeats

    def _annotators(self, i):
        m = self.c.meta
        a0, n = m["starts"][i], m["counts"][i]
        mode = self.target_mode
        if mode == "mixed":
            mode = "medoid" if random.random() < 0.5 else "random"
        if mode == "random":
            return [a0 + random.randrange(n)], False
        if mode == "majority":
            return list(range(a0, a0 + n)), True
        return [a0 + m["medoid"][i]], False

    def __getitem__(self, j):
        i = self.idx[j % len(self.idx)]
        f = self.files[j % len(self.idx)]
        cx, cy, r = self.c.meta["disks"][i]
        pos = self.c.pos[i]
        ann, vote = self._annotators(i)
        teacher = None
        if self.teacher_dir:
            tp = os.path.join(self.teacher_dir, stem_of(f) + "_m.png")
            if os.path.exists(tp):
                teacher = cv2.imread(tp, cv2.IMREAD_GRAYSCALE)
        t = self.tile
        xb, yb = [], []
        for _ in range(self.k):
            if len(pos) and random.random() < self.pos_frac:
                py, px = pos[random.randrange(len(pos))]
                y0 = int(np.clip(py - t // 2 + random.randint(-t // 3, t // 3), 0, H - t))
                x0 = int(np.clip(px - t // 2 + random.randint(-t // 3, t // 3), 0, W - t))
            else:
                y0 = random.randint(int(np.clip(cy - r, 0, H - t)), int(np.clip(cy + r - t, 0, H - t)))
                x0 = random.randint(int(np.clip(cx - r, 0, W - t)), int(np.clip(cx + r - t, 0, W - t)))
            sl = (slice(y0, y0 + t), slice(x0, x0 + t))
            q = np.asarray(self.c.inp[i, :, sl[0], sl[1]])
            yy, xx = np.ogrid[y0:y0 + t, x0:x0 + t]
            rad = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
            xi = np.empty((3, t, t), np.float32)
            xi[:2] = (q.astype(np.float32) - 128.0) / 32.0
            xi[2] = np.clip(rad, 0.0, 1.25)
            maps = [np.asarray(self.c.tgt[a, sl[0], sl[1]]) for a in ann]
            if vote:
                mask = (np.mean([(m & 1) for m in maps], 0) >= 0.5).astype(np.float32)
                spine = np.max([(m >> 1) & 1 for m in maps], 0).astype(np.float32)
            else:
                mask = (maps[0] & 1).astype(np.float32)
                spine = ((maps[0] >> 1) & 1).astype(np.float32)
            weight = np.ones_like(mask)
            if self.ignore_limb_band:
                band = ((maps[0] >> 2) & 1).astype(bool)
                weight[band & (mask < 0.5)] = 0.0
            chans = [mask, spine, weight]
            if teacher is not None:
                chans.append(teacher[sl].astype(np.float32) / 255.0)
            xi, yi = augment(xi, np.stack(chans, 0))
            xb.append(xi); yb.append(yi)
        return torch.from_numpy(np.stack(xb)), torch.from_numpy(np.stack(yb))


def augment(x: np.ndarray, y: np.ndarray):
    """D4 group + photometric jitter (only channels 0/1 are intensities) + seeing blur / noise."""
    k = random.randint(0, 3)
    if k:
        x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k, (1, 2))
    if random.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, :, ::-1]
    x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
    if random.random() < 0.8:
        g = 1.0 + random.uniform(-0.2, 0.2)
        x[0] = x[0] * g + random.uniform(-0.3, 0.3)
        x[1] = x[1] * (1.0 + random.uniform(-0.2, 0.2))
    if random.random() < 0.25:  # atmospheric seeing
        s = random.uniform(0.5, 1.5)
        x[0] = cv2.GaussianBlur(x[0], (0, 0), s)
        x[1] = cv2.GaussianBlur(x[1], (0, 0), s)
    if random.random() < 0.25:
        x[:2] += np.random.normal(0, random.uniform(0.02, 0.08), x[:2].shape).astype(np.float32)
    return x, y


def collate_tiles(batch):
    xs, ys = zip(*batch)
    return torch.cat(xs, 0), torch.cat(ys, 0)
