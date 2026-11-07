"""Host metric (Panoptic Quality pooled over every annotator set of every image) + diagnostics.

Replicates the organisers' self-evaluation notebook exactly:
  * each annotator set of an image is an independent GT "image";
  * a (gt, pred) pair is a hit when IoU > 0.5; every hit counts as a TP (masks are disjoint,
    so at most one hit per pair at IoU > 0.5);
  * FP = predictions hitting no GT, FN = GT hitting no prediction;
  * an annotator set without filaments turns all predictions into FPs.
"""
from __future__ import annotations

import collections
import json
from dataclasses import dataclass, field

import numpy as np
from pycocotools import mask as mask_util

from .data import ann_rle

IOU_T = 0.5


def build_gt_sets(ann_path: str, files: list[str] | None = None) -> dict[str, list[list[dict]]]:
    d = json.load(open(ann_path))
    img = {im["id"]: im for im in d["images"]}
    anns = collections.defaultdict(list)
    for a in d["annotations"]:
        if not a.get("iscrowd", 0):
            anns[a["image_id"]].append(a)
    by_file = collections.defaultdict(list)
    for iid, im in img.items():
        by_file[im["file_name"]].append(iid)
    wanted = files if files is not None else sorted(by_file)
    return {f: [[ann_rle(a, int(img[i]["height"]), int(img[i]["width"])) for a in anns.get(i, [])]
                for i in sorted(by_file[f])] for f in wanted}


@dataclass
class PQResult:
    pq: float
    tp: int
    fp: int
    fn: int
    sq: float
    rq: float
    tp_iou: list = field(default_factory=list)
    one_to_many: int = 0   # GT overlapped (IoU>0) by >1 prediction  (fragmentation)
    many_to_one: int = 0   # prediction overlapping (IoU>0) >1 GT      (over-merging)
    n_gt: int = 0

    def __str__(self):
        return (f"PQ {self.pq:.4f} | SQ {self.sq:.4f} RQ {self.rq:.4f} | TP {self.tp} FP {self.fp} FN {self.fn} "
                f"| 1:n {self.one_to_many} n:1 {self.many_to_one} / {self.n_gt} GT")


def iou_matrix(gt: list, preds: list) -> np.ndarray:
    if not gt or not preds:
        return np.zeros((len(gt), len(preds)))
    return np.asarray(mask_util.iou(gt, preds, [0] * len(preds)), dtype=float).reshape(len(gt), len(preds))


def panoptic_quality(preds_by_file: dict[str, list], gt_sets: dict[str, list[list]]) -> PQResult:
    tp_iou, fp, fn, o2m, m2o, n_gt = [], 0, 0, 0, 0, 0
    for f, sets in gt_sets.items():
        preds = preds_by_file.get(f, [])
        for gt in sets:
            n_gt += len(gt)
            if not gt:
                fp += len(preds)
                continue
            if not preds:
                fn += len(gt)
                continue
            iou = iou_matrix(gt, preds)
            hit = iou > IOU_T
            tp_iou.extend(iou[hit].tolist())
            fp += int((hit.sum(0) == 0).sum())
            fn += int((hit.sum(1) == 0).sum())
            ov = iou > 0
            o2m += int((ov.sum(1) > 1).sum())
            m2o += int((ov.sum(0) > 1).sum())
    tp = len(tp_iou)
    den = tp + 0.5 * fp + 0.5 * fn
    pq = sum(tp_iou) / den if den else 0.0
    sq = float(np.mean(tp_iou)) if tp else 0.0
    rq = tp / den if den else 0.0
    return PQResult(pq, tp, fp, fn, sq, rq, tp_iou, o2m, m2o, n_gt)


def encode(mask: np.ndarray):
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
