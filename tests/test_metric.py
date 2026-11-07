"""Our PQ (RLE-based, cached) must equal the organisers' self-evaluation logic (dense matrices)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from filanet.evaluate import ImageComps, pq_select  # noqa: E402
from filanet.metric import encode, iou_matrix, panoptic_quality  # noqa: E402


def host_pq(pairs):
    """Organisers' get_pq_score on dense (gt_layers, pred_layers) per GT image."""
    tp, fp, fn = [], 0, 0
    for gt, pr in pairs:
        if len(gt) == 0:
            fp += len(pr); continue
        if len(pr) == 0:
            fn += len(gt); continue
        g = torch.tensor(np.stack(gt), dtype=torch.float32).reshape(len(gt), -1)
        p = torch.tensor(np.stack(pr), dtype=torch.float32).reshape(len(pr), -1)
        inter = g @ p.t()
        union = g.sum(1, keepdim=True) + p.sum(1)[None] - inter
        iou = torch.where(union == 0, torch.tensor(0.0), inter / union)
        hit = iou > 0.5
        tp += iou[hit].tolist()
        fp += int((hit.sum(0) == 0).sum()); fn += int((hit.sum(1) == 0).sum())
    den = len(tp) + 0.5 * fp + 0.5 * fn
    return sum(tp) / den if den else 0.0


def rand_masks(rng, n, h=64, w=64):
    out = []
    for _ in range(n):
        m = np.zeros((h, w), np.uint8)
        y, x = rng.integers(0, h - 12), rng.integers(0, w - 12)
        m[y:y + rng.integers(4, 12), x:x + rng.integers(4, 12)] = 1
        out.append(m)
    return out


def test_pq_matches_host():
    rng = np.random.default_rng(0)
    preds, gts, pairs, cache = {}, {}, [], {}
    for i in range(30):
        f = f"img{i}"
        pr = rand_masks(rng, rng.integers(0, 5))
        # make predictions disjoint like a real submission
        occ = np.zeros((64, 64), np.uint8); dis = []
        for m in pr:
            m = m & (1 - occ); occ |= m
            if m.sum():
                dis.append(m)
        sets = [rand_masks(rng, rng.integers(0, 5)) + [d.copy() for d in dis[:1]] for _ in range(rng.integers(1, 4))]
        pr_rle = [encode(m) for m in dis]
        gt_rle = [[encode(m) for m in s] for s in sets]
        preds[f], gts[f] = pr_rle, gt_rle
        pairs += [(s, dis) for s in sets]
        cache[f] = ImageComps({}, [iou_matrix(g, pr_rle) for g in gt_rle], len(pr_rle))
    ours = panoptic_quality(preds, gts).pq
    host = host_pq(pairs)
    cached = pq_select(cache, {f: np.ones(c.n, bool) for f, c in cache.items()}).pq
    assert abs(ours - host) < 1e-9, (ours, host)
    assert abs(cached - host) < 1e-9, (cached, host)
    print("PQ ours", ours, "host", host, "cached", cached)


if __name__ == "__main__":
    test_pq_matches_host()
