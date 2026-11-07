"""Losses: masked BCE + soft Dice for the filament mask, spine supervision, skeleton recall,
and optional knowledge distillation from a larger teacher's probability map.

Target tensor y (B, C, H, W): y[:,0] mask, y[:,1] spine, y[:,2] pixel weight (0 = ignore),
optional y[:,3] teacher probability.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce(logit, target, w, pos_weight: float = 1.0):
    pw = torch.tensor(pos_weight, device=logit.device, dtype=logit.dtype)
    l = F.binary_cross_entropy_with_logits(logit, target, reduction="none", pos_weight=pw)
    return (l * w).sum() / w.sum().clamp_min(1.0)


def soft_dice(prob, target, w, eps: float = 1.0):
    p, t = prob * w, target * w
    inter = (p * t).sum()
    return 1.0 - (2 * inter + eps) / (p.sum() + t.sum() + eps)


def skeleton_recall(prob, skel, eps: float = 1.0):
    """Skeleton Recall loss (Kirchhoff et al., ECCV 2024): the mask must cover the GT skeleton.
    MAGFiLO ships hand-drawn spines, so no skeletonisation is needed."""
    return 1.0 - ((prob * skel).sum() + eps) / (skel.sum() + eps)


class FilamentLoss(torch.nn.Module):
    def __init__(self, w_dice=1.0, w_bce=1.0, w_spine=0.5, w_srec=0.3, w_kd=0.0, kd_temp=1.0):
        super().__init__()
        self.w = dict(dice=w_dice, bce=w_bce, spine=w_spine, srec=w_srec, kd=w_kd)
        self.kd_temp = kd_temp

    def forward(self, logits, y):
        logits = logits.float()
        m_logit, s_logit = logits[:, 0], logits[:, 1]
        mask, spine, w = y[:, 0], y[:, 1], y[:, 2]
        pm = torch.sigmoid(m_logit)
        ps = torch.sigmoid(s_logit)
        parts = {
            "bce": masked_bce(m_logit, mask, w),
            "dice": soft_dice(pm, mask, w),
            "spine": masked_bce(s_logit, spine, w, pos_weight=3.0) + soft_dice(ps, spine, w),
            "srec": skeleton_recall(pm, spine * w),
        }
        if self.w["kd"] > 0 and y.shape[1] > 3:
            t = y[:, 3]
            parts["kd"] = masked_bce(m_logit / self.kd_temp, t, w)
        total = sum(self.w[k] * v for k, v in parts.items())
        return total, {k: float(v.detach()) for k, v in parts.items()}
