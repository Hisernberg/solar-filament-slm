"""Solar-disk geometry and physics-aware input channels.

GONG H-alpha frames are 2048x2048 with the solar disk roughly centred. We fit the disk per
image (centre + radius) from a thresholded, downsampled copy, then build three input channels:

  0. disk-normalised intensity  (I - mean_disk) / std_disk         -> instrument/site invariance
  1. local contrast             I - blur(I, ~48 px), / std_disk    -> removes limb darkening,
                                                                       filaments are *locally* dark
  2. radial position            r / R_sun  (clipped to [0, 1.25])  -> lets the net learn limb effects

All three are invariant to the D4 group (flips / 90-degree rotations) about the disk centre, so
D4 augmentation and D4 test-time augmentation remain valid.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SIN70 = float(np.sin(np.deg2rad(70.0)))  # MAGFiLO annotators skip filaments beyond 70 deg longitude


@dataclass
class Disk:
    cx: float
    cy: float
    r: float


def fit_disk(img_u8: np.ndarray) -> Disk:
    """Fit the solar disk with a robust threshold on a 1/4-res copy (fast, ~5 ms)."""
    h, w = img_u8.shape
    s = 4
    small = cv2.resize(img_u8, (w // s, h // s), interpolation=cv2.INTER_AREA)
    thr = max(8.0, 0.3 * float(np.percentile(small, 90)))
    m = (small > thr).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return Disk(w / 2.0, h / 2.0, 0.45 * min(h, w))
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = (lab == k).astype(np.uint8)
    # fill holes (dark filaments / sunspots must not shrink the disk estimate)
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (cx, cy), rad = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    area = float(cv2.contourArea(max(cnts, key=cv2.contourArea)))
    r_area = np.sqrt(area / np.pi)
    r = 0.5 * (rad + r_area)  # average the two estimators: robust to clipped / ragged limbs
    return Disk(cx * s + (s - 1) / 2.0, cy * s + (s - 1) / 2.0, r * s)


def coord_maps(shape: tuple[int, int], disk: Disk) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (r/R, |dx|/R, |dy|/R) float32 maps."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xx - disk.cx) / disk.r
    dy = (yy - disk.cy) / disk.r
    return np.sqrt(dx * dx + dy * dy), np.abs(dx), np.abs(dy)


def radial_map(shape: tuple[int, int], disk: Disk) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[0:h, 0:w]
    return (np.sqrt(((xx - disk.cx) ** 2 + (yy - disk.cy) ** 2)) / disk.r).astype(np.float32)


def longitude_ignore_mask(shape: tuple[int, int], disk: Disk) -> np.ndarray:
    """1 where MAGFiLO annotators were told not to annotate (beyond ~70 deg longitude, on disk).

    Approximation: heliographic longitude from the central meridian with B0 ~ 0 and P ~ 0
    (GONG frames are solar-north up). On the disk at height y, |sin(lon)| = |dx| / sqrt(1 - dy^2).
    """
    h, w = shape
    yy, xx = np.ogrid[0:h, 0:w]
    dx = (xx - disk.cx) / disk.r
    dy = (yy - disk.cy) / disk.r
    rr = dx * dx + dy * dy
    cos_lat = np.sqrt(np.clip(1.0 - dy * dy, 1e-6, None))
    sin_lon = np.abs(dx) / cos_lat
    return ((sin_lon > SIN70) & (rr <= 1.0)).astype(np.uint8)


def make_inputs(img_u8: np.ndarray, disk: Disk | None = None) -> tuple[np.ndarray, Disk]:
    """uint8 HxW -> float32 3xHxW network input (see module docstring)."""
    if disk is None:
        disk = fit_disk(img_u8)
    img = img_u8.astype(np.float32) / 255.0
    r = radial_map(img.shape, disk)
    on = r <= 0.98
    v = img[on]
    mu, sd = (float(v.mean()), max(float(v.std()), 1e-3)) if v.size > 1000 else (0.5, 0.25)
    norm = (img - mu) / sd
    # local background via a cheap large blur at 1/4 resolution; off-disk filled with disk mean
    filled = np.where(r <= 1.0, img, mu).astype(np.float32)
    h, w = img.shape
    small = cv2.resize(filled, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    bg = cv2.GaussianBlur(small, (0, 0), sigmaX=12.0)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    local = (img - bg) / sd
    local[r > 1.0] = 0.0
    norm[r > 1.02] = -mu / sd  # flatten the sky to the value of a zero-intensity pixel
    x = np.stack([norm, local * 2.0, np.clip(r, 0.0, 1.25)], 0).astype(np.float32)
    return x, disk
