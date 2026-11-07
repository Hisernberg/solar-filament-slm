"""Inference: the fully-convolutional net runs on the whole 2048x2048 disk in one pass (no tiles,
no seams) on GPU; a cosine-blended tiled path is used on CPU / small GPUs. Optional D4 TTA."""
from __future__ import annotations

import numpy as np
import torch

from .geometry import make_inputs

D4 = [(0, False), (1, False), (2, False), (3, False), (0, True), (1, True), (2, True), (3, True)]


def _fwd(model, x, k, flip):
    z = torch.flip(x, dims=[3]) if flip else x
    z = torch.rot90(z, k, dims=[2, 3]) if k else z
    p = torch.sigmoid(model(z).float())
    p = torch.rot90(p, -k, dims=[2, 3]) if k else p
    return torch.flip(p, dims=[3]) if flip else p


def _cos_window(t: int) -> torch.Tensor:
    w = torch.hann_window(t, periodic=False)
    return (w[:, None] * w[None, :]).clamp_min(1e-3)


@torch.no_grad()
def predict_maps(model, img_u8: np.ndarray, device: str | torch.device = "cuda", tta: int = 1,
                 tile: int | None = None, stride: int | None = None, amp: bool = True):
    """-> (mask_prob uint8 HxW, spine_prob uint8 HxW, disk). Probabilities are stored as round(p*255)."""
    x_np, disk = make_inputs(img_u8, method=getattr(model, "disk_fit", None))
    x = torch.from_numpy(x_np)[None].to(device)
    dev_type = x.device.type
    dtype = torch.bfloat16 if dev_type == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    use_amp = amp and dev_type == "cuda"
    views = D4[:tta]

    def run(z):
        with torch.autocast(dev_type, dtype=dtype, enabled=use_amp):
            return sum(_fwd(model, z, k, f) for k, f in views) / len(views)

    if tile is None:
        p = run(x)[0]
    else:
        stride = stride or tile * 3 // 4
        _, _, Hh, Ww = x.shape
        acc = torch.zeros((2, Hh, Ww), device=device)
        wsum = torch.zeros((1, Hh, Ww), device=device)
        win = _cos_window(tile).to(device)
        ys = list(range(0, Hh - tile + 1, stride)); xs = list(range(0, Ww - tile + 1, stride))
        if ys[-1] != Hh - tile: ys.append(Hh - tile)
        if xs[-1] != Ww - tile: xs.append(Ww - tile)
        for y0 in ys:
            for x0 in xs:
                q = run(x[:, :, y0:y0 + tile, x0:x0 + tile])[0]
                acc[:, y0:y0 + tile, x0:x0 + tile] += q * win
                wsum[:, y0:y0 + tile, x0:x0 + tile] += win
        p = acc / wsum
    p = (p.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
    return p[0], p[1], disk
