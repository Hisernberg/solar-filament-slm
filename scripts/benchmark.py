#!/usr/bin/env python3
"""Efficiency table: parameters, GFLOPs at 2048x2048, latency (GPU full-frame / CPU tiled), file size."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from filanet.infer import predict_maps  # noqa: E402
from filanet.model import build_model, count_params, load_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--encoder", default="convnext_atto.d2_in1k")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.weights:
        m, _ = load_checkpoint(a.weights, dev)
        size_mb = os.path.getsize(a.weights) / 1e6
    else:
        m = build_model({"encoder": a.encoder}, pretrained=False).eval().to(dev)
        size_mb = None
    x = torch.zeros(1, 3, 512, 512, device=dev)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m(x)
    gflops_2048 = fc.get_total_flops() / 1e9 * 16 / 2  # MACs->FLOPs convention: report GMACs
    img = (np.random.rand(2048, 2048) * 255).astype(np.uint8)
    tile = None if dev == "cuda" else 1024
    predict_maps(m, img, dev, tile=tile)
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(a.reps):
        predict_maps(m, img, dev, tile=tile)
    if dev == "cuda":
        torch.cuda.synchronize()
    out = dict(params_M=count_params(m) / 1e6, GMACs_2048=gflops_2048, device=dev,
               sec_per_image=(time.time() - t) / a.reps, checkpoint_MB=size_mb,
               gpu=torch.cuda.get_device_name() if dev == "cuda" else None)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
