"""Train one FilaNet fold (or the full-data model) on one GPU.

    python -m filanet.train --data-dir <MAGFiLO root> --out-dir runs/f0 --fold 0
    python -m filanet.train --data-dir <MAGFiLO root> --out-dir runs/full --fold -1 --epochs 40

On a 4-GPU box `scripts/train_folds.py` launches folds 0..3 in parallel (one process per GPU),
which gives out-of-fold maps for every training image in one wall-clock training run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import cv2
import numpy as np
import torch
import copy
from torch.utils.data import DataLoader

from .cache import Cache, build_cache
from .data import (TileDataset, collate_tiles, day_folds, find_annotation_json, find_dir, fold_of, load_image,
                   stem_of)
from .evaluate import grid_search
from .geometry import make_inputs
from .infer import predict_maps
from .losses import FilamentLoss
from .metric import build_gt_sets
from .model import build_model, count_params, save_checkpoint


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="runs/f0")
    ap.add_argument("--cache-dir", default=None, help="rendered per-annotator targets (shared by folds)")
    ap.add_argument("--fold", type=int, default=0, help="-1 = train on all images")
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--encoder", default="convnext_atto.d2_in1k")
    ap.add_argument("--dec-channels", default="128,80,48,24,16")
    ap.add_argument("--no-strip-pool", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=2, help="visits of each training image per epoch")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--tiles-per-image", type=int, default=4)
    ap.add_argument("--images-per-batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--target-mode", default="medoid", choices=["medoid", "majority", "random", "mixed"])
    ap.add_argument("--no-limb-ignore", action="store_true")
    ap.add_argument("--w-spine", type=float, default=0.5)
    ap.add_argument("--w-srec", type=float, default=0.3)
    ap.add_argument("--w-kd", type=float, default=0.0)
    ap.add_argument("--teacher-dir", default=None)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--val-limit", type=int, default=0, help="validate on at most N images (0 = all)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train-images", type=int, default=0, help="debug: subsample")
    ap.add_argument("--time-limit-min", type=float, default=0, help="stop training after N minutes")
    return ap.parse_args()


class EMA:
    """EMA of weights *and* BN buffers with warmup: decay_t = min(decay, (1 + t) / (10 + t))."""

    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.t = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.t += 1
        d = min(self.decay, (1 + self.t) / (10 + self.t))
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def worker_init(i):
    s = torch.initial_seed() % 2 ** 32
    random.seed(s + i); np.random.seed((s + i) % 2 ** 32)


@torch.no_grad()
def predict_files(model, files, image_dir, device):
    maps, locs = {}, {}
    for f in files:
        img = load_image(os.path.join(image_dir, f))
        pm, sm, disk = predict_maps(model, img, device=device, tta=1, tile=None if device.type == "cuda" else 1024)
        maps[f] = (pm, sm, disk)
        locs[f] = make_inputs(img, disk)[0][1]
    return maps, locs


def main():
    a = parse()
    set_seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ann = find_annotation_json(a.data_dir)
    image_dir = find_dir(os.path.dirname(ann), "train_images") if os.path.isdir(
        os.path.join(os.path.dirname(ann), "train_images")) else find_dir(a.data_dir, "train_images")
    cache_dir = a.cache_dir or os.path.join(a.out_dir, "cache")
    cache = Cache(build_cache(ann, image_dir, cache_dir, workers=os.cpu_count() or 4))
    files = cache.files
    folds = day_folds(files, a.n_folds)
    if a.fold >= 0:
        tr = [f for f in files if fold_of(f, folds) != a.fold]
        va = [f for f in files if fold_of(f, folds) == a.fold]
    else:
        tr, va = files, []
    if a.max_train_images:
        tr = tr[:a.max_train_images]
    va_eval = va[:a.val_limit] if a.val_limit else va
    print(f"train {len(tr)} | val {len(va)} | fold {a.fold} | device {device}", flush=True)
    gt_sets = build_gt_sets(ann, va) if va else {}

    cfg = dict(encoder=a.encoder, dec_channels=tuple(int(c) for c in a.dec_channels.split(",")),
               strip_pool=not a.no_strip_pool)
    model = build_model(cfg, pretrained=True).to(device).to(memory_format=torch.channels_last)
    print(f"params {count_params(model) / 1e6:.2f}M  cfg {cfg}", flush=True)
    ema = EMA(model, a.ema)
    fwd = torch.compile(model) if a.compile else model

    ds = TileDataset(cache, tr, tile=a.tile, tiles_per_image=a.tiles_per_image, repeats=a.repeats,
                     target_mode=a.target_mode, ignore_limb_band=not a.no_limb_ignore, teacher_dir=a.teacher_dir)
    dl = DataLoader(ds, batch_size=a.images_per_batch, shuffle=True, num_workers=a.workers,
                    collate_fn=collate_tiles, pin_memory=True, drop_last=True, persistent_workers=a.workers > 0,
                    worker_init_fn=worker_init, prefetch_factor=4 if a.workers > 0 else None)
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.99))
    total = a.epochs * len(dl)
    warm = int(a.warmup_epochs * len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm else
        0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    crit = FilamentLoss(w_spine=a.w_spine, w_srec=a.w_srec, w_kd=a.w_kd)
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    log, best, best_rule = [], -1.0, None
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        run, nb, parts_sum = 0.0, 0, {}
        for x, y in dl:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                out = fwd(x)
            loss, parts = crit(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update(); sched.step()
            ema.update(model)
            run += float(loss.detach()); nb += 1
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v
        msg = dict(epoch=ep, loss=run / max(nb, 1), **{k: v / max(nb, 1) for k, v in parts_sum.items()},
                   lr=sched.get_last_lr()[0], sec=time.time() - t0)
        timeout = a.time_limit_min and (time.time() - t0) / 60 > a.time_limit_min
        last = ep == a.epochs - 1 or timeout
        if va_eval and ((ep + 1) % a.val_every == 0 or last):
            ema.module.eval()
            maps, locs = predict_files(ema.module, va_eval, image_dir, device)
            res, rule, _ = grid_search(maps, {f: gt_sets[f] for f in va_eval}, locs)
            msg.update(val_pq=res.pq, val_sq=res.sq, val_rq=res.rq, tp=res.tp, fp=res.fp, fn=res.fn, rule=rule)
            if res.pq > best:
                best, best_rule = res.pq, rule
                save_checkpoint(os.path.join(a.out_dir, "best.pt"), ema.module, cfg,
                                dict(epoch=ep, val_pq=res.pq, rule=rule, fold=a.fold))
        save_checkpoint(os.path.join(a.out_dir, "last.pt"), ema.module, cfg, dict(epoch=ep, fold=a.fold))
        print(json.dumps(msg, default=float), flush=True)
        log.append(msg)
        json.dump(log, open(os.path.join(a.out_dir, "log.json"), "w"), default=float, indent=1)
        if timeout:
            print("time limit reached", flush=True)
            break

    # out-of-fold maps from the best checkpoint, for keep-rule tuning and the rescorer
    if va:
        from .model import load_checkpoint
        m, ck = load_checkpoint(os.path.join(a.out_dir, "best.pt"), device=str(device))
        od = os.path.join(a.out_dir, "oof")
        os.makedirs(od, exist_ok=True)
        disks = {}
        for f in va:
            img = load_image(os.path.join(image_dir, f))
            pm, sm, disk = predict_maps(m, img, device=device, tta=1, tile=None if device.type == "cuda" else 1024)
            cv2.imwrite(os.path.join(od, stem_of(f) + "_m.png"), pm)
            cv2.imwrite(os.path.join(od, stem_of(f) + "_s.png"), sm)
            disks[f] = [disk.cx, disk.cy, disk.r]
        json.dump(disks, open(os.path.join(od, "disks.json"), "w"))
    json.dump(dict(best_val_pq=best, rule=best_rule, minutes=(time.time() - t0) / 60, params=count_params(model),
                   cfg=cfg, args=vars(a)), open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1,
              default=str)
    print(f"done. best val PQ {best:.4f} rule {best_rule}", flush=True)


if __name__ == "__main__":
    main()
