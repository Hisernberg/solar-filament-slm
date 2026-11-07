#!/usr/bin/env python3
"""Train several folds / variants in parallel, one process per GPU.

    python scripts/train_folds.py --data-dir <root> --out-root runs --jobs "0:--fold 0" "1:--fold 1" ...

Each job is "<gpu>:<extra args>"; every job writes to <out-root>/<job name>.
Default (no --jobs): folds 0..N-1 on GPUs 0..N-1.
"""
import argparse
import os
import subprocess
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--jobs", nargs="*", default=None)
    ap.add_argument("--common", default="", help="args appended to every job")
    a = ap.parse_args()
    n_gpu = max(torch.cuda.device_count(), 1)
    jobs = a.jobs or [f"{i}:--fold {i}" for i in range(n_gpu)]
    cache = a.cache_dir or os.path.join(a.out_root, "cache")
    os.makedirs(a.out_root, exist_ok=True)
    # render targets once before forking the trainers
    sys.path.insert(0, ROOT)
    from filanet.cache import build_cache
    from filanet.data import find_annotation_json, find_dir
    t = time.time()
    build_cache(find_annotation_json(a.data_dir), find_dir(a.data_dir, "train_images"), cache,
                workers=os.cpu_count() or 4)
    print(f"cache ready in {time.time() - t:.0f}s", flush=True)
    workers = max(2, (os.cpu_count() or 8) // len(jobs) - 1)
    procs = []
    for j, job in enumerate(jobs):
        gpu, extra = job.split(":", 1)
        name = extra.strip().replace("--", "").replace(" ", "_").replace("/", "-") or f"job{j}"
        out = os.path.join(a.out_root, name)
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, "-u", "-m", "filanet.train", "--data-dir", a.data_dir, "--out-dir", out,
               "--cache-dir", cache, "--workers", str(workers)] + extra.split() + a.common.split()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=ROOT)
        print("launch", gpu, " ".join(cmd), flush=True)
        procs.append((name, subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=open(os.path.join(out, "stdout.log"), "w"),
                                             stderr=subprocess.STDOUT)))
    while any(p.poll() is None for _, p in procs):
        time.sleep(60)
        for name, _ in procs:
            try:
                lines = open(os.path.join(a.out_root, name, "stdout.log")).read().strip().splitlines()
                print(f"[{name}] {lines[-1][:300] if lines else ''}", flush=True)
            except OSError:
                pass
    for name, p in procs:
        print(name, "exit", p.returncode, flush=True)
        if p.returncode:
            print(open(os.path.join(a.out_root, name, "stdout.log")).read()[-3000:], flush=True)


if __name__ == "__main__":
    main()
