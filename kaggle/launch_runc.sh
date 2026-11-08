#!/bin/bash
# Run C: isolate the disk-fit fix (GPU0) and DINOv3 distillation (GPU1) on fold 0.
# Teacher maps come from Run B's kernel output (mounted under /kaggle/input).
set -e
cd "$(dirname "$0")/.."
TEACHER='$(dirname $(find /kaggle/input -path "*fold_0_encoder*" -name "maps_all" -type d | head -1))/maps_all'
python3 kaggle/build_kernel.py --slug filanet-runc --no-comp \
  --datasets dasshovon/magfilo-kaggle2026-private --kernels dasshovon/filanet-runb \
  --cmd "ls /kaggle/input; find /kaggle/input -maxdepth 4 -type d | head -30" \
  --cmd "python scripts/train_folds.py --data-dir {DATA} --out-root /kaggle/working/runs --cache-dir /tmp/fcache --jobs '0:--fold 0 --oof-tta 4' \"1:--fold 0 --oof-tta 4 --w-kd 0.5 --teacher-dir $TEACHER\" --common '--n-folds 5 --epochs 40 --time-budget-min 120 --time-limit-min 150 --val-every 6 --val-limit 60 --tile 640 --tiles-per-image 4 --images-per-batch 4'" \
  --cmd "du -sh /kaggle/working/runs/*"
