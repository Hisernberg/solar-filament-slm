#!/bin/bash
# Run C (v2 = limb-ring disk fit), fold 0 ablations vs Run A fold 0:
#   GPU0: v2 medoid targets            -> effect of the disk-fit fix
#   GPU1: v2 mixed (medoid/random) targets -> effect of multi-annotator training
# Both dump 4-view TTA OOF maps to measure test-time augmentation.
set -e
cd "$(dirname "$0")/.."
python3 kaggle/build_kernel.py --slug filanet-runc --no-comp --datasets dasshovon/magfilo-kaggle2026-private \
  --cmd "python scripts/train_folds.py --data-dir {DATA} --out-root /kaggle/working/runs --cache-dir /tmp/fcache --jobs '0:--fold 0 --target-mode medoid' '1:--fold 0 --target-mode mixed' --common '--n-folds 5 --epochs 40 --time-budget-min 125 --time-limit-min 150 --val-every 6 --val-limit 60 --tile 640 --tiles-per-image 4 --images-per-batch 4 --oof-tta 4'" \
  --cmd "du -sh /kaggle/working/runs/*"
