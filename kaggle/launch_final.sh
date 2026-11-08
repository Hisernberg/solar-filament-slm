#!/bin/bash
# Final round (after the Nov 14 quota reset). Recipe chosen from Runs A-C on held-out fold 0:
#   v2 limb-ring disk fit, mixed multi-annotator targets (+0.009 PQ), ~2x longer training
#   (val PQ still rising at 31 epochs), 4-view TTA OOF maps for every fold (TTA-aware rescorer).
# Usage: bash kaggle/launch_final.sh D|E|F
set -e
cd "$(dirname "$0")/.."
COMMON="--n-folds 5 --epochs 70 --time-budget-min 235 --time-limit-min 265 --val-every 8 --val-limit 60 --tile 640 --tiles-per-image 4 --images-per-batch 4 --target-mode mixed --oof-tta 4"
case "$1" in
  D) JOBS="'0:--fold 1' '1:--fold 2'";;
  E) JOBS="'0:--fold 3' '1:--fold 4'";;
  F) JOBS="'0:--fold 0' '1:--fold -1 --dump-all-maps'";;
  *) echo "usage: $0 D|E|F"; exit 1;;
esac
SLUG=filanet-final-$(echo "$1" | tr A-Z a-z)
python3 kaggle/build_kernel.py --slug "$SLUG" --no-comp --datasets dasshovon/magfilo-kaggle2026-private \
  --cmd "python scripts/train_folds.py --data-dir {DATA} --out-root /kaggle/working/runs --cache-dir /tmp/fcache --jobs $JOBS --common '$COMMON'" \
  --cmd "du -sh /kaggle/working/runs/*"
kaggle kernels push -p "kaggle/out/$SLUG" --accelerator NvidiaTeslaT4
