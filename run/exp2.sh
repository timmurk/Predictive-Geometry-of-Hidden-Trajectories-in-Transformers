#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.sh"
for m in "${MODELS[@]}"; do
  for d in "${DATASETS[@]}"; do
    python scripts/exp2_rank_allocation.py --model "$m" --dataset "$d" --n "${N:-32}" \
      --taus 0.01,0.05,0.1,0.2 --rank-frac "${RANK_FRAC:-0.25}" --device "$DEVICE" \
      --out "$OUT/exp2/$(short "$m")_$d.csv"
  done
done
