#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.sh"
for m in "${MODELS[@]}"; do
  for d in "${DATASETS[@]}"; do
    python scripts/exp3_token_pruning.py --model "$m" --dataset "$d" --n "${N:-32}" \
      --layers "${LAYERS:-every4}" --drops 0.05,0.1,0.2,0.3 --device "$DEVICE" \
      --out "$OUT/exp3/$(short "$m")_$d.csv"
  done
done
