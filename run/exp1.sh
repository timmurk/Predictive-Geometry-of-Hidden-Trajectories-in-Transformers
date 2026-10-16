#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.sh"
for m in "${MODELS[@]}"; do
  for d in "${DATASETS[@]}"; do
    python scripts/exp1_anisotropy.py --model "$m" --dataset "$d" --n "${N:-16}" \
      --k 16 --probes 16 --rho 1e-2 --device "$DEVICE" \
      --out "$OUT/exp1/$(short "$m")_$d.csv"
  done
done
