#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.sh"
for m in "${MODELS[@]}"; do
  for d in wikitext fineweb; do
    python scripts/exp0_taylor.py --model "$m" --dataset "$d" --n "${N:-16}" \
      --layers "${LAYERS:-every4}" --device "$DEVICE" \
      --out "$OUT/exp0/$(short "$m")_$d.csv"
  done
done
