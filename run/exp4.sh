#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.sh"
for m in "${KD_MODELS[@]}"; do
  for init in svd asvd; do
    for kd in fkl rkl skewkl; do
      for r in 16 32 64; do
        for geo in "" "--geo"; do
          tag="$(short "$m")_${init}_${kd}_r${r}${geo:+_geo}"
          [ -f "$OUT/exp4/$tag.csv" ] && continue
          python scripts/exp4_distill.py --model "$m" --init "$init" --kd "$kd" --rank "$r" $geo \
            --steps "${STEPS:-1000}" --device "$DEVICE" --out "$OUT/exp4/$tag.csv"
        done
      done
    done
  done
done
