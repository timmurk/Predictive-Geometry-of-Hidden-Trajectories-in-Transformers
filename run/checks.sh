#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.sh"
for m in "${CHECK_MODELS[@]}"; do
  s=$(short "$m")
  python scripts/check_hessian_fisher.py --model "$m" --dataset openwebtext --n 3 --rho 1e-3 \
    --device "$DEVICE" --out "$OUT/checks/hessian_$s.csv"
  python scripts/check_future_leakage.py --model "$m" --dataset openwebtext --n 3 --rho 1e-3 \
    --horizon 8 --data-driven --device "$DEVICE" --out "$OUT/checks/future_$s.csv"
  python scripts/check_effective_rank.py --model "$m" --dataset openwebtext --n 3 --rho 1e-3 \
    --device "$DEVICE" --out "$OUT/checks/effrank_$s.csv"
done
