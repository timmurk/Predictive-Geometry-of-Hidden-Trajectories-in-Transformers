#!/usr/bin/env bash
set -euo pipefail
dir="$(dirname "$0")"
for step in exp0 exp1 exp2 exp3 exp4 checks report; do
  bash "$dir/$step.sh"
done
