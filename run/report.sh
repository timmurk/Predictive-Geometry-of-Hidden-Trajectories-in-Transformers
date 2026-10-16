#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.sh"
python scripts/summarize.py --root "$OUT" --out "$OUT/tables"
python scripts/plot.py --root "$OUT" --out "$OUT/figures"
