#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.sh"
m=${SMOKE_MODEL:-HuggingFaceTB/SmolLM2-135M}
d=${SMOKE_DATA:-wikitext}
o=${OUT}/smoke
c="--dataset $d --seq-len 64 --dtype fp32 --device $DEVICE"
python scripts/exp0_taylor.py --model "$m" $c --n 1 --layers 0,2 --dirs 2 --radii 1e-3,1e-2 --out "$o/exp0/x.csv"
python scripts/exp1_anisotropy.py --model "$m" $c --n 1 --layers 0,2 --k 4 --probes 4 --dirs 2 --out "$o/exp1/x.csv"
python scripts/exp2_rank_allocation.py --model "$m" $c --n 2 --n-eval 2 --kmax 8 --probes 4 --ft-steps 2 --out "$o/exp2/x.csv"
python scripts/exp3_token_pruning.py --model "$m" $c --n 1 --layers 1 --out "$o/exp3/x.csv"
python scripts/exp4_distill.py --model "$m" $c --steps 8 --rank 8 --kd skewkl --init asvd --n-eval 2 --n-calib 2 --out "$o/exp4/a.csv"
python scripts/exp4_distill.py --model "$m" $c --steps 8 --rank 8 --kd skewkl --init asvd --geo --geo-layers 1,2 --n-eval 2 --n-calib 2 --out "$o/exp4/b.csv"
python scripts/check_hessian_fisher.py --model "$m" $c --n 1 --layers 1 --k 4 --out "$o/checks/hessian_x.csv"
python scripts/check_future_leakage.py --model "$m" $c --n 2 --layers 1 --k 4 --horizon 4 --data-driven --out "$o/checks/future_x.csv"
python scripts/check_effective_rank.py --model "$m" $c --n 1 --layers 1 --probes 8 --out "$o/checks/effrank_x.csv"
python scripts/summarize.py --root "$o" --out "$o/tables"
python scripts/plot.py --root "$o" --out "$o/figures"
