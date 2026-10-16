# Predictive Geometry of Hidden Trajectories in Transformers

Code and results for the NeurIPS 2026 paper

**Predictive Geometry of Hidden Trajectories in Transformers**
Timur Mudarisov, Mikhail Burtsev, Tatiana Petrova, Radu State
[arXiv:2609.37717](https://arxiv.org/abs/2609.37717) · [OpenReview](https://openreview.net/forum?id=2abHWd81WV)

The repository contains a matrix-free implementation of the pullback Fisher operator
`K_l = J_l^T F J_l`, scripts for all experiments in the paper, and the reported results.

## Layout

```
pgeom/             library code
  model.py         model loading, layer-input capture, downstream map X_l -> target logits
  geometry.py      K_l v via JVP/VJP, Hutchinson traces, top eigenvectors, token scores kappa_{l,s}
  data.py          WikiText / OpenWebText / FineWeb loaders (or any local .txt file)
  lowrank.py       SVD / activation-aware SVD factorization, rank budgets
  kd.py            FKL / RKL / SkewKL losses and the readout-Fisher geometry term
scripts/           one script per experiment, plus summarize.py and plot.py
run/               shell entry points (config.sh holds the model and dataset lists)
results/paper/     tables and figures reported in the paper
figures/, tables/  additional validation (Hessian vs Fisher, future leakage, effective rank)
```

## Install

```
pip install -r requirements.txt
```

Gated checkpoints (LLaMA, Gemma, Mistral) need `huggingface-cli login`.
Set `OWT_PATH` if your OpenWebText copy lives somewhere other than `Skylion007/openwebtext`.

## Tests

```
PYTHONPATH=. python tests/test_geometry.py /path/to/small/model [...]
```

compares `K_l v`, `v^T K_l v`, the top eigenvalues, the Hutchinson trace and the token scores `kappa_{l,s}`
with the dense `J^T F J` built by autograd. It needs a model small enough for a dense Jacobian
(a randomly initialised 4-layer config is enough). Tested with torch 2.5.1 and transformers 4.46.3 / 4.57.6
on LLaMA, Gemma-2, Phi-3, Qwen2 and OLMo-3 architectures.

## Running

```
bash run/smoke.sh            # tiny end-to-end run on SmolLM2-135M
bash run/exp0.sh             # Exp. 0: local Taylor validation
bash run/exp1.sh             # Exp. 1: observable vs null anisotropy, effective rank
bash run/exp2.sh             # Exp. 2: geometry-guided rank allocation
bash run/exp3.sh             # Exp. 3: token-update pruning with kappa
bash run/exp4.sh             # Exp. 4: low-rank student distillation with the geometry term
bash run/checks.sh           # Exp. 1b: Hessian vs Fisher, future leakage, effective rank
bash run/report.sh           # tables -> outputs/tables, figures -> outputs/figures
```

Every script writes raw per-example rows to `outputs/<experiment>/*.csv`.
`ONLY_MODEL`, `ONLY_DATASET`, `N`, `LAYERS`, `OUT` and `DEVICE` override the defaults in `run/config.sh`, e.g.

```
ONLY_MODEL=meta-llama/Llama-2-7b-hf ONLY_DATASET=wikitext N=8 bash run/exp1.sh
```

A single script can also be called directly:

```
PYTHONPATH=. python scripts/exp1_anisotropy.py --model Qwen/Qwen2.5-3B --dataset fineweb \
    --n 16 --layers every2 --out outputs/exp1/qwen_fineweb.csv
```

`--layers` accepts `all`, `everyK` or a comma-separated list. Any path to a text file can be used as `--dataset`.

## Experiments

| Paper | Script | What is measured |
|---|---|---|
| Exp. 0, Fig. 4, 7 | `exp0_taylor.py` | true ΔCE / ΔKL under random perturbations vs. linear, Fisher-quadratic and combined predictors |
| Exp. 1, Fig. 2 | `exp1_anisotropy.py` | ΔKL for equal-norm perturbations in the top-k eigenspace of K_l and its complement; Hutchinson r_eff |
| Exp. 2, Fig. 3 | `exp2_rank_allocation.py` | ΔPPL of per-layer ranks from the spectral rule (13) vs. uniform SVD and uniform LoRA-style ranks at the same budget |
| Exp. 3, Fig. 5, 6 | `exp3_token_pruning.py` | ΔCE after skipping the block update for the lowest-scored tokens (kappa, JᵀJ, diagonal Fisher, attention, grad×act, top-logit, activation norm, random) |
| Exp. 4, Tab. 2, 3–6 | `exp4_distill.py` | PPL of SVD/ASVD low-rank students trained with FKL / RKL / SkewKL, with and without the geometry term |
| Exp. 1b, Tab. 1, App. G | `check_hessian_fisher.py` | v^T K v vs. v^T H_CE v, cos(H_CE v, K v), finite-difference teacher-KL and CE curvature |
| Exp. 1b, Tab. 1, App. G | `check_future_leakage.py` | curvature at the target and at the next h positions for single-target, horizon, null, random and data-driven directions |
| Exp. 1b, Tab. 1, App. G | `check_effective_rank.py` | r_eff from Hutchinson probes vs. moments of finite-difference curvatures |

Implementation notes:

* `K_l v` is never formed explicitly: one JVP through the downstream blocks, a softmax-Fisher product, one VJP.
* Layer inputs and the extra arguments of every decoder block (rotary embeddings, masks) are captured with forward pre-hooks,
  so the downstream map works for any Hugging Face model with `model.model.layers`, `model.model.norm` and `lm_head`.
* Single-target experiments truncate the sequence at the target position; the future-leakage check keeps `h` extra positions
  and restricts all directions to tokens up to the target.
* Token scores `kappa_{l,s}` are estimated with sampled-label probes: `E||P_s J^T (e_y - p)||^2` with `y ~ p` equals `tr(P_s K_l P_s^T)`.
* Checks run in fp32 by default because a relative radius of 1e-3 is below bf16 resolution. In fp32 the
  finite-difference curvature of null directions sits at the numerical floor; `--dtype fp64` removes it
  (the immediate curvature then matches `v^T K v` to 4-5 digits).

Compute for the main geometry statistics is listed in `results/paper/tables/compute_cost.csv`.

## Results

`results/paper/tables` contains the numbers reported in the paper (KD sweep, paired RKL/SkewKL comparisons,
geometry-advantage summary, model checkpoints, compute cost). `results/paper/figures` contains the paper figures.

`figures/` and `tables/` contain the additional validation on LLaMA-2-7B and Mistral-7B (OpenWebText, all 32 layers,
3 sequences per model, rho = 1e-3) used in Section 4.3 and Appendix G:

| Model | cos(H_CE v, K v) top | Spearman(v^T K v, ΔKL) | null/top KL, target | null/horizon-top KL, future | r_eff FD / Hutchinson |
|---|---|---|---|---|---|
| LLaMA-2-7B | 0.87 | 0.98 | 1.1e-3 | 2.6e-3 | 22.7 / 10.1 |
| Mistral-7B | 0.64 | 0.98 | 4.9e-4 | 1.8e-3 | 30.3 / 33.2 |

## Acknowledgments

We thank Nebius for providing the computational resources used in this work.

## Citation

```
@inproceedings{mudarisov2026predictive,
  title     = {Predictive Geometry of Hidden Trajectories in Transformers},
  author    = {Mudarisov, Timur and Burtsev, Mikhail and Petrova, Tatiana and State, Radu},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
