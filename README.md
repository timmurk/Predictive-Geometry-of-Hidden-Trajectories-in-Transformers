# Reviewer Follow-up Results

This repository contains only the additional reviewer-follow-up tables and figures for the predictive-geometry experiments.

## Scope

- Models: LLaMA-2-7B and Mistral-7B.
- Dataset: OpenWebText.
- Layers: all detected decoder layers.
- Evaluation examples per model: {'LLaMA-2-7B': 3, 'Mistral-7B': 3}.
- Main reporting radius: `rho = 1e-3`.

The files here are intended to support the rebuttal/revision with newly generated comparison results only. The repository intentionally excludes paper source, author information, raw model checkpoints, and private environment details.

## Directory structure

```text
figures/
  PNG figures used for visual comparison.

tables/
  CSV, Markdown, and LaTeX tables with the same summarized results.

summary_info.json
  Minimal machine-readable summary of the included result set.
```

## Main figures

- `figures/fig_effective_rank_fd_moment_by_layer.png`
- `figures/fig_effective_rank_hutchinson_by_layer.png`
- `figures/fig_effective_rank_moment_vs_hutchinson_scatter.png`
- `figures/fig_future_fd2_by_direction_rho_1e-3.png`
- `figures/fig_future_leakage_single_null_ratio_by_layer_rho_1e-3.png`
- `figures/fig_hessian_rel_error_by_direction_rho_1e-3.png`
- `figures/fig_hessian_top_fisher_cosine_by_layer.png`
- `figures/fig_immediate_vs_future_fd2_scatter_rho_1e-3.png`
- `figures/fig_teacherKL_fd2_vs_vKv_top_fisher_rho_1e-3.png`

## Main tables

- `tables/table_compact_reviewer_comparison.csv`
- `tables/table_compact_reviewer_comparison.md`
- `tables/table_compact_reviewer_comparison.tex`
- `tables/table_effective_rank_summary.csv`
- `tables/table_effective_rank_summary.md`
- `tables/table_effective_rank_summary.tex`
- `tables/table_future_leakage_summary_rho_1e-3.csv`
- `tables/table_future_leakage_summary_rho_1e-3.md`
- `tables/table_future_leakage_summary_rho_1e-3.tex`
- `tables/table_hessian_fisher_direct_summary_rho_1e-3.csv`
- `tables/table_hessian_fisher_direct_summary_rho_1e-3.md`
- `tables/table_hessian_fisher_direct_summary_rho_1e-3.tex`

## Notes

- The results are based on OpenWebText runs for the two 7B models.
- Cross-corpus stability is not included in this package because the uploaded result archive contains only OpenWebText runs.
- Very small finite-difference radii can be noisy; the main tables therefore report `rho = 1e-3`.
