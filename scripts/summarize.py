import argparse
import glob
import os

import numpy as np
import pandas as pd


def read(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def r2(y, yhat):
    ss = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - yhat) ** 2).sum() / ss if ss > 0 else np.nan


def exp0(df):
    out = []
    for (m, rho), g in df.groupby(["model", "rho"]):
        out.append(dict(
            model=m, rho=rho,
            r2_ce_linear=r2(g.d_ce, g.lin),
            r2_ce_quad=r2(g.d_ce, g.quad),
            r2_ce_lin_quad=r2(g.d_ce, g.lin + g.quad),
            r2_kl_quad=r2(g.d_kl, g.quad),
            spearman_kl_quad=g.d_kl.corr(g.quad, method="spearman"),
        ))
    return pd.DataFrame(out)


def exp1(df):
    g = df.groupby(["model", "layer"])
    return g.agg(kl_obs=("kl_obs", "mean"), kl_null=("kl_null", "mean"), reff=("reff", "mean")).reset_index()


def exp2(df):
    rows = []
    for (m, ds), g in df.groupby(["model", "dataset"]):
        base = {k: g[g.method == k].delta_ppl.min() for k in ["svd", "lora"]}
        for _, r in g[g.method == "geom"].iterrows():
            rows.append(dict(model=m, dataset=ds, tau=r.tau, delta_ppl_geom=r.delta_ppl,
                             adv_vs_svd=base["svd"] - r.delta_ppl,
                             adv_vs_lora=base["lora"] - r.delta_ppl,
                             adv_vs_best=min(base.values()) - r.delta_ppl))
    return pd.DataFrame(rows)


def exp3(df):
    per_model = df.groupby(["model", "method", "drop"]).d_ce.mean().reset_index()
    out = per_model.groupby(["method", "drop"]).d_ce.mean().unstack("drop")
    fisher = out.loc["kappa"]
    adv = (out - fisher).add_prefix("adv_drop_")
    return pd.concat([out.add_prefix("dce_drop_"), adv], axis=1).reset_index()


def exp4(df):
    df = df.copy()
    rows = []
    for kd in ["fkl", "rkl", "skewkl"]:
        a = df[(df.kd == kd) & (df.geo == 0)].set_index(["model", "init", "rank", "seed"]).final_ppl
        b = df[(df.kd == kd) & (df.geo == 1)].set_index(["model", "init", "rank", "seed"]).final_ppl
        d = (a - b).dropna()
        if d.empty:
            continue
        inits = d.index.get_level_values("init")
        rows.append(dict(kd=kd, pairs=len(d), wins=int((d > 0).sum()), mean=d.mean(),
                         median=d.median(), svd_mean=d[inits == "svd"].mean(),
                         asvd_mean=d[inits == "asvd"].mean()))
    return pd.DataFrame(rows)


def checks(df):
    num = df.select_dtypes("number").columns.difference(["example", "layer"])
    med = df.groupby(["model", "direction_type"])[list(num)].median()
    med.insert(0, "n", df.groupby(["model", "direction_type"]).size())
    return med.reset_index()


def hessian(df):
    out = checks(df)
    sp = df.groupby(["model", "direction_type"]).apply(
        lambda g: g.vKv_fisher.corr(g.teacherKL_fd2, method="spearman")).rename("spearman_vKv_vs_teacherKL_fd2")
    cos = df.groupby(["model", "direction_type"]).cos_HCEv_Kv.mean().rename("mean_cos_HCEv_Kv")
    return out.merge(sp.reset_index()).merge(cos.reset_index())


def effrank(df):
    per_layer = df.groupby(["model", "layer"])[["fd_reff", "hutch_reff"]].mean().reset_index()
    rows = []
    for m, g in per_layer.groupby("model"):
        rel = (g.fd_reff - g.hutch_reff).abs() / g.hutch_reff
        rows.append(dict(model=m, layers=len(g), mean_fd_reff=g.fd_reff.mean(),
                         median_fd_reff=g.fd_reff.median(), mean_hutch_reff=g.hutch_reff.mean(),
                         median_hutch_reff=g.hutch_reff.median(), mean_rel_diff=rel.mean(),
                         median_rel_diff=rel.median(),
                         spearman=g.fd_reff.corr(g.hutch_reff, method="spearman")))
    return pd.DataFrame(rows)


JOBS = {
    "exp0": ("exp0/*.csv", exp0),
    "exp1": ("exp1/*.csv", exp1),
    "exp2": ("exp2/*.csv", exp2),
    "exp3": ("exp3/*.csv", exp3),
    "exp4": ("exp4/*.csv", exp4),
    "hessian_fisher": ("checks/hessian_*.csv", hessian),
    "future_leakage": ("checks/future_*.csv", checks),
    "effective_rank": ("checks/effrank_*.csv", effrank),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs")
    p.add_argument("--out", default="outputs/tables")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, (pattern, fn) in JOBS.items():
        df = read(os.path.join(args.root, pattern))
        if df is None:
            continue
        table = fn(df)
        table.to_csv(os.path.join(args.out, f"{name}.csv"), index=False)
        with open(os.path.join(args.out, f"{name}.md"), "w") as f:
            f.write(table.to_markdown(index=False, floatfmt=".4g"))
        print(name, len(table), "rows")


if __name__ == "__main__":
    main()
