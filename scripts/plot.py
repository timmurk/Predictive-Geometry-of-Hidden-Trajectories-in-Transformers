import argparse
import glob
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator, NullFormatter

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})
RED, BLUE, GREEN, PURPLE, GRAY = "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#7f8c8d"


def read(pattern):
    files = sorted(glob.glob(pattern))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


def grid(n):
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.8 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    return fig, axes.flat


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def exp0(tables, out):
    df = pd.read_csv(os.path.join(tables, "exp0.csv"))
    mean = df.groupby("rho").mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col, lab, c in [("r2_ce_linear", "CE linear", BLUE), ("r2_ce_lin_quad", "CE linear + quad", PURPLE),
                        ("r2_ce_quad", "CE quad", GRAY), ("r2_kl_quad", "teacher-KL quad", RED)]:
        ax.plot(mean.index, mean[col].clip(-1, 1), "o-", color=c, lw=2.5, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel(r"relative radius $\rho$")
    ax.set_ylabel(r"$R^2$")
    ax.legend()
    save(fig, os.path.join(out, "exp0_r2_by_radius.png"))


def exp1(tables, out):
    df = pd.read_csv(os.path.join(tables, "exp1.csv"))
    models = list(df.model.unique())
    fig, axes = grid(len(models))
    for ax, m in zip(axes, models):
        g = df[df.model == m]
        ax.plot(g.layer, g.kl_obs, color=RED, lw=2.5, label="observable")
        ax.plot(g.layer, g.kl_null, "--", color=BLUE, lw=2.5, label="null")
        ax.set_yscale("log")
        ax.set_title(m)
        ax.set_xlabel("Layer")
        ax.set_ylabel(r"$\Delta$KL")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].legend()
    save(fig, os.path.join(out, "exp1_kl.png"))
    fig, axes = grid(len(models))
    for ax, m in zip(axes, models):
        g = df[df.model == m]
        ax.plot(g.layer, g.reff, color=GREEN, lw=2.5)
        ax.set_title(m)
        ax.set_xlabel("Layer")
        ax.set_ylabel(r"$r_{\mathrm{eff}}$")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    save(fig, os.path.join(out, "exp1_rank.png"))


def exp2(tables, out):
    df = pd.read_csv(os.path.join(tables, "exp2.csv"))
    for (ds, tau), g in df.groupby(["dataset", "tau"]):
        ds = os.path.splitext(os.path.basename(str(ds)))[0]
        g = g.iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 0.55 * len(g) + 1.8))
        y = range(len(g))
        ax.barh([i + 0.2 for i in y], g.adv_vs_lora, height=0.38, color=RED, label="vs LoRA")
        ax.barh([i - 0.2 for i in y], g.adv_vs_svd, height=0.38, color=BLUE, label="vs SVD")
        ax.set_yticks(list(y))
        ax.set_yticklabels(g.model)
        ax.axvline(0, color="black", lw=1.5)
        lim = max(g.adv_vs_lora.abs().max(), g.adv_vs_svd.abs().max()) * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_xlabel(r"$\Delta$PPL advantage")
        ax.set_title(fr"$\tau={tau}$ ({ds})", pad=36)
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
        save(fig, os.path.join(out, f"exp2_advantage_{ds}_tau{str(tau).replace('.', '_')}.png"))


def exp3(tables, out):
    df = pd.read_csv(os.path.join(tables, "exp3.csv")).set_index("method")
    adv = df[[c for c in df.columns if c.startswith("adv_drop_")]].drop("kappa")
    drops = [c.replace("adv_drop_", "") for c in adv.columns]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.8 / len(adv)
    for j, (m, r) in enumerate(adv.iterrows()):
        ax.bar([i + j * width for i in range(len(drops))], r.values, width, label=m)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(drops))])
    ax.set_xticklabels(drops)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("drop fraction")
    ax.set_ylabel(r"$\Delta$CE$_{\mathrm{base}}-\Delta$CE$_{\kappa}$")
    ax.legend(ncol=2, fontsize=12)
    save(fig, os.path.join(out, "exp3_fisher_advantage.png"))


def checks(root, out):
    df = read(os.path.join(root, "checks", "hessian_*.csv"))
    if df is not None:
        top = df[df.direction_type == "top_fisher"]
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for m, g in top.groupby("model"):
            ax.scatter(g.vKv_fisher, g.teacherKL_fd2, s=30, alpha=0.8, label=m)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$v^\top K v$")
        ax.set_ylabel("teacher-KL curvature (FD)")
        ax.legend()
        save(fig, os.path.join(out, "check_teacherKL_vs_vKv.png"))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for m, g in top.groupby("model"):
            c = g.groupby("layer").cos_HCEv_Kv.mean()
            ax.plot(c.index, c.values, "o-", lw=2.5, label=m)
        ax.set_xlabel("Layer")
        ax.set_ylabel(r"cos$(H_{CE}v, Kv)$")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        save(fig, os.path.join(out, "check_hessian_cosine_by_layer.png"))
    df = read(os.path.join(root, "checks", "future_*.csv"))
    if df is not None:
        med = df.groupby(["model", "direction_type"]).future_sum_fd2.median().reset_index()
        fig, ax = plt.subplots(figsize=(9, 0.45 * len(med) + 1.5))
        ax.barh(med.model + " / " + med.direction_type, med.future_sum_fd2, color=BLUE)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("median future-position KL curvature")
        save(fig, os.path.join(out, "check_future_by_direction.png"))
    df = read(os.path.join(root, "checks", "effrank_*.csv"))
    if df is not None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for m, g in df.groupby("model"):
            g = g.groupby("layer").mean(numeric_only=True)
            axes[0].plot(g.index, g.hutch_reff, "o-", lw=2.5, label=m)
            axes[1].plot(g.index, g.fd_reff, "o-", lw=2.5, label=m)
        axes[0].set_title("Hutchinson")
        axes[1].set_title("finite-difference moments")
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("Layer")
            ax.set_ylabel(r"$r_{\mathrm{eff}}$")
        axes[0].legend()
        save(fig, os.path.join(out, "check_effective_rank.png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs")
    p.add_argument("--out", default="outputs/figures")
    args = p.parse_args()
    tables = os.path.join(args.root, "tables")
    for fn in [exp0, exp1, exp2, exp3]:
        if os.path.exists(os.path.join(tables, fn.__name__ + ".csv")):
            fn(tables, args.out)
    checks(args.root, args.out)


if __name__ == "__main__":
    main()
