#!/usr/bin/env python3
"""Produce the Layer-2 sweep figures and summary tables.

Reads sweep/results/summary.csv (no header), writes:
    analysis/figures/heatmap.png       — val_loss by (config, WD), seed-averaged
    analysis/figures/pareto.png        — val_loss vs wall-clock by config
    analysis/figures/dropout.png       — dropout=0 vs dropout=0.1, by config x WD
    analysis/figures/seed_variance.png — std across 3 seeds per cell
    analysis/figures/summary_table.txt — best-cell-per-config table

Run from project root:
    python analysis/plot_sweep.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "sweep" / "results" / "summary.csv"
FIG = ROOT / "analysis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

COLS = ["idx", "name", "optimizer", "use_sam", "wd", "dropout", "seed",
        "val_loss", "val_bpb", "train_time"]


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV, header=None, names=COLS)
    # train_time is like "33.72m" — strip suffix
    df["train_time_min"] = df["train_time"].str.rstrip("m").astype(float)
    # Derive a config label per row
    df["config"] = df.apply(
        lambda r: f"{r['optimizer']}{'+SAM' if r['use_sam'] == 1 else ''}", axis=1,
    )
    return df.sort_values(["config", "wd", "dropout", "seed"]).reset_index(drop=True)


CONFIG_ORDER = ["adamw", "adamw+SAM", "muon", "muon+SAM"]
CONFIG_COLOR = {
    "adamw":     "#1f77b4",
    "adamw+SAM": "#aec7e8",
    "muon":      "#d62728",
    "muon+SAM":  "#ff9896",
}


def plot_heatmap(df: pd.DataFrame) -> None:
    """Val-loss heatmap: rows = (config, dropout), cols = WD, color = mean val_loss."""
    agg = (df.groupby(["config", "dropout", "wd"], as_index=False)["val_loss"]
             .mean())
    wds = sorted(df["wd"].unique())
    dropouts = sorted(df["dropout"].unique())

    fig, axes = plt.subplots(1, len(dropouts), figsize=(7 * len(dropouts), 4),
                             sharey=True)
    if len(dropouts) == 1:
        axes = [axes]
    vmin = agg["val_loss"].min()
    vmax = agg["val_loss"].max()
    for ax, dr in zip(axes, dropouts):
        sub = agg[agg["dropout"] == dr]
        mat = sub.pivot(index="config", columns="wd", values="val_loss")
        mat = mat.reindex(CONFIG_ORDER)
        im = ax.imshow(mat.values, aspect="auto", cmap="viridis_r",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(wds)), [str(w) for w in wds])
        ax.set_yticks(range(len(CONFIG_ORDER)), CONFIG_ORDER)
        ax.set_xlabel("weight decay")
        ax.set_title(f"dropout = {dr}")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=9, color="white" if v > (vmin+vmax)/2 else "black")
    fig.colorbar(im, ax=axes, label="val loss (mean over 3 seeds)", shrink=0.85)
    fig.suptitle("Layer-2 sweep: val loss by (config, WD, dropout)", y=1.02)
    fig.savefig(FIG / "heatmap.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG/'heatmap.png'}")


def plot_pareto(df: pd.DataFrame) -> None:
    """Val-loss vs wall-clock. Each point = one (cell, seed); markers = config."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for cfg in CONFIG_ORDER:
        sub = df[df["config"] == cfg]
        ax.scatter(sub["train_time_min"], sub["val_loss"],
                   s=42, alpha=0.75, edgecolor="black", linewidth=0.4,
                   color=CONFIG_COLOR[cfg], label=cfg)
    # Pareto front (lower-left) — naive sort-and-sweep
    pts = df.sort_values("train_time_min")[["train_time_min", "val_loss"]].values
    best = np.inf
    front_x, front_y = [], []
    for x, y in pts:
        if y < best:
            best = y
            front_x.append(x)
            front_y.append(y)
    ax.plot(front_x, front_y, "--", color="black", linewidth=1, alpha=0.5,
            label="empirical Pareto front")
    ax.set_xlabel("wall-clock time (min)")
    ax.set_ylabel("val loss")
    ax.set_title("Val loss vs wall-clock — does SAM's 2× cost pay off?")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG / "pareto.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG/'pareto.png'}")


def plot_dropout_interaction(df: pd.DataFrame) -> None:
    """For each config, plot val_loss vs WD with one line per dropout value."""
    fig, axes = plt.subplots(1, len(CONFIG_ORDER), figsize=(4*len(CONFIG_ORDER), 4),
                             sharey=True)
    for ax, cfg in zip(axes, CONFIG_ORDER):
        sub = df[df["config"] == cfg]
        for dr, marker in zip(sorted(df["dropout"].unique()), ["o", "s"]):
            sub_dr = sub[sub["dropout"] == dr]
            agg = sub_dr.groupby("wd")["val_loss"].agg(["mean", "std"]).reset_index()
            ax.errorbar(agg["wd"], agg["mean"], yerr=agg["std"], marker=marker,
                        capsize=3, label=f"dropout={dr}", linewidth=1.5)
        ax.set_xscale("log")
        ax.set_xlabel("weight decay")
        ax.set_title(cfg)
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("val loss (mean ± std over 3 seeds)")
        ax.legend(fontsize=9)
    fig.suptitle("WD curves by config, split by dropout", y=1.02)
    fig.savefig(FIG / "dropout.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG/'dropout.png'}")


def plot_seed_variance(df: pd.DataFrame) -> None:
    """Per-cell std across 3 seeds — tells you which differences are noise."""
    agg = (df.groupby(["config", "wd", "dropout"])["val_loss"]
             .std().reset_index().rename(columns={"val_loss": "std"}))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = {cfg: i for i, cfg in enumerate(CONFIG_ORDER)}
    for cfg in CONFIG_ORDER:
        sub = agg[agg["config"] == cfg]
        x_jitter = positions[cfg] + np.random.uniform(-0.18, 0.18, size=len(sub))
        ax.scatter(x_jitter, sub["std"], s=46, alpha=0.75,
                   edgecolor="black", linewidth=0.4,
                   color=CONFIG_COLOR[cfg], label=cfg)
    ax.set_xticks(list(positions.values()), CONFIG_ORDER)
    ax.set_ylabel("std of val_loss across 3 seeds (per cell)")
    ax.set_title("Seed variance per (config, WD, dropout) cell")
    ax.grid(True, alpha=0.3)
    median_std = agg["std"].median()
    ax.axhline(median_std, color="black", linestyle=":",
               label=f"median = {median_std:.4f}")
    ax.legend(fontsize=9)
    fig.savefig(FIG / "seed_variance.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG/'seed_variance.png'}")


def summary_table(df: pd.DataFrame) -> None:
    """Per-config: best cell, mean val_loss, seed std, wall-clock."""
    # Best cell per config: argmin over seed-averaged val_loss
    cell_mean = (df.groupby(["config", "wd", "dropout"])["val_loss"]
                   .agg(["mean", "std", "count"]).reset_index())
    best_per_config = (cell_mean.sort_values("mean")
                                .groupby("config", as_index=False).first()
                                .set_index("config")
                                .reindex(CONFIG_ORDER))

    # Average wall-clock per config
    wallclock = df.groupby("config")["train_time_min"].mean().reindex(CONFIG_ORDER)

    lines = []
    lines.append("LAYER-2 SWEEP SUMMARY")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Best (WD, dropout) cell per config — seed-averaged:")
    lines.append("")
    lines.append(f"  {'config':<14} {'WD':>6} {'drop':>6}  {'mean':>8} {'± std':>8} {'wall':>8}")
    lines.append("  " + "-" * 60)
    for cfg in CONFIG_ORDER:
        r = best_per_config.loc[cfg]
        wc = wallclock.loc[cfg]
        lines.append(f"  {cfg:<14} {r['wd']:>6.2f} {r['dropout']:>6.1f}  "
                     f"{r['mean']:>8.4f} {r['std']:>8.4f} {wc:>6.1f}m")
    lines.append("")

    # Pairwise gap matrix (best-vs-best)
    bests = {cfg: best_per_config.loc[cfg, "mean"] for cfg in CONFIG_ORDER}
    lines.append("Pairwise gap (row − col) at each config's best cell:")
    lines.append("")
    lines.append(f"  {'':<14}" + "".join(f"{c:>11}" for c in CONFIG_ORDER))
    for r_cfg in CONFIG_ORDER:
        row = f"  {r_cfg:<14}"
        for c_cfg in CONFIG_ORDER:
            gap = bests[r_cfg] - bests[c_cfg]
            row += f"{gap:>+11.4f}"
        lines.append(row)
    lines.append("")

    # SAM verdict
    lines.append("SAM verdict (lower = better):")
    lines.append("")
    sam_adamw_delta = bests["adamw+SAM"] - bests["adamw"]
    sam_muon_delta  = bests["muon+SAM"]  - bests["muon"]
    lines.append(f"  AdamW+SAM vs AdamW: {sam_adamw_delta:+.4f}  ({'SAM hurts' if sam_adamw_delta > 0 else 'SAM helps'})")
    lines.append(f"  Muon +SAM vs Muon : {sam_muon_delta:+.4f}  ({'SAM hurts' if sam_muon_delta > 0 else 'SAM helps'})")
    lines.append(f"  Muon vs AdamW (best-vs-best): {bests['muon'] - bests['adamw']:+.4f}")
    lines.append(f"  Muon+SAM vs AdamW+SAM        : {bests['muon+SAM'] - bests['adamw+SAM']:+.4f}")
    lines.append("")

    out = "\n".join(lines)
    print(out)
    (FIG / "summary_table.txt").write_text(out + "\n")
    print(f"  wrote {FIG/'summary_table.txt'}")


def main():
    if not CSV.exists():
        print(f"ERROR: {CSV} not found", file=sys.stderr)
        sys.exit(1)
    df = load()
    print(f"Loaded {len(df)} rows, {df['config'].nunique()} configs, "
          f"WDs={sorted(df['wd'].unique())}, dropouts={sorted(df['dropout'].unique())}, "
          f"seeds={sorted(df['seed'].unique())}")
    print()
    plot_heatmap(df)
    plot_pareto(df)
    plot_dropout_interaction(df)
    plot_seed_variance(df)
    print()
    summary_table(df)


if __name__ == "__main__":
    main()
