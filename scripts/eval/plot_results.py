"""Generate publication-quality figures from compare_methods.py results.csv.

Figures produced
----------------
  fig1_success_rate.pdf       Bar chart: success rate per method
  fig2_conflict_count.pdf     Violin: conflict count distribution per method
  fig3_conflict_time_ratio.pdf Bar chart: mean conflict time ratio per method
  fig4_success_vs_npieces.pdf  Line: success rate vs N_pieces
  fig5_time_vs_npieces.pdf     Line: mean planning time vs N_pieces (log y)
  fig6_time_vs_conflicts.pdf   Scatter: planning time vs num_conflicts (Pareto)

All figures saved as both .pdf (for LaTeX) and .png (for quick inspection).

Usage
-----
    python scripts/eval/plot_results.py \\
        --results eval_output/results.csv \\
        --output_dir eval_output/figures

    # Custom method order / labels:
    python scripts/eval/plot_results.py \\
        --results eval_output/results.csv \\
        --output_dir eval_output/figures \\
        --methods rrt rrt_cbs diff diff_cbs \\
        --labels "RRT" "RRT+CBS" "Diffusion" "Diff+CBS"
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
from matplotlib.lines import Line2D
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
#  Style constants
# ---------------------------------------------------------------------------

# Colorblind-safe palette (Wong 2011)
PALETTE = {
    "rrt":      "#E69F00",  # orange
    "rrt_cbs":  "#56B4E9",  # sky blue
    "diff":     "#009E73",  # green
    "diff_cbs": "#CC79A7",  # pink/purple
}
DEFAULT_METHODS = ["rrt", "rrt_cbs", "diff", "diff_cbs"]
DEFAULT_LABELS = {
    "rrt":      "RRT",
    "rrt_cbs":  "RRT+CBS",
    "diff":     "Diffusion",
    "diff_cbs": "Diff+CBS",
}

FIGURE_WIDTH = 3.5   # inches — fits a two-column IEEE / NeurIPS column
FIGURE_HEIGHT = 2.8
DPI = 300
FONT_SIZE = 9

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.dpi": DPI,
    "pdf.fonttype": 42,   # embed fonts for PDF
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
#  Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate publication figures from results.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results", type=str, required=True,
                   help="Path to results.csv produced by compare_methods.py.")
    p.add_argument("--output_dir", type=str, default="eval_output/figures",
                   help="Directory to save figures.")
    p.add_argument("--methods", type=str, nargs="+", default=DEFAULT_METHODS,
                   help="Methods to include (in display order).")
    p.add_argument("--labels", type=str, nargs="+", default=None,
                   help="Display labels for each method (must match --methods length).")
    p.add_argument("--no_pdf", action="store_true",
                   help="Skip PDF output; save PNG only.")
    return p.parse_args()


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, path_no_ext: str, no_pdf: bool) -> None:
    """Save figure as PNG (and optionally PDF)."""
    fig.savefig(path_no_ext + ".png", bbox_inches="tight", dpi=DPI)
    if not no_pdf:
        fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _color(method: str) -> str:
    return PALETTE.get(method, "#333333")


def _valid_rows(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Return rows for method with valid (non-error) results."""
    mask = (df["method"] == method) & (df["num_conflicts"] >= 0)
    return df[mask]


def _add_significance_bar(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    h: float,
    p_val: float,
) -> None:
    """Draw a significance bracket between positions x1 and x2."""
    if p_val < 0.001:
        sig_label = "***"
    elif p_val < 0.01:
        sig_label = "**"
    elif p_val < 0.05:
        sig_label = "*"
    else:
        return  # not significant; skip
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c="k")
    ax.text((x1 + x2) / 2, y + h, sig_label, ha="center", va="bottom",
            fontsize=FONT_SIZE - 1)


# ---------------------------------------------------------------------------
#  Figure 1: Success rate bar chart
# ---------------------------------------------------------------------------

def _wilson_ci(k: int, n: int, alpha: float = 0.05):
    """Wilson score confidence interval for a proportion.

    Returns (lo, hi) in [0, 1].  Falls back to normal approximation if
    statsmodels is not installed.
    """
    if n == 0:
        return 0.0, 0.0
    try:
        from statsmodels.stats.proportion import proportion_confint  # type: ignore
        return proportion_confint(k, n, alpha=alpha, method="wilson")
    except ImportError:
        pass
    # Normal approximation fallback
    z = scipy_stats.norm.ppf(1 - alpha / 2)
    p = k / n
    margin = z * (p * (1 - p) / n) ** 0.5
    return max(0.0, p - margin), min(1.0, p + margin)


def fig_success_rate(
    df: pd.DataFrame,
    methods: List[str],
    labels: Dict[str, str],
    out_dir: str,
    no_pdf: bool,
) -> None:
    """Bar chart: success rate per method with 95% Wilson CI error bars."""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    x = np.arange(len(methods))
    success_rates, ci_lo, ci_hi = [], [], []

    for m in methods:
        sub = _valid_rows(df, m)
        n = len(sub)
        k = int(sub["success"].sum())
        rate = k / n if n > 0 else 0.0
        success_rates.append(rate * 100)
        if n > 0:
            lo, hi = _wilson_ci(k, n, alpha=0.05)
            ci_lo.append((rate - lo) * 100)
            ci_hi.append((hi - rate) * 100)
        else:
            ci_lo.append(0)
            ci_hi.append(0)

    bars = ax.bar(
        x, success_rates,
        color=[_color(m) for m in methods],
        edgecolor="black",
        linewidth=0.6,
        width=0.55,
        yerr=[ci_lo, ci_hi],
        capsize=3,
        error_kw={"elinewidth": 0.8},
    )

    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in methods], rotation=15, ha="right")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Success rate (collision-free plans)")

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig1_success_rate"), no_pdf)


# ---------------------------------------------------------------------------
#  Figure 2: Conflict count violin
# ---------------------------------------------------------------------------

def fig_conflict_violin(
    df: pd.DataFrame,
    methods: List[str],
    labels: Dict[str, str],
    out_dir: str,
    no_pdf: bool,
) -> None:
    """Violin plot: distribution of num_conflicts per method."""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    data = [_valid_rows(df, m)["num_conflicts"].values for m in methods]

    parts = ax.violinplot(
        data,
        positions=range(len(methods)),
        showmedians=True,
        showextrema=False,
    )

    # Color each violin
    for pc, method in zip(parts["bodies"], methods):
        pc.set_facecolor(_color(method))
        pc.set_alpha(0.7)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.5)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.0)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([labels[m] for m in methods], rotation=15, ha="right")
    ax.set_ylabel("Number of conflicts")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Conflict count distribution")

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig2_conflict_count"), no_pdf)


# ---------------------------------------------------------------------------
#  Figure 3: Conflict time ratio bar chart
# ---------------------------------------------------------------------------

def fig_conflict_time_ratio(
    df: pd.DataFrame,
    methods: List[str],
    labels: Dict[str, str],
    out_dir: str,
    no_pdf: bool,
) -> None:
    """Bar chart: mean conflict time ratio (± SEM) per method.

    The conflict time ratio is the fraction of timesteps during which at least
    one pair of pieces overlaps.  It captures *temporal coverage* of conflicts,
    not just their count.  A plan with conflicts concentrated in one timestep
    scores much lower than one with conflicts spread across many timesteps.
    """
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    x = np.arange(len(methods))
    means, sems = [], []

    for m in methods:
        vals = _valid_rows(df, m)["conflict_time_ratio"].dropna().values
        means.append(vals.mean() if len(vals) > 0 else 0.0)
        sems.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)

    ax.bar(
        x, means,
        color=[_color(m) for m in methods],
        edgecolor="black",
        linewidth=0.6,
        width=0.55,
        yerr=sems,
        capsize=3,
        error_kw={"elinewidth": 0.8},
    )
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in methods], rotation=15, ha="right")
    ax.set_ylabel("Conflict time ratio")
    ax.set_ylim(0, max(means) * 1.3 + 0.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Conflict time ratio\n(fraction of timesteps with any overlap)")

    # Add Wilcoxon p-values comparing CBS vs non-CBS within same backbone
    pairs = [("rrt", "rrt_cbs"), ("diff", "diff_cbs")]
    y_offset = max(means) * 1.15 + 0.02
    for (m1, m2) in pairs:
        if m1 not in methods or m2 not in methods:
            continue
        v1 = _valid_rows(df, m1)["conflict_time_ratio"].dropna().values
        v2 = _valid_rows(df, m2)["conflict_time_ratio"].dropna().values
        if len(v1) < 5 or len(v2) < 5:
            continue
        # Paired Wilcoxon (same episodes)
        min_n = min(len(v1), len(v2))
        _, p = scipy_stats.wilcoxon(v1[:min_n], v2[:min_n])
        _add_significance_bar(
            ax,
            methods.index(m1), methods.index(m2),
            y_offset, 0.01, p,
        )

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig3_conflict_time_ratio"), no_pdf)


# ---------------------------------------------------------------------------
#  Figure 4: Success rate vs N_pieces
# ---------------------------------------------------------------------------

def fig_success_vs_npieces(
    df: pd.DataFrame,
    methods: List[str],
    labels: Dict[str, str],
    out_dir: str,
    no_pdf: bool,
) -> None:
    """Line chart: success rate vs number of puzzle pieces."""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    for m in methods:
        sub = _valid_rows(df, m)
        grouped = sub.groupby("n_pieces")["success"].agg(
            rate=lambda x: x.mean() * 100,
            n="count",
        ).reset_index()
        if grouped.empty:
            continue
        ax.plot(
            grouped["n_pieces"], grouped["rate"],
            marker="o", markersize=4, linewidth=1.2,
            color=_color(m), label=labels[m],
        )

    ax.set_xlabel("Number of pieces")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("Success rate vs puzzle complexity")
    ax.legend(loc="lower left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig4_success_vs_npieces"), no_pdf)


# ---------------------------------------------------------------------------
#  Figure 5: Planning time vs N_pieces (log y)
# ---------------------------------------------------------------------------

def fig_time_vs_npieces(
    df: pd.DataFrame,
    methods: List[str],
    labels: Dict[str, str],
    out_dir: str,
    no_pdf: bool,
) -> None:
    """Line chart: mean planning time vs N_pieces with log-y axis."""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    for m in methods:
        sub = _valid_rows(df, m)
        grouped = sub.groupby("n_pieces")["planning_time_s"].agg(
            mean="mean",
            sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0,
        ).reset_index()
        if grouped.empty:
            continue
        ax.errorbar(
            grouped["n_pieces"], grouped["mean"],
            yerr=grouped["sem"],
            marker="o", markersize=4, linewidth=1.2, capsize=3,
            color=_color(m), label=labels[m],
        )

    ax.set_xlabel("Number of pieces")
    ax.set_ylabel("Planning time (s)")
    ax.set_yscale("log")
    ax.set_title("Planning time vs puzzle complexity")
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig5_time_vs_npieces"), no_pdf)


# ---------------------------------------------------------------------------
#  Figure 6: Time vs conflicts scatter (Pareto front)
# ---------------------------------------------------------------------------

def fig_time_vs_conflicts(
    df: pd.DataFrame,
    methods: List[str],
    labels: Dict[str, str],
    out_dir: str,
    no_pdf: bool,
) -> None:
    """Scatter: per-episode (planning_time, num_conflicts) per method.

    Draws a 50th percentile marker and, for methods with enough points,
    a rough Pareto front annotation.
    """
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH * 1.2, FIGURE_HEIGHT))

    legend_handles = []
    for m in methods:
        sub = _valid_rows(df, m)
        if sub.empty:
            continue
        t = sub["planning_time_s"].values
        c = sub["num_conflicts"].values
        ax.scatter(
            t, c,
            color=_color(m), alpha=0.35, s=12, edgecolors="none",
        )
        # Median marker
        ax.scatter(
            [np.median(t)], [np.median(c)],
            color=_color(m), s=60, marker="D",
            edgecolors="black", linewidths=0.5, zorder=5,
        )
        patch = mpatches.Patch(color=_color(m), label=labels[m])
        legend_handles.append(patch)

    ax.set_xlabel("Planning time (s)")
    ax.set_ylabel("Number of conflicts")
    ax.set_title("Time–quality trade-off\n(◆ = median)")
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig6_time_vs_conflicts"), no_pdf)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.results)
    # Cast types in case CSV loaded everything as strings
    df["success"] = df["success"].astype(bool)
    df["num_conflicts"] = pd.to_numeric(df["num_conflicts"], errors="coerce").fillna(-1).astype(int)
    for col in ("conflict_time_ratio", "planning_time_s", "arc_length",
                "smoothness", "final_pose_error"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["n_pieces"] = pd.to_numeric(df["n_pieces"], errors="coerce").astype(int)

    # Filter to requested methods and order
    methods = [m for m in args.methods if m in df["method"].unique()]
    if not methods:
        raise ValueError(
            f"None of the requested methods {args.methods} found in {args.results}. "
            f"Available: {df['method'].unique().tolist()}"
        )

    # Build label map
    if args.labels is not None:
        if len(args.labels) != len(args.methods):
            raise ValueError("--labels must have the same length as --methods")
        label_map = dict(zip(args.methods, args.labels))
    else:
        label_map = {**DEFAULT_LABELS}
        for m in methods:
            if m not in label_map:
                label_map[m] = m

    print(f"Loaded {len(df)} rows for methods: {methods}")
    print(f"Figures will be saved to: {args.output_dir}")

    fig_success_rate(df, methods, label_map, args.output_dir, args.no_pdf)
    print("  [1/6] fig1_success_rate")

    fig_conflict_violin(df, methods, label_map, args.output_dir, args.no_pdf)
    print("  [2/6] fig2_conflict_count")

    fig_conflict_time_ratio(df, methods, label_map, args.output_dir, args.no_pdf)
    print("  [3/6] fig3_conflict_time_ratio")

    fig_success_vs_npieces(df, methods, label_map, args.output_dir, args.no_pdf)
    print("  [4/6] fig4_success_vs_npieces")

    fig_time_vs_npieces(df, methods, label_map, args.output_dir, args.no_pdf)
    print("  [5/6] fig5_time_vs_npieces")

    fig_time_vs_conflicts(df, methods, label_map, args.output_dir, args.no_pdf)
    print("  [6/6] fig6_time_vs_conflicts")

    print("Done.")


if __name__ == "__main__":
    main()
