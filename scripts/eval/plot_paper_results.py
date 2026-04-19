#!/usr/bin/env python3
"""Conference-paper figures from the three results text files.

Sources
-------
  results_increase_pieces.txt  — success rate vs n_pieces (3–12)
  results_id.txt               — in-distribution scalar metrics  (N=50)
  results_ood.txt              — out-of-distribution scalar metrics (N=30)

Output
------
  figures/paper_results.png / .pdf          — combined 3-panel figure
  figures/plot_scaling.png / .pdf           — panel (a) alone
  figures/plot_success.png / .pdf           — panel (b) alone
  figures/plot_time.png / .pdf              — panel (c) alone
  figures/table_id.png / .pdf              — in-distribution results table
  figures/table_ood.png / .pdf             — out-of-distribution results table
  figures/table_scaling.png / .pdf         — scaling with n_pieces table
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import norm as _norm

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except ImportError:
    print("[warn] scienceplots not found — using default style")

os.makedirs("figures", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Colours — Wong (2011) colorblind-safe palette
# ─────────────────────────────────────────────────────────────────────────────
_C = {
    "UNet + CBS":        "#0072B2",   # blue
    "RRT-Connect + CBS": "#E69F00",   # amber
    "UNet only":         "#009E73",   # green
}
_MK = {
    "UNet + CBS":        "o",
    "RRT-Connect + CBS": "s",
    "UNet only":         "^",
}
_METHODS = ["UNet + CBS", "RRT-Connect + CBS", "UNet only"]
_SHORT   = ["UNet+CBS", "RRT+CBS", "UNet only"]

# ─────────────────────────────────────────────────────────────────────────────
#  Data — transcribed verbatim from results files; every number double-checked
# ─────────────────────────────────────────────────────────────────────────────

# results_increase_pieces.txt
_N_PIECES = [3,  4,  5,  6, 7, 8, 9, 10, 11, 12]
_IP_COUNTS: dict[str, list[tuple[int, int]]] = {
    #                     n=3      n=4      n=5       n=6    n=7    n=8    n=9   n=10   n=11   n=12
    "UNet + CBS":        [(22,22),(11,12),(14,16),(3, 4),(5,6),(8,9),(3,3),(1,2),(3,3),(3,5)],
    "RRT-Connect + CBS": [(22,22),(12,12),(14,16),(4, 4),(5,6),(7,9),(3,3),(0,2),(2,3),(3,5)],
    "UNet only":         [( 9,22),( 1,12),( 0,16),(0, 4),(0,6),(0,9),(0,3),(0,2),(0,3),(0,5)],
}

# results_id.txt  (in-distribution, N=50)
_ID: dict[str, dict] = {
    "UNet + CBS":        {"k": 50, "n": 50, "succ": 100.0, "conf":  0.00, "time":   5.80, "fpe": 0.041, "arc": 0.173, "nodes":  6.3},
    "RRT-Connect + CBS": {"k": 48, "n": 50, "succ":  96.0, "conf":  0.22, "time":   1.54, "fpe": 0.000, "arc": 0.188, "nodes": 29.7},
    "UNet only":         {"k":  5, "n": 50, "succ":  10.0, "conf": 19.94, "time":   0.29, "fpe": 0.046, "arc": 0.186, "nodes":  0.0},
}

# results_ood.txt  (out-of-distribution, N=30)
_OOD: dict[str, dict] = {
    "UNet + CBS":        {"k": 26, "n": 30, "succ":  86.7, "conf":  1.17, "time": 556.29, "fpe": 0.000, "arc": 0.188, "nodes": 104.6},
    "RRT-Connect + CBS": {"k": 23, "n": 30, "succ":  76.7, "conf":  1.17, "time":  28.77, "fpe": 0.000, "arc": 0.178, "nodes": 125.7},
    "UNet only":         {"k":  0, "n": 30, "succ":   0.0, "conf": 75.07, "time":   0.64, "fpe": 0.000, "arc": 0.202, "nodes":   0.0},
}


# ─────────────────────────────────────────────────────────────────────────────
#  Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = float(_norm.ppf(1.0 - alpha / 2.0))
    p = k / n
    denom  = 1.0 + z ** 2 / n
    centre = (p + z ** 2 / (2.0 * n)) / denom
    half   = z * np.sqrt(p * (1.0 - p) / n + z ** 2 / (4.0 * n ** 2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _ci_errorbars(k: int, n: int) -> tuple[float, float]:
    p_pct = k / n * 100.0 if n > 0 else 0.0
    lo, hi = wilson_ci(k, n)
    return p_pct - lo * 100.0, hi * 100.0 - p_pct


# ─────────────────────────────────────────────────────────────────────────────
#  Shared grouped-bar geometry
# ─────────────────────────────────────────────────────────────────────────────
_bw   = 0.28
_gap  = 0.06
_x    = np.arange(len(_METHODS))
_x_id  = _x - (_bw + _gap) / 2
_x_ood = _x + (_bw + _gap) / 2

_leg_id_ood = [
    Patch(facecolor="grey", edgecolor="black", linewidth=0.5, label="ID ($N$=50)"),
    Patch(facecolor="grey", edgecolor="black", linewidth=0.5,
          alpha=0.45, hatch="//", label="OOD ($N$=30)"),
]


# ─────────────────────────────────────────────────────────────────────────────
#  Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_scaling(ax, title_prefix="(a) "):
    for m in _METHODS:
        xs, ys = [], []
        for x, (k, n) in zip(_N_PIECES, _IP_COUNTS[m]):
            xs.append(x)
            ys.append(k / n * 100.0)
        ax.plot(xs, ys, color=_C[m], marker=_MK[m],
                markersize=4.5, linewidth=1.4, label=m, zorder=3)
    ax.set_xlabel("Number of pieces")
    ax.set_ylabel("Success rate (\\%)")
    ax.set_xlim(2.5, 12.5)
    ax.set_ylim(-5, 113)
    ax.set_xticks(_N_PIECES)
    ax.legend(loc="lower left", frameon=False, fontsize=6.5,
              handlelength=1.5, handletextpad=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{title_prefix}Scaling with puzzle complexity",
                 loc="left", fontsize=7.5, pad=4)


def _draw_success(ax, title_prefix="(b) "):
    for i, m in enumerate(_METHODS):
        id_lo, id_hi = _ci_errorbars(_ID[m]["k"], _ID[m]["n"])
        ax.bar(_x_id[i], _ID[m]["succ"], width=_bw,
               color=_C[m], edgecolor="black", linewidth=0.5,
               yerr=[[id_lo], [id_hi]], capsize=3,
               error_kw={"elinewidth": 0.9}, zorder=3)
        ood_lo, ood_hi = _ci_errorbars(_OOD[m]["k"], _OOD[m]["n"])
        ax.bar(_x_ood[i], _OOD[m]["succ"], width=_bw,
               color=_C[m], edgecolor="black", linewidth=0.5,
               alpha=0.45, hatch="//",
               yerr=[[ood_lo], [ood_hi]], capsize=3,
               error_kw={"elinewidth": 0.9}, zorder=3)
    ax.set_xticks(_x)
    ax.set_xticklabels(_SHORT, fontsize=6.5, rotation=20, ha="right")
    ax.set_ylabel("Success rate (\\%)")
    ax.set_ylim(0, 118)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{title_prefix}Success rate", loc="left", fontsize=7.5, pad=4)
    ax.legend(handles=_leg_id_ood, loc="upper right", frameon=False,
              fontsize=6.0, handlelength=1.2, handletextpad=0.4)


def _draw_time(ax, title_prefix="(c) "):
    for i, m in enumerate(_METHODS):
        ax.bar(_x_id[i],  _ID[m]["time"],  width=_bw,
               color=_C[m], edgecolor="black", linewidth=0.5, zorder=3)
        ax.bar(_x_ood[i], _OOD[m]["time"], width=_bw,
               color=_C[m], edgecolor="black", linewidth=0.5,
               alpha=0.45, hatch="//", zorder=3)
        for x_pos, val in [(_x_id[i], _ID[m]["time"]), (_x_ood[i], _OOD[m]["time"])]:
            ax.text(x_pos, val * 2.2, f"{val:.3g}s",
                    ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(_x)
    ax.set_xticklabels(_SHORT, fontsize=6.5, rotation=20, ha="right")
    ax.set_ylabel("Planning time (s)")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 8000)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{title_prefix}Planning time", loc="left", fontsize=7.5, pad=4)
    ax.legend(handles=_leg_id_ood, loc="upper left", frameon=False,
              fontsize=6.0, handlelength=1.2, handletextpad=0.4)


def _save(fig, stem):
    fig.savefig(f"figures/{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"figures/{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: figures/{stem}.png / .pdf")


# ─────────────────────────────────────────────────────────────────────────────
#  Combined 3-panel figure
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(7.5, 3.0))
gs = gridspec.GridSpec(1, 3, left=0.08, right=0.98, top=0.88, bottom=0.20,
                       wspace=0.44, width_ratios=[1.6, 1.0, 1.0])
_draw_scaling(fig.add_subplot(gs[0]))
_draw_success(fig.add_subplot(gs[1]))
_draw_time(fig.add_subplot(gs[2]))
_save(fig, "paper_results")


# ─────────────────────────────────────────────────────────────────────────────
#  Individual plot figures
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(4.0, 3.0))
plt.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.16)
_draw_scaling(ax, title_prefix="")
ax.set_title("Scaling with puzzle complexity", loc="left", fontsize=8, pad=4)
_save(fig, "plot_scaling")

fig, ax = plt.subplots(figsize=(3.2, 3.0))
plt.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.22)
_draw_success(ax, title_prefix="")
ax.set_title("Success rate", loc="left", fontsize=8, pad=4)
_save(fig, "plot_success")

fig, ax = plt.subplots(figsize=(3.2, 3.0))
plt.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.22)
_draw_time(ax, title_prefix="")
ax.set_title("Planning time", loc="left", fontsize=8, pad=4)
_save(fig, "plot_time")


# ─────────────────────────────────────────────────────────────────────────────
#  Table helper — booktabs style via matplotlib table widget
# ─────────────────────────────────────────────────────────────────────────────

def _best_cells_by_col(cell_data, col_directions):
    """Return set of (row, col) to bold.
    col_directions: list of 'max'/'min'/None per column (None = skip).
    Handles ties: all tied-best values are bolded.
    """
    bold = set()
    n_rows = len(cell_data)
    for j, direction in enumerate(col_directions):
        if direction is None:
            continue
        vals = []
        for i in range(n_rows):
            try:
                vals.append(float(cell_data[i][j].rstrip("%").split()[0]))
            except (ValueError, IndexError):
                vals.append(None)
        valid = [v for v in vals if v is not None]
        if not valid:
            continue
        best = max(valid) if direction == "max" else min(valid)
        for i, v in enumerate(vals):
            if v == best:
                bold.add((i, j))
    return bold


def _best_cells_by_row(cell_data, direction="max"):
    """Bold best value in each row (for scaling table)."""
    bold = set()
    for i, row in enumerate(cell_data):
        vals = []
        for j, cell in enumerate(row):
            try:
                vals.append(float(cell.split()[0].rstrip("%")))
            except (ValueError, IndexError):
                vals.append(None)
        valid = [v for v in vals if v is not None]
        if not valid:
            continue
        best = max(valid) if direction == "max" else min(valid)
        for j, v in enumerate(vals):
            if v == best:
                bold.add((i, j))
    return bold


def _save_table(col_labels, row_labels, cell_data, title, stem,
                row_label_header="Method", figsize=(6.5, None), fontsize=8.5,
                bold_cells=None):
    n_rows = len(row_labels)
    h = figsize[1] or max(1.6, n_rows * 0.34 + 1.0)
    fig, ax = plt.subplots(figsize=(figsize[0], h))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.5)

    for (i, j), cell in tbl.get_celld().items():
        cell.set_edgecolor("none")
        cell.set_facecolor("white")
        if i == 0:
            cell.set_text_props(fontweight="bold")
        elif j == -1:
            cell.set_text_props(fontweight="bold", ha="right")
        elif bold_cells and (i - 1, j) in bold_cells:
            cell.set_text_props(fontweight="bold")

    ax.set_title(title, fontsize=9, fontweight="bold", pad=4, loc="left", x=0.0)
    fig.tight_layout(pad=0.5)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()

    tbl_bb = tbl.get_window_extent(renderer)
    hdr_bb = tbl[0, 0].get_window_extent(renderer)

    def _wx(x, y): return inv.transform((x, y))

    x0    = _wx(tbl_bb.x0, 0)[0]
    x1    = _wx(tbl_bb.x1, 0)[0]
    y_top = _wx(0, tbl_bb.y1)[1]
    y_mid = _wx(0, hdr_bb.y0)[1]
    y_bot = _wx(0, tbl_bb.y0)[1]

    for y, lw in [(y_top, 1.2), (y_mid, 0.6), (y_bot, 1.2)]:
        ax.plot([x0, x1], [y, y], color="black", linewidth=lw,
                transform=ax.transAxes, clip_on=False, zorder=10)

    _save(fig, stem)


# ─────────────────────────────────────────────────────────────────────────────
#  Table: in-distribution results
# ─────────────────────────────────────────────────────────────────────────────

_scalar_cols = ["N", "Succ %", "Conflicts", "Plan (s)", "FPE", "Arc (m)", "Nodes"]
_id_rows = [
    [str(_ID[m]["n"]),
     f"{_ID[m]['succ']:.1f}%",
     f"{_ID[m]['conf']:.2f}",
     f"{_ID[m]['time']:.2f}",
     f"{_ID[m]['fpe']:.3f}",
     f"{_ID[m]['arc']:.3f}",
     "N/A" if m == "UNet only" else f"{_ID[m]['nodes']:.1f}"]
    for m in _METHODS
]
# N: skip; Succ%: max; Conf: min; Plan: min; FPE: min; Arc: min; Nodes: min
_id_bold = _best_cells_by_col(_id_rows, [None, "max", "min", "min", "min", "min", "min"])
_save_table(_scalar_cols, _METHODS, _id_rows,
            title="In-distribution results (ID, N=50)",
            stem="table_id", figsize=(7.2, 1.8), bold_cells=_id_bold)

# ─────────────────────────────────────────────────────────────────────────────
#  Table: out-of-distribution results
# ─────────────────────────────────────────────────────────────────────────────

_ood_rows = [
    [str(_OOD[m]["n"]),
     f"{_OOD[m]['succ']:.1f}%",
     f"{_OOD[m]['conf']:.2f}",
     f"{_OOD[m]['time']:.2f}",
     f"{_OOD[m]['fpe']:.3f}",
     f"{_OOD[m]['arc']:.3f}",
     "N/A" if m == "UNet only" else f"{_OOD[m]['nodes']:.1f}"]
    for m in _METHODS
]
_ood_bold = _best_cells_by_col(_ood_rows, [None, "max", "min", "min", "min", "min", "min"])
_save_table(_scalar_cols, _METHODS, _ood_rows,
            title="Out-of-distribution results (OOD, N=30)",
            stem="table_ood", figsize=(7.2, 1.8), bold_cells=_ood_bold)

# ─────────────────────────────────────────────────────────────────────────────
#  Table: scaling with n_pieces
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_cell(k, n):
    pct = k / n * 100.0
    return f"{pct:.1f}%  ({k}/{n})"

_scale_row_labels = [str(n) for n in _N_PIECES] + ["All"]
_scale_cell_data  = []
for np_idx in range(len(_N_PIECES)):
    _scale_cell_data.append([_fmt_cell(*_IP_COUNTS[m][np_idx]) for m in _METHODS])

_all_row = []
for m in _METHODS:
    k_tot = sum(k for k, _ in _IP_COUNTS[m])
    n_tot = sum(n for _, n in _IP_COUNTS[m])
    _all_row.append(_fmt_cell(k_tot, n_tot))
_scale_cell_data.append(_all_row)

_scale_bold = _best_cells_by_row(_scale_cell_data, direction="max")
_save_table(
    col_labels=["UNet + CBS", "RRT-Connect + CBS", "UNet only"],
    row_labels=_scale_row_labels,
    cell_data=_scale_cell_data,
    title="Success rate vs. number of pieces",
    stem="table_scaling",
    row_label_header="Pieces",
    figsize=(7.2, None),
    fontsize=8.0,
    bold_cells=_scale_bold,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Table: success rate — ID vs OOD side by side  (matches plot b)
# ─────────────────────────────────────────────────────────────────────────────

_succ_rows = [
    [f"{_ID[m]['succ']:.1f}%", f"{_OOD[m]['succ']:.1f}%"]
    for m in _METHODS
]
_succ_bold = _best_cells_by_col(_succ_rows, ["max", "max"])
_save_table(
    col_labels=["ID (N=50)  Succ %", "OOD (N=30)  Succ %"],
    row_labels=_METHODS,
    cell_data=_succ_rows,
    title="Success rate",
    stem="table_success",
    figsize=(4.8, 1.8),
    bold_cells=_succ_bold,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Table: planning time — ID vs OOD side by side  (matches plot c)
# ─────────────────────────────────────────────────────────────────────────────

_time_rows = [
    [f"{_ID[m]['time']:.2f}", f"{_OOD[m]['time']:.2f}"]
    for m in _METHODS
]
_time_bold = _best_cells_by_col(_time_rows, ["min", "min"])
_save_table(
    col_labels=["ID (N=50)  Plan (s)", "OOD (N=30)  Plan (s)"],
    row_labels=_METHODS,
    cell_data=_time_rows,
    title="Planning time",
    stem="table_time",
    figsize=(4.8, 1.8),
    bold_cells=_time_bold,
)
