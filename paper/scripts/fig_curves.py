"""Figure: Brier score per checkpoint (paper Fig. curves), one-response runs pairs_R7_{std,high}.npz.
Checkpoints are placed on the official log2(1+k) axis."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
BUDGETS = np.array([0, 1, 3, 7, 15, 31])
X = np.log2(1 + BUDGETS)
SERIES = [("emp_mean", "Empirical Mean (official)", "#eb6834", "s"),
          ("beta11", "Beta(1,1) pair mean", "#eda100", "D"),
          ("online0", "hierarchical model, random acquisition", "#1baf7a", "^"),
          ("online0_f@g1c1", "v6 proxy (no encoder)", "#2a78d6", "o")]
panels = [("std", "(a) Standard regime"), ("high", "(b) High-overlap regime")]
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.25), sharey=True)
for ax, (reg, title) in zip(axes, panels):
    f = ROOT / f"logbooks/runs/pairs_R7_{reg}.npz"
    ax.set_title(title, fontsize=8, loc="left")
    ax.axhline(0.25, color=INK2, lw=0.8, ls=":", zorder=1)
    if not f.exists():
        ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", color=INK2)
        continue
    P = np.load(f)
    n = len(P["online0"])
    for key, lab, col, mk in SERIES:
        y = P[key].mean(0)
        se = P[key].std(0, ddof=1) / np.sqrt(n)
        ax.fill_between(X, y - 1.96 * se, y + 1.96 * se, color=col, alpha=0.13, lw=0)
        ax.plot(X, y, color=col, lw=1.5, marker=mk, ms=3.4, mec="white", mew=0.5, label=lab, zorder=3)
    ax.text(X[-1], 0.2515, "constant 0.5", ha="right", va="bottom", color=INK2, fontsize=6.5)
    ax.text(0.98, 0.97, f"{n:,} pairs", transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=6.5)
    ax.set_xticks(X, BUDGETS)
    ax.set_xlabel("labels acquired per pair (log$_2$(1+k) spacing)")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Mean Brier score")
axes[0].set_ylim(0.13, 0.385)
h, l = axes[1].get_legend_handles_labels() or axes[0].get_legend_handles_labels()
if not h:
    h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, 1.01))
fig.tight_layout(pad=0.3, w_pad=1.0, rect=(0, 0, 1, 0.86))
fig.savefig(ROOT / "paper/figs/curves.pdf")
print("ok")
