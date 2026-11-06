"""Figure (appendix): official formative scores of all uploaded packages; identical-v3 repeats and the selected v6."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
runs = {"v1": [0.197811], "v2": [0.194774], "v3": [0.169708, 0.192029, 0.176251, 0.199922],
        "v4": [0.186192, 0.185276], "v3+tok": [0.204440], "v3 cap1": [0.186417],
        "v3+name": [0.180691], "v3$-x$": [0.145828], "v6\n(selected)": [0.208649]}
allr = np.concatenate(list(runs.values()))
m, sd = allr.mean(), allr.std(ddof=1)
fig, a = plt.subplots(figsize=(3.6, 2.3))
a.axhspan(m - sd, m + sd, color=GRID, lw=0, zorder=0)
a.axhline(m, color=INK2, lw=0.8, ls="--", zorder=1)
for j, (k, v) in enumerate(runs.items()):
    c = BLUE if k == "v3" else (ORANGE if k.startswith("v6") else INK2)
    mk = "D" if k.startswith("v6") else "o"
    a.scatter(np.full(len(v), j) + np.linspace(-0.13, 0.13, len(v)) * (len(v) > 1), v, s=26, color=c, marker=mk,
              edgecolor="white", linewidth=0.6, zorder=3)
a.set_xticks(range(len(runs)), list(runs), rotation=40, ha="right", fontsize=7.5)
a.set_ylabel("Formative Brier ALC")
a.text(len(runs) - 0.6, m - sd - 0.0015, f"pooled mean {m:.3f} $\\pm$ 1 SD", ha="right", va="top", color=INK2, fontsize=7)
a.text(2.25, 0.166, "identical\nv3 zip", color=BLUE, fontsize=7.5, ha="left", va="top", linespacing=0.9)
a.set_ylim(0.14, 0.215)
a.grid(axis="y", color=GRID, lw=0.5)
a.set_axisbelow(True)
fig.tight_layout(pad=0.3)
fig.savefig(ROOT / "paper/figs/noise.pdf")
print("ok")
