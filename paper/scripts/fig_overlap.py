"""Figure 2: controlled overlap study (held-out base logit), from logbooks/runs/overlap_R9_s0.json."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
o = json.load(open(ROOT / "logbooks/runs/overlap_R9_s0.json"))
rows = o["rows"]
m = np.array([r["systems"] for r in rows])
cov = 100 * np.array([r["coverage"] for r in rows])
x = np.arange(len(m))
fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 2.25))
for key, lab, col, mk in [("focal only", "focal pair's labels only", ORANGE, "s"),
                          ("no item", "all labels, item effects suppressed", AQUA, "^"),
                          ("full", "all labels (full model)", BLUE, "o")]:
    a.plot(x, [r["alc"][key] for r in rows], color=col, lw=1.6, marker=mk, ms=5, mec="white", mew=0.7,
           label=lab, zorder=3)
a.set_xticks(x, [f"{k}" for k in m])
a.set_xlabel("systems sharing the item set")
a.set_ylabel("focal-pair Brier ALC")
a.set_title("(a) Same predictor, different evidence", fontsize=8, loc="left")
a.legend(frameon=False, fontsize=6.6, loc="lower left")
for key, lab, col, mk in [("partner evidence", "all partner evidence", BLUE, "o"),
                          ("item effects", "via shared item effects", AQUA, "^")]:
    g = np.array([r[key] for r in rows])
    b.errorbar(cov, g[:, 0], yerr=[g[:, 0] - g[:, 1], g[:, 2] - g[:, 0]], color=col, lw=1.6, marker=mk, ms=5,
               mec="white", mew=0.7, capsize=2.5, elinewidth=0.9, label=lab, zorder=3)
for c_, g_, n_ in zip(cov, [r["partner evidence"][0] for r in rows], m):
    b.annotate(f"{n_} sys.", (c_, g_), textcoords="offset points", xytext=(-4, 7), ha="right", fontsize=6.5,
               color=INK2)
b.axhline(0, color=INK2, lw=0.8)
b.set_xlabel("coverage at budget 31 (%)")
b.set_ylabel("ALC reduction")
b.set_title("(b) Gain with 95% benchmark intervals", fontsize=8, loc="left")
b.legend(frameon=False, fontsize=6.6, loc="upper left")
for ax in (a, b):
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
fig.tight_layout(pad=0.3, w_pad=1.4)
fig.savefig(ROOT / "paper/figs/overlap.pdf")
print("ok")
