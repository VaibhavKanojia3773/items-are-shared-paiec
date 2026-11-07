"""Independent check of the numbers in the paper.

1. Recompute every ALC macro from the raw per-pair arrays (own code path) and compare with results.tex.
2. Recompute overlap macros from the overlap JSON.
3. Check hard-coded numbers in main.tex against their source files.
    python paper/scripts/verify_numbers.py
"""
import json, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
W = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
mac = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (ROOT / "paper/results.tex").read_text(encoding="utf-8")))
tex = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
bad = []


def check(name, expected, got, tol=5e-5):
    try:
        g = float(str(got).replace("$-$", "-").replace("$+$", "+"))
    except ValueError:
        bad.append(f"{name}: unparsable {got!r}")
        return
    if abs(g - expected) > tol:
        bad.append(f"{name}: paper {g} vs recomputed {expected:.5f}")


M = {"Const": "const", "EM": "emp_mean", "Beta": "beta11", "EB": "eb_bench", "Hier": "online0", "HierU": "online0@g1",
     "Tok": "online0_f@g1", "TokRand": "online0_f", "Six": "online0_f@g1c1", "Sev": "online_name_f@g1c1"}
n_checked = 0
for tag, reg in {"Std": "std", "High": "high", "SpTwo": "sp2", "SpOne": "sp1"}.items():
    P = np.load(ROOT / f"logbooks/runs/pairs_R7_{reg}.npz")
    alc = {k: float((P[v] @ W).mean()) for k, v in M.items()}
    for k, v in alc.items():
        check(f"a{tag}{k}", v, mac[f"a{tag}{k}"]); n_checked += 1
    pairs = {"Unc": ("HierU", "Hier"), "Tokens": ("Tok", "HierU"), "Cov": ("Six", "Tok"), "Name": ("Sev", "Six"),
             "Act": ("Six", "TokRand")}
    for c, (x, y) in pairs.items():
        check(f"d{tag}{c}", alc[x] - alc[y], mac[f"d{tag}{c}"], 6e-5); n_checked += 1
    for c, (x, y) in {"SixVsBeta": ("Six", "Beta"), "SixVsEM": ("Six", "EM"), "SixVsEB": ("Six", "EB")}.items():
        check(f"p{tag}{c}", alc[y] - alc[x], mac[f"p{tag}{c}"], 6e-5); n_checked += 1
    check(f"n{tag}", len(P["online0"]), mac[f"n{tag}"].replace("{,}", ""), 0.5); n_checked += 1
    g8 = np.load(ROOT / f"logbooks/runs/pairs_R8_{reg}.npz")["online_name_f@g1c1o"] @ W
    check(f"d{tag}Gate", float(g8.mean()) - alc["Sev"], mac[f"d{tag}Gate"], 6e-5); n_checked += 1
o = json.load(open(ROOT / "logbooks/runs/overlap_R9_s0.json"))
for r in o["rows"]:
    w = {1: "One", 2: "Two", 4: "Four", 10: "Ten"}[r["systems"]]
    pb = o["per_bench"][str(r["systems"])]
    part = np.mean(np.array(pb["focal only"]) - np.array(pb["full"]))
    item = np.mean(np.array(pb["no item"]) - np.array(pb["full"]))
    check(f"ovPart{w}", part, mac[f"ovPart{w}"]); check(f"ovItem{w}", item, mac[f"ovItem{w}"])
    check(f"ovCov{w}", 100 * r["coverage"], mac[f"ovCov{w}"], 0.51); n_checked += 3
o1 = json.load(open(ROOT / "logbooks/runs/overlap_R9_s1.json"))
check("ovPartTenB", o1["rows"][-1]["partner evidence"][0], mac["ovPartTenB"]); n_checked += 1

# hard-coded numbers in the text
a1 = json.load(open(ROOT / "logbooks/runs/kaggle/a1/eval_A1.json"))["summary"]
a2 = json.load(open(ROOT / "logbooks/runs/kaggle/a2/eval_A2.json"))["summary"]
hard = {"A1 v6 0.1942": (a1["online0_propf@g1c1"]["alc"], 0.1942), "A2 v6 0.1847": (a2["online0_propf@g1c1"]["alc"], 0.1847),
        "A1 EM 0.2438": (a1["emp_mean"]["alc"], 0.2438), "A2 EM 0.2416": (a2["emp_mean"]["alc"], 0.2416),
        "A1 propagation alone -0.0022": (a1["online0_prop@g1"]["alc"] - a1["online0@g1"]["alc"], -0.0022)}
for k, (v, paper) in hard.items():
    check(k, v, paper, 6e-5); n_checked += 1
v3 = np.array([0.169708, 0.192029, 0.176251, 0.199922])
check("v3 mean 0.184", v3.mean(), 0.184, 6e-4); check("v3 sd 0.014", v3.std(ddof=1), 0.014, 6e-4)
check("runs per mean 30", (1.96 * v3.std(ddof=1) / 0.005) ** 2, 30, 1.0)
check("runs per arm 60", 2 * (1.96 * v3.std(ddof=1) / 0.005) ** 2, 60, 1.5); n_checked += 4
em_gap = {r: float((np.load(ROOT / f"logbooks/runs/pairs_R7_{r}.npz")["emp_mean"] @ W).mean()) for r in ["std", "high"]}
print(f"one-response vs whole-cell Empirical Mean gap: std {em_gap['std'] - a1['emp_mean']['alc']:+.4f}, "
      f"high {em_gap['high'] - a2['emp_mean']['alc']:+.4f}")
eda = __import__("pandas").read_csv(ROOT / "logbooks/runs/final_eda_per_bench.csv")
print("EDA: benchmarks", len(eda), "| interactor benchmarks", int((eda.interactors > 0).sum()),
      "| asset-heavy (>50%)", int((eda.assets > 0.5).sum()), "| 4-key fuses >=50%:", int(((eda.subj_four / eda.subj_rows) < 0.5).sum()))
print(f"checked {n_checked} values; mismatches: {len(bad)}")
for b in bad:
    print("  MISMATCH", b)
if bad:
    raise SystemExit(1)
