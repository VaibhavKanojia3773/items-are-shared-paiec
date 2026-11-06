"""Round-7 analysis -> single results manifest for the paper.

Reads logbooks/runs/pairs_R7_<regime>.npz (pairs x 6 Brier; one-response reveals) and overlap_R7.json,
rebuilds each pair's benchmark and subject from the deterministic sampler (same seeds as evaluate.py),
computes ALC, paired contrasts and benchmark-bootstrap 95% intervals, and writes
    logbooks/runs/r7_summary.json   and   paper/results.tex (LaTeX macros used by main.tex)
    python paper/scripts/analyze_r7.py
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paiec"))
from data import load_cells          # noqa: E402
from sim import build_pairs, build_pairs_formative, WEIGHTS  # noqa: E402
from evaluate import fold_of          # noqa: E402

REGIMES = {"Std": ("std", None, None), "High": ("high", 100, 1000), "SpTwo": ("sp2", 100, 200), "SpOne": ("sp1", 100, 100)}
METHODS = {"Const": "const", "EM": "emp_mean", "Beta": "beta11", "EB": "eb_bench", "Hier": "online0",
           "HierU": "online0@g1", "Tok": "online0_f@g1", "TokRand": "online0_f", "Six": "online0_f@g1c1",
           "Sev": "online_name_f@g1c1"}
CONTRASTS = {"Unc": ("HierU", "Hier"), "Tokens": ("Tok", "HierU"), "Cov": ("Six", "Tok"), "Name": ("Sev", "Six"),
             "Act": ("Six", "TokRand"), "VsEM": ("Sev", "EM"), "VsBeta": ("Sev", "Beta"), "VsEB": ("Sev", "EB"),
             "VsHier": ("Sev", "Hier"), "Gate": ("Gate", "Sev"), "SixVsBeta": ("Six", "Beta"), "SixVsEM": ("Six", "EM"), "SixVsEB": ("Six", "EB")}
cells, meta = load_cells(pd.read_parquet(ROOT / "data/proc/subset10k.parquet")["iid_code"].to_numpy())
folds = fold_of(cells["bench"].unique(), 5)
cells["fold"] = cells["bench"].map(folds).astype(int)
name_of = meta["normalized_name"].fillna("").astype(str).to_numpy()


def pairs_meta(tag):
    _, k, rows = REGIMES[tag]
    b, known = [], []
    for f in range(5):
        test, train = cells[cells.fold == f], cells[cells.fold != f]
        train_names = set(name_of[train.sid.unique()]) - {""}
        rng = np.random.default_rng(f)
        ps = build_pairs_formative(test, rng, k, rows) if k else build_pairs(test, rng, 20, 200, 100)
        b += [p["bench"] for p in ps]
        known += [name_of[p["sid"]] in train_names for p in ps]
    return np.array(b), np.array(known)


def boot_ci(x, b, n=10000, seed=0):
    ub, inv = np.unique(b, return_inverse=True)
    sums, cnt = np.bincount(inv, x), np.bincount(inv)
    idx = np.random.default_rng(seed).integers(0, len(ub), (n, len(ub)))
    s = sums[idx].sum(1) / cnt[idx].sum(1)
    return float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


macros, summary = {}, {}
f3 = lambda x: f"{x:.4f}"  # intervals use the same 4 decimals as scores
f4 = lambda x: f"{x:.4f}"
sg = lambda x: ("$+$" if x >= 0 else "$-$") + (f"{abs(x):.5f}" if 0 < abs(x) < 0.0001 else f"{abs(x):.4f}")
for tag, (reg, _, _) in REGIMES.items():
    f = ROOT / f"logbooks/runs/pairs_R7_{reg}.npz"
    if not f.exists():
        print("missing", f.name)
        continue
    P = np.load(f)
    b, known = pairs_meta(tag)
    alc = {k: P[v] @ WEIGHTS for k, v in METHODS.items()}
    g8 = ROOT / f"logbooks/runs/pairs_R8_{reg}.npz"  # overlap-gate run on the same pairs, streams and draws
    if g8.exists():
        alc["Gate"] = np.load(g8)["online_name_f@g1c1o"] @ WEIGHTS
    assert len(b) == len(alc["Six"]), (tag, len(b), len(alc["Six"]))
    s = {"pairs": int(len(b)), "benchmarks": int(len(np.unique(b))), "known_frac": float(known.mean()),
         "alc": {}, "contrast": {}, "curves": {k: P[v].mean(0).tolist() for k, v in METHODS.items()}}
    macros[f"n{tag}"] = f"{len(b):,}".replace(",", "{,}")
    macros[f"nb{tag}"] = str(len(np.unique(b)))
    for k, a in alc.items():
        lo, hi = boot_ci(a, b)
        s["alc"][k] = [float(a.mean()), lo, hi]
        macros[f"a{tag}{k}"] = f4(a.mean())
        macros[f"ci{tag}{k}"] = f"[{f3(lo)}, {f3(hi)}]"
    for c, (x, y) in CONTRASTS.items():
        if x not in alc:
            continue
        d = alc[x] - alc[y]
        lo, hi = boot_ci(d, b)
        s["contrast"][c] = [float(d.mean()), lo, hi]
        macros[f"d{tag}{c}"] = sg(d.mean())
        macros[f"p{tag}{c}"] = f4(-d.mean())  # improvement (positive = better)
        mm = lambda x: ("$-$" if x < 0 else "") + f"{abs(x):.4f}"
        macros[f"pci{tag}{c}"] = f"[{mm(-hi)}, {mm(-lo)}]"
        macros[f"dci{tag}{c}"] = f"[{sg(lo)}, {sg(hi)}]"
    for grp, mask in (("Known", known), ("Unknown", ~known)):
        if mask.sum() > 1:
            d = (alc["Sev"] - alc["Six"])[mask]
            lo, hi = boot_ci(d, b[mask])
            s["contrast"]["Name" + grp] = [float(d.mean()), lo, hi, int(mask.sum())]
            macros[f"d{tag}Name{grp}"] = sg(d.mean())
            macros[f"dci{tag}Name{grp}"] = f"[{sg(lo)}, {sg(hi)}]"
            macros[f"n{tag}Name{grp}"] = str(int(mask.sum()))
    macros[f"rel{tag}EM"] = f"{100 * (1 - alc['Sev'].mean() / alc['EM'].mean()):.1f}"
    macros[f"relSix{tag}EM"] = f"{100 * (1 - alc['Six'].mean() / alc['EM'].mean()):.1f}"
    macros[f"relSix{tag}Beta"] = f"{100 * (1 - alc['Six'].mean() / alc['Beta'].mean()):.1f}"
    macros[f"rel{tag}Beta"] = f"{100 * (1 - alc['Sev'].mean() / alc['Beta'].mean()):.1f}"
    summary[tag] = s
    print(f"== {tag}: {s['pairs']} pairs / {s['benchmarks']} benchmarks, known names {known.mean():.2f}")
    for k, v in s["alc"].items():
        print(f"   {k:8s} {v[0]:.4f} [{v[1]:.4f}, {v[2]:.4f}]")
    for k, v in s["contrast"].items():
        print(f"   d{k:12s} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]")

words = {1: "One", 2: "Two", 4: "Four", 10: "Ten"}
for seed, suf in ((0, ""), (1, "B")):
    ov = ROOT / f"logbooks/runs/overlap_R9_s{seed}.json"
    if not ov.exists():
        continue
    o = json.load(open(ov))
    summary[f"overlap_s{seed}"] = {k: v for k, v in o.items() if k != "per_bench"}
    macros[f"ovNb{suf}"] = str(o["benchmarks"])
    for r in o["rows"]:
        w = words[r["systems"]] + suf
        macros[f"ovCov{w}"] = f"{100 * r['coverage']:.0f}"
        for k, v in r["alc"].items():
            macros["ovA" + {"full": "Full", "no item": "NoItem", "focal only": "Focal"}[k] + w] = f4(v)
        for key, short in (("partner evidence", "Part"), ("item effects", "Item"), ("benchmark level", "Lev")):
            g, lo, hi = r[key]
            macros[f"ov{short}{w}"] = f4(g)
            macros[f"ov{short}CI{w}"] = f"[{f4(lo)}, {f4(hi)}]"
json.dump(summary, open(ROOT / "logbooks/runs/r7_summary.json", "w"), indent=1)
with open(ROOT / "paper/results.tex", "w") as fh:
    fh.write("% Generated by paper/scripts/analyze_r7.py -- do not edit by hand.\n")
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(macros)} macros")
