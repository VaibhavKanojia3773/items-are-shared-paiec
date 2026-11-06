"""Controlled overlap experiment: does a focal pair gain from partner systems' labels on the same items?

Benchmarks where >= 10 subjects answered a common set of K items. Each subject keeps a fixed 50/50 split
and stream; the focal subject (rank 0) is scored while 0, 1, 3 or 9 partners are added. Acquisition is the
evaluator's random rule (identical trajectories for every predictor) and reveals one binary response.
The scalar base logit is fitted on the other benchmark folds (held out from the scored benchmark).

Predictors (same hyperparameters unless noted):
  full            all revealed labels, shared benchmark and item effects
  no item         all revealed labels, item effects suppressed (s_w ~ 0)
  focal only      the full predictor fed only the focal pair's own labels (partner evidence withheld)
Reports focal-pair ALC and paired deltas with benchmark-bootstrap 95% intervals, plus realized coverage
(share of focal evaluation items revealed for at least one partner by budget 31).

    python paiec/overlap.py --seed 0   -> logbooks/runs/overlap_R9_s0.json
"""
import argparse, json
import numpy as np
import pandas as pd

from online import OnlineModel
from sim import run_episode, WEIGHTS
from data import load_cells
from evaluate import fold_of

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0, help="selection of items, subjects and focal order")
ap.add_argument("--tag", default="R9")
args = ap.parse_args()
K, NS, PARTNERS = 100, 10, [1, 2, 4, 10]
HP = {"a": 0.4, "s_u": 1.4, "s_t": 0.5, "s_v": 1.0, "s_w": 2.2, "c": 0.1, "n_cap": 1.0}

cells, meta = load_cells(pd.read_parquet("data/proc/subset10k.parquet")["iid_code"].to_numpy())
folds = fold_of(cells["bench"].unique(), 5)
cells["fold"] = cells["bench"].map(folds).astype(int)
mu_fold = {}
for f in range(5):  # base logit fitted on the other folds only
    tr = cells[cells.fold != f]
    mu_fold[f] = float(np.log(tr.k.sum() / (tr.n.sum() - tr.k.sum())))
mu_bench = {b: mu_fold[f] for b, f in folds.items()}

rng = np.random.default_rng(args.seed)
ranks = {r: [] for r in range(NS)}
for bench, g in cells.groupby("bench", sort=True, observed=True):
    cover = g.groupby("iid").size().sort_values(ascending=False, kind="stable")
    if len(cover) < K:
        continue
    items = rng.choice(cover.index.to_numpy()[: 2 * K], K, replace=False)
    has = g[g.iid.isin(items)].groupby("sid").size()
    subj = has.index[has.to_numpy() == K].to_numpy()
    if len(subj) < NS:
        continue
    subj = rng.permutation(subj)[:NS]
    for r, sid in enumerate(subj):
        perm = rng.permutation(items)
        ranks[r].append({"sid": sid, "bench": bench, "acq": perm[: K // 2], "eval": perm[K // 2:]})
nb = len(ranks[0])
focal = {(p["sid"], p["bench"]) for p in ranks[0]}
print(f"seed {args.seed}: benchmarks with {NS} subjects sharing {K} items: {nb}", flush=True)
eta = lambda df: np.array([mu_bench[b] for b in df.bench.to_numpy()]) if len(df) else np.zeros(0)


def predictor(hp, focal_only=False):
    def pred(t, l):
        if focal_only and len(l):
            keep = np.fromiter(((s, b) in focal for s, b in zip(l.sid, l.bench)), bool, len(l))
            l = l[keep]
        mod = OnlineModel(hp).fit(l.sid, l.iid, l.bench, l.n, l.k, eta(l))
        return mod.predict(t.sid, t.iid, t.bench, eta(t))
    return pred


VARIANTS = {"full": predictor(HP), "no item": predictor(dict(HP, s_w=1e-3)), "focal only": predictor(HP, True)}
res, coverage = {}, {}
for m in PARTNERS:
    pairs = [p for r in range(m) for p in ranks[r]]  # rank-major: focal pairs first (fixed draws)
    cap = {}

    def capture(t, l):
        cap["l"] = l
        return np.full(len(t), 0.5)
    run_episode(cells, pairs, capture, policy=None, rng=np.random.default_rng(1 + args.seed), single_seed=2 + args.seed)
    lab = cap["l"]
    cov = []
    for j in range(nb):
        p = pairs[j]
        others = set(lab[(lab.bench == p["bench"]) & (lab.sid != p["sid"])].iid.tolist())
        cov.append(np.mean([i in others for i in p["eval"]]))
    coverage[m] = float(np.mean(cov))
    for name, fn in VARIANTS.items():
        b = run_episode(cells, pairs, fn, policy=None, rng=np.random.default_rng(1 + args.seed),
                        single_seed=2 + args.seed)["brier"]
        res[(m, name)] = b[:nb] @ WEIGHTS
        print(f"systems={m:2d} coverage={coverage[m]:.3f} {name:11s} focal ALC={res[(m, name)].mean():.4f}", flush=True)


def boot(d, n=10000):
    idx = np.random.default_rng(3).integers(0, len(d), (n, len(d)))
    return [float(np.quantile(d[idx].mean(1), q)) for q in (0.025, 0.975)]


out = {"seed": args.seed, "benchmarks": nb, "K": K, "rows": [],
       "per_bench": {str(m): {k: res[(m, k)].tolist() for k in VARIANTS} for m in PARTNERS}}
for m in PARTNERS:
    full = res[(m, "full")]
    row = {"systems": m, "coverage": coverage[m], "alc": {k: float(res[(m, k)].mean()) for k in VARIANTS}}
    for key, (x, y) in {"partner evidence": ("focal only", "full"), "item effects": ("no item", "full"),
                        "benchmark level": ("focal only", "no item")}.items():
        d = res[(m, x)] - res[(m, y)]
        row[key] = [float(d.mean())] + boot(d)
    out["rows"].append(row)
    print(json.dumps({k: v for k, v in row.items() if k != "alc"}), flush=True)
json.dump(out, open(f"logbooks/runs/overlap_{args.tag}_s{args.seed}.json", "w"), indent=1)
