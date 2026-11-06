"""Stacked meta-model over simulated episodes (cross-fold, benchmark held out).

    python paiec/stack.py --tag S001 [--emb GLOB] [--gamma 1]

1. For each fold, run an episode with the online model; at every checkpoint
   record features of every evaluation target (online-model components,
   label counts / empirical means at pair, benchmark and item level, budget,
   prior) together with its outcome (n, k).
2. For each fold, train LightGBM on the other folds' records and score the
   held-out fold; report Brier ALC of base vs. stacked predictions.
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd

from sim import build_pairs, run_episode, WEIGHTS, BUDGETS
from online import OnlineModel, sigmoid
from priors import NamePrior, logit
from data import load_cells, load_embeddings
from evaluate import fold_of, Emb
from acq import UncertaintyPolicy

OUT = Path(os.environ.get("PAIEC_OUT", Path(__file__).resolve().parents[1] / "logbooks" / "runs"))
HP = {"a": 0.6, "s_u": 1.0, "s_t": 0.5, "s_v": 0.5, "s_w": 1.2, "c": 0.2, "n_cap": 2.0}


def platt_fit(z, y, w, iters=25, l2=1e-3):
    """Weighted logistic regression of outcome rate y on [z, 1] (Newton). Returns (slope, bias)."""
    X = np.column_stack([z, np.ones_like(z)])
    beta = np.array([1.0, 0.0])
    for _ in range(iters):
        p = sigmoid(X @ beta)
        g = X.T @ (w * (y - p)) - l2 * (beta - [1.0, 0.0])
        H = (X * (w * p * (1 - p))[:, None]).T @ X + l2 * np.eye(2)
        beta = beta + np.linalg.solve(H, g)
    return beta


def calibrate(df, regime_cols):
    """Cross-fold per-regime Platt scaling of the base logit; returns calibrated p."""
    z = df.logit_base.to_numpy()
    y = (df.k / df.n).to_numpy()
    w = (df.n / df.groupby(["pair_id", "budget"])["n"].transform("sum")).to_numpy()
    key = df[regime_cols].astype(str).agg("|".join, axis=1).to_numpy()
    out = df.p_base.to_numpy().copy()
    params = {}
    for f in range(5):
        tr, te = (df.fold != f).to_numpy(), (df.fold == f).to_numpy()
        for r in np.unique(key):
            a, b = tr & (key == r), te & (key == r)
            if a.sum() < 500 or not b.any():
                continue
            beta = platt_fit(z[a], y[a], w[a])
            out[b] = sigmoid(beta[0] * z[b] + beta[1])
            params.setdefault(r, []).append([round(float(x), 3) for x in beta])
    return out, params


def group_stats(l, keys, t):
    """Count and smoothed mean of revealed labels per key group, aligned to targets t."""
    if len(l) == 0:
        return np.zeros(len(t)), np.full(len(t), np.nan)
    g = l.groupby(keys)[["n", "k"]].sum()
    idx = pd.MultiIndex.from_frame(t[keys]) if len(keys) > 1 else pd.Index(t[keys[0]])
    r = g.reindex(idx)
    n = r["n"].fillna(0).to_numpy()
    mean = np.where(n > 0, (r["k"].fillna(0).to_numpy() + 0.5) / (n + 1.0), np.nan)
    return n, mean


def features(t, l, model, eta, emb, budget):
    parts = {}
    p = model.predict(t.sid, t.iid, t.bench, eta(t), emb=emb, parts=parts)
    f = {"budget": np.full(len(t), budget), "eta": eta(t), "p_base": p, "logit_base": logit(p)}
    for k, v in parts.items():
        f[k] = np.asarray(v, float)
    for name, keys in [("pair", ["sid", "bench"]), ("bench", ["bench"]), ("item", ["bench", "iid"])]:
        n, m = group_stats(l, keys, t)
        f[f"n_{name}"], f[f"mean_{name}"] = n, m
    if len(l):
        nsub = l.groupby("bench")["sid"].nunique()
        f["n_subj_bench"] = t.bench.map(nsub).fillna(0).to_numpy()
    else:
        f["n_subj_bench"] = np.zeros(len(t))
    return pd.DataFrame(f), p


NO_NAME = False


def collect(cells, meta, emb, gamma, seed):
    folds = fold_of(cells["bench"].unique(), 5)
    cells = cells.assign(fold=cells["bench"].map(folds).astype(int))
    recs = []
    for f in range(5):
        test = cells[cells.fold == f]
        pairs = build_pairs(test, np.random.default_rng(seed + f), 20, 200, 100)
        pri = NamePrior().fit(cells[cells.fold != f], meta)
        eta = (lambda df, mu=pri.mu: np.full(len(df), mu)) if NO_NAME else pri.logit
        look = test.set_index(["sid", "bench", "iid"])[["n", "k"]]
        state = {"b": 0}

        def pred(t, l):
            m = OnlineModel(HP).fit(l.sid, l.iid, l.bench, l.n, l.k, eta(l))
            budget = BUDGETS[state["b"]]
            state["b"] += 1
            X, p = features(t.reset_index(drop=True), l, m, eta, emb, budget)
            nk = look.loc[pd.MultiIndex.from_arrays([t.sid, t.bench, t.iid])].to_numpy()
            X["n"], X["k"], X["fold"] = nk[:, 0], nk[:, 1], f
            X["pair_id"] = f * 100000 + pd.factorize(list(zip(t.sid, t.bench)))[0]
            recs.append(X)
            return p

        policy = UncertaintyPolicy(eta, HP, gamma, emb=emb) if gamma > 0 else None
        r = run_episode(test, pairs, pred, policy=policy, rng=np.random.default_rng(100 + f))
        print(f"fold {f}: base ALC={r['alc']:.4f}", flush=True)
    return pd.concat(recs, ignore_index=True)


def alc_of(df, p):
    sq = df.n * p ** 2 - 2 * p * df.k + df.k
    g = pd.DataFrame({"pair": df.pair_id, "budget": df.budget, "sq": sq, "n": df.n})
    per = g.groupby(["budget", "pair"]).agg(sq=("sq", "sum"), n=("n", "sum"))
    per = (per.sq / per.n).groupby("budget").mean().reindex(BUDGETS).to_numpy()
    return float(per @ WEIGHTS), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="")
    ap.add_argument("--subset", default="data/proc/subset10k.parquet")
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--tag", default="S001")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hp", default="")
    ap.add_argument("--no_name_prior", action="store_true")
    ap.add_argument("--calib_only", action="store_true")
    args = ap.parse_args()
    if args.hp:
        HP.clear(); HP.update(json.loads(args.hp))
    emb = Emb(*load_embeddings(args.emb)) if args.emb else None
    subset = emb.codes if emb is not None else pd.read_parquet(args.subset)["iid_code"].to_numpy()
    cells, meta = load_cells(subset)
    t0 = time.time()
    global NO_NAME
    NO_NAME = args.no_name_prior
    df = collect(cells, meta, emb, args.gamma, args.seed)
    print(f"collected {len(df):,} target-budget rows in {time.time()-t0:.0f}s", flush=True)
    df["item_known"] = (df.get("k_w", pd.Series(0, index=df.index)) > 0).astype(int)
    base_alc, base_curve = alc_of(df, df.p_base)
    print(f"BASE        ALC={base_alc:.4f} curve={np.round(base_curve, 4)}")
    for cols in (["budget"], ["budget", "item_known"]):
        pc, params = calibrate(df, cols)
        a, c = alc_of(df, pc)
        print(f"PLATT {'+'.join(cols):18s} ALC={a:.4f} curve={np.round(c, 4)}")
        print("   params (fold-wise slope, bias):", {k: v[0] for k, v in params.items()})
    if args.calib_only:
        return
    import lightgbm as lgb
    feats = [c for c in df.columns if c not in ("n", "k", "fold", "pair_id")]
    out = np.zeros(len(df))
    params = {"objective": "cross_entropy", "learning_rate": 0.05, "num_leaves": 31,
              "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1}
    importance = None
    for f in range(5):
        tr, te = df.fold != f, df.fold == f
        # Equal weight per pair: each pair's responses sum to 1 per budget.
        w = df.n / df.groupby(["pair_id", "budget"])["n"].transform("sum")
        ds = lgb.Dataset(df.loc[tr, feats], label=(df.k / df.n)[tr], weight=w[tr])
        booster = lgb.train(params, ds, num_boost_round=300)
        out[te.to_numpy()] = booster.predict(df.loc[te, feats])
        imp = pd.Series(booster.feature_importance("gain"), index=feats)
        importance = imp if importance is None else importance + imp
    base_alc, base_curve = alc_of(df, df.p_base)
    st_alc, st_curve = alc_of(df, np.clip(out, 1e-4, 1 - 1e-4))
    print(f"BASE  ALC={base_alc:.4f} curve={np.round(base_curve, 4)}")
    print(f"STACK ALC={st_alc:.4f} curve={np.round(st_curve, 4)}")
    print("top features:", importance.sort_values(ascending=False).head(12).round(0).to_dict())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"stack_{args.tag}.json").write_text(json.dumps({
        "args": vars(args), "base": [base_alc, list(map(float, base_curve))],
        "stack": [st_alc, list(map(float, st_curve))]}, indent=1))
    df.to_parquet(OUT / f"stack_{args.tag}.parquet")


if __name__ == "__main__":
    main()
