"""X1: does a linear model on item text features transfer difficulty across benchmarks?

    python paiec/lobo_ridge.py --emb GLOB [--tag X1]

Targets (training cells only, per item): easiness = logit of smoothed mean success
over subjects. Features: embedding (+ handcrafted format features from item text).
Leave-one-benchmark-out (grouped 10-fold over benchmarks) ridge -> R^2 for
(a) item easiness and (b) benchmark mean easiness (from the mean of item predictions).
"""
import argparse, json, os, re
from pathlib import Path
import numpy as np
import pandas as pd

from data import load_cells, load_embeddings, find_dir
from priors import logit

OUT = Path(os.environ.get("PAIEC_OUT", Path(__file__).resolve().parents[1] / "logbooks" / "runs"))


def format_features(text, feats):
    t = text.fillna("")
    f = pd.DataFrame({
        "len_log": np.log1p(t.str.len()),
        "n_lines": np.log1p(t.str.count("\n")),
        "mc_letters": t.str.count(r"(?m)^\s*\(?[A-Ha-h][\).:]\s").clip(0, 10),
        "has_options_word": t.str.contains(r"(?i)\boptions?\b|\bchoices?\b").astype(float),
        "yes_no": t.str.contains(r"(?i)\byes or no\b|\btrue or false\b").astype(float),
        "code": t.str.contains(r"def |class |\{|\}|;\s*$|```|import ").astype(float),
        "math": t.str.count(r"[\$\\=\^]").clip(0, 50) / 50,
        "non_ascii": t.map(lambda x: sum(ord(c) > 127 for c in x[:2000]) / max(1, len(x[:2000]))),
        "digits": t.map(lambda x: sum(c.isdigit() for c in x[:2000]) / max(1, len(x[:2000]))),
        "question_mark": t.str.contains(r"\?").astype(float),
        "has_feats": (feats.fillna("").str.len() > 0).astype(float),
        "url": t.str.contains(r"https?://").astype(float),
        "json": t.str.contains(r"^\s*[\[{]").astype(float),
    })
    return f


def ridge_lobo(X, y, groups, lam, k=10, seed=0):
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(ug), np.arange(len(ug)) % k))
    fold = np.array([fold_of[g] for g in groups])
    pred = np.zeros_like(y)
    for f in range(k):
        tr, te = fold != f, fold == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xt = (X[tr] - mu) / sd
        A = Xt.T @ Xt + lam * len(Xt) * np.eye(X.shape[1])
        w = np.linalg.solve(A, Xt.T @ (y[tr] - y[tr].mean()))
        pred[te] = (X[te] - mu) / sd @ w + y[tr].mean()
    return pred


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--per_bench", type=int, default=600)
    ap.add_argument("--tag", default="X1")
    args = ap.parse_args()
    codes, E = load_embeddings(args.emb)
    cells, meta = load_cells(codes)
    g = cells.assign(rate=cells.k / cells.n).groupby("iid").agg(bench=("bench", "first"), m=("rate", "size"),
                                                                  q=("rate", "mean"))
    g = g[g.m >= 3]
    g = g.groupby("bench", group_keys=False).apply(lambda x: x.sample(min(len(x), args.per_bench), random_state=0))
    y = logit(((g.m * g.q + 1.0) / (g.m + 2.0)).to_numpy())
    items = pd.read_parquet(find_dir("items_all.parquet") / "items_all.parquet",
                            columns=["iid_code", "text", "item_features"],
                            filters=[("iid_code", "in", g.index.to_list())]).set_index("iid_code").loc[g.index]
    F = format_features(items.text, items.item_features).to_numpy(float)
    Xe = E[np.searchsorted(codes, g.index.to_numpy())]
    groups = g.bench.to_numpy()
    bench_mean = pd.Series(y).groupby(groups).transform("mean").to_numpy()
    res = {"n_items": int(len(y)), "n_bench": int(len(np.unique(groups)))}
    for name, X in [("format", F), ("emb", Xe), ("emb+format", np.hstack([Xe, F]))]:
        for lam in [0.01, 0.1, 1.0]:
            p = ridge_lobo(X, y, groups, lam)
            pb = pd.Series(p).groupby(groups).transform("mean").to_numpy()
            ub = np.unique(groups, return_index=True)[1]
            rec = {"item_r2": r2(y, p), "bench_r2": r2(bench_mean[ub], pb[ub]),
                   "within_r2": r2(y - bench_mean, p - pb)}
            res[f"{name}_lam{lam}"] = rec
            print(f"{name:11s} lam={lam:<5} item R2={rec['item_r2']:.3f} bench-mean R2={rec['bench_r2']:.3f} "
                  f"within-bench R2={rec['within_r2']:.3f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"lobo_{args.tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
