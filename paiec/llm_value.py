"""Value of an LLM item-difficulty prior: simulate ratings correlated rho with true item easiness
(within-benchmark z-scores) and feed beta*rating as the item's prior mean (through eta).

    python paiec/llm_value.py --formative 100 --rows 100      # sparse: one pair per benchmark
"""
import argparse
import numpy as np
import pandas as pd

import online as _online
from online import OnlineModel, feature_tokens, text_tokens
from sim import build_pairs_formative, run_episode, WEIGHTS
from data import load_cells, find_dir
from evaluate import fold_of
from acq import UncertaintyPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--formative", type=int, default=100)
ap.add_argument("--rows", type=int, default=100)
args = ap.parse_args()
HP = {"a": 1.0, "s_u": 1.4, "s_t": 0.5, "s_v": 1.0, "s_w": 2.2, "c": 0.1, "n_cap": 1.0, "s_f": 0.7, "s_g": 0.3}
cells, meta = load_cells(pd.read_parquet("data/proc/subset10k.parquet")["iid_code"].to_numpy())
_online.TEMPLATE_TOKENS = True
it = pd.read_parquet(find_dir("items_all.parquet") / "items_all.parquet", columns=["iid_code", "item_features", "text"],
                     filters=[("iid_code", "in", cells.iid.unique().tolist())])
TOK = {int(r.iid_code): tuple(feature_tokens(r.item_features) + text_tokens(r.text)) for r in it.itertuples(index=False)}
# true item easiness: smoothed logit of the item's success rate over all subjects, z-scored within benchmark
g = cells.groupby(["bench", "iid"])[["k", "n"]].sum().reset_index()
g["d"] = np.log((g.k + 0.5) / (g.n - g.k + 0.5))
g["z"] = g.groupby("bench")["d"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
g["sd"] = g.groupby("bench")["d"].transform("std").fillna(0)
Z = dict(zip(g.iid, g.z))
folds = fold_of(cells.bench.unique(), 5)
cells["fold"] = cells.bench.map(folds).astype(int)
for rho in [0.0, 0.3, 0.5, 0.7]:
    rng = np.random.default_rng(7)
    noise = {i: rng.standard_normal() for i in g.iid}
    rating = {i: rho * Z[i] + np.sqrt(1 - rho ** 2) * noise[i] for i in g.iid}
    out = []
    for f in range(5):
        test, train = cells[cells.fold == f], cells[cells.fold != f]
        mu = float(np.log(train.k.sum() / (train.n.sum() - train.k.sum())))
        beta = rho * 1.0  # E[z | rating] = rho * rating; z in units of the within-benchmark SD (~1 logit)
        eta = lambda df: 0.4 * mu + beta * np.array([rating.get(int(i), 0.0) for i in df.iid.to_numpy()])
        tok = lambda df: [TOK.get(int(i), ()) for i in df.iid.to_numpy()]

        def pred(t, l):
            m = OnlineModel(HP).fit(l.sid, l.iid, l.bench, l.n, l.k, eta(l), tokens=tok(l))
            return m.predict(t.sid, t.iid, t.bench, eta(t), tokens=tok(t))
        pairs = build_pairs_formative(test, np.random.default_rng(f), args.formative, args.rows)
        pol = UncertaintyPolicy(eta, HP, 1.0, beta=1.0)
        out.append(run_episode(test, pairs, pred, policy=pol, rng=np.random.default_rng(100 + f))["brier"])
    b = np.concatenate(out).mean(0)
    print(f"rho={rho:.1f} pairs={sum(len(o) for o in out)} ALC={b @ WEIGHTS:.4f} curve={np.round(b, 4)}", flush=True)
