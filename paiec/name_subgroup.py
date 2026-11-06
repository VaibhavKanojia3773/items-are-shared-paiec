"""Does a name-based ability prior help for widely evaluated models? Per-pair paired comparison of
online0_f vs online_name_f (both @g1c1, tokens on), split by how many TRAINING benchmarks the
subject's model name appears in.  python paiec/name_subgroup.py [--formative 100]
"""
import argparse, json
import numpy as np
import pandas as pd

import evaluate as ev
import online as _online
from online import feature_tokens, text_tokens
from sim import build_pairs, build_pairs_formative, run_episode, WEIGHTS
from data import load_cells, find_dir

ap = argparse.ArgumentParser()
ap.add_argument("--formative", type=int, default=0)
ap.add_argument("--hp", default='{"a":0.4,"s_u":1.4,"s_t":0.5,"s_v":1.0,"s_w":2.2,"c":0.1,"n_cap":1.0,"s_f":0.7,"s_g":0.3}')
args = ap.parse_args()
hp = json.loads(args.hp)
cells, meta = load_cells(pd.read_parquet("data/proc/subset10k.parquet")["iid_code"].to_numpy())
_online.TEMPLATE_TOKENS = True
it = pd.read_parquet(find_dir("items_all.parquet") / "items_all.parquet", columns=["iid_code", "item_features", "text"],
                     filters=[("iid_code", "in", cells.iid.unique().tolist())])
for r in it.itertuples(index=False):
    ev.TOKENS[int(r.iid_code)] = tuple(feature_tokens(r.item_features) + text_tokens(r.text))
name = meta["normalized_name"].fillna("").astype(str).to_numpy()
folds = ev.fold_of(cells.bench.unique(), 5)
cells["fold"] = cells.bench.map(folds).astype(int)
rows = []
for f in range(5):
    test, train = cells[cells.fold == f], cells[cells.fold != f]
    rng = np.random.default_rng(f)
    pairs = build_pairs_formative(test, rng, args.formative) if args.formative else build_pairs(test, rng, 20, 200, 100)
    nb = pd.Series(name[train.sid.to_numpy()]).groupby(train.bench.to_numpy()).unique().explode()
    freq = nb[nb != ""].value_counts()
    ms = ev.make_methods(train, meta, ["online0_f@g1c1", "online_name_f@g1c1"], hp, None, None, {})
    res = {m: run_episode(test, pairs, fn, policy=pol, rng=np.random.default_rng(100 + f))["brier"] @ WEIGHTS
           for m, (fn, pol) in ms.items()}
    for j, p in enumerate(pairs):
        nm = name[p["sid"]]
        rows.append((f, nm, int(freq.get(nm, 0)) if nm else 0, res["online0_f@g1c1"][j], res["online_name_f@g1c1"][j]))
d = pd.DataFrame(rows, columns=["fold", "name", "train_benches", "const", "name_prior"])
d["diff"] = d.name_prior - d.const
d["bucket"] = pd.cut(d.train_benches, [-1, 0, 4, 19, 1000], labels=["0", "1-4", "5-19", "20+"])
g = d.groupby("bucket", observed=True)["diff"].agg(["count", "mean", lambda x: x.std(ddof=1) / np.sqrt(len(x))])
print(f"regime={'formative' if args.formative else 'standard'}  pairs={len(d)}  overall diff={d['diff'].mean():+.4f}")
print(g.rename(columns={"<lambda_0>": "se"}).round(4).to_string())
d.to_csv(f"logbooks/runs/name_subgroup_{'F' if args.formative else 'S'}.csv", index=False)
