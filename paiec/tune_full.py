"""Random search over the FULL system (tokens + subject x token + acquisition + optional name effect),
scored on both simulator regimes; the objective is the mean of the two ALCs (robust choice).

    python paiec/tune_full.py --emb "/kaggle/input/**/emb_*.npy" --n 60 --seed 1 --tag T100
The embeddings only define the item subset (the final system's propagation is redundant with tokens),
so the model evaluated is online0_f (+n when s_n > 0). Results append to tune_<tag>.jsonl.
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd

import online as _online
from online import OnlineModel, feature_tokens, text_tokens
from sim import build_pairs, build_pairs_formative, run_episode, WEIGHTS
from data import load_cells, load_embeddings, find_dir
from evaluate import fold_of
from acq import UncertaintyPolicy

OUT = Path(os.environ.get("PAIEC_OUT", Path(__file__).resolve().parents[1] / "logbooks" / "runs"))
V6 = {"a": 0.4, "s_u": 1.4, "s_t": 0.5, "s_v": 1.0, "s_w": 2.2, "c": 0.1, "n_cap": 1.0,
      "s_f": 0.7, "s_g": 0.3, "gamma": 1.0, "beta": 1.0, "s_n": 0.0}


def sample(rng):
    ch = lambda xs: float(rng.choice(xs))
    return {"a": ch([0.2, 0.4, 0.6]), "s_u": ch([1.0, 1.4, 2.0]), "s_t": ch([0.3, 0.5, 0.8]),
            "s_v": ch([0.7, 1.0, 1.3]), "s_w": ch([1.6, 2.2, 3.0]), "c": ch([0.05, 0.1, 0.2]),
            "n_cap": ch([1.0, 2.0, 4.0]), "s_f": ch([0.4, 0.7, 1.0, 1.4]), "s_g": ch([0.0, 0.3, 0.5, 0.8]),
            "gamma": ch([0.8, 1.0, 1.2]), "beta": ch([0.5, 1.0, 2.0]), "s_n": ch([0.0, 0.0, 0.4, 0.8])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="")
    ap.add_argument("--subset", default="data/proc/subset10k.parquet")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="T100")
    args = ap.parse_args()
    subset = load_embeddings(args.emb)[0] if args.emb else pd.read_parquet(args.subset)["iid_code"].to_numpy()
    cells, meta = load_cells(subset)
    _online.TEMPLATE_TOKENS = True
    it = pd.read_parquet(find_dir("items_all.parquet") / "items_all.parquet",
                         columns=["iid_code", "item_features", "text"],
                         filters=[("iid_code", "in", cells.iid.unique().tolist())])
    tokens = {int(r.iid_code): tuple(feature_tokens(r.item_features) + text_tokens(r.text))
              for r in it.itertuples(index=False)}
    names = meta["normalized_name"].fillna("").astype(str).to_numpy()
    folds = fold_of(cells["bench"].unique(), 5)
    cells["fold"] = cells["bench"].map(folds).astype(int)
    setups = []
    for f in range(5):
        test = cells[cells.fold == f]
        mu = float(np.log((cells[cells.fold != f].k.sum()) / (cells[cells.fold != f].n.sum()
                                                               - cells[cells.fold != f].k.sum())))
        setups.append(("standard", f, test, build_pairs(test, np.random.default_rng(f), 20, 200, 100), mu))
        setups.append(("formative", f, test, build_pairs_formative(test, np.random.default_rng(f), 100), mu))
    print(f"loaded; {len(tokens):,} tokenised items; setups ready", flush=True)
    rng = np.random.default_rng(args.seed)
    configs = [dict(V6)] + [sample(rng) for _ in range(args.n)]
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f"tune_{args.tag}.jsonl"
    best = None
    for c, hp in enumerate(configs):
        t0 = time.time()
        res = {"standard": [], "formative": []}
        for regime, f, test, pairs, mu in setups:
            eta = lambda df, mu=mu: np.full(len(df), mu)
            tok = lambda df: [tokens.get(int(i), ()) for i in df.iid.to_numpy()]
            nk = (lambda df: list(names[df.sid.to_numpy().astype(np.int64)])) if hp["s_n"] > 0 else (lambda df: None)

            def pred(t, l, hp=hp, eta=eta):
                m = OnlineModel(hp).fit(l.sid, l.iid, l.bench, l.n, l.k, eta(l), tokens=tok(l), nkey=nk(l))
                return m.predict(t.sid, t.iid, t.bench, eta(t), tokens=tok(t), nkey=nk(t))
            pol = UncertaintyPolicy(eta, hp, hp["gamma"], beta=hp["beta"])
            res[regime].append(run_episode(test, pairs, pred, policy=pol, rng=np.random.default_rng(100 + f))["brier"])
        alc = {r: float(np.concatenate(v).mean(0) @ WEIGHTS) for r, v in res.items()}
        rec = {"hp": hp, "alc_std": alc["standard"], "alc_form": alc["formative"],
               "obj": 0.5 * (alc["standard"] + alc["formative"])}
        with open(log, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if best is None or rec["obj"] < best["obj"]:
            best = rec
        print(f"[{c:3d}] std={alc['standard']:.4f} form={alc['formative']:.4f} obj={rec['obj']:.4f} "
              f"best={best['obj']:.4f} {time.time()-t0:.0f}s {hp}", flush=True)
    print("BEST", json.dumps(best))


if __name__ == "__main__":
    main()
