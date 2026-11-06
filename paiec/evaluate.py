"""Real-data evaluation: benchmark-level K-fold, simulated PAIEC episodes.

    python paiec/evaluate.py --methods emp_mean,online0,online_name --folds 5
    python paiec/evaluate.py --emb "/kaggle/input/**/emb_*.npy" --methods ...,online_knn

Each fold holds out whole benchmarks (cold start). Priors are fitted only on the
training benchmarks of that fold; the online model only sees revealed labels.
With --emb, evaluation is restricted to the embedded item subset.
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd

from sim import build_pairs, build_pairs_formative, run_episode, WEIGHTS
from online import OnlineModel, sigmoid
from priors import NamePrior
from data import load_cells, load_embeddings
from acq import UncertaintyPolicy

TOKENS = {}  # iid code -> shareable item_features tokens (filled in main when needed)
OUT = Path(os.environ.get("PAIEC_OUT", Path(__file__).resolve().parents[1] / "logbooks" / "runs"))


def fold_of(benches, k, seed=0):
    b = np.array(sorted(benches))
    np.random.default_rng(seed).shuffle(b)
    return {x: i % k for i, x in enumerate(b)}


class Emb:
    def __init__(self, codes, mat):
        self.codes, self.mat = codes, mat

    def __call__(self, iid):
        return self.mat[np.searchsorted(self.codes, np.asarray(iid, dtype=np.int64))]


def make_methods(train, meta, names, hp, emb, test_items, knn_kw, hp2=None):
    out = {}
    pri = NamePrior().fit(train, meta)
    mu = pri.mu
    knn_eta = None
    if emb is not None and any("knn" in m for m in names):
        from knn_prior import KnnPrior
        knn = KnnPrior(**knn_kw).fit(train, meta, emb.codes, emb.mat)
        easy = knn.item_easiness(emb(test_items))
        easy_of = pd.Series(easy, index=test_items)

        def knn_eta(df):
            return knn.logit(df.sid.to_numpy(), easy_of.loc[df.iid.to_numpy()].to_numpy())

    name_of = meta["normalized_name"].fillna("").astype(str).to_numpy()

    def tkey_of(df):  # model name, else the subject itself
        sid = df.sid.to_numpy().astype(np.int64)  # empty frames at budget 0 have object dtype
        return [n if n else f"sid{s}" for n, s in zip(name_of[sid], sid)]

    hp_x = dict(hp, x_scale=hp.get("x_scale") or 1.0)  # *_propx variants always use the residual term

    def hp_for(resid, resid_y):
        h = dict(hp_x) if resid else dict(hp)
        if resid_y:
            h["y_scale"] = hp.get("y_scale") or 1.0
        return h

    def toks(df):
        return [TOKENS.get(int(i), ()) for i in df.iid.to_numpy()]

    def nkey_of(df):  # model name ('' when unknown) for the nested name effect
        return list(name_of[df.sid.to_numpy().astype(np.int64)])

    def online_with(eta, use_emb=False, by_name=False, resid=False, resid_y=False, lowrank=False, ftok=False,
                    name_n=False, ens=False):
        if ens:  # average the probabilities of the hp and hp2 models
            f1 = online_with(eta, use_emb, by_name, resid, resid_y, lowrank, ftok, name_n)
            f2 = _online(eta, use_emb, by_name, resid, resid_y, lowrank, ftok, name_n, hp_override=hp2)
            return lambda t, l: 0.5 * (np.asarray(f1(t, l)) + np.asarray(f2(t, l)))
        return _online(eta, use_emb, by_name, resid, resid_y, lowrank, ftok, name_n)

    def _online(eta, use_emb=False, by_name=False, resid=False, resid_y=False, lowrank=False, ftok=False,
                name_n=False, hp_override=None):
        def f(t, l):
            tk_l, tk_t = (tkey_of(l), tkey_of(t)) if by_name else (None, None)
            f_l, f_t = (pri.subject_feats(l.sid), pri.subject_feats(t.sid)) if lowrank else (None, None)
            o_l, o_t = (toks(l), toks(t)) if ftok else (None, None)
            h = hp_for(resid, resid_y) if hp_override is None else dict(hp_override)
            if ftok and hp_override is None:
                h["s_f"] = hp.get("s_f") or 1.0
                h["s_g"] = hp.get("s_g", 0.0)
            n_l, n_t = (nkey_of(l), nkey_of(t)) if name_n else (None, None)
            m = OnlineModel(h).fit(l.sid, l.iid, l.bench, l.n, l.k, eta(l), tkey=tk_l, feats=f_l, tokens=o_l,
                                   nkey=n_l)
            return m.predict(t.sid, t.iid, t.bench, eta(t), emb=emb if use_emb else None, tkey=tk_t, feats=f_t,
                             tokens=o_t, nkey=n_t)
        return f

    const_eta = lambda df: np.full(len(df), mu)
    etas = {"online0": const_eta, "online_name": pri.logit, "online_name_t": pri.logit, "online0_prop": const_eta,
            "online0_propx": const_eta, "online_name_propx": pri.logit,
            "online_name_y": pri.logit, "online0_propxy": const_eta, "online_name_lr": pri.logit,
            "online0_proplr": const_eta, "online0_propxlr": const_eta, "online_name_f": pri.logit,
            "online_name_prop_t": pri.logit,
            "online0_propf": const_eta, "online0_y": const_eta, "online0_f": const_eta, "online0_propxf": const_eta}
    if knn_eta is not None:
        etas.update({"online_knn": knn_eta, "online_knn_prop": knn_eta, "online_knn_prop_t": knn_eta,
                     "online_knn_propx": knn_eta})
    table = {
        "const": lambda t, l: np.full(len(t), 0.5),
        "emp_mean": None,
        "online0": online_with(const_eta),
        "online_name": online_with(pri.logit),
        "online_name_t": online_with(pri.logit, by_name=True),
        "online0_prop": online_with(const_eta, True),
        "online0_propx": online_with(const_eta, True, resid=True),
        "online_name_propx": online_with(pri.logit, True, resid=True),
        "online_name_y": online_with(pri.logit, resid_y=True),
        "online0_y": online_with(const_eta, resid_y=True),
        "online0_f": online_with(const_eta, ftok=True),
        "online_name_lr": online_with(pri.logit, lowrank=True),
        "online_name_f": online_with(pri.logit, ftok=True),
        "online_name_prop_t": online_with(pri.logit, True, by_name=True),
        "online0_propf": online_with(const_eta, True, ftok=True),
        "online0_propxf": online_with(const_eta, True, resid=True, ftok=True),
        "online0_proplr": online_with(const_eta, True, lowrank=True),
        "online0_propxlr": online_with(const_eta, True, resid=True, lowrank=True),
        "online0_propxy": online_with(const_eta, True, resid=True, resid_y=True),
    }
    if knn_eta is not None:
        table.update({"knn": lambda t, l: sigmoid(knn_eta(t)),
                      "online_knn": online_with(knn_eta),
                      "online_knn_prop": online_with(knn_eta, True),
                      "online_knn_prop_t": online_with(knn_eta, True, True),
                      "online_knn_propx": online_with(knn_eta, True, resid=True)})

    def emp_mean(t, l):
        if len(l) == 0:
            return np.full(len(t), 0.5)
        a = l.groupby(["sid", "bench"])[["k", "n"]].sum()
        m = (a.k / a.n).to_dict()
        return np.array([m.get(x, 0.5) for x in zip(t.sid, t.bench)])
    table["emp_mean"] = emp_mean

    def beta11(t, l):  # Beta(1,1)-smoothed pair mean: (k+1)/(n+2)
        if len(l) == 0:
            return np.full(len(t), 0.5)
        a = l.groupby(["sid", "bench"])[["k", "n"]].sum()
        m = ((a.k + 1) / (a.n + 2)).to_dict()
        return np.array([m.get(x, 0.5) for x in zip(t.sid, t.bench)])

    def eb_bench(t, l):  # pair mean shrunk to the acquired benchmark rate: p0=(S+2)/(N+4), p=(s+2p0)/(n+2)
        if len(l) == 0:
            return np.full(len(t), 0.5)
        b = l.groupby("bench")[["k", "n"]].sum()
        p0 = ((b.k + 2) / (b.n + 4)).to_dict()
        a = l.groupby(["sid", "bench"])[["k", "n"]].sum()
        pr = {key: (r.k + 2 * p0[key[1]]) / (r.n + 2) for key, r in a.iterrows()}
        return np.array([pr.get((s, bb), p0.get(bb, 0.5)) for s, bb in zip(t.sid, t.bench)])
    table["beta11"], table["eb_bench"] = beta11, eb_bench
    flags = {"online0": {}, "online0_prop": {"use_emb": True}, "online0_f": {"ftok": True},
             "online0_propf": {"use_emb": True, "ftok": True},
             "online0_propxf": {"use_emb": True, "ftok": True, "resid": True}}
    for m in names:  # modifiers: online0_propf+n+e@g1c1
        b0 = m.split("@")[0]
        if "+" in b0:
            root, *mods = b0.split("+")
            table[b0] = online_with(const_eta, name_n="n" in mods, ens="e" in mods, **flags[root])
            etas[b0] = const_eta
    for m in names:
        base, _, g = m.partition("@g")  # e.g. online_name@g1.5 -> uncertainty acquisition, gamma 1.5
        g, _, beta = g.partition("c")  # e.g. online0@g1c0.5 -> coverage-aware acquisition, beta 0.5
        gate = beta.endswith("o")  # e.g. online0@g1c1o -> overlap gate (random rule without partners)
        beta = beta.rstrip("o")
        fade = beta.endswith("f")  # e.g. online0@g1c1f -> coverage weight fades with the budget used
        beta = beta.rstrip("f")
        vopt = g.endswith("v")  # e.g. online0_prop@g1v -> V-optimal-style acquisition
        g = g.rstrip("v")
        if base in table:
            by_name = base.endswith("_t")
            policy = UncertaintyPolicy(etas[base], hp_for("x" in base.split("_")[-1], base.endswith("y")), float(g),
                                       tkey=tkey_of if by_name else None,
                                       emb=emb if "prop" in base else None, vopt=vopt,
                                       beta=float(beta or 0), fade=fade, gate=gate) if g else None
            out[m] = (table[base], policy)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="const,emp_mean,online0,online_name")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only_folds", default="")
    ap.add_argument("--max_subj", type=int, default=20)
    ap.add_argument("--max_acq", type=int, default=200)
    ap.add_argument("--max_eval", type=int, default=100)
    ap.add_argument("--hp", default="{}")
    ap.add_argument("--single", action="store_true", help="acquisition reveals one binary response")
    ap.add_argument("--no_provider", action="store_true", help="name prior without provider fallback (v7)")
    ap.add_argument("--hp2", default="{}", help="second HP set for '+e' (ensemble) methods")
    ap.add_argument("--knn", default="{}")
    ap.add_argument("--emb", default="")
    ap.add_argument("--subset", default="", help="parquet with iid_code column to restrict items")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--template", action="store_true", help="X7: add first/last-word template tokens")
    ap.add_argument("--merged", action="store_true", help="subjects keyed by the competition definition")
    ap.add_argument("--formative", type=int, default=0, help="K>0: formative-like pairs sharing K common items")
    ap.add_argument("--rows", type=int, default=1000, help="subject-item rows per benchmark in --formative mode")
    ap.add_argument("--tok", default="feat", choices=["feat", "text", "both"],
                    help="token sources for *_f methods: item_features, derived text tokens, or both")
    args = ap.parse_args()
    hp, knn_kw = json.loads(args.hp), json.loads(args.knn)
    names = args.methods.split(",")
    t0 = time.time()
    emb = Emb(*load_embeddings(args.emb)) if args.emb else None
    subset = emb.codes if emb is not None else (
        pd.read_parquet(args.subset)["iid_code"].to_numpy() if args.subset else None)
    cells, meta = load_cells(subset, merged=args.merged)
    if args.no_provider:
        import priors as _priors
        _priors.PROVIDER_FALLBACK = False
    print(f"loaded {len(cells):,} cells in {time.time()-t0:.0f}s", flush=True)
    if any(m.split("@")[0].split("+")[0].endswith("f") for m in names):
        import online as _online
        _online.TEMPLATE_TOKENS = args.template
        from online import feature_tokens, text_tokens
        from data import find_dir
        need = cells.iid.unique().tolist()
        cols = ["iid_code", "item_features"] + (["text"] if args.tok != "feat" else [])
        it = pd.read_parquet(find_dir("items_all.parquet") / "items_all.parquet", columns=cols,
                             filters=[("iid_code", "in", need)])
        for row in it.itertuples(index=False):
            toks = feature_tokens(row.item_features) if args.tok != "text" else []
            if args.tok != "feat":
                toks = toks + text_tokens(row.text)
            TOKENS[int(row.iid_code)] = tuple(toks)
        print(f"item feature tokens for {sum(1 for v in TOKENS.values() if v):,} items", flush=True)
    folds = fold_of(cells["bench"].unique(), args.folds)
    cells["fold"] = cells["bench"].map(folds).astype(int)
    results = {m: [] for m in names}
    fold_ids = [int(x) for x in args.only_folds.split(",")] if args.only_folds else range(args.folds)
    for f in fold_ids:
        test = cells[cells.fold == f]
        rng = np.random.default_rng(args.seed + f)
        if args.formative:
            pairs = build_pairs_formative(test, rng, args.formative, args.rows)
        else:
            pairs = build_pairs(test, rng, args.max_subj, args.max_acq, args.max_eval)
        test_items = np.unique(np.concatenate([np.r_[p["acq"], p["eval"]] for p in pairs]))
        methods = make_methods(cells[cells.fold != f], meta, names, hp, emb, test_items, knn_kw,
                               dict(hp, **json.loads(args.hp2)))
        for m, (fn, policy) in methods.items():
            t1 = time.time()
            r = run_episode(test, pairs, fn, policy=policy, rng=np.random.default_rng(100 + f),
                            single_seed=910 + f if args.single else None)
            results[m].append(r["brier"])
            print(f"fold {f} {m:16s} pairs={len(pairs):4d} ALC={r['alc']:.4f} "
                  f"curve={np.round(r['mean_brier'], 4)} ({time.time()-t1:.0f}s)", flush=True)
    print("\n=== pooled over folds (equal weight per pair) ===")
    summary = {}
    for m in names:
        if not results[m]:
            continue
        b = np.concatenate(results[m]).mean(0)
        summary[m] = {"alc": float(b @ WEIGHTS), "curve": [round(float(x), 4) for x in b]}
        print(f"{m:16s} ALC={summary[m]['alc']:.4f} curve={np.round(b, 4)}")
    OUT.mkdir(parents=True, exist_ok=True)
    # per-pair Brier curves (pairs x budgets) for paired standard errors
    np.savez_compressed(OUT / f"pairs_{args.tag or int(time.time())}.npz",
                        **{m: np.concatenate(results[m]) for m in names if results[m]})
    (OUT / f"eval_{args.tag or int(time.time())}.json").write_text(json.dumps(
        {"args": vars(args), "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
