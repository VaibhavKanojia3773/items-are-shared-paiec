"""Random search over OnlineModel hyperparameters on simulated episodes (5 folds).

    python paiec/tune.py --n 40 --prior name --tag T001
    python paiec/tune.py --emb "/kaggle/input/**/emb_*.npy" --space prop --n 60 --seed 3 --tag T010
Results append to <PAIEC_OUT or logbooks/runs>/tune_<tag>.jsonl, best printed at the end.
--space base: random-effect HP only; prop: + text propagation (tau_w, kappa_w) and
subject-residual propagation (x_scale, tau_x, kappa_x); needs --emb.
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd

from sim import build_pairs, build_pairs_formative, run_episode, WEIGHTS
from online import OnlineModel, DEFAULT_HP
from priors import NamePrior
from data import load_cells, load_embeddings
from evaluate import fold_of, Emb
from acq import UncertaintyPolicy

OUT = Path(os.environ.get("PAIEC_OUT", Path(__file__).resolve().parents[1] / "logbooks" / "runs"))
BEST_BASE = {"a": 0.6, "s_u": 1.0, "s_t": 0.5, "s_v": 0.5, "s_w": 1.2, "c": 0.2, "n_cap": 2.0}


V3 = {"a": 0.4, "s_u": 1.4, "s_t": 0.5, "s_v": 0.7, "s_w": 2.2, "c": 0.1, "n_cap": 4.0,
      "tau_w": 0.02, "kappa_w": 1.0, "x_scale": 1.0, "tau_x": 0.03, "kappa_x": 10.0}


def sample_hp(rng, space):
    if space == "fine":  # local search around v3
        ch = lambda xs: float(rng.choice(xs))
        return {"a": ch([0.3, 0.4, 0.5]), "s_u": ch([1.0, 1.4, 2.0]), "s_t": ch([0.3, 0.5, 0.8]),
                "s_v": ch([0.5, 0.7, 1.0]), "s_w": ch([1.6, 2.2, 3.0, 4.0]), "c": ch([0.05, 0.1, 0.15]),
                "n_cap": ch([2.0, 4.0]), "tau_w": ch([0.01, 0.015, 0.02, 0.03]),
                "kappa_w": ch([0.3, 0.6, 1.0, 2.0]), "x_scale": ch([0.5, 1.0, 1.5]),
                "tau_x": ch([0.02, 0.03, 0.05]), "kappa_x": ch([5.0, 10.0, 20.0])}
    if space == "form":  # around v3, plus the item-discrimination residual y (high item overlap)
        ch = lambda xs: float(rng.choice(xs))
        return {"a": ch([0.2, 0.4, 0.6]), "s_u": ch([1.0, 1.4, 2.0]), "s_t": ch([0.3, 0.5, 0.8]),
                "s_v": ch([0.5, 0.7, 1.0]), "s_w": ch([1.6, 2.2, 3.0, 4.0]), "c": ch([0.05, 0.1, 0.2]),
                "n_cap": ch([2.0, 4.0, 8.0]), "tau_w": ch([0.01, 0.02, 0.03, 0.05]),
                "kappa_w": ch([0.3, 1.0, 2.0]), "x_scale": ch([0.0, 0.5, 1.0, 1.5]),
                "tau_x": ch([0.02, 0.03, 0.05]), "kappa_x": ch([3.0, 10.0, 20.0]),
                "y_scale": ch([0.0, 0.0, 0.5, 1.0]), "ell_y": ch([0.5, 1.0, 2.0]), "kappa_y": ch([1.0, 2.0, 5.0])}
    hp = {"a": float(rng.choice([0.4, 0.6, 0.8, 1.0])),
          "s_u": float(rng.choice([0.5, 0.8, 1.0, 1.4, 2.0])),
          "s_t": float(rng.choice([0.2, 0.5, 0.8])),
          "s_v": float(rng.choice([0.3, 0.5, 0.7, 1.0])),
          "s_w": float(rng.choice([0.6, 0.9, 1.2, 1.6, 2.2])),
          "c": float(rng.choice([0.1, 0.2, 0.3, np.pi / 8])),
          "n_cap": float(rng.choice([1.0, 2.0, 4.0]))}
    if space == "prop":
        hp.update({"tau_w": float(rng.choice([0.02, 0.05, 0.1, 0.2])),
                   "kappa_w": float(rng.choice([0.3, 1.0, 3.0, 10.0])),
                   "x_scale": float(rng.choice([0.0, 0.0, 0.5, 1.0])),
                   "tau_x": float(rng.choice([0.03, 0.05, 0.1])),
                   "kappa_x": float(rng.choice([1.0, 3.0, 10.0]))})
    return hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--prior", default="name", choices=["const", "name"])
    ap.add_argument("--space", default="base", choices=["base", "prop", "fine", "form"])
    ap.add_argument("--emb", default="")
    ap.add_argument("--subset", default="data/proc/subset10k.parquet")
    ap.add_argument("--gamma", type=float, default=0.0, help="uncertainty acquisition (0 = random)")
    ap.add_argument("--formative", type=int, default=0, help="K>0: formative-like pairs sharing K items")
    ap.add_argument("--tag", default="T001")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    emb = Emb(*load_embeddings(args.emb)) if args.emb else None
    subset = emb.codes if emb is not None else pd.read_parquet(args.subset)["iid_code"].to_numpy()
    cells, meta = load_cells(subset)
    folds = fold_of(cells["bench"].unique(), 5)
    cells["fold"] = cells["bench"].map(folds).astype(int)
    setups = []
    for f in range(5):
        test = cells[cells.fold == f]
        pairs = (build_pairs_formative(test, np.random.default_rng(f), args.formative) if args.formative
                 else build_pairs(test, np.random.default_rng(f), 20, 200, 100))
        pri = NamePrior().fit(cells[cells.fold != f], meta)
        eta = pri.logit if args.prior == "name" else (lambda df, mu=pri.mu: np.full(len(df), mu))
        setups.append((f, test, pairs, eta))
    rng = np.random.default_rng(args.seed)
    first = dict(DEFAULT_HP, **BEST_BASE)
    if args.space == "prop":
        first.update({"tau_w": 0.05, "kappa_w": 1.0, "x_scale": 0.0})
    if args.space in ("fine", "form"):
        first = dict(DEFAULT_HP, **V3)
    configs = [first] + [sample_hp(rng, args.space) for _ in range(args.n)]
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f"tune_{args.tag}.jsonl"
    best = None
    for c, hp in enumerate(configs):
        t0 = time.time()
        briers = []
        for f, test, pairs, eta in setups:
            def pred(t, l, eta=eta, hp=hp):
                m = OnlineModel(hp).fit(l.sid, l.iid, l.bench, l.n, l.k, eta(l))
                return m.predict(t.sid, t.iid, t.bench, eta(t), emb=emb)
            policy = UncertaintyPolicy(eta, hp, args.gamma, emb=emb) if args.gamma > 0 else None
            briers.append(run_episode(test, pairs, pred, policy=policy,
                                      rng=np.random.default_rng(100 + f))["brier"])
        curve = np.concatenate(briers).mean(0)
        alc = float(curve @ WEIGHTS)
        rec = {"hp": {k: float(v) for k, v in hp.items()}, "alc": alc,
               "curve": [round(float(x), 4) for x in curve], "prior": args.prior,
               "space": args.space, "gamma": args.gamma}
        with open(log, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if best is None or alc < best["alc"]:
            best = rec
        print(f"[{c:3d}] ALC={alc:.4f} best={best['alc']:.4f} {time.time()-t0:.0f}s {hp}", flush=True)
    print("BEST", json.dumps(best))


if __name__ == "__main__":
    main()
