"""End-to-end test of a built submission through the dict-based competition API.

    python paiec/test_submission.py dist/stage [--benches 4 --subj 6 --eval 40]

Streams acquisition items through labeling.acquisition_function (or the random
rule), calls model.predict(input, labeled) per evaluation target at every
checkpoint, and reports the Brier curve and ALC. `labeled` spans all pairs.
"""
import argparse, hashlib, importlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim import BUDGETS, WEIGHTS, random_policy
from data import load_cells, find_dir

FIELDS = ["normalized_name", "provider", "release_date", "access_date", "harness",
          "harness_version", "reasoning_effort", "subject_features_extra"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--benches", type=int, default=4)
    ap.add_argument("--subj", type=int, default=6)
    ap.add_argument("--eval", type=int, default=40)
    ap.add_argument("--acq", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    sys.path.insert(0, args.stage)
    model = importlib.import_module("model")
    labeling = importlib.import_module("labeling") if (Path(args.stage) / "labeling.py").exists() else None
    rng = np.random.default_rng(args.seed)
    cells, meta = load_cells()
    benches = rng.choice(cells.bench.unique(), args.benches, replace=False)
    pairs = []
    for b in benches:
        cb = cells[cells.bench == b]
        sizes = cb.groupby("sid").size()
        for s in rng.permutation(sizes[sizes >= 80].index.to_numpy())[: args.subj]:
            its = rng.permutation(cb.loc[cb.sid == s, "iid"].to_numpy())
            half = len(its) // 2
            pairs.append((b, s, its[:half][: args.acq], its[half:][: args.eval]))
    look = cells.set_index(["sid", "iid"])[["n", "k"]]
    need = sorted({int(i) for p in pairs for i in np.r_[p[2], p[3]]})
    items = pd.read_parquet(find_dir() / "items_all.parquet", columns=["iid_code", "text", "item_features"],
                            filters=[("iid_code", "in", need)]).set_index("iid_code")

    def subj_dict(s):
        row = meta.loc[s]
        return {f: ("" if pd.isna(row.get(f)) else str(row.get(f))) for f in FIELDS}

    def item_dict(b, i):
        r = items.loc[i]
        return {"item_content": r.text, "item_features": r.item_features, "interactors": "",
                "benchmark_id": hashlib.sha1(str(b).encode()).hexdigest()[:12]}

    by_bench = {}
    labeled, acquired = [], [0] * len(pairs)
    streams = [list(p[2]) for p in pairs]
    curve = []
    t0 = time.time()
    for budget in BUDGETS:
        for j, (b, s, acq, ev) in enumerate(pairs):
            while acquired[j] < budget and streams[j]:
                i = streams[j].pop(0)
                inp = [subj_dict(s), item_dict(b, i)]
                ctx = {"subject_id": str(s), "benchmark_id": str(b), "labels_acquired": acquired[j],
                       "labels_remaining": 31 - acquired[j], "max_labels": 31,
                       "items_remaining": len(streams[j]) + 1}
                if labeling is None:
                    take = random_policy(ctx, i)
                else:
                    p = model.predict(inp, labeled)
                    take = labeling.acquisition_function(inp, p, labeled, ctx)
                if take:
                    acquired[j] += 1
                    n, k = look.loc[(s, i)]
                    labeled += [[inp, 1]] * int(k) + [[inp, 0]] * int(n - k)
        per_pair = []
        for b, s, acq, ev in pairs:
            num = den = 0.0
            for i in ev:
                p = model.predict([subj_dict(s), item_dict(b, i)], labeled)
                assert isinstance(p, float) and 0.0 <= p <= 1.0
                n, k = look.loc[(s, i)]
                num += n * p * p - 2 * p * k + k
                den += n
            per_pair.append(num / den)
            by_bench.setdefault(int(b), []).append((budget, num / den))
        curve.append(float(np.mean(per_pair)))
        print(f"budget {budget:2d}: brier={curve[-1]:.4f} labels={len(labeled)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"ALC={np.array(curve) @ WEIGHTS:.4f} pairs={len(pairs)}")
    w = dict(zip(BUDGETS, WEIGHTS))
    for b, recs in sorted(by_bench.items()):
        alc = sum(w[bu] * v for bu, v in recs) / (len(recs) / len(BUDGETS))
        print(f"  bench {b}: ALC={alc:.4f} pairs={len(recs) // len(BUDGETS)}")


if __name__ == "__main__":
    main()
