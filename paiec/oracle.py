"""Headroom analysis: Brier of oracle predictors on the simulator's evaluation pools.

O_pair : the pair's true success rate over all its items (subject x benchmark known).
O_item : additive logit of the pair rate and the item's rate among *other* subjects
         (item difficulty known from everyone else), relative to the benchmark rate.
O_cell : the cell's own rate (irreducible noise from repeated responses only).
"""
import numpy as np
import pandas as pd

from sim import build_pairs
from data import load_cells
from evaluate import fold_of
from priors import logit
from online import sigmoid


def main():
    cells, meta = load_cells(pd.read_parquet("data/proc/subset10k.parquet")["iid_code"].to_numpy())
    folds = fold_of(cells["bench"].unique(), 5)
    cells["fold"] = cells["bench"].map(folds).astype(int)
    res = {"O_pair": [], "O_item": [], "O_cell": [], "const": []}
    for f in range(5):
        test = cells[cells.fold == f]
        pairs = build_pairs(test, np.random.default_rng(f), 20, 200, 100)
        pr = test.groupby(["sid", "bench"])[["k", "n"]].sum()
        pr = (pr.k / pr.n)
        br = test.groupby("bench")[["k", "n"]].sum()
        br = (br.k / br.n)
        it = test.groupby(["bench", "iid"])[["k", "n"]].sum()
        for p in pairs:
            ev = test[(test.sid == p["sid"]) & (test.bench == p["bench"]) & test.iid.isin(p["eval"])]
            n, k = ev.n.to_numpy(float), ev.k.to_numpy(float)
            rate = k / n
            r_pair = pr.loc[(p["sid"], p["bench"])]
            oth = it.loc[[(p["bench"], i) for i in ev.iid]].to_numpy() - ev[["k", "n"]].to_numpy()
            r_item = np.where(oth[:, 1] > 0, (oth[:, 0] + 0.5) / (oth[:, 1] + 1.0), br.loc[p["bench"]])
            p_item = sigmoid(logit(r_pair) + logit(r_item) - logit(br.loc[p["bench"]]))
            for name, q in [("O_pair", np.full(len(n), r_pair)), ("O_item", p_item),
                            ("O_cell", rate), ("const", np.full(len(n), 0.5))]:
                res[name].append(((n * q ** 2 - 2 * q * k + k).sum()) / n.sum())
    for name, v in res.items():
        print(f"{name:7s} mean per-pair Brier = {np.mean(v):.4f}")


if __name__ == "__main__":
    main()
