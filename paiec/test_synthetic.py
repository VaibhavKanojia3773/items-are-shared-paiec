"""Sanity check of the simulator and online model on synthetic IRT data.

Ground truth: logit = theta_s + phi_b - d_i. The "offline prior" sees theta_s and
a noisy benchmark guess, but not item difficulty, mimicking the cold start.
"""
import numpy as np
import pandas as pd
from sim import build_pairs, run_episode
from online import OnlineModel


def make_data(rng, n_subj=30, n_bench=12):
    theta = rng.normal(0, 1.0, n_subj)
    rows, prior = [], {}
    for b in range(n_bench):
        phi = rng.normal(0, 1.5)
        guess = phi + rng.normal(0, 1.0)  # imperfect text-based benchmark prior
        n_items = rng.integers(90, 250)
        d = rng.normal(0, 1.3, n_items)
        subj = rng.choice(n_subj, rng.integers(8, n_subj), replace=False)
        for s in subj:
            p = 1 / (1 + np.exp(-(theta[s] + phi - d)))
            y = (rng.random(n_items) < p).astype(int)
            for i in range(n_items):
                rows.append((f"s{s}", f"b{b}_i{i}", f"b{b}", 1, y[i]))
            prior[(f"s{s}", f"b{b}")] = theta[s] + guess
    return pd.DataFrame(rows, columns=["sid", "iid", "bench", "n", "k"]), prior


def main():
    rng = np.random.default_rng(1)
    resp, prior = make_data(rng)
    pairs = build_pairs(resp, rng)
    eta = lambda df: np.array([prior[(s, b)] for s, b in zip(df.sid, df.bench)])

    def const(t, l):
        return np.full(len(t), 0.5)

    def emp_mean(t, l):
        g = (l.groupby(["sid", "bench"])["k"].sum() / l.groupby(["sid", "bench"])["n"].sum()).to_dict()
        return np.array([g.get((s, b), 0.5) for s, b in zip(t.sid, t.bench)])

    def prior_only(t, l):
        return 1 / (1 + np.exp(-eta(t)))

    def online(t, l):
        m = OnlineModel().fit(l.sid, l.iid, l.bench, l.n, l.k, eta(l))
        return m.predict(t.sid, t.iid, t.bench, eta(t))

    for name, f in [("const0.5", const), ("empirical_mean", emp_mean),
                    ("prior_only", prior_only), ("online", online)]:
        r = run_episode(resp, pairs, f, rng=np.random.default_rng(2))
        print(f"{name:15s} ALC={r['alc']:.4f}  per-budget={np.round(r['mean_brier'], 4)}")


if __name__ == "__main__":
    main()
