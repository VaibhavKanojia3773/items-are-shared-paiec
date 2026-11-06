"""Local simulator of the PAIEC streaming_alc_v1 protocol.

Mirrors the published rules: per subject-benchmark pair with >= 80 distinct items,
items are split 50/50 into acquisition and evaluation pools; acquisition items
stream one at a time and the policy answers yes/no; at each budget checkpoint
(0, 1, 3, 7, 15, 31 labels per pair) every evaluation response is predicted with
the frozen `labeled` set, which spans all sampled pairs. Score = Brier ALC with
weights 0.1/0.2/0.2/0.2/0.2/0.1, equal weight per pair.

Data are aggregated cells (sid, iid, bench, n, k): n responses of subject sid on
item iid, k of them correct. Repeated responses stay together: acquiring an item
reveals the whole cell for one budget unit, and every response is an evaluation
target, so a cell's Brier contribution is (n p^2 - 2 p k + k) over n responses.
"""
import hashlib
import numpy as np
import pandas as pd

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
MIN_ITEMS = 80


def brier_alc(per_budget_brier):
    """per_budget_brier: array (..., 6) -> weighted ALC (lower is better)."""
    return np.asarray(per_budget_brier) @ WEIGHTS


def random_policy(context, item_key):
    """The evaluator's default: uniform sampling without replacement (hash-based)."""
    key = f"{context['subject_id']}|{context['benchmark_id']}|{item_key}"
    u = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2 ** 64
    return u < min(1.0, context["labels_remaining"] / context["items_remaining"])


def build_pairs(cells, rng, max_subjects_per_bench=None, max_acq=None, max_eval=None):
    """Sample evaluation pairs and split each pair's items 50/50.

    max_acq / max_eval subsample the pools for speed (the official evaluation
    also samples); the split itself is made on the full item set first.
    """
    pairs = []
    for (bench, sid), g in cells.groupby(["bench", "sid"], sort=False, observed=True):
        if len(g) < MIN_ITEMS:
            continue
        pairs.append((bench, sid, g["iid"].to_numpy()))
    if max_subjects_per_bench:
        by_b = {}
        for p in pairs:
            by_b.setdefault(p[0], []).append(p)
        pairs = []
        for b, ps in by_b.items():
            if len(ps) > max_subjects_per_bench:
                ps = [ps[i] for i in sorted(rng.choice(len(ps), max_subjects_per_bench, replace=False))]
            pairs += ps
    out = []
    for bench, sid, items in pairs:
        perm = rng.permutation(items)
        half = len(perm) // 2
        acq, ev = perm[:half], perm[half:]
        if max_acq and len(acq) > max_acq:
            acq = acq[:max_acq]
        if max_eval and len(ev) > max_eval:
            ev = ev[:max_eval]
        out.append({"sid": sid, "bench": bench, "acq": acq, "eval": ev})
    return out


def build_pairs_formative(cells, rng, k_items=100, n_rows=1000):
    """Formative-like sample: per benchmark, ~n_rows/k_items subjects that all answered the
    same k_items items (high item overlap across pairs, small pools); each pair split 50/50."""
    out = []
    n_subj = max(1, n_rows // k_items)  # 1 = a pair alone on its benchmark (common in hidden draws)
    for bench, g in cells.groupby("bench", sort=True, observed=True):
        cover = g.groupby("iid").size().sort_values(ascending=False, kind="stable")
        if len(cover) < k_items:
            continue
        top = cover.index.to_numpy()[: 2 * k_items]
        items = rng.choice(top, k_items, replace=False)
        has = g[g.iid.isin(items)].groupby("sid").size()
        subj = has.index[has.to_numpy() == k_items].to_numpy()
        if len(subj) < min(2, n_subj):
            continue
        subj = rng.choice(subj, min(n_subj, len(subj)), replace=False)
        for sid in sorted(subj):
            perm = rng.permutation(items)
            out.append({"sid": sid, "bench": bench, "acq": perm[: k_items // 2], "eval": perm[k_items // 2:]})
    return out


class LabeledView:
    """Revealed rows; the DataFrame is built only when needed (cached by length)."""

    def __init__(self, rows):
        self.rows, self._df, self._n = rows, None, -1

    def __len__(self):
        return len(self.rows)

    def df(self):
        if self._n != len(self.rows):
            self._df = pd.DataFrame(self.rows, columns=["sid", "iid", "bench", "n", "k"])
            self._n = len(self.rows)
        return self._df


def run_episode(cells, pairs, predictor, policy=None, rng=None, budgets=BUDGETS, single_seed=None):
    """Simulate one evaluation.

    predictor(targets, labeled) -> probabilities, where
      targets: DataFrame (sid, iid, bench) to predict,
      labeled: DataFrame (sid, iid, bench, n, k) revealed so far (all pairs).
    policy(pair, iid, labeled, context, predictor) -> bool, or None for random;
      labeled is a LabeledView (len() is cheap, .df() builds the DataFrame).
    Returns per-pair Brier (n_pairs, 6), the mean curve and the ALC score.
    single_seed: if set, an acquired item reveals ONE binary response drawn from its cell
    (Bernoulli(k/n), fixed per candidate and identical across methods) instead of the whole
    (n, k) cell, as in the official protocol. Evaluation cells keep all responses.
    """
    rng = rng or np.random.default_rng(0)
    lookup = cells.set_index(["sid", "bench", "iid"])[["n", "k"]]
    ev = pd.concat([pd.DataFrame({"pair": j, "sid": p["sid"], "bench": p["bench"], "iid": p["eval"]})
                    for j, p in enumerate(pairs)], ignore_index=True)
    nk = lookup.loc[pd.MultiIndex.from_arrays([ev.sid, ev.bench, ev.iid])].to_numpy()
    ev["n"], ev["k"] = nk[:, 0], nk[:, 1]
    stream = [list(rng.permutation(p["acq"])) for p in pairs]
    draw = None
    if single_seed is not None:  # one fixed binary draw per acquisition candidate
        brng = np.random.default_rng(single_seed)
        draw = []
        for j, p in enumerate(pairs):
            nk = lookup.loc[pd.MultiIndex.from_arrays([np.full(len(p["acq"]), p["sid"]),
                                                       np.full(len(p["acq"]), p["bench"]), p["acq"]])].to_numpy()
            u = brng.random(len(p["acq"]))
            draw.append(dict(zip(p["acq"].tolist(), (u < nk[:, 1] / nk[:, 0]).astype(float))))
    acquired = [0] * len(pairs)
    labeled_rows = []
    view = LabeledView(labeled_rows)
    briers = np.zeros((len(pairs), len(budgets)))
    denom = np.bincount(ev["pair"], weights=ev["n"], minlength=len(pairs))
    for b_idx, budget in enumerate(budgets):
        for j, p in enumerate(pairs):
            while acquired[j] < budget and stream[j]:
                iid = stream[j].pop(0)
                context = {"subject_id": str(p["sid"]), "benchmark_id": str(p["bench"]),
                           "labels_acquired": acquired[j],
                           "labels_remaining": budgets[-1] - acquired[j],
                           "max_labels": budgets[-1], "items_remaining": len(stream[j]) + 1}
                if policy is None:
                    take = random_policy(context, iid)
                else:
                    take = policy(p, iid, view, context, predictor)
                if take:
                    acquired[j] += 1
                    if draw is None:
                        n, k = lookup.loc[(p["sid"], p["bench"], iid)]
                    else:
                        n, k = 1.0, draw[j][int(iid)]
                    labeled_rows.append((p["sid"], iid, p["bench"], n, k))
        labeled = pd.DataFrame(labeled_rows, columns=["sid", "iid", "bench", "n", "k"])
        prob = np.asarray(predictor(ev[["sid", "iid", "bench"]], labeled), float)
        if not np.all(np.isfinite(prob)) or prob.min() < 0 or prob.max() > 1:
            raise ValueError("predictor returned invalid probabilities")
        sq = ev["n"] * prob ** 2 - 2 * prob * ev["k"] + ev["k"]
        briers[:, b_idx] = np.bincount(ev["pair"], weights=sq, minlength=len(pairs)) / denom
    return {"brier": briers, "mean_brier": briers.mean(0),
            "alc": float(brier_alc(briers.mean(0))), "n_acquired": np.array(acquired)}
