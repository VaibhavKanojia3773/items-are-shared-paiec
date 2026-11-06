"""Acquisition policies for the simulator (mirror submission/labeling.py).

UncertaintyPolicy: accept an item when its outcome is uncertain enough,
    1 - 4 p (1 - p) <= gamma * labels_remaining / items_remaining,
i.e. the acceptance threshold matches the remaining labelling rate (gamma > 1
spends labels earlier). Always takes everything once items <= labels left.
For speed the online model is refitted every `refit_every` new labels and a
pair's whole acquisition pool is predicted in one vectorised call per refit
(the real evaluator refits on every call).
"""
import numpy as np
import pandas as pd

from online import OnlineModel
from sim import random_policy


def accept(p, context, gamma, var_item=None, cover=1.0):
    """Uncertainty rule; with var_item, a V-optimal-style score that discounts items whose
    difficulty is still uncertain: info = w / (1 + w * var_item), w = p(1-p), normalised by
    its maximum (p = 0.5) so the threshold keeps the same rate semantics."""
    remaining, items = context["labels_remaining"], context["items_remaining"]
    if remaining <= 0:
        return False
    if items <= remaining:
        return True
    w = p * (1.0 - p)
    if var_item is None:
        q = 4.0 * w
    else:
        q = (w / (1.0 + w * var_item)) / (0.25 / (1.0 + 0.25 * var_item))
    return 1.0 - q * cover <= gamma * remaining / items


class UncertaintyPolicy:
    def __init__(self, eta, hp, gamma=1.0, refit_every=100, tkey=None, emb=None, vopt=False, beta=0.0, fade=False, gate=False):
        self.eta, self.hp, self.gamma, self.refit_every = eta, hp, gamma, refit_every
        # beta: coverage weight; the score is scaled by exp(-beta * min(labels on this item, 3)),
        # so beta > 0 prefers items no other subject has revealed yet (their offsets help every pair).
        # fade: scale beta by labels_remaining / max_labels (coverage early, pure uncertainty late).
        self.beta, self.fade, self.n_seen, self.item_n = beta, fade, 0, {}
        # gate: use the evaluator's random rule until another system has a label on this benchmark
        # (without partners, uncertainty selection biases the pair's own ability estimate)
        self.gate, self.g_seen, self.bench_sids = gate, 0, {}
        self.vopt = vopt
        self.tkey, self.emb = tkey, emb
        self.model, self.n_fit, self.version, self.cache = None, -1, 0, {}

    def _refit(self, labeled):
        l = labeled.df()
        tk = self.tkey(l) if self.tkey else None
        self.model = OnlineModel(self.hp).fit(l.sid, l.iid, l.bench, l.n, l.k, self.eta(l), tkey=tk)
        self.n_fit = len(labeled)
        self.version += 1

    def __call__(self, pair, iid, labeled, context, predictor):
        if context["labels_remaining"] <= 0:
            return False
        if context["items_remaining"] <= context["labels_remaining"]:
            return True
        if self.gate:
            for row in labeled.rows[self.g_seen:]:
                self.bench_sids.setdefault(row[2], set()).add(row[0])
            self.g_seen = len(labeled.rows)
            if not (self.bench_sids.get(pair['bench'], set()) - {pair['sid']}):
                return random_policy(context, iid)
        if self.model is None or len(labeled) - self.n_fit >= self.refit_every:
            self._refit(labeled)
        key = (pair["sid"], pair["bench"])
        hit = self.cache.get(key)
        if hit is None or hit[0] != self.version:
            rows = pd.DataFrame({"sid": pair["sid"], "iid": pair["acq"], "bench": pair["bench"]})
            tk = self.tkey(rows) if self.tkey else None
            parts = {}
            p = self.model.predict(rows.sid, rows.iid, rows.bench, self.eta(rows), emb=self.emb, tkey=tk,
                                   parts=parts)
            vw = parts["v_w"] if self.vopt else np.full(len(p), np.nan)
            hit = (self.version, dict(zip(pair["acq"], zip(p, vw))))
            self.cache[key] = hit
        p, vw = hit[1][iid]
        cover = 1.0
        if self.beta:
            for row in labeled.rows[self.n_seen:]:
                self.item_n[(row[2], row[1])] = self.item_n.get((row[2], row[1]), 0) + 1
            self.n_seen = len(labeled.rows)
            beta = self.beta * (context["labels_remaining"] / context["max_labels"] if self.fade else 1.0)
            cover = float(np.exp(-beta * min(self.item_n.get((pair["bench"], iid), 0), 3)))
        return accept(float(p), context, self.gamma, float(vw) if self.vopt else None, cover)
