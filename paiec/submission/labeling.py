"""Streaming acquisition: reveal the items whose outcome is most uncertain.

Fisher information of a Bernoulli outcome is p(1-p), maximal at p = 0.5, so
labels near 0.5 tell the most about the subject's ability on this benchmark.
An item is accepted when 1 - 4p(1-p) * cover <= gamma * labels_remaining / items_remaining:
the threshold tracks the remaining labelling rate, and every remaining item is
taken once items_remaining <= labels_remaining, so the full budget is spent.

cover = exp(-beta * min(n_item, 3)) where n_item counts labels already revealed on
this item (any subject): a first label on an item also fixes its offset for every
other subject whose evaluation pool contains it. With fade, beta shrinks as the
pair's budget is spent. With gate, items are sampled uniformly at the remaining rate until another
system has a label on the benchmark: without partner evidence, uncertainty selection biases the pair's
own ability estimate.
"""
import hashlib
import json
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "artifacts", "config.json")) as fh:
    _ACQ = json.load(fh).get("acq", {})
_GAMMA = float(_ACQ.get("gamma", 1.0))
_BETA = float(_ACQ.get("beta", 0.0))
_FADE = bool(_ACQ.get("fade", False))
_GATE = bool(_ACQ.get("gate", False))  # random rule until another system has labels on the benchmark


def _item_key(item):
    # same identity as model._item_key, scoped by benchmark
    text = "\x1f".join(item.get(f) or "" for f in ("item_content", "item_features", "interactors"))
    return (item.get("benchmark_id") or "", hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest())


_COUNTS = {"n": 0, "last": None, "counts": {}, "bench_subjects": {}}


def _subject_key(subject):
    return json.dumps(subject, sort_keys=True)


def _item_counts(labeled):
    """Labels per item key; `labeled` only grows, so new entries are added incrementally."""
    c = _COUNTS
    n = len(labeled)
    if n < c["n"] or (c["n"] and json.dumps(labeled[c["n"] - 1], sort_keys=True) != c["last"]):
        c.update(n=0, last=None, counts={}, bench_subjects={})
    for entry in labeled[c["n"]:]:
        key = _item_key(entry[0][1])
        c["counts"][key] = c["counts"].get(key, 0) + 1
        c["bench_subjects"].setdefault(key[0], set()).add(_subject_key(entry[0][0]))
    if n:
        c["n"], c["last"] = n, json.dumps(labeled[n - 1], sort_keys=True)
    return c["counts"]


def acquisition_function(input, prediction, labeled, context) -> bool:
    remaining = context["labels_remaining"]
    items = context["items_remaining"]
    if remaining <= 0:
        return False
    if items <= remaining:
        return True
    if _GATE:
        _item_counts(labeled)
        bench = input[1].get("benchmark_id") or ""
        if not (_COUNTS["bench_subjects"].get(bench, set()) - {_subject_key(input[0])}):
            # no partner evidence yet: uniform sampling at the remaining rate (hash-based, reproducible)
            h = hashlib.sha256((_subject_key(input[0]) + "|" + "|".join(_item_key(input[1]))).encode()).digest()
            return int.from_bytes(h[:8], "big") / 2 ** 64 < remaining / items
    p = float(prediction)
    cover = 1.0
    if _BETA and labeled:
        beta = _BETA * (remaining / context["max_labels"] if _FADE and context.get("max_labels") else 1.0)
        cover = math.exp(-beta * min(_item_counts(labeled).get(_item_key(input[1]), 0), 3))
    return 1.0 - 4.0 * p * (1.0 - p) * cover <= _GAMMA * remaining / items
