"""Streaming acquisition: reveal the items whose outcome is most uncertain.

Fisher information of a Bernoulli outcome is p(1-p), maximal at p = 0.5, so
labels near 0.5 tell the most about the subject's ability on this benchmark.
An item is accepted when 1 - 4p(1-p) <= gamma * labels_remaining / items_remaining:
the threshold tracks the remaining labelling rate, and every remaining item is
taken once items_remaining <= labels_remaining, so the full budget is spent.
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "artifacts", "config.json")) as fh:
    _GAMMA = float(json.load(fh).get("acq", {}).get("gamma", 1.0))


def acquisition_function(input, prediction, labeled, context) -> bool:
    remaining = context["labels_remaining"]
    items = context["items_remaining"]
    if remaining <= 0:
        return False
    if items <= remaining:
        return True
    p = float(prediction)
    return 1.0 - 4.0 * p * (1.0 - p) <= _GAMMA * remaining / items
