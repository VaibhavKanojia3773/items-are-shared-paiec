"""PAIEC submission: offline prior + test-time hierarchical logistic model.

predict(input, labeled) -> P(correct) for one (subject, item) target.

    logit p = a * eta0(s, i) + u_bench + t_subject + v_subject,bench - w_item
eta0 is the offline prior: subject ability by normalized_name (fitted on the
public measurement-db) plus, when the text encoder is available, item easiness
from nearest public items. The random effects are fitted on `labeled` (which
spans all sampled pairs) at each checkpoint; unlabelled items borrow offsets of
similar labelled items of the same benchmark. No state is shared across
checkpoints; within one worker the fit is cached per frozen `labeled` list.
"""
import hashlib
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from online import OnlineModel, feature_tokens, text_tokens  # noqa: E402

with open(os.path.join(_HERE, "artifacts", "config.json")) as fh:
    CFG = json.load(fh)
with open(os.path.join(_HERE, "artifacts", "subject_prior.json")) as fh:
    _SP = json.load(fh)
_MU, _THETA, _PROV = _SP["mu"], _SP["theta"], _SP["prov_theta"]
_CLIP = CFG.get("clip", [0.005, 0.995])

_encoder = None
_knn = None
if CFG.get("use_text", False):  # encoder: needed for test-time text propagation
    try:
        import text_prior
        _encoder = text_prior.Encoder(_HERE, CFG)
        if CFG.get("use_knn", False):  # optional kNN easiness prior (weak across benchmarks)
            _knn = text_prior.Memory(_HERE, CFG)
    except Exception as exc:  # degrade to the no-text model
        print(f"[paiec] text features disabled: {type(exc).__name__}: {exc}", file=sys.stderr)
        _encoder = _knn = None


def _subject_key(subject):
    return json.dumps(subject, sort_keys=True)


def _item_key(item):
    # interactors (e.g. attacker=GCG) change the task, so they are part of the item identity
    text = "\x1f".join(item.get(f) or "" for f in ("item_content", "item_features", "interactors"))
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def _tokens(item):
    """Benchmark-scoped categorical tokens: item_features, interactors, optional text buckets."""
    toks = feature_tokens(item.get("item_features")) + feature_tokens(item.get("interactors"))
    if CFG.get("text_tokens", False):
        toks += text_tokens(item.get("item_content"))
    return toks


def _tkey(subject):
    """Pool the subject offset across benchmarks by model name (harness strings differ)."""
    name = (subject.get("normalized_name") or "").strip()
    if not CFG.get("pool_by_name", True) or not name:
        return _subject_key(subject)
    return name


def _theta(subject):
    if not CFG.get("use_name_prior", True):
        return 0.0
    name = (subject.get("normalized_name") or "").strip()
    if name in _THETA:
        return _THETA[name]
    return _PROV.get((subject.get("provider") or "").strip(), 0.0)


class _ItemText:
    """Embeddings and kNN easiness per item key, computed lazily and cached."""

    def __init__(self):
        self.emb, self.easy = {}, {}

    def add(self, items):
        new = [(k, it) for k, it in items.items() if k not in self.emb]
        if not new or _encoder is None:
            return
        vecs = _encoder.encode([it for _, it in new])
        easy = _knn.easiness(vecs) if _knn is not None else np.zeros(len(new))
        for (k, _), v, e in zip(new, vecs, easy):
            self.emb[k], self.easy[k] = v, float(e)

    def eta(self, subject, key):
        easy = self.easy.get(key, _knn.mu_easy) if _knn is not None else 0.0
        return _MU + _theta(subject) + easy

    def lookup(self, keys):
        return np.stack([self.emb[k] for k in keys])


_TEXT = _ItemText()
_CACHE = {"sig": None, "model": None}


def _signature(labeled):
    h = hashlib.sha1()
    h.update(str(len(labeled)).encode())
    for entry in labeled[:: max(1, len(labeled) // 64)] + labeled[-3:]:
        h.update(json.dumps(entry, sort_keys=True).encode())
    return h.hexdigest()


def _fit(labeled):
    sig = _signature(labeled)
    if _CACHE["sig"] == sig:
        return _CACHE["model"]
    cells, subjects, items = {}, {}, {}
    for (subject, item), label in labeled:
        sk, ik = _subject_key(subject), _item_key(item)
        bench = item.get("benchmark_id") or ""
        key = (sk, ik, bench)
        n, k = cells.get(key, (0, 0.0))
        cells[key] = (n + 1, k + float(label))
        subjects[sk], items[ik] = subject, item
    _TEXT.add(items)
    keys = list(cells)
    sid = [k[0] for k in keys]
    iid = [k[1] for k in keys]
    bench = [k[2] for k in keys]
    n = np.array([cells[k][0] for k in keys], float)
    k_ = np.array([cells[k][1] for k in keys], float)
    eta = np.array([_TEXT.eta(subjects[s], i) for s, i in zip(sid, iid)])
    tkey = [_tkey(subjects[s]) for s in sid]
    tokens = [_tokens(items[i]) for i in iid] if CFG["hp"].get("s_f") else None
    model = OnlineModel(CFG["hp"]).fit(sid, iid, bench, n, k_, eta, tkey=tkey, tokens=tokens)
    _CACHE.update(sig=sig, model=model)
    return model


def predict(input: list[dict], labeled: list | None = None) -> float:
    subject, item = input
    labeled = labeled or []
    sk, ik = _subject_key(subject), _item_key(item)
    bench = item.get("benchmark_id") or ""
    _TEXT.add({ik: item})
    eta = np.array([_TEXT.eta(subject, ik)])
    model = _fit(labeled)
    use_emb = _encoder is not None and CFG.get("propagate", True)
    tokens = [_tokens(item)] if CFG["hp"].get("s_f") else None
    p = model.predict([sk], [ik], [bench], eta, emb=_TEXT.lookup if use_emb else None,
                      tkey=[_tkey(subject)], tokens=tokens)[0]
    if not math.isfinite(p):
        p = 1.0 / (1.0 + math.exp(-_MU))
    return float(min(max(p, _CLIP[0]), _CLIP[1]))
