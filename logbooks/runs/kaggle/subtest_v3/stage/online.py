"""Test-time hierarchical logistic model fitted on the revealed labels.

    logit p(s, i) = a * eta0(s, i) + u_b + t_s + v_{s,b} - w_i

eta0 is the offline prior logit (budget-0 assessor). The random effects are
Gaussian with variances tuned on simulated episodes:
    u_b      benchmark offset           (pooled over all subjects on b)
    t_s      subject offset             (pooled over all benchmarks in the run)
    v_{s,b}  subject x benchmark offset (the pair's own labels)
    w_i      item offset                (other subjects' labels on item i)
Labels are binomial cells (n responses, k correct); n is capped at n_cap so a
heavily resampled item counts like a few labels. MAP fit by blockwise Newton
steps (numpy only); predictions integrate the Laplace posterior variance with a
probit-style correction, which keeps low-budget predictions calibrated.
"""
import numpy as np

DEFAULT_HP = {"a": 1.0, "s_u": 1.0, "s_t": 0.5, "s_v": 0.7, "s_w": 1.2,
              "iters": 12, "c": np.pi / 8, "n_cap": 4.0}
GROUPS = ("u", "t", "v", "w")
SIGN = {"u": 1.0, "t": 1.0, "v": 1.0, "w": -1.0}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _keys(sid, iid, bench, tkey=None):
    """Group keys. tkey (e.g. model name) pools the subject offset across benchmarks."""
    return {"u": list(bench), "t": list(sid if tkey is None else tkey),
            "v": list(zip(sid, bench)), "w": list(zip(bench, iid))}


class OnlineModel:
    def __init__(self, hp=None):
        self.hp = dict(DEFAULT_HP, **(hp or {}))

    def fit(self, sid, iid, bench, n, k, eta0, tkey=None, feats=None):
        """feats: optional (rows, d) subject features; adds a per-benchmark loading
        vector lambda_b ~ N(0, s_L^2 I) so that logit += feats . lambda_b, i.e. each new
        benchmark learns, from all subjects' labels, which subject skills matter."""
        hp = self.hp
        n = np.asarray(n, float)
        k = np.asarray(k, float)
        scale = np.minimum(n, hp["n_cap"]) / np.maximum(n, 1)
        n_eff, k_eff = n * scale, k * scale
        base = hp["a"] * np.asarray(eta0, float)
        self.var = {"u": hp["s_u"] ** 2, "t": hp["s_t"] ** 2, "v": hp["s_v"] ** 2, "w": hp["s_w"] ** 2}
        self.vocab, idx = {}, {}
        for g, keys in _keys(sid, iid, bench, tkey).items():
            vocab = {}
            idx[g] = np.fromiter((vocab.setdefault(x, len(vocab)) for x in keys), np.int64, len(keys))
            self.vocab[g] = vocab
        size = {g: len(self.vocab[g]) for g in GROUPS}
        theta = {g: np.zeros(size[g]) for g in GROUPS}
        prec = {g: np.full(size[g], 1.0 / self.var[g]) for g in GROUPS}
        F = None if feats is None else np.asarray(feats, float)
        lam_L = None
        if F is not None:
            bidx = idx["u"]
            nbench, d = size["u"], F.shape[1]
            lam_L = np.zeros((nbench, d))
            prec_L = np.zeros((nbench, d, d))
        lin_L = lambda: 0.0 if F is None else (F * lam_L[bidx]).sum(1)
        for _ in range(hp["iters"] if len(n) else 0):
            if F is not None:
                lin = base + sum(SIGN[h] * theta[h][idx[h]] for h in GROUPS) + lin_L()
                p = sigmoid(lin)
                wgt = n_eff * p * (1 - p)
                H = np.zeros((nbench, d, d)) + np.eye(d) / hp.get("s_L", 0.5) ** 2
                np.add.at(H, bidx, F[:, :, None] * F[:, None, :] * wgt[:, None, None])
                G = np.zeros((nbench, d))
                np.add.at(G, bidx, F * (k_eff - n_eff * p)[:, None])
                G -= lam_L / hp.get("s_L", 0.5) ** 2
                lam_L = lam_L + np.linalg.solve(H, G[..., None])[..., 0]
                prec_L = H
            for g in GROUPS:
                lin = base + sum(SIGN[h] * theta[h][idx[h]] for h in GROUPS) + lin_L()
                p = sigmoid(lin)
                grad = np.bincount(idx[g], weights=SIGN[g] * (k_eff - n_eff * p), minlength=size[g])
                hess = np.bincount(idx[g], weights=n_eff * p * (1 - p), minlength=size[g])
                grad -= theta[g] / self.var[g]
                prec[g] = hess + 1.0 / self.var[g]
                theta[g] += grad / prec[g]
        self.theta, self.prec = theta, prec
        self.lam_L = lam_L
        self.cov_L = None if lam_L is None else np.linalg.inv(prec_L)
        if F is not None:
            base = base + lin_L()
        # Per-row logit-scale residuals for subject-specific propagation (x term).
        if len(n):
            lin = base + sum(SIGN[h] * theta[h][idx[h]] for h in GROUPS)
            p = sigmoid(lin)
            h = np.maximum(n_eff * p * (1 - p), 1e-6)
            abil = base + sum(SIGN[g] * theta[g][idx[g]] for g in ("u", "t", "v"))
            self.res = {"sid": np.asarray(sid), "iid": np.asarray(iid), "bench": np.asarray(bench),
                        "z": (k_eff - n_eff * p) / h, "h": h, "abil": abil}
        else:
            self.res = None
        return self

    def predict(self, sid, iid, bench, eta0, emb=None, tkey=None, parts=None, feats=None):
        """Probabilities for targets; unseen groups fall back to their prior.

        emb: optional callable iid array -> unit-norm embeddings. Unlabelled
        target items then borrow the fitted offsets of similar labelled items
        of the same benchmark (kernel exp((cos-1)/tau_w), shrinkage kappa_w).
        """
        hp = self.hp
        mu = hp["a"] * np.asarray(eta0, float)
        s2 = np.zeros_like(mu)
        if feats is not None and self.lam_L is not None:
            F = np.asarray(feats, float)
            j = np.fromiter((self.vocab["u"].get(b, -1) for b in bench), np.int64, len(mu))
            known = j >= 0
            sL2 = hp.get("s_L", 0.5) ** 2
            s2 += np.where(known, 0.0, sL2 * (F * F).sum(1))
            if known.any():
                Fk = F[known]
                mu[known] += (Fk * self.lam_L[j[known]]).sum(1)
                s2[known] += np.einsum("nd,nde,ne->n", Fk, self.cov_L[j[known]], Fk)
        abil = mu.copy()
        for g, keys in _keys(sid, iid, bench, tkey).items():
            vocab = self.vocab[g]
            j = np.fromiter((vocab.get(x, -1) for x in keys), np.int64, len(keys))
            known = j >= 0
            m = np.zeros_like(mu)
            v = np.full_like(mu, self.var[g])
            m[known] = self.theta[g][j[known]]
            v[known] = 1.0 / self.prec[g][j[known]]
            if g == "w" and emb is not None and (~known).any() and len(vocab):
                m_new, v_new = self._propagate(np.asarray(iid)[~known], np.asarray(bench)[~known], emb)
                m[~known], v[~known] = m_new, v_new
            mu += SIGN[g] * m
            s2 += v
            if g in ("u", "t", "v"):
                abil += SIGN[g] * m
            if parts is not None:  # per-component diagnostics for stacking
                parts[f"m_{g}"], parts[f"v_{g}"], parts[f"k_{g}"] = m, v, known.astype(float)
        if emb is not None and hp.get("x_scale", 0.0) > 0 and self.res is not None:
            x = self._subject_residual(np.asarray(sid), np.asarray(iid), np.asarray(bench), emb)
            mu += hp["x_scale"] * x
            if parts is not None:
                parts["m_x"] = x
        if hp.get("y_scale", 0.0) > 0 and self.res is not None:
            y = self._item_residual(np.asarray(sid), np.asarray(iid), np.asarray(bench), abil)
            mu += hp["y_scale"] * y
            if parts is not None:
                parts["m_y"] = y
        if parts is not None:
            parts["mu"], parts["s2"] = mu.copy(), s2.copy()
        return sigmoid(mu / np.sqrt(1.0 + hp["c"] * s2))

    def _item_residual(self, sid, iid, bench, abil):
        """Other subjects' residuals on the same item, weighted by ability similarity.

        An online stand-in for item discrimination: a hard item that only strong
        subjects solved lowers the prediction for weak subjects more than strong ones.
        """
        hp, r = self.hp, self.res
        ell, kappa = hp.get("ell_y", 1.0), hp.get("kappa_y", 2.0)
        out = np.zeros(len(sid))
        by_item = {}
        for j, key in enumerate(zip(r["bench"].tolist(), r["iid"].tolist())):
            by_item.setdefault(key, []).append(j)
        for t, key in enumerate(zip(bench.tolist(), iid.tolist())):
            l = by_item.get(key)
            if not l:
                continue
            l = np.array([j for j in l if r["sid"][j] != sid[t]])
            if len(l) == 0:
                continue
            w = np.exp(-0.5 * ((abil[t] - r["abil"][l]) / ell) ** 2) * r["h"][l]
            out[t] = (w @ r["z"][l]) / (w.sum() + kappa)
        return out

    def _subject_residual(self, sid, iid, bench, emb):
        """Kernel-smoothed residual of the same subject on similar labelled items of the same benchmark."""
        hp, r = self.hp, self.res
        tau, kappa = hp.get("tau_x", 0.05), hp.get("kappa_x", 2.0)
        out = np.zeros(len(sid))
        lab_groups = {}
        for j, key in enumerate(zip(r["sid"].tolist(), r["bench"].tolist())):
            lab_groups.setdefault(key, []).append(j)
        tgt_groups = {}
        for j, key in enumerate(zip(sid.tolist(), bench.tolist())):
            tgt_groups.setdefault(key, []).append(j)
        for key, t in tgt_groups.items():
            l = lab_groups.get(key)
            if not l:
                continue
            l = np.array(l)
            sims = emb(iid[t]) @ emb(r["iid"][l]).T
            w = np.exp((sims - 1.0) / tau) * r["h"][l][None, :]
            out[t] = (w @ r["z"][l]) / (w.sum(1) + kappa)
        return out

    def _propagate(self, iid, bench, emb):
        hp = self.hp
        tau, kappa = hp.get("tau_w", 0.05), hp.get("kappa_w", 1.0)
        lab = np.array(list(self.vocab["w"].keys()), dtype=object)
        lab_b = np.array([k[0] for k in lab])
        lab_i = np.array([k[1] for k in lab])
        lab_w = self.theta["w"]
        m = np.zeros(len(iid))
        v = np.full(len(iid), self.var["w"])
        for b in np.unique(bench):
            t = np.flatnonzero(bench == b)
            l = np.flatnonzero(lab_b == b)
            if len(l) == 0:
                continue
            sims = emb(iid[t]) @ emb(lab_i[l]).T
            w = np.exp((sims - 1.0) / tau)
            tot = w.sum(1) + kappa
            m[t] = (w @ lab_w[l]) / tot
            v[t] = self.var["w"] * kappa / tot
        return m, v
