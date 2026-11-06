"""Offline priors (budget-0 predictions) fitted on training benchmarks only.

NamePrior: two-way logit model on subject-benchmark success rates,
    logit(rate_{s,b}) = mu + beta_b + theta_name(s),
fitted by ridge-regularised alternating least squares. On a held-out benchmark
beta_b is unknown (0), so the prior is mu + theta_name; unseen names fall back
to the mean theta of their provider, then to 0.

Data are integer-coded (see data.py): cells (bench, sid, iid, n, k) and a
subject metadata table indexed by sid code with normalized_name and provider.
"""
import numpy as np
import pandas as pd


PROVIDER_FALLBACK = True  # unknown names: provider mean (False: neutral 0, as in package v7)


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


class NamePrior:
    def __init__(self, lam_name=2.0, lam_bench=0.5, iters=30):
        self.lam_name, self.lam_bench, self.iters = lam_name, lam_bench, iters

    def fit(self, cells, meta):
        self.name_of = meta["normalized_name"].fillna("").astype(str).to_numpy()
        self.prov_of = meta["provider"].fillna("").astype(str).to_numpy()
        pair = cells.groupby(["bench", "sid"], observed=True)[["k", "n"]].sum().reset_index()
        y = logit((pair.k / pair.n).to_numpy())
        names = self.name_of[pair.sid.to_numpy()]
        bcode, bvocab = pd.factorize(pair.bench)
        ncode, nvocab = pd.factorize(names)
        self.mu = float(pd.Series(y).groupby(bcode).mean().mean())
        beta, theta = np.zeros(len(bvocab)), np.zeros(len(nvocab))
        nb = np.bincount(bcode, minlength=len(bvocab))
        nn = np.bincount(ncode, minlength=len(nvocab))
        for _ in range(self.iters):
            beta = np.bincount(bcode, y - self.mu - theta[ncode], len(bvocab)) / (nb + self.lam_bench)
            theta = np.bincount(ncode, y - self.mu - beta[bcode], len(nvocab)) / (nn + self.lam_name)
        self.theta = dict(zip(nvocab, theta))
        self.theta.pop("", None)
        self._fit_lowrank(y, bcode, ncode, beta, theta, nvocab, len(bvocab))
        provs = pd.Series(self.prov_of[pair.sid.to_numpy()]).groupby(names).first()
        t = pd.DataFrame({"theta": theta}, index=nvocab).join(provs.rename("prov"))
        self.prov_theta = t[t.prov != ""].groupby("prov")["theta"].mean().to_dict()
        return self

    def _fit_lowrank(self, y, bcode, ncode, beta, theta, nvocab, nb, K=3, lam=3.0, iters=25):
        """Rank-K skill vectors per name from residual pair logits (ALS with ridge)."""
        r = y - self.mu - beta[bcode] - theta[ncode]
        rng = np.random.default_rng(0)
        U = rng.normal(0, 0.1, (len(nvocab), K))
        V = rng.normal(0, 0.1, (nb, K))
        for _ in range(iters):
            for X, Y, own, other, n_own in ((U, V, ncode, bcode, len(nvocab)), (V, U, bcode, ncode, nb)):
                A = np.zeros((n_own, K, K)) + lam * np.eye(K)
                np.add.at(A, own, Y[other][:, :, None] * Y[other][:, None, :])
                b = np.zeros((n_own, K))
                np.add.at(b, own, Y[other] * r[:, None])
                X[:] = np.linalg.solve(A, b[..., None])[..., 0]
        sd = U.std(0) + 1e-9
        self.skills = {n: U[j] / sd for j, n in enumerate(nvocab) if n}
        self.K = K

    def subject_feats(self, sid):
        """[theta, skills...] per sid code; zeros for names not seen in training."""
        sid = np.asarray(sid).astype(np.int64)
        uniq, inv = np.unique(sid, return_inverse=True)
        th = self.subject_theta(uniq)
        sk = np.array([self.skills.get(self.name_of[s], np.zeros(self.K)) for s in uniq]).reshape(len(uniq), self.K)
        return np.column_stack([th, sk])[inv]

    def subject_theta(self, sid):
        """Ability offset for sid codes (name, else provider mean, else 0)."""
        sid = np.asarray(sid).astype(np.int64)
        uniq, inv = np.unique(sid, return_inverse=True)
        fb = (lambda s: self.prov_theta.get(self.prov_of[s], 0.0)) if PROVIDER_FALLBACK else (lambda s: 0.0)
        val = np.array([self.theta.get(self.name_of[s], fb(s)) for s in uniq])
        return val[inv]

    def logit(self, df):
        return self.mu + self.subject_theta(df.sid.to_numpy())
