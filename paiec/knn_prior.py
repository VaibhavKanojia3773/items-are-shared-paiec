"""Text-similarity prior: item easiness from nearest training items.

Offline (training benchmarks only):
    theta(s)   subject ability by normalized_name (NamePrior)
    easy_j     for each memory item j: logit of its smoothed mean success over
               subjects, minus the mean ability of those subjects
Query (new item i, subject s):
    eta(s, i) = theta(s) + (sum_j w_ij easy_j + kappa * mu) / (sum_j w_ij + kappa)
with w_ij = exp((cos_ij - 1) / tau) over the top-K neighbours; kappa shrinks
toward the global mean easiness when no neighbour is close.
"""
import numpy as np
import pandas as pd

from priors import NamePrior, logit


class KnnPrior:
    def __init__(self, k=32, tau=0.05, kappa=0.5, smooth=2.0, max_mem_per_bench=4000, seed=0):
        self.k, self.tau, self.kappa, self.smooth = k, tau, kappa, smooth
        self.max_mem, self.seed = max_mem_per_bench, seed

    def fit(self, cells, meta, emb_codes, emb):
        """cells: training rows; emb_codes: sorted iid codes indexing the rows of emb."""
        self.names = NamePrior().fit(cells, meta)
        g = pd.DataFrame({"iid": cells.iid.to_numpy(), "bench": cells.bench.to_numpy(),
                          "rate": (cells.k / cells.n).to_numpy(),
                          "th": self.names.subject_theta(cells.sid.to_numpy())})
        g = g[np.isin(g.iid.to_numpy(), emb_codes)]
        item = g.groupby("iid").agg(bench=("bench", "first"), m=("rate", "size"),
                                    q=("rate", "mean"), th=("th", "mean"))
        keep = item.groupby("bench", group_keys=False).apply(
            lambda x: x.sample(min(len(x), self.max_mem), random_state=self.seed))
        q = (keep.m * keep.q + self.smooth * 0.5) / (keep.m + self.smooth)
        self.easy = (logit(q.to_numpy()) - self.names.mu - keep.th.to_numpy()).astype(np.float32)
        self.mu_easy = float(np.mean(self.easy))
        self.mem = emb[np.searchsorted(emb_codes, keep.index.to_numpy())].astype(np.float32)
        self.mem_bench = keep.bench.to_numpy()
        self.alpha, self.lam = 0.0, 1.0
        self._calibrate()
        return self

    def _calibrate(self, n=40000):
        """Leave-one-benchmark-out calibration of the raw kNN average.

        Held-out benchmarks are unseen at test time, so for a sample of memory
        items the kNN average is computed from other benchmarks only, and true
        easiness is regressed on it: easy ~ alpha + lam * knn_loo.
        """
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(self.mem), min(n, len(self.mem)), replace=False)
        raw = self._raw(self.mem[idx], exclude=self.mem_bench[idx])
        y = self.easy[idx]
        x = raw - raw.mean()
        self.lam = float((x * (y - y.mean())).sum() / max((x * x).sum(), 1e-9))
        self.alpha = float(y.mean() - self.lam * raw.mean())
        pred = self.alpha + self.lam * raw
        self.r2 = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        print(f"[knn] LOBO calibration: lam={self.lam:.3f} alpha={self.alpha:.3f} R2={self.r2:.3f}", flush=True)

    def item_easiness(self, q_emb, batch=4096):
        """Calibrated kNN easiness for query embeddings (n, d)."""
        return self.alpha + self.lam * self._raw(q_emb, batch=batch)

    def _raw(self, q_emb, exclude=None, batch=4096):
        """Shrunk kNN average of memory easiness; `exclude` masks same-benchmark memory."""
        out = np.empty(len(q_emb), np.float32)
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            mem = torch.from_numpy(self.mem).to(dev)
            easy = torch.from_numpy(self.easy).to(dev)
            mb = torch.from_numpy(self.mem_bench.astype(np.int64)).to(dev) if exclude is not None else None
            for a in range(0, len(q_emb), batch):
                qb = torch.from_numpy(np.ascontiguousarray(q_emb[a:a + batch], np.float32)).to(dev)
                sims = qb @ mem.T
                if exclude is not None:
                    eb = torch.from_numpy(exclude[a:a + batch].astype(np.int64)).to(dev)
                    sims = sims.masked_fill(eb[:, None] == mb[None, :], -2.0)
                s, j = torch.topk(sims, self.k, dim=1)
                w = torch.exp((s - 1) / self.tau)
                v = (w * easy[j]).sum(1) + self.kappa * self.mu_easy
                out[a:a + batch] = (v / (w.sum(1) + self.kappa)).cpu().numpy()
        except ImportError:
            for a in range(0, len(q_emb), batch):
                sims = q_emb[a:a + batch] @ self.mem.T
                if exclude is not None:
                    sims[exclude[a:a + batch, None] == self.mem_bench[None, :]] = -2.0
                j = np.argpartition(-sims, self.k, axis=1)[:, :self.k]
                s = np.take_along_axis(sims, j, 1)
                w = np.exp((s - 1) / self.tau)
                out[a:a + batch] = ((w * self.easy[j]).sum(1) + self.kappa * self.mu_easy) / (w.sum(1) + self.kappa)
        return out

    def logit(self, sid, item_easy):
        return self.names.mu + self.names.subject_theta(sid) + item_easy
