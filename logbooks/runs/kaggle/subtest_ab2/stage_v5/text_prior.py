"""Item text encoder and kNN easiness memory for the submission.

Preprocessing matches kaggle/embed_template.py exactly: optional "[features] "
prefix, content truncated to 3000 head + 500 tail chars, e5 "query: " prefix,
max 512 tokens, L2-normalised embeddings.
"""
import os

import numpy as np

HEAD, TAIL = 3000, 500


def item_text(item, e5_prefix):
    content = item.get("item_content") or ""
    if len(content) > HEAD + TAIL:
        content = content[:HEAD] + " ... " + content[-TAIL:]
    feats = item.get("item_features") or ""
    text = (f"[{feats[:200]}] " if feats else "") + content
    return ("query: " + text) if e5_prefix else text


class Encoder:
    def __init__(self, root, cfg):
        import torch
        from sentence_transformers import SentenceTransformer
        local = os.path.join(root, "artifacts", "encoder")
        name = local if os.path.isdir(local) else cfg["encoder"]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = {"torch_dtype": torch.float16} if dev == "cuda" else {}
        self.model = SentenceTransformer(name, device=dev, model_kwargs=kwargs)
        self.model.max_seq_length = 512
        self.e5 = "e5" in cfg["encoder"]

    def encode(self, items):
        texts = [item_text(it, self.e5) for it in items]
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                 show_progress_bar=False).astype(np.float32)


class Memory:
    def __init__(self, root, cfg):
        z = np.load(os.path.join(root, "artifacts", "knn_memory.npz"))
        self.mem = z["emb"].astype(np.float32)
        self.easy = z["easy"].astype(np.float32)
        self.mu_easy = float(z["mu_easy"])
        self.k, self.tau, self.kappa = int(cfg["knn_k"]), float(cfg["knn_tau"]), float(cfg["knn_kappa"])

    def easiness(self, q):
        sims = q @ self.mem.T
        k = min(self.k, sims.shape[1] - 1)
        j = np.argpartition(-sims, k, axis=1)[:, :k]
        s = np.take_along_axis(sims, j, 1)
        w = np.exp((s - 1.0) / self.tau)
        return ((w * self.easy[j]).sum(1) + self.kappa * self.mu_easy) / (w.sum(1) + self.kappa)
