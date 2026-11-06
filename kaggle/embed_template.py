"""PAIEC item-embedding kernel (Kaggle GPU). Generated per shard by scripts/kaggle_push_embed.py.

Reads items_all.parquet from the private paiec-items dataset, embeds `text`
(prefixed with item_features when present) and writes float16 embeddings for
this shard to /kaggle/working.
"""
import os, sys, time, glob, subprocess
import numpy as np
import pandas as pd

MODEL = "__MODEL__"
SHARD, NSHARDS = __SHARD__, __NSHARDS__
MAXLEN = 512



def main():
    paths = glob.glob("/kaggle/input/**/items_all.parquet", recursive=True)
    CAP = 10000  # items per benchmark; same deterministic subset as scripts/subset.py
    items = pd.read_parquet(paths[0], columns=["iid_code", "dir", "text", "item_features"])
    items = items.groupby("dir", group_keys=False).apply(lambda g: g.sample(min(len(g), CAP), random_state=0))
    items = items.sort_values("iid_code").iloc[SHARD::NSHARDS].reset_index(drop=True)
    text = (items["item_features"].where(items["item_features"].str.len() > 0, "")
            .map(lambda f: f"[{f[:200]}] " if f else "") + items["text"]).tolist()
    if "e5" in MODEL:
        text = ["query: " + t for t in text]
    print(f"shard {SHARD}/{NSHARDS}: {len(text):,} items, model {MODEL}", flush=True)

    import torch
    from sentence_transformers import SentenceTransformer
    print("torch", torch.__version__, "gpus", torch.cuda.device_count(), flush=True)
    model = SentenceTransformer(MODEL, device="cuda", model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = MAXLEN
    # Sort by length for efficient batching, then restore order.
    order = np.argsort([len(t) for t in text])
    t0 = time.time()
    pool = model.start_multi_process_pool() if torch.cuda.device_count() > 1 else None
    sorted_text = [text[i] for i in order]
    if pool is not None:
        emb = model.encode(sorted_text, pool=pool, batch_size=64, normalize_embeddings=True, chunk_size=4096)
        model.stop_multi_process_pool(pool)
    else:
        emb = model.encode(sorted_text, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    out = np.empty_like(emb)
    out[order] = emb
    print(f"embedded in {time.time()-t0:.0f}s, dim {out.shape[1]}", flush=True)
    tag = MODEL.split("/")[-1]
    np.save(f"/kaggle/working/emb_{tag}_s{SHARD}.npy", out.astype(np.float16))
    np.save(f"/kaggle/working/codes_{tag}_s{SHARD}.npy", items["iid_code"].to_numpy())
    print("done", flush=True)


if __name__ == "__main__":  # required: the multi-GPU pool spawns processes
    main()
