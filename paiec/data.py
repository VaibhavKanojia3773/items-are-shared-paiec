"""Load the integer-coded tables (local data/proc or a Kaggle dataset dir)."""
import glob, os
from pathlib import Path
import numpy as np
import pandas as pd


def find_dir(name="cells.parquet"):
    for base in [Path(__file__).resolve().parents[1] / "data" / "proc", Path("/kaggle/input")]:
        hits = glob.glob(str(base / "**" / name), recursive=True)
        if hits:
            return Path(hits[0]).parent
    raise FileNotFoundError(name)


MERGE_COLS = ["normalized_name", "harness", "harness_version", "reasoning_effort"]


def merge_subjects(cells, meta):
    """Re-key subjects by the competition definition (model, harness, harness version,
    reasoning effort) within each benchmark; repeated runs become extra responses."""
    m = meta.copy()
    for c in MERGE_COLS:
        m[c] = m[c].fillna("").astype(str)
    m["merged"] = pd.factorize(m["dir"] + "|" + m[MERGE_COLS].agg("|".join, axis=1))[0]
    msid = m["merged"].to_numpy()[cells.sid.to_numpy()]
    agg = (cells.assign(sid=msid).groupby(["sid", "iid"], sort=False)
           .agg(bench=("bench", "first"), n=("n", "sum"), k=("k", "sum")).reset_index())
    new_meta = m.drop_duplicates("merged").sort_values("merged").reset_index(drop=True)
    new_meta["sid_code"] = new_meta["merged"]
    return agg[["bench", "sid", "iid", "n", "k"]], new_meta


def load_cells(subset=None, merged=False):
    """cells: bench, sid, iid (int codes), n, k; plus subject metadata keyed by sid code.

    subset: optional array of iid codes to keep (the embedded-item subset).
    """
    d = find_dir()
    cells = pd.read_parquet(d / "cells.parquet").rename(
        columns={"bench_code": "bench", "sid_code": "sid", "iid_code": "iid"})
    if subset is not None:
        cells = cells[cells.iid.isin(subset)]
    sid = pd.read_parquet(d / "sid_codes.parquet")
    subj = pd.read_parquet(d / "subjects_all.parquet").drop_duplicates(["dir", "subject_id"])
    meta = sid.merge(subj, on=["dir", "subject_id"], how="left").sort_values("sid_code")
    meta = meta.reset_index(drop=True)
    cells = cells.reset_index(drop=True)
    if merged:
        cells, meta = merge_subjects(cells, meta)
    return cells, meta


def load_embeddings(pattern):
    """Concatenate emb_*_s*.npy / codes_*_s*.npy shards -> (codes, float32 matrix)."""
    embs, codes = [], []
    for f in sorted(glob.glob(pattern, recursive=True)):
        if "/emb_" not in f.replace("\\", "/"):
            continue
        embs.append(np.load(f).astype(np.float32))
        codes.append(np.load(f.replace("emb_", "codes_")))
    codes = np.concatenate(codes)
    order = np.argsort(codes)
    return codes[order], np.concatenate(embs)[order]
