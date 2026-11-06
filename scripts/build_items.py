"""Extract compact item text for embedding: data/proc/items_all.parquet.

Only fields visible at test time are kept (content -> item_content, item_features).
grading_criterion / verifier are reference answers and are never used.
Content is truncated to HEAD chars (plus a short tail) because the encoder sees
at most ~512 tokens; the full length is kept as a feature.
"""
import glob
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
HEAD, TAIL = 3000, 500
eligible = set(pd.read_parquet(ROOT / "data/proc/bench_codes.parquet")["dir"])
codes = pd.read_parquet(ROOT / "data/proc/iid_codes.parquet")
parts = []
for path in sorted(glob.glob(str(ROOT / "data/mdb/*/formatted_tables/items.parquet"))):
    d = Path(path).parents[1].name
    if d not in eligible:
        continue
    cols = [c for c in ["item_id", "content", "item_features"] if c in pq.read_schema(path).names]
    df = pd.read_parquet(path, columns=cols)
    c = df["content"].fillna("").astype(str)
    df["n_chars"] = c.str.len()
    df["text"] = c.where(c.str.len() <= HEAD + TAIL, c.str[:HEAD] + " ... " + c.str[-TAIL:])
    df["item_features"] = df.get("item_features", pd.Series("", index=df.index)).fillna("").astype(str)
    df["dir"] = d
    parts.append(df[["dir", "item_id", "text", "item_features", "n_chars"]])
items = pd.concat(parts, ignore_index=True)
items["item_id"] = items["item_id"].astype(str)
items = codes.merge(items, on=["dir", "item_id"], how="left")
print(f"items with responses but no text row: {items.text.isna().sum():,}")
items["text"] = items["text"].fillna(""); items["item_features"] = items["item_features"].fillna(""); items["n_chars"] = items["n_chars"].fillna(0).astype(int)
items.to_parquet(ROOT / "data/proc/items_all.parquet", compression="zstd")
print(f"items: {len(items):,} over {items.dir.nunique()} benchmarks; "
      f"text MB: {items.text.str.len().sum()/1e6:.0f}; missing benchmarks: {sorted(eligible - set(items.dir))}")
print("n_chars quantiles:", items.n_chars.quantile([.5, .9, .99]).to_dict())
