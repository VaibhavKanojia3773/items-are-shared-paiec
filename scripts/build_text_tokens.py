"""Cache derived text tokens (online.text_tokens) for the embedded subset: data/proc/text_tokens.parquet."""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paiec"))
from online import text_tokens
sub = pd.read_parquet(ROOT / "data/proc/subset10k.parquet")["iid_code"].tolist()
it = pd.read_parquet(ROOT / "data/proc/items_all.parquet", columns=["iid_code", "text"], filters=[("iid_code", "in", sub)])
it["tokens"] = it.text.map(lambda t: ";".join(text_tokens(t)))
it[["iid_code", "tokens"]].to_parquet(ROOT / "data/proc/text_tokens.parquet")
print(len(it), it.tokens.head(3).tolist())
