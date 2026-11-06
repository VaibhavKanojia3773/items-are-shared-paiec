"""The deterministic embedded-item subset (<= 10k items per benchmark), identical to the Kaggle kernel."""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
CAP = 10000
items = pd.read_parquet(ROOT / "data/proc/items_all.parquet", columns=["iid_code", "dir", "n_chars"])
sub = items.groupby("dir", group_keys=False).apply(lambda g: g.sample(min(len(g), CAP), random_state=0))
sub.sort_values("iid_code")[["iid_code", "dir", "n_chars"]].to_parquet(ROOT / "data/proc/subset10k.parquet")
print(len(sub))
