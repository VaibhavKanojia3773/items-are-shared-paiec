"""Consolidate per-benchmark core tables into data/proc/*.parquet and print dataset stats.

Keeps only competition-eligible benchmarks (binary responses, item granularity),
as tools/prepare_data.py in the official baseline does.
"""
import glob, json, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MDB, OUT = ROOT / "data" / "mdb", ROOT / "data" / "proc"
OUT.mkdir(parents=True, exist_ok=True)

bench_rows, subj_rows, resp_parts, skipped, nonbin_counts = [], [], [], {}, {}
for bdir in sorted(glob.glob(str(MDB / "*" / "formatted_tables"))):
    name = Path(bdir).parent.name
    try:
        b = pd.read_parquet(f"{bdir}/benchmarks.parquet").iloc[0].to_dict()
    except FileNotFoundError:
        skipped[name] = "no benchmarks.parquet"; continue
    b["dir"] = name
    bench_rows.append(b)
    if not (b["response_type"] == "binary" and b["granularity"] == "item"):
        skipped[name] = f"{b['response_type']}/{b['granularity']}"; continue
    if not os.path.exists(f"{bdir}/responses.parquet"):
        skipped[name] = "no responses"; continue
    s = pd.read_parquet(f"{bdir}/subjects.parquet"); s["dir"] = name
    subj_rows.append(s)
    # Aggregate repeated responses (trials, test conditions) per subject-item:
    # Brier over repeats depends only on (n, k), so nothing is lost for scoring.
    # test_condition is not visible in the competition input, so it is pooled.
    r = pd.read_parquet(f"{bdir}/responses.parquet", columns=["subject_id", "item_id", "response"])
    nonbin = int((~r.response.isin([0, 1])).sum())
    r = r[r.response.isin([0, 1])]
    a = r.groupby(["subject_id", "item_id"], sort=False, observed=True)["response"].agg(["size", "sum"])
    a = a.reset_index().rename(columns={"size": "n", "sum": "k"})
    a["n"] = a["n"].astype("int32"); a["k"] = a["k"].astype("float32")
    a["dir"] = name
    nonbin_counts[name] = (nonbin, len(r))
    resp_parts.append(a)
    del r

bench = pd.DataFrame(bench_rows)
subj = pd.concat(subj_rows, ignore_index=True)
resp = pd.concat(resp_parts, ignore_index=True)
for c in ["subject_id", "item_id", "dir"]:
    resp[c] = resp[c].astype("category")
bench.drop(columns=[c for c in bench.columns if bench[c].map(type).eq(np.ndarray).any()], errors="ignore")     .to_parquet(OUT / "benchmarks_all.parquet")
bench[["dir", "modality", "domain"]].assign(modality=bench.modality.map(lambda x: list(x) if x is not None else []),
                                            domain=bench.domain.map(lambda x: list(x) if x is not None else []))     .to_parquet(OUT / "benchmark_tags.parquet")
subj.to_parquet(OUT / "subjects_all.parquet")
resp.to_parquet(OUT / "responses_agg.parquet")

# ---- stats -------------------------------------------------------------------
nb_tot = sum(v[0] for v in nonbin_counts.values()); raw_tot = sum(v[1] for v in nonbin_counts.values())
print(f"benchmark dirs: {len(bench)}; eligible with responses: {resp['dir'].nunique()}; skipped: {len(skipped)}")
print("skip reasons:", pd.Series(skipped).value_counts().to_dict())
print(f"binary responses kept: {raw_tot:,}; non-binary dropped: {nb_tot:,}; aggregated subject-item cells: {len(resp):,}")
print(f"cells with n>1: {(resp.n > 1).mean():.1%}; max n: {resp.n.max()}")
print(f"global mean response (per response): {resp.k.sum()/resp.n.sum():.3f}; per cell: {(resp.k/resp.n).mean():.3f}")
print(f"subject rows: {len(subj):,}; unique subject_id: {subj.subject_id.nunique():,}; "
      f"unique normalized_name: {subj.normalized_name.nunique():,}")
sb = subj.groupby("subject_id")["dir"].nunique()
print(f"subject_id appearing in >1 benchmark: {(sb > 1).sum():,} ({(sb > 1).mean():.1%})")
nb = subj.groupby("normalized_name")["dir"].nunique()
print(f"normalized_name in >1 benchmark: {(nb > 1).sum():,}; in >=5: {(nb >= 5).sum():,}; in >=20: {(nb >= 20).sum():,}")
print("most widespread names:", nb.sort_values().tail(10).to_dict())
for c in ["provider", "harness", "reasoning_effort", "harness_version", "release_date", "access_date", "subject_features_extra"]:
    print(f"  subjects.{c}: non-null {subj[c].notna().mean():.1%}, unique {subj[c].nunique()}")
pi = resp.groupby(["dir", "subject_id"], observed=True)["item_id"].size()
elig = pi[pi >= 80]
print(f"subject-benchmark pairs: {len(pi):,}; with >=80 items (eval-eligible): {len(elig):,} "
      f"over {elig.index.get_level_values(0).nunique()} benchmarks")
resp["p"] = resp.k / resp.n
per_b = resp.groupby("dir", observed=True).agg(cells=("p", "size"), mean=("p", "mean"),
                                                 subj=("subject_id", "nunique"), items=("item_id", "nunique"))
per_b["elig_pairs"] = elig.groupby(level=0).size().reindex(per_b.index).fillna(0).astype(int)
print("per-benchmark mean quantiles:", per_b["mean"].quantile([0, .1, .25, .5, .75, .9, 1]).round(3).to_dict())
print("per-benchmark subjects quantiles:", per_b["subj"].quantile([0, .25, .5, .75, .9, 1]).to_dict())
print("per-benchmark items quantiles:", per_b["items"].quantile([0, .25, .5, .75, .9, 1]).to_dict())
per_b.to_csv(OUT / "per_benchmark_stats.csv")
# Variance decomposition over eval-eligible pairs (cell-level success rate):
e = resp.set_index(["dir", "subject_id"]).loc[elig.index].reset_index()
pair_mean = e.groupby(["dir", "subject_id"], observed=True)["p"].transform("mean")
bench_mean = e.groupby("dir", observed=True)["p"].transform("mean")
tot = e.p.var()
print(f"[eligible cells] total var {tot:.4f}; between-benchmark {bench_mean.var():.4f} ({bench_mean.var()/tot:.0%}); "
      f"subject-within-benchmark {(pair_mean-bench_mean).var():.4f} ({(pair_mean-bench_mean).var()/tot:.0%}); "
      f"within-pair {(e.p-pair_mean).var():.4f} ({(e.p-pair_mean).var()/tot:.0%})")
