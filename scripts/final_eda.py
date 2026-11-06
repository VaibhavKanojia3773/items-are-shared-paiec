"""Final EDA on the raw measurement-db tables (before the raw download is deleted).

    python scripts/final_eda.py   -> logbooks/runs/final_eda_per_bench.csv (+ printed summary)

Per eligible benchmark (binary, item granularity), streaming the responses in batches:
- response level: repeated trials, test_condition and interactors coverage;
- competition targets: subject = full subject dict (model, provider, dates, harness, version,
  reasoning effort, features_extra) or the 4-field definition; item = (item_id, interactors);
  repeats per target and the irreducible Brier floor sum k(n-k)/n / sum n;
- pair structure: items per pair, eligible pairs (>= 80 items), pair base rates, matrix density,
  and item overlap (share of a pair's items answered by >= 5 other eligible subjects);
- items: duplicated content_hash, image/asset manifests, item_features coverage.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
MDB, PROC = ROOT / "data" / "mdb", ROOT / "data" / "proc"
OUT = ROOT / "logbooks" / "runs"
FULL = ["normalized_name", "provider", "release_date", "access_date", "harness", "harness_version",
        "reasoning_effort", "subject_features_extra"]
FOUR = ["normalized_name", "harness", "harness_version", "reasoning_effort"]


def nonempty(s):
    return s.notna() & (s.astype(str).str.strip() != "") & (s.astype(str) != "[]") & (s.astype(str) != "{}")


def bench_stats(d):
    t = MDB / d / "formatted_tables"
    subj = pq.read_table(t / "subjects.parquet").to_pandas().drop_duplicates("subject_id")
    for c in FULL:
        subj[c] = subj[c].fillna("").astype(str) if c in subj else ""
    full_key = dict(zip(subj.subject_id, pd.factorize(subj[FULL].agg("|".join, axis=1))[0]))
    four_key = dict(zip(subj.subject_id, pd.factorize(subj[FOUR].agg("|".join, axis=1))[0]))
    r = {"dir": d, "subj_rows": len(subj), "subj_full": len(set(full_key.values())),
         "subj_four": len(set(four_key.values()))}
    rows = trial_gt1 = tc = inter = 0
    inter_vals, parts_full, parts_four = set(), [], []
    pf = pq.ParquetFile(t / "responses.parquet")
    cols = [c for c in ["subject_id", "item_id", "trial", "test_condition", "interactors", "response"]
            if c in pf.schema_arrow.names]
    for batch in pf.iter_batches(batch_size=2_000_000, columns=cols):
        b = batch.to_pandas()
        b["response"] = pd.to_numeric(b["response"], errors="coerce")
        b = b[b.response.isin([0, 1])]
        rows += len(b)
        if "trial" in b:
            trial_gt1 += int((pd.to_numeric(b.trial, errors="coerce").fillna(0) > 1).sum())
        if "test_condition" in b:
            tc += int(nonempty(b.test_condition).sum())
        ia = b["interactors"].fillna("").astype(str) if "interactors" in b else pd.Series("", index=b.index)
        inter += int(nonempty(ia).sum())
        if len(inter_vals) < 1000:
            inter_vals.update(ia.unique()[:1000].tolist())
        b = b.assign(ia=ia, k=b.response.astype(float))
        for keymap, parts in ((full_key, parts_full), (four_key, parts_four)):
            g = b.assign(s=b.subject_id.map(keymap)).groupby(["s", "item_id", "ia"], sort=False)["k"]
            parts.append(pd.DataFrame({"n": g.size(), "k": g.sum()}))
    if rows == 0:
        return None
    r.update(resp=rows, trial_gt1=trial_gt1 / rows, test_cond=tc / rows, interactors=inter / rows,
             n_interactor_vals=len(inter_vals - {""}))
    for name, parts in (("full", parts_full), ("four", parts_four)):
        c = pd.concat(parts).groupby(level=[0, 1, 2], sort=False).sum().reset_index()
        r[f"{name}_targets_in_rep_cells"] = float(c.n[c.n > 1].sum() / c.n.sum())
        r[f"{name}_brier_floor"] = float((c.k * (c.n - c.k) / c.n).sum() / c.n.sum())
        r[f"{name}_cells"] = len(c)
        if name != "full":
            continue
        c["item"] = c.item_id.astype(str) + "\x1f" + c.ia
        per_pair = c.groupby("s").agg(items=("item", "size"), n=("n", "sum"), k=("k", "sum"))
        el = per_pair[per_pair["items"] >= 80]
        r["pairs"], r["elig_pairs"] = len(per_pair), len(el)
        r["items_per_elig_pair_med"] = float(el["items"].median()) if len(el) else np.nan
        n_items = c["item"].nunique()
        r["n_items_resp"] = n_items
        if len(el):
            rate = el.k / el.n
            r["pair_rate_med"] = float(rate.median())
            r["pair_rate_extreme"] = float(((rate < 0.1) | (rate > 0.9)).mean())  # near-deterministic pairs
            r["pair_rate_brier"] = float((rate * (1 - rate)).mean())  # Brier of the pair-rate oracle (approx.)
            ce = c[c.s.isin(el.index)]
            r["density"] = float(len(ce) / (len(el) * ce["item"].nunique()))
            cover = ce.groupby("item").size()
            r["overlap5"] = float((ce["item"].map(cover) - 1 >= 5).mean())
    it = pq.read_table(t / "items.parquet", columns=["item_id", "content_hash", "asset_manifest",
                                                      "item_features"]).to_pandas()
    dup = it.content_hash.duplicated(keep=False) & it.content_hash.notna()
    r.update(items_table=len(it), dup_content=float(dup.mean()), assets=float(nonempty(it.asset_manifest).mean()),
             item_feats=float(nonempty(it.item_features).mean()))
    return r


def main():
    dirs = pd.read_csv(PROC / "per_benchmark_stats.csv")["dir"].tolist()
    meta = pd.read_parquet(PROC / "benchmarks_all.parquet").set_index("dir")
    out = []
    for i, d in enumerate(dirs):
        try:
            r = bench_stats(d)
        except Exception as exc:  # keep going; record the failure
            r = {"dir": d, "error": f"{type(exc).__name__}: {exc}"[:200]}
        if r:
            for c in ("modality", "domain", "multi_single_turn", "max_trial", "coverage"):
                if c in meta.columns:
                    r[c] = meta.at[d, c] if d in meta.index else None
            out.append(r)
        print(f"[{i+1}/{len(dirs)}] {d} {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in (r or {}).items() if k in ('resp', 'elig_pairs', 'full_targets_in_rep_cells', 'interactors', 'overlap5', 'error')})}", flush=True)
    df = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "final_eda_per_bench.csv", index=False)
    print(f"wrote {OUT / 'final_eda_per_bench.csv'} ({len(df)} benchmarks)")


if __name__ == "__main__":
    main()
