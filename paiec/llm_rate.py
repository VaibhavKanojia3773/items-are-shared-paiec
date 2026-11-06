"""Offline study: can an LLM rank item difficulty within a benchmark? (NVIDIA API, open models)

    python paiec/llm_rate.py --model google/gemma-3-12b-it --benches 10 --per 50
Ratings are cached in logbooks/runs/llm_ratings/<model>.jsonl; prints the within-benchmark Spearman
correlation between the rating and true item easiness (smoothed logit over all public subjects).
The key is read from .env (NVIDIA_API_KEY) and never printed.
"""
import argparse, json, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logbooks" / "runs" / "llm_ratings"
PROMPT = ("You estimate how hard a benchmark item is for current AI language models.\n"
          "Benchmark: {bench}\n\nItem:\n\"\"\"\n{text}\n\"\"\"\n\n"
          "On a scale from 1 (almost every AI system answers correctly) to 10 (almost none does), "
          "how difficult is this item? Reply with a single integer only.")


def key():
    for line in open(ROOT / ".env", encoding="utf-8"):
        if line.startswith("NVIDIA_API_KEY"):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise KeyError("NVIDIA_API_KEY")


def ask(model, prompt, k, tries=5):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "max_tokens": 400}).encode()
    for t in range(tries):
        try:
            req = urllib.request.Request("https://integrate.api.nvidia.com/v1/chat/completions", data=body,
                                         headers={"Authorization": "Bearer " + k, "Content-Type": "application/json"})
            txt = json.load(urllib.request.urlopen(req, timeout=60))["choices"][0]["message"]["content"] or ""
            nums = [int(x) for x in re.findall(r"\d+", txt) if 1 <= int(x) <= 10]
            return float(nums[-1]) if nums else None
        except Exception as exc:  # rate limits / transient errors: back off
            time.sleep(2 * (t + 1))
            err = type(exc).__name__
    return None


def sample_items(n_bench, per, seed=0):
    d = ROOT / "data" / "proc"
    cells = pd.read_parquet(d / "cells.parquet").rename(columns={"bench_code": "bench", "iid_code": "iid", "sid_code": "sid"})
    sub = set(pd.read_parquet(d / "subset10k.parquet")["iid_code"])
    cells = cells[cells.iid.isin(sub)]
    st = cells.groupby("bench").agg(subj=("sid", "nunique"), items=("iid", "nunique"))
    ok = st[(st["subj"] >= 15) & (st["items"] >= 200)].index.to_numpy()
    rng = np.random.default_rng(seed)
    benches = rng.choice(ok, min(n_bench, len(ok)), replace=False)
    g = cells[cells.bench.isin(benches)].groupby(["bench", "iid"])[["k", "n"]].sum().reset_index()
    g = g[g.n >= 10]
    g["easy"] = np.log((g.k + 0.5) / (g.n - g.k + 0.5))
    g = g.groupby("bench", group_keys=False).apply(lambda x: x.sample(min(per, len(x)), random_state=seed))
    names = pd.read_parquet(d / "bench_codes.parquet")
    bname = dict(zip(names.bench_code, names["dir"]))
    it = pd.read_parquet(d / "items_all.parquet", columns=["iid_code", "text"], filters=[("iid_code", "in", g.iid.tolist())])
    g = g.merge(it, left_on="iid", right_on="iid_code")
    g["bname"] = g.bench.map(lambda b: str(bname.get(b, b)))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benches", type=int, default=10)
    ap.add_argument("--per", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    g = sample_items(args.benches, args.per)
    OUT.mkdir(parents=True, exist_ok=True)
    cache_f = OUT / (args.model.replace("/", "__") + ".jsonl")
    cache = {}
    if cache_f.exists():
        for line in open(cache_f):
            r = json.loads(line)
            cache[r["iid"]] = r["rating"]
    todo = [r for r in g.itertuples() if int(r.iid) not in cache]
    k = key()

    def job(r):
        return int(r.iid), ask(args.model, PROMPT.format(bench=r.bname, text=str(r.text)[:2000]), k)
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex, open(cache_f, "a") as fh:
        for fut in as_completed([ex.submit(job, r) for r in todo]):
            iid, rating = fut.result()
            if rating is None:  # failed calls are retried on the next run
                continue
            cache[iid] = rating
            fh.write(json.dumps({"iid": iid, "rating": rating}) + "\n")
            fh.flush()
    g["rating"] = g.iid.map(lambda i: cache.get(int(i)))
    ok = g.dropna(subset=["rating"])
    rho = pd.Series({b: (x["rating"].rank().corr(-x["easy"].rank()) if x["rating"].nunique() > 1 else np.nan)
                     for b, x in ok.groupby("bench")})
    print(f"model={args.model} items={len(ok)}/{len(g)} ({time.time()-t0:.0f}s) "
          f"within-bench Spearman(rating, difficulty): mean {rho.mean():.3f} median {rho.median():.3f} "
          f"per bench {np.round(rho.values, 2).tolist()}")


if __name__ == "__main__":
    main()
