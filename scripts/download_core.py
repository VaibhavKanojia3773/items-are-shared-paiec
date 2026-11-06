"""Download core measurement-db tables per file, using the API index in data/hf_ft_index.json.

Usage: python scripts/download_core.py responses,subjects,benchmarks[,items] [--skip cseo_bench]
Skips traces.parquet, assets.parquet and raw/ (see PAIEC_dataset_files.md).
"""
import concurrent.futures as cf, json, os, sys, time
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
for line in (ROOT / ".env").read_text().splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

tables = set(sys.argv[1].split(","))
skip = set(sys.argv[sys.argv.index("--skip") + 1].split(",")) if "--skip" in sys.argv else set()
index = json.loads((ROOT / "data" / "hf_ft_index.json").read_text())
todo = [(p, s) for b, files in index.items() if b not in skip for p, s in files
        if p.rsplit("/", 1)[-1].removesuffix(".parquet") in tables]
out = ROOT / "data" / "mdb"
print(f"{len(todo)} files, {sum(s for _, s in todo)/1e9:.2f} GB", flush=True)

def get(ps):
    p, s = ps
    dst = out / p
    if dst.exists() and dst.stat().st_size == s:
        return 0
    for attempt in range(4):
        try:
            hf_hub_download("aims-foundations/measurement-db", p, repo_type="dataset",
                            local_dir=str(out), token=os.environ["HF_TOKEN"])
            return s
        except Exception as e:
            if attempt == 3:
                print("FAIL", p, type(e).__name__, flush=True)
            time.sleep(2 ** attempt)
    return 0

t, done = time.time(), 0
with cf.ThreadPoolExecutor(8) as ex:
    for i, n in enumerate(ex.map(get, todo), 1):
        done += n
        if i % 100 == 0 or i == len(todo):
            print(f"{i}/{len(todo)} files, {done/1e9:.2f} GB, {time.time()-t:.0f}s", flush=True)
print("done")
