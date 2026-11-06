"""Sync paiec code to a private Kaggle dataset and push an evaluation kernel.

    python scripts/kaggle_push_eval.py VN1 E040 "e5" "--methods ... --hp '{...}'"
EMB names the embedding kernels on that account: e5 or qwen. The kernel reads
paiec-proc, paiec-code and the two embedding shards' outputs; GPU is on.
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USERS = {"VN1": "<kaggle-user-1>", "VN2": "<kaggle-user-2>"}
EMB = {"e5": "paiec-emb-multilingual-e5-smal", "qwen": "paiec-emb-qwen3-embedding-06b"}
CODE = ["sim.py", "online.py", "priors.py", "data.py", "knn_prior.py", "evaluate.py", "acq.py", "tune.py", "lobo_ridge.py", "stack.py", "tune_full.py"]


def kaggle(acct, *args, cwd=None):
    # Upload paths must be bare folder names relative to cwd (Windows resume-path bug in the CLI).
    env = dict(os.environ, KAGGLE_CONFIG_DIR=str(Path.home() / ".kaggle" / acct), PYTHONUTF8="1")
    r = subprocess.run(["kaggle", *args], env=env, capture_output=True, text=True, cwd=cwd)
    return (r.stdout + r.stderr).strip()


def sync_code(acct):
    user = USERS[acct]
    d = ROOT / "kaggle" / "code_ds" / acct
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for f in CODE:
        shutil.copy(ROOT / "paiec" / f, d / f)
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": "paiec-code", "id": f"{user}/paiec-code", "licenses": [{"name": "other"}]}))
    out = kaggle(acct, "datasets", "version", "-p", acct, "-m", time.strftime("%H%M%S"), cwd=d.parent)
    if "being created" not in out:
        out = kaggle(acct, "datasets", "create", "-p", acct, cwd=d.parent)
    print("code:", [l for l in out.splitlines() if "%|" not in l][-1:])


def main():
    acct, tag, emb, args = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    gpu = "--cpu" not in sys.argv[5:]
    module = next((a.split("=", 1)[1] for a in sys.argv[5:] if a.startswith("--module=")), "evaluate")
    user = USERS[acct]
    sync_code(acct)
    slug = f"paiec-{'eval' if module == 'evaluate' else module}-{tag.lower()}"
    d = ROOT / "kaggle" / "build" / slug
    d.mkdir(parents=True, exist_ok=True)
    emb_glob = f"/kaggle/input/**/emb_*.npy"
    full_args = f'--emb "{emb_glob}" {args} --tag {tag}' if emb else f"{args} --tag {tag}"
    (d / "run.py").write_text((ROOT / "kaggle" / "eval_template.py").read_text().replace("__ARGS__", full_args).replace("__MODULE__", module))
    sources = [f"{user}/{EMB[emb]}-s0", f"{user}/{EMB[emb]}-s1"] if emb else []
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
        "enable_internet": False, "dataset_sources": [f"{user}/paiec-proc", f"{user}/paiec-code"],
        "kernel_sources": sources, "competition_sources": [], "model_sources": []}, indent=1))
    print("kernel:", kaggle(acct, "kernels", "push", "-p", str(d)).splitlines()[-1])


if __name__ == "__main__":
    main()
