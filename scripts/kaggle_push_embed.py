"""Generate and push PAIEC embedding kernels: python scripts/kaggle_push_embed.py MODEL NSHARDS PROFILE
All shards of one model run on one Kaggle account; PROFILE is a folder under ~/.kaggle holding its kaggle.json."""
import json, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
model, nshards = sys.argv[1], int(sys.argv[2])
acct_sel = sys.argv[3]  # credentials profile: all shards of this model run on one account
only = range(nshards)
tag = model.split("/")[-1].lower().replace(".", "")[:20]
tpl = (ROOT / "kaggle" / "embed_template.py").read_text()
for s in only:
    acct, user = acct_sel, json.loads((Path.home() / ".kaggle" / acct_sel / "kaggle.json").read_text())["username"]
    slug = f"paiec-emb-{tag}-s{s}"
    d = ROOT / "kaggle" / "build" / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(tpl.replace("__MODEL__", model).replace("__SHARD__", str(s)).replace("__NSHARDS__", str(nshards)))
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "dataset_sources": [f"{user}/paiec-proc"],
        "kernel_sources": [], "competition_sources": [], "model_sources": []}, indent=1))
    env = dict(os.environ, KAGGLE_CONFIG_DIR=str(Path.home() / ".kaggle" / acct))
    r = subprocess.run(["kaggle", "kernels", "push", "-p", str(d)], env=env, capture_output=True, text=True)
    print(acct, slug, (r.stdout + r.stderr).strip().splitlines()[-1])
