"""Build the submission ZIP from all public data.

    python paiec/build_submission.py --hp '{...}' [--emb "data/emb/e5/**/emb_*.npy"] --out dist/paiec.zip

Without --emb the package uses the subject-only prior (no text encoder needed).
"""
import argparse, json, shutil, zipfile
from pathlib import Path
import numpy as np

from data import load_cells, load_embeddings
from priors import NamePrior
from online import DEFAULT_HP

ROOT = Path(__file__).resolve().parents[1]
SUB = Path(__file__).resolve().parent / "submission"
ENCODER = "intfloat/multilingual-e5-small"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hp", default="{}")
    ap.add_argument("--acq", default='{"gamma": 1.0}')
    ap.add_argument("--emb", default="", help="embeddings for the kNN prior memory (optional)")
    ap.add_argument("--text", action="store_true", help="ship the e5 encoder (models.txt) for propagation")
    ap.add_argument("--knn", default='{"k": 32, "tau": 0.05, "kappa": 0.5, "max_mem_per_bench": 2000}')
    ap.add_argument("--no_labeling", action="store_true", help="use the evaluator's random acquisition")
    ap.add_argument("--no_name_prior", action="store_true", help="constant prior mu (the online0 variants)")
    ap.add_argument("--no_provider_prior", action="store_true", help="no provider fallback for unknown names")
    ap.add_argument("--no_name_pool", action="store_true", help="subject offset keyed by full subject dict")
    ap.add_argument("--text_tokens", action="store_true", help="add derived text-format tokens")
    ap.add_argument("--template_tokens", action="store_true", help="add first/last-word template tokens")
    ap.add_argument("--out", default=str(ROOT / "dist" / "paiec.zip"))
    args = ap.parse_args()
    stage = ROOT / "dist" / "stage"
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "artifacts").mkdir(parents=True)
    knn_kw = json.loads(args.knn)
    emb = load_embeddings(args.emb) if args.emb else None
    cells, meta = load_cells(emb[0] if emb else None)
    pri = NamePrior().fit(cells, meta)
    (stage / "artifacts" / "subject_prior.json").write_text(json.dumps(
        {"mu": pri.mu, "theta": pri.theta, "prov_theta": pri.prov_theta}))
    cfg = {"hp": dict(DEFAULT_HP, **json.loads(args.hp)), "acq": json.loads(args.acq),
           "use_text": bool(args.text or emb is not None), "use_knn": emb is not None,
           "propagate": bool(args.text or emb is not None), "encoder": ENCODER,
           "knn_k": knn_kw["k"], "knn_tau": knn_kw["tau"], "knn_kappa": knn_kw["kappa"],
           "clip": [0.005, 0.995], "use_name_prior": not args.no_name_prior,
           "pool_by_name": not args.no_name_pool, "provider_prior": not args.no_provider_prior, "text_tokens": args.text_tokens or args.template_tokens,
           "template_tokens": args.template_tokens}
    if emb is not None:
        from knn_prior import KnnPrior
        knn = KnnPrior(**knn_kw).fit(cells, meta, *emb)
        # The memory easiness is relative to names.mu; the submission adds _MU itself.
        np.savez_compressed(stage / "artifacts" / "knn_memory.npz", emb=knn.mem.astype(np.float16),
                            easy=knn.easy, mu_easy=knn.mu_easy)
    if cfg["use_text"]:
        (stage / "models.txt").write_text(ENCODER + "\n")
    (stage / "artifacts" / "config.json").write_text(json.dumps(cfg, indent=1))
    for f in ["model.py", "text_prior.py"] + ([] if args.no_labeling else ["labeling.py"]):
        shutil.copy(SUB / f, stage / f)
    shutil.copy(Path(__file__).resolve().parent / "online.py", stage / "online.py")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(stage).as_posix())
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB); config: {json.dumps(cfg)[:300]}")


if __name__ == "__main__":
    main()
