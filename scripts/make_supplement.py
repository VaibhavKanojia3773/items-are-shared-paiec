"""Build the OpenReview supplementary archive (code + package + logged outputs + instructions).

    python scripts/make_supplement.py   -> dist/paiec_supplementary.zip
Excludes data, credentials, caches and local backups; checks that no secret files are included.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "paiec_supplementary.zip"
INCLUDE = ["README.md", "MANIFEST.md", "paper/title.tex", "paiec", "scripts", "kaggle/embed_template.py", "kaggle/eval_template.py",
           "paper/main.tex", "paper/main.pdf", "paper/refs.bib", "paper/neurips_2026.sty",
           "paper/results.tex", "paper/figs", "paper/scripts", "logbooks", "dist/paiec_v6.zip",
           "PAIEC_task_analysis.md", "PAIEC_dataset_files.md"]
SKIP_PARTS = {"__pycache__", ".cache"}
SKIP_SUFFIX = {".pyc", ".aux", ".log", ".fls", ".fdb_latexmk", ".out", ".synctex.gz"}
KEEP_LOG = {"logbooks"}  # run logs are kept, LaTeX logs are not


def files():
    for inc in INCLUDE:
        p = ROOT / inc
        for f in ([p] if p.is_file() else sorted(p.rglob("*"))):
            if not f.is_file() or SKIP_PARTS & set(f.parts) or "_bak" in f.name or f.name == ".env":
                continue
            rel = f.relative_to(ROOT)
            if f.suffix in SKIP_SUFFIX and not (f.suffix == ".log" and rel.parts[0] in KEEP_LOG):
                continue
            yield f, rel


SECRETS = []  # values from .env must never enter the archive
if (ROOT / ".env").exists():
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            v = line.split("=", 1)[1].strip().strip("'\"")
            if len(v) >= 12:
                SECRETS.append(v.encode())
OUT.parent.mkdir(exist_ok=True)
n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for f, rel in files():
        data = f.read_bytes()
        assert not any(sec in data for sec in SECRETS), f"secret found in {rel}"
        zf.write(f, "paiec_entry/" + rel.as_posix())
        n += 1
print(f"wrote {OUT} with {n} files, {OUT.stat().st_size / 1e6:.1f} MB")
