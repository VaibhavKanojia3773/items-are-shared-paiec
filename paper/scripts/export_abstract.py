"""Expand the LaTeX abstract (with results macros) to plain text for the OpenReview form."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
mac = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (ROOT / "results.tex").read_text(encoding="utf-8")))
tex = (ROOT / "main.tex").read_text(encoding="utf-8")
txt = tex.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
txt = re.sub(r"\\(\w+)\{\}", lambda m: mac.get(m.group(1), m.group(0)), txt)
txt = re.sub(r"\\(\w+)", lambda m: mac.get(m.group(1), m.group(0)), txt)
txt = txt.replace("\\emph{other}", "other").replace("--", "–").replace("$-$", "−").replace("$", "")
txt = re.sub(r"\\emph\{([^}]*)\}", r"\1", txt)
txt = " ".join(txt.split())
(ROOT.parent / "report" / "abstract_plain.txt").write_text(txt, encoding="utf-8")
print(txt)
