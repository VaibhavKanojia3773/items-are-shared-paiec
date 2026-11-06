from pathlib import Path
p = Path("main.tex"); s = p.read_text(encoding="utf-8")
rep = [
 (r"\usepackage[sglblindworkshop]{neurips_2026}", r"\usepackage[sglblindworkshop, final]{neurips_2026}"),
 (r"\title{Borrowing Strength Across Subjects: Test-Time Item Response\Models for Label-Efficient Prediction on Unseen Benchmarks}",
  r"\title{Borrowing Strength Across Subjects:\Test-Time Item Response Models for\Label-Efficient Prediction on Unseen Benchmarks}"),
 (r"\resizebox{\linewidth}{!}{\input{figs/overview.tex}}", r"\includegraphics[width=\linewidth]{figs/standalone_overview.pdf}"),
 ("on a benchmark it has never been seen on,", "on a benchmark on which it has never been evaluated,"),
]
for a, b in rep:
    assert a in s, a[:40]
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("ok")
