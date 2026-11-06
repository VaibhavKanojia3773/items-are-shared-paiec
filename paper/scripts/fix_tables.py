"""One-off layout fixes for main.tex: title break, table widths, two-line component cells."""
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "main.tex"
s = p.read_text(encoding="utf-8")
reps = [
    (r"\title{When Cross-Subject Labels Help on Unseen Benchmarks:\\A PAIEC Competition Report}",
     r"\title{When Cross-Subject Labels Help on Unseen\\Benchmarks: A PAIEC Competition Report}"),
    ("  \\centering\\small\n  \\setlength{\\tabcolsep}{4pt}\n  \\begin{tabular}{@{}lllll@{}}\n    \\toprule\n"
     "    Method & Standard & High overlap & Sparse, 2 systems & Sparse, 1 system \\\\",
     "  \\centering\\footnotesize\n  \\setlength{\\tabcolsep}{3.5pt}\n  \\begin{tabular}{@{}lllll@{}}\n    \\toprule\n"
     "    Method & Standard & High overlap & Sparse, 2 sys. & Sparse, 1 sys. \\\\"),
    ("Empirical Mean (official baseline)        &", "Empirical Mean (official)                 &"),
    ("Pair mean shrunk to benchmark rate        &", "Pair mean shrunk to bench.\\ rate          &"),
    ("Hierarchical model, random acquisition    &", "Hierarchical model, random acq.           &"),
    ("\\quad + known-name prior (\\texttt{v7}, submitted) &", "\\quad + known-name prior (\\texttt{v7})     &"),
    ("  \\centering\\small\n  \\setlength{\\tabcolsep}{3pt}\n  \\begin{tabular}{@{}lllll@{}}\n    \\toprule\n"
     "    Component & Standard & High overlap & Sparse, 2 systems & Sparse, 1 system \\\\",
     "  \\centering\\footnotesize\n  \\setlength{\\tabcolsep}{4pt}\n  \\begin{tabular}{@{}lcccc@{}}\n    \\toprule\n"
     "    Component & Standard & High overlap & Sparse, 2 sys. & Sparse, 1 sys. \\\\"),
    ("kernel temperature $\\tau_w$, shrinkage $\\kappa_w$ & 0.02, 0.3 &", "kernel $\\tau_w$, shrinkage $\\kappa_w$ & 0.02, 0.3 &"),
    ("known-name prior / provider fallback & on / off & name pooling of $t_s$ & off \\\\",
     "known-name prior & on & provider fallback, name pooling & off, off \\\\"),
    ("item key & hash(content, features, interactors) & subject key & full dictionary \\\\",
     "item key & content+features+interactors & subject key & full dictionary \\\\"),
]
for a, b in reps:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
# component table rows: value over interval
a = s.index("    Uncertainty acquisition  &")
b = s.index("    \\bottomrule", a)
rows = [("Uncertainty acquisition", "Unc"), ("Item tokens", "Tokens"), ("Coverage term", "Cov"),
        ("Known-name prior", "Name"), ("\\makecell[l]{Active acquisition\\\\(final predictor)}", "Act")]
body = ""
for lab, c in rows:
    cells = " & ".join("\\makecell{\\d%s%s\\\\[-1pt] {\\scriptsize\\dci%s%s}}" % (r, c, r, c)
                       for r in ["Std", "High", "SpTwo", "SpOne"])
    body += f"    {lab} & {cells} \\\\\n"
s = s[:a] + body + s[b:]
p.write_text(s, encoding="utf-8")
print("fixed")
