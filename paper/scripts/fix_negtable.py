from pathlib import Path

p = Path(__file__).resolve().parents[1] / "main.tex"
s = p.read_text(encoding="utf-8")
reps = [
    ("\\caption{\\textbf{Ideas that did not help} (development runs with whole-cell reveals unless noted).",
     "\\caption{\\textbf{Ideas that did not help or were not adopted} (development runs with whole-cell reveals unless noted)."),
    ("\\footnotesize $^{\\dagger}$Adds", "{\\raggedright\\footnotesize $^{\\dagger}$Adds"),
    ("Neither is part of the submitted package.\n", "Neither is part of the submitted package.\\par}\n"),
    ("Known-name ability offset$^{\\dagger}$ & high overlap (R7) &", "Known-name ability offset$^{\\dagger}$ & high overlap (R7) &"),
    ("    \\quad among known names only & high overlap (R7) &", "    \\quad known names only & high overlap (R7) &"),
    ("& sparse 1 / standard (R8) &", "& sparse 1 / std.\\ (R8) &"),
    ("& sparse 2 / high overlap (R8) &", "& sparse 2 / high (R8) &"),
]
for a, b in reps:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
s = s.replace("{\\scriptsize\\dciHighName}", "{\\tiny\\dciHighName}").replace("{\\scriptsize\\dciHighNameKnown}", "{\\tiny\\dciHighNameKnown}")
p.write_text(s, encoding="utf-8")
print("ok")
