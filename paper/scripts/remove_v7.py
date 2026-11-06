"""Remove v7 as a named package: the paper describes only the submitted v6; the known-name offset and the
overlap gate become appendix ablations."""
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "main.tex"
s = p.read_text(encoding="utf-8")
reps = [
    # method: drop the v7 variant sentence
    ("A variant (\\texttt{v7}) adds a known-name ability offset $a\\,\\theta_s$ to the base logit, the shrunken mean of benchmark-centred logits of the subject's \\texttt{normalized\\_name} over public benchmarks, with $\\theta_s=0$ for unseen names.\n", ""),
    (" \\texttt{v7} was built after the last possible upload and is reported as an exploratory variant. \\Cref{app:hp} lists the configuration and hashes.",
     " \\Cref{app:hp} lists the configuration and the package hash."),
    # table 1
    ("The model rows are embedding-free proxies; \\texttt{v6} is the selected configuration and \\texttt{v7} the exploratory variant.}",
     "The model rows are embedding-free proxies; the last row is the selected configuration (\\texttt{v6}).}"),
    ("    \\quad + known-name offset (\\texttt{v7} proxy) & \\aStdSev & \\aHighSev & \\aSpTwoSev & \\aSpOneSev \\\\\n", ""),
    # table 2: drop the known-name row
    ("    Known-name offset (\\texttt{v7}) & \\makecell{\\dStdName\\\\[-1pt] {\\scriptsize\\dciStdName}} & \\makecell{\\dHighName\\\\[-1pt] {\\scriptsize\\dciHighName}} & \\makecell{\\dSpTwoName\\\\[-1pt] {\\scriptsize\\dciSpTwoName}} & \\makecell{\\dSpOneName\\\\[-1pt] {\\scriptsize\\dciSpOneName}} \\\\\n", ""),
]
for a, b in reps:
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)
# components paragraph: replace the gate + name-offset sentences with a pointer
a = s.index("We therefore tested a gate on the \\texttt{v7} proxy")
b = s.index("\\paragraph{Negative results.}")
s = s[:a] + ("A gate that uses the evaluator's random rule until another system has a label on the benchmark "
             "did not help consistently (\\cref{app:negative}).\n\n") + s[b:]
reps2 = [
    # official results unchanged; appendix A
    ("The \\texttt{v7} variant differs functionally only by the known-name offset. Values were chosen",
     "Values were chosen"),
    ("    known-name offset & off (on in \\texttt{v7}) & name pooling of $t_s$ & off \\\\\n",
     "    name-based ability prior & off & name pooling of $t_s$ & off \\\\\n"),
    # appendix E: add the two ablations
    ("    Open-LLM difficulty ratings, 4B model & 9 benchmarks (R6) & rank correlation & 0.06 \\\\\n",
     "    Open-LLM difficulty ratings, 4B model & 9 benchmarks (R6) & rank correlation & 0.06 \\\\\n"
     "    \\midrule\n"
     "    Known-name ability offset$^{\\dagger}$ & high overlap (R7) & $\\Delta$ & \\dHighName{} {\\scriptsize\\dciHighName} \\\\\n"
     "    \\quad among known names only & high overlap (R7) & $\\Delta$ & \\dHighNameKnown{} {\\scriptsize\\dciHighNameKnown} \\\\\n"
     "    Overlap gate on acquisition$^{\\ddagger}$ & sparse 1 / standard (R8) & $\\Delta$ & \\dSpOneGate{} / \\dStdGate{} \\\\\n"
     "    & sparse 2 / high overlap (R8) & $\\Delta$ & \\dSpTwoGate{} / \\dHighGate{} \\\\\n"),
]
for a, b in reps2:
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)
a = "  \\label{tab:negative}\n"
add_note = ("\\footnotesize $^{\\dagger}$Adds $a\\,\\theta_s$ to the base logit, the shrunken mean of benchmark-centred logits of the subject's "
            "\\texttt{normalized\\_name} over public benchmarks ($\\theta_s=0$ for unseen names); one-response reveals, embedding-free, "
            "benchmark-bootstrap intervals. $^{\\ddagger}$Uses the evaluator's random rule until another system has a label on the "
            "benchmark; tested with the name offset enabled; all four intervals include zero. Neither is part of the submitted package.\n")
end_tab = s.index("\\end{table}", s.index("\\label{tab:negative}"))
s = s[:end_tab] + add_note + s[end_tab:]
p.write_text(s, encoding="utf-8")
print("ok", s.count("v7"))
