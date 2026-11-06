"""Final edits: new title, submitted package = v6 (Codabench 1031141), checklist removed."""
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "main.tex"
s = p.read_text(encoding="utf-8")
reps = [
    (r"\title{When Cross-Subject Labels Help on Unseen\\Benchmarks: A PAIEC Competition Report}",
     r"\title{No Benchmark Is an Island: Test-Time Item Response\\Models That Borrow Other Systems' Answers}"),
    # abstract
    (r"our submitted configuration reaches a Brier area under the learning curve of \aStdSev{} (standard sampling) and \aHighSev{} (high item overlap)",
     r"our submitted configuration reaches a Brier area under the learning curve of \aStdSix{} (standard sampling) and \aHighSix{} (high item overlap)"),
    # figure 1 caption
    (r"offline, public data are used only to tune shrinkage and a known-name ability prior.",
     r"offline, public data are used only to tune shrinkage (and, in a variant we also evaluate, a known-name ability prior)."),
    # method: offline part
    (r"The offline part is a scaled base logit: $\mu$ is the global mean logit of the public data, and $\theta_s$ is a known-name ability, the shrunken mean of benchmark-centred logits of the subject's \texttt{normalized\_name} over public benchmarks. Unseen names get $\theta_s=0$.",
     r"The offline part is a scaled base logit: $\mu$ is the global mean logit of the public data. The optional $\theta_s$ is a known-name ability, the shrunken mean of benchmark-centred logits of the subject's \texttt{normalized\_name} over public benchmarks, with $\theta_s=0$ for unseen names. The submitted package uses $\theta_s=0$ for all subjects."),
    # submitted package paragraph
    (r"The selected package (\texttt{v7}) implements \cref{eq:model,eq:acq} with the known-name prior, no provider fallback for unseen names, and the e5 encoder declared in \texttt{models.txt}. It falls back to the text-free model if the encoder cannot be loaded. \Cref{app:hp} lists its configuration and hash.",
     r"The selected package (\texttt{v6}, Codabench submission 1031141) implements \cref{eq:model,eq:acq} with $\theta_s=0$, embedding propagation, item tokens and the coverage term, with the e5 encoder declared in \texttt{models.txt}. It falls back to the text-free model if the encoder cannot be loaded. \Cref{app:hp} lists its configuration and hash. A variant with the known-name prior (\texttt{v7}) was built after the last possible upload and is reported as an evaluated alternative."),
    # simulator differences (iv)
    (r"(iv) public subjects are often unnamed checkpoints, so the known-name prior is exercised on fewer pairs than it likely would be in the hidden set;",
     r"(iv) public subjects are often unnamed checkpoints, so the known-name variant is exercised on fewer pairs than it would be on named systems;"),
    # table 1
    (r"Text-free proxies of the packages (embedding propagation omitted; \cref{sec:validation}). \texttt{v7} is the submitted configuration.}",
     r"Text-free proxies of the packages (embedding propagation omitted; \cref{sec:validation}). \texttt{v6} is the submitted configuration; \texttt{v7} adds the known-name prior.}"),
    (r"\quad + coverage term (\texttt{v6})       &", r"\quad + coverage term (\texttt{v6}, submitted) &"),
    (r"\quad + known-name prior (\texttt{v7})     & \aStdSev \ {\scriptsize\ciStdSev} & \aHighSev \ {\scriptsize\ciHighSev} &",
     r"\quad + known-name prior (\texttt{v7})     & \aStdSev & \aHighSev &"),
    (r"\quad + coverage term (\texttt{v6}, submitted) & \aStdSix & \aHighSix &",
     r"\quad + coverage term (\texttt{v6}, submitted) & \aStdSix \ {\scriptsize\ciStdSix} & \aHighSix \ {\scriptsize\ciHighSix} &"),
    # main comparison numbers: submitted = Six
    (r"The submitted configuration improves on the official baseline by \relStdEM\% (standard) and \relHighEM\% (high overlap), and on the Beta-smoothed pair mean by \relStdBeta\% and \relHighBeta\%.",
     r"The submitted configuration improves on the official baseline by \relSixStdEM\% (standard) and \relSixHighEM\% (high overlap), and on the Beta-smoothed pair mean by \relSixStdBeta\% and \relSixHighBeta\%."),
    (r"The remaining gap to the Beta baseline, \dStdVsBeta{} \dciStdVsBeta{} (standard) and \dHighVsBeta{} \dciHighVsBeta{} (high overlap), is what the item response model adds. It shrinks to \dSpOneVsBeta{} \dciSpOneVsBeta{}",
     r"The remaining gap to the Beta baseline, \dStdSixVsBeta{} \dciStdSixVsBeta{} (standard) and \dHighSixVsBeta{} \dciHighSixVsBeta{} (high overlap), is what the item response model adds. It shrinks to \dSpOneSixVsBeta{} \dciSpOneSixVsBeta{}"),
    # name prior paragraph
    (r"We submitted it because hidden systems are named releases, for which the public data provide ability estimates. Its expected effect is small either way.",
     r"It was developed after the last possible upload, so the submitted package does not include it; its expected effect on named systems is small but favourable."),
    (r"so the submitted package does not use it.", r"so we do not use it."),
    # official results
    (r"These runs cannot separate small package effects from resampling variability, so we selected \texttt{v7} on the simulator evidence above. Its common-subset summative score is determined by the organisers.",
     r"These runs cannot separate small package effects from resampling variability, so we selected \texttt{v6} (submission 1031141, one formative run at 0.209) on the simulator evidence above: among the uploaded packages it is the best configuration in both regimes. Its common-subset summative score is determined by the organisers."),
    # appendix A
    (r"\caption{\textbf{Submitted configuration} (\texttt{v7}; SHA-256 of the ZIP \texttt{2cf5a1bf\ldots91cc74}).",
     r"\caption{\textbf{Submitted configuration} (\texttt{v6}, Codabench submission 1031141; SHA-256 of the ZIP \texttt{d091ae06\ldots b742543}). The \texttt{v7} variant differs only by the known-name prior (SHA-256 \texttt{2cf5a1bf\ldots91cc74})."),
    (r"scale $a$ on $\mu+\theta_s$ & 0.4 &", r"scale $a$ on $\mu$ & 0.4 &"),
    (r"known-name prior & on & provider fallback, name pooling & off, off \\", r"known-name prior & off (on in \texttt{v7}) & name pooling of $t_s$ & off \\"),
    # remove checklist
    ("\\newpage\n\\input{checklist.tex}\n", ""),
]
for a, b in reps:
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("ok")
