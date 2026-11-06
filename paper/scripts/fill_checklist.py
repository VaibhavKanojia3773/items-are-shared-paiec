"""Fill the NeurIPS checklist (paper/checklist.tex) from the template, in order of the 16 questions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
s = (ROOT / "template" / "checklist.tex").read_text(encoding="utf-8")
a = s.index("%%% BEGIN INSTRUCTIONS %%%")
b = s.index("%%% END INSTRUCTIONS %%%") + len("%%% END INSTRUCTIONS %%%")
s = s[:a] + s[b:]
ANSWERS = [
    ("Yes", r"The abstract and introduction state the claims; \cref{tab:main,tab:components} and \cref{fig:overlap,fig:curves} support them, and \cref{sec:validation} states the assumptions about the hidden sampling design and the simulator's differences from the hosted evaluator."),
    ("Yes", r"The limitations paragraph and \cref{sec:validation} discuss the unknown sampling design, simulator--package differences, reuse of the public benchmark pool for tuning, visual benchmarks and chance-level budget 0."),
    ("NA", r"The paper makes no theoretical claims."),
    ("Yes", r"\Cref{sec:method,sec:validation} and \cref{app:hp,app:sim,app:repro} give the model, inference, acquisition rule, simulator, seeds and the submitted configuration; the supplementary archive contains the code and commands that regenerate every table and figure."),
    ("Yes", r"The supplementary archive contains the package, simulator, experiment and figure scripts, logged outputs and a README; the data are the public measurement-db \citep{measurementdb2026}."),
    ("Yes", r"Folds, sampling regimes, pool sizes, seeds, hyperparameters and the selection procedure are in \cref{sec:validation} and \cref{app:hp,app:sim}."),
    ("Yes", r"All simulator results report 95\% intervals from resampling whole benchmarks (10{,}000 draws); \cref{fig:curves} shows per-checkpoint 95\% intervals over pairs; \cref{app:noise} quantifies the variability of official scores."),
    ("Yes", r"The disclosures list the compute: a laptop CPU and Kaggle CPU/T4 sessions; the text-free runs take minutes per regime and the package runs on CPU."),
    ("Yes", r"The work uses public benchmark-response data and conforms to the NeurIPS Code of Ethics."),
    ("Yes", r"Better performance prediction can reduce evaluation cost, but could encourage trusting predictions instead of running evaluations; we report that predictions on unseen benchmarks stay near chance without labels."),
    ("NA", r"We release no model or dataset with a high risk of misuse."),
    ("Yes", r"We cite measurement-db, the competition and its baselines \citep{measurementdb2026,paiec2026,paiecbaseline2026}, the e5 encoder (MIT licence) \citep{wang2024me5} and the participant reports we draw on \citep{brierrabbit2026,doctawho2026}."),
    ("Yes", r"The released code is documented by a README and docstrings, and includes the per-benchmark data statistics and logged run outputs."),
    ("NA", r"No crowdsourcing or human-subject research was conducted."),
    ("NA", r"No human-subject research was conducted."),
    ("Yes", r"No LLM is part of the submitted method. An offline pilot rated public items with open LLMs and did not influence the method; Claude Code was used as a coding and writing assistant (Disclosures)."),
]
assert s.count(r"\answerTODO{}") == len(ANSWERS) == s.count(r"\justificationTODO{}")
for yn, just in ANSWERS:
    s = s.replace(r"\answerTODO{}", "\\answer" + yn + "{}", 1).replace(r"\justificationTODO{}", just, 1)
(ROOT / "checklist.tex").write_text(s, encoding="utf-8")
print("checklist filled")
