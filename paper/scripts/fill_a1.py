"""Fill the standard-regime (A1) column of Table 2 and the dependent text in main.tex."""
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "main.tex"
s = p.read_text(encoding="utf-8")
rows = {
    r"Constant 0.5                         & 0.2500 & \tbd{} & 0.2500 & $+0.0653$":
        r"Constant 0.5                         & 0.2500 & $+0.0558_{\pm.0016}$ & 0.2500 & $+0.0653_{\pm.0022}$",
    r"Empirical Mean (official baseline)   & \tbd{} & \tbd{} &":
        r"Empirical Mean (official baseline)   & 0.2438 & $+0.0497_{\pm.0014}$ &",
    r"Hierarchical model, random acquisition & \tbd{} & \tbd{} &":
        r"Hierarchical model, random acquisition & 0.1988 & $+0.0046_{\pm.0005}$ &",
    r"+ uncertainty acquisition ($\gamma{=}1$) & \tbd{} & \tbd{} &":
        r"+ uncertainty acquisition ($\gamma{=}1$) & 0.1980 & $+0.0039_{\pm.0003}$ &",
    r"+ text propagation              & \tbd{} & \tbd{} &":
        r"+ text propagation              & 0.1958 & $+0.0017_{\pm.0002}$ &",
    r"+ item tokens, subject$\times$token & \tbd{} & \tbd{} &":
        r"+ item tokens, subject$\times$token & 0.1941 & $-0.0001_{\pm.0001}$ &",
    r"= \textbf{final} & \tbd{\textbf{}} & -- &":
        r"= \textbf{final} & \textbf{0.1942} & -- &",
    r"$-$ text propagation                  & \tbd{} & \tbd{} &":
        r"$-$ text propagation                  & 0.1940 & $-0.0002_{\pm.0002}$ &",
    r"$-$ tokens                            & \tbd{} & \tbd{} &":
        r"$-$ tokens                            & 0.1960 & $+0.0019_{\pm.0002}$ &",
    r"$-$ active acquisition (random)       & \tbd{} & \tbd{} &":
        r"$-$ active acquisition (random)       & 0.1949 & $+0.0008_{\pm.0004}$ &",
    r"$+$ subject residual $x$              & \tbd{} & \tbd{} &":
        r"$+$ subject residual $x$              & 0.1941 & $-0.0001_{\pm.0001}$ &",
    "1{,}713 pairs": "1{,}738 pairs",
    r"$\tbd{0.24}\to\tbd{0.199}$ (standard) and $0.242\to0.190$ (formative-like).":
        r"$0.244\to0.199$ (standard) and $0.242\to0.190$ (formative-like).",
    r"Our final system reduces the Brier area under the learning curve from 0.242 (official Empirical Mean baseline) to 0.185 in simulation;":
        r"Our final system reduces the Brier area under the learning curve of the official Empirical Mean baseline from 0.242 to 0.185 (formative-like regime) and from 0.244 to 0.194 (standard regime);",
    r"We selected v6 because it is best in both simulator regimes (\cref{tab:main}), not because of its single official score.":
        r"We selected v6 because it is best in the formative-like regime and tied for best in the standard regime (\cref{tab:main}), not because of its single official score.",
}
old_gain = s[s.index("Uncertainty acquisition, item tokens and coverage acquisition"):s.index("We keep both because they are cheap")]
rows[old_gain] = (
    r"Which additions matter depends on the overlap between pairs. "
    r"With low overlap (standard), text propagation is the largest addition ($-0.0022$), followed by tokens ($-0.0017$) and uncertainty acquisition ($-0.0008$). "
    r"With full overlap (formative-like), acquisition matters more: uncertainty acquisition $-0.0017$, tokens $-0.0015$ and coverage acquisition $-0.0010$, all with paired SE $\le0.0004$. "
    r"Coverage is neutral in the standard regime ($+0.0001$), as expected when items are rarely shared. "
    r"Two components are redundant given the others. Text propagation helps on its own but adds nothing once tokens are present (ablation: $-0.0002$ and $-0.0001$), because template and feature tokens capture the same within-benchmark structure. The subject-residual term $x$ (\cref{app:hp}) is likewise neutral. "
)
for a, b in rows.items():
    assert a in s, a[:60]
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("filled; remaining tbd:", s.count(r"\tbd{"))
