# Release manifest

## Selected submission
| Item | Value |
|---|---|
| Codabench submission ID | **1031141** (uploaded 2026-11-04, formative Brier ALC 0.208649 on one draw) |
| Package file | `dist/paiec_v6.zip` (the exact uploaded file) |
| SHA-256 | `d091ae066618c26d11c7bbc26d926e5a169b4bd96c64034f18cbbfcbab742543` |

## Inputs and environment
- Data: measurement-db core tables (`aims-foundations/measurement-db`, main branch, downloaded 2026-11-02; per-file index in the download script). Public data only.
- Encoder: `intfloat/multilingual-e5-small` (frozen; declared in the package's `models.txt`).
- Environment used for all simulator runs: Python 3.10.11, NumPy 2.2.6, pandas 2.3.3, pyarrow 25.0.1 (laptop CPU), plus Kaggle CPU/T4 sessions for embeddings and some runs. Kaggle and local runs of the standard regime agree to 1e-15.

## Paper inputs
| Paper element | Input files | Produced by |
|---|---|---|
| Table 1, Table 2, Figure 3, abstract numbers | `logbooks/runs/pairs_R7_{std,high,sp2,sp1}.npz` (one-response, embedding-free proxies) | `paiec/evaluate.py --single --no_provider ...` (see README) |
| Gate contrast (Section 6) | `logbooks/runs/pairs_R8_{std,high,sp2,sp1}.npz` | same, method `online_name_f@g1c1o` |
| Figure 2, overlap numbers | `logbooks/runs/overlap_R9_s0.json` (replicate `_s1`) | `paiec/overlap.py --seed 0/1` |
| All numbers in the PDF | `paper/results.tex` | `paper/scripts/analyze_r7.py` |
| Appendix B embedding runs | `logbooks/runs/kaggle/a1`, `a2` | `paiec/evaluate.py --emb ...` (Kaggle) |
| Figure 4 (formative scores) | values in `paper/scripts/fig_noise.py` | Codabench submission list |
| Data statistics | `logbooks/runs/final_eda_per_bench.csv`, `logbooks/03_data_analysis.md` | `scripts/build_tables.py`, `scripts/final_eda.py` |

## SHA-256 of the main result arrays
```
a4a8bafb233a9f3291024f8ebe0b7503836a3fca8d719cb89a6b2a00cc0481cc  logbooks/runs/pairs_R7_high.npz
8663c849e99cf0f9029d9da07177897d8c6b00f5c069338f37a907bd48afdfd8  logbooks/runs/pairs_R7_sp1.npz
1ce30881a358a50b4c51fd3db65260010bfc43a85e1e724e5f4ad1ee43f43d88  logbooks/runs/pairs_R7_sp2.npz
f0404a69c4958a7a21243c7ca845f07d8674e69e71429a7df97f3c7ec8f02057  logbooks/runs/pairs_R7_std.npz
d30c53474d3d285adc1c8b8349ec475a1b775b7caeb71460720e05567457870c  logbooks/runs/pairs_R8_high.npz
6b76970b24e57b52bad5fb8042b6382c0a0e1cf32d2e2b07dcce08bf4531ed19  logbooks/runs/pairs_R8_sp1.npz
788927cf310858894a9a3e39d3b1818e63ee55a703b2bb464e0f1cabc3ba1201  logbooks/runs/pairs_R8_sp2.npz
cbbb8c79f94906a52450943e886f93d59e1b9c515ea2542d38519301ab24a516  logbooks/runs/pairs_R8_std.npz
```

## Note on the archive contents
`logbooks/` also contains exploratory and superseded runs from development (kept for transparency; the logbooks explain each). Only the files listed above feed the paper.
