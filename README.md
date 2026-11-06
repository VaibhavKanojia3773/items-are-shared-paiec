# Items Are Shared, Abilities Are Not: Predicting AI Correctness on Unseen Benchmarks from Other Systems' Labels

Repository: https://github.com/VaibhavKanojia3773/items-are-shared-paiec · Paper: `paper/main.pdf` · Selected Codabench submission: 1031141

See `MANIFEST.md` for hashes, inputs and which files produce each table and figure. Code, submitted package and experiment records for our NeurIPS 2026 Predictive AI Evaluation Competition (PAIEC) entry, described in `paper/main.pdf`.

At every call, the submission refits a test-time item response model on all revealed labels, with benchmark, system, pair, item and item-category effects. Items without labels borrow difficulty from similar items via `intfloat/multilingual-e5-small` embeddings. Labels are acquired by a budget-paced uncertainty rule with a coverage term that prefers items no other system has revealed. The package runs on CPU.

## Submitted package

| File | Description |
|---|---|
| `dist/paiec_v6.zip` | **Submitted package (v6), Codabench submission 1031141.** The exact uploaded file, SHA-256 `d091ae066618c26d11c7bbc26d926e5a169b4bd96c64034f18cbbfcbab742543`. Passes the official `check_submission_zip.py` |

Rebuild an equivalent v6 package from source (needs the processed tables in `data/proc/`, see below).:
```
python paiec/build_submission.py \
  --hp '{"a":0.4,"s_u":1.4,"s_t":0.5,"s_v":1.0,"s_w":2.2,"c":0.1,"n_cap":1.0,"tau_w":0.02,"kappa_w":0.3,"s_f":0.7,"s_g":0.3}' \
  --acq '{"gamma":1.0,"beta":1.0}' --text --no_name_prior --no_name_pool --text_tokens --template_tokens \
  --out dist/paiec_v6_rebuilt.zip
python paiec_baseline/check_submission_zip.py dist/paiec_v6_rebuilt.zip
```
`paiec_baseline/` is the organisers' public baseline repository (https://github.com/aims-foundations/paiec_baseline); clone it there to use the checker.

CPU smoke test of the package through the official dictionary interface (public data, 6 benchmarks × 4 systems):
```
mkdir -p /tmp/v6 && unzip -o dist/paiec_v6.zip -d /tmp/v6
python paiec/test_submission.py /tmp/v6 --benches 6 --subj 4 --eval 30 --seed 5
```

## Layout

| Path | Contents |
|---|---|
| `paiec/submission/` | Package sources: `model.py` (`predict`), `labeling.py` (`acquisition_function`), `text_prior.py` (encoder) |
| `paiec/online.py` | The test-time model, shared by the package and the simulator |
| `paiec/sim.py` | Protocol simulator: splits, streaming acquisition (one-response reveals with `single_seed`), checkpoints, Brier ALC; sampling regimes `build_pairs` (standard) and `build_pairs_formative` (high overlap / sparse) |
| `paiec/evaluate.py` | Benchmark-held-out 5-fold evaluation; writes summaries (`eval_*.json`) and per-pair curves (`pairs_*.npz`) |
| `paiec/overlap.py` | Controlled overlap study (Figure 2) |
| `paiec/acq.py` | Acquisition policies used by the simulator |
| `paiec/tune.py`, `paiec/tune_full.py` | Hyperparameter searches |
| `paiec/llm_rate.py`, `paiec/llm_value.py` | Offline LLM difficulty-rating pilot and its value simulation (not part of the package) |
| `scripts/` | Data download, table building, EDA, Kaggle orchestration |
| `paper/` | LaTeX source; `paper/scripts/` regenerates every figure and `paper/results.tex` (all numbers in the paper) |
| `logbooks/` | Every experiment, including negative results: `02_experiments.md`, data analysis `03_data_analysis.md`, competitor notes `05_competitor_research.md`; raw outputs in `logbooks/runs/` |

## Reproduce the paper

Python ≥ 3.10 with `numpy pandas pyarrow huggingface_hub matplotlib`; add `sentence-transformers` for embeddings.

1. **Data.** The public measurement-db core tables (about 8.5 GB) need an HF token with access:
   ```
   python scripts/download_core.py
   python scripts/build_tables.py   # data/proc/cells.parquet: 31.8M (subject, item, n, k) cells
   python scripts/build_items.py    # data/proc/items_all.parquet
   python scripts/subset.py         # data/proc/subset10k.parquet (<= 10k items per benchmark)
   ```
2. **Main tables** (Tables 1–2, one-response reveals, text-free proxies; minutes per regime on a laptop CPU):
   ```
   HP='{"a":0.4,"s_u":1.4,"s_t":0.5,"s_v":1.0,"s_w":2.2,"c":0.1,"n_cap":1.0,"s_f":0.7,"s_g":0.3}'
   M=const,emp_mean,beta11,eb_bench,online0,online0@g1,online0_f@g1,online0_f,online0_f@g1c1,online_name_f@g1c1
   E="python paiec/evaluate.py --subset data/proc/subset10k.parquet --tok both --template --single --no_provider --methods $M --hp $HP"
   $E --tag R7_std
   $E --formative 100 --tag R7_high
   $E --formative 100 --rows 200 --tag R7_sp2
   $E --formative 100 --rows 100 --tag R7_sp1
   ```
   Method names: `online0` is the hierarchical model; `_f` adds item tokens; `online_name_` adds the known-name prior; `@g1` means uncertainty acquisition with γ = 1, and `c1` adds the coverage term with β = 1.
3. **Overlap study** (Figure 2): `python paiec/overlap.py --seed 0` and `--seed 1` (replicate). **Gate runs:** the same `evaluate.py` command with `--methods online_name_f@g1c1o,online_name_f --tag R8_<regime>`
4. **Intervals, macros and figures:**
   ```
   python paper/scripts/analyze_r7.py
   python paper/scripts/fig_overlap.py
   python paper/scripts/fig_curves.py
   python paper/scripts/fig_noise.py
   cd paper && latexmk -pdf main.tex
   ```
5. **Embedding runs** (Appendix B) used `kaggle/embed_template.py` via `scripts/kaggle_push_embed.py intfloat/multilingual-e5-small 2 <account>` and `paiec/evaluate.py --emb ...`; outputs are in `logbooks/runs/kaggle/a1`, `a2`.

## Disclosures
- Pretrained model: `intfloat/multilingual-e5-small` (MIT licence), frozen. Data: public measurement-db only. No LLM is called at scoring time.
- An offline pilot rated public items with open LLMs through the NVIDIA API. It did not influence the submitted method.
- Development used Claude Code (Anthropic) as a coding and writing assistant.
