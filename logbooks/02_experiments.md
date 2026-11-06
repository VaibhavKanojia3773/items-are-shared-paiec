# Logbook 02: Experiments and results

Every run gets an entry: date, ID, what changed, data, score, verdict. Metric is **Brier ALC** (lower is better; 0.25 = constant 0.5) from `paiec/sim.py` unless stated otherwise. Raw outputs are in `logbooks/runs/`.

Real-data protocol (unless noted): 5-fold split **by benchmark** (test benchmarks unseen by any prior), ≤ 20 subjects per benchmark, acquisition pool ≤ 200 and evaluation pool ≤ 100 items per pair, equal weight per pair. "Subset" = embedded items only (≤ 10k items per benchmark, 597k items, `scripts/subset.py`).

## Results table

| ID | Date | Data | Method | ALC | B0 | B1 | B3 | B7 | B15 | B31 | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E000 | 11-02 | synthetic IRT (30 subj, 12 bench) | constant 0.5 | 0.2500 | .250 | .250 | .250 | .250 | .250 | .250 | reference |
| E001 | 11-02 | synthetic | Empirical Mean (official baseline logic) | 0.2331 | .250 | .335 | .232 | .199 | .186 | .178 | budget-1 blow-up |
| E002 | 11-02 | synthetic | prior only (noisy benchmark guess + true ability) | 0.1961 | .196 | .196 | .196 | .196 | .196 | .196 | reference |
| E003 | 11-02 | synthetic | **online hierarchical model** (`online.py`) | **0.1716** | .190 | .178 | .174 | .170 | .164 | .155 | code path validated |
| E010 | 11-02 | real, 5-fold, all items | constant 0.5 | 0.2500 | .250 | .250 | .250 | .250 | .250 | .250 | reference |
| E010 | 11-02 | same | global mean from training | 0.2508 | .251 | .251 | .251 | .251 | .251 | .251 | worse than 0.5: base rates vary too much |
| E010 | 11-02 | same | **Empirical Mean (official baseline)** | **0.2416** | .250 | .353 | .239 | .205 | .193 | .187 | budget-1 blow-up |
| E010 | 11-02 | same | name prior only | 0.2537 | .254 | .254 | .254 | .254 | .254 | .254 | harmful alone |
| E010 | 11-02 | same | online model, constant prior | **0.2001** | .251 | .211 | .200 | .192 | .185 | .177 | **−17% vs. official baseline** |
| E010 | 11-02 | same | online model + name prior | 0.2003 | .251 | .211 | .200 | .192 | .185 | .177 | ≈ constant prior |
| E011 | 11-02 | subset, fold 0 | Empirical Mean / online0 / online_name | 0.2520 / 0.2097 / 0.2099 | | | | | | | refactor to integer-coded data reproduces E010 |
| T001 | 11-02 | subset, 5-fold | random search, 41 online-model HP configs | 0.2003 (best) vs. 0.2014 (default) | .251 | .211 | .200 | .193 | .185 | .176 | flat; HP are not the bottleneck |
| E030 | 11-02 | subset, fold 0 | online_name: random vs. uncertainty γ=0.5 / 1 / 2 | 0.2092 / 0.2089 / **0.2083** / 0.2084 | | | | | | | small gain from uncertainty sampling |
| E031 | 11-02 | subset, 5-fold | online_name, random acquisition | 0.2003 | .251 | .211 | .200 | .193 | .185 | .176 | reference |
| E031 | 11-02 | same | + uncertainty acquisition γ=1 | **0.1995** | .251 | .210 | .199 | .191 | .184 | .176 | consistent small gain |
| E031 | 11-02 | same | + uncertainty acquisition γ=2 | 0.1996 | | | | | | | ≈ γ=1 |
| E031 | 11-02 | same | name-pooled subject offset (`_t`), ± γ=1 | 0.2003 / 0.1997 | | | | | | | neutral |
| E040 | 11-02 | subset, 5-fold, Kaggle GPU, e5-small | kNN text prior alone (uncalibrated) | 0.2677 | .268 | | | | | | **worse than 0.5**: overconfident |
| E040 | 11-02 | same | online + kNN prior (uncalibrated) | 0.2022 | .253 | .213 | .202 | .194 | .186 | .177 | hurts |
| E040 | 11-02 | same | **online + test-time text propagation** (`online0_prop`) | **0.1988** | .250 | .209 | .199 | .191 | .183 | .175 | helps at every budget ≥ 1 |
| E041 | 11-02 | same | kNN prior with leave-one-benchmark-out calibration | 0.2547 | .255 | | | | | | LOBO R² = 0.003–0.021, λ = 0.08–0.26 |
| E041 | 11-02 | same | online + calibrated kNN + propagation | 0.1990 | .250 | .209 | .199 | .191 | .184 | .175 | ≈ propagation alone |
| E041 | 11-02 | same | **online0_prop + uncertainty γ=1** | **0.1983** | .250 | .208 | .199 | .190 | .183 | .175 | **current best (−19% vs. official 0.2438)** |

| E050 | 11-03 | subset, 5-fold, **Qwen3-Embedding-0.6B** | online0_prop / online_name_propx / prop@g1 / propx@g1 | 0.1990 / 0.1987 / 0.1985 / 0.1980 | | | | | | | Qwen ≈ e5 (slightly worse) |
| E051 | 11-03 | subset, 5-fold, **e5-small** | online0_prop / online0_propx / prop@g1 / **propx@g1** | 0.1988 / 0.1982 / 0.1983 / **0.1976** | .250 | .208 | .198 | .189 | .182 | .173 | e5 kept; residual term x helps a little |
| T010/T011 | 11-03 | subset, 5-fold, e5, Kaggle CPU ×2 | random search, 122 configs, propagation + residual space (random acquisition) | best **0.1967** vs. 0.1988 default | .250 | .208 | .197 | .188 | .180 | .171 | sharper kernel (τ_w 0.02, κ_w 0.3–1), larger s_w 2.2, a 0.4, c 0.1 |
| E061 | 11-03 | same | 2nd-best T010 HP: prop / prop@g1 / propx / **propx@g1** | 0.1973 / 0.1966 / 0.1972 / **0.1964** | .250 | .208 | .197 | .188 | .179 | .170 | **new local best → v3** |
| E060 | 11-03 | subset, 5-fold, e5 | best T010 HP: prop / prop@g1 / propx / propx@g1 / name_propx@g1 | 0.1967 / **0.1958** / 0.1967 / 0.1959 / 0.1958 | .250 | .207 | .196 | .188 | .179 | .170 | best HP set ("BEST") |
| X1 | 11-03 | subset, LOBO ridge, e5 & Qwen | item text → easiness / benchmark base rate (embeddings, format features, both) | R² ≤ 0.04 everywhere (mostly < 0) | | | | | | | **budget 0 cannot be predicted from text**; stop |
| X2 | 11-03 | subset, 5-fold | per-budget Platt scaling, old HP / v3 HP | 0.2003→0.1995 / 0.1964→0.1967 | | | | | | | the tuned c=0.1 already calibrates; dropped |
| X3 | 11-03 | subset, folds 0–1 | V-optimal-style acquisition (item-variance-discounted info), γ 1/2/4 | 0.2030 / 0.2030 / 0.2029 vs. 0.2028 | | | | | | | acquisition saturated |
| ORACLE | 11-03 | subset, 5-fold pools | pair-rate oracle / + item difficulty from other subjects / cell noise floor | 0.181 / **0.127** / 0.007 | | | | | | | headroom is item difficulty within benchmark |
| X5 | 11-03 | subset, folds 0–1 | **item_features key=value token random effects** (bench-scoped, ID-like filtered), s_f sweep | 0.2036 → **0.2006** (s_f 0.7) | .247 | .211 | .202 | .194 | .185 | .176 | first version diverged (Jacobi overshoot); fixed with a curvature majorizer |
| X6 | 11-03 | subset, folds 0–1 | derived text tokens (length/lines/options/code/math buckets) / both token types | 0.2024 / **0.1999** | | | | | | | additive with item_features |
| E090 | 11-03 | subset, 5-fold, e5 | v3 HP: propx@g1 / propxf@g1 / propf@g1 | 0.1964 / **0.1952** / 0.1954 | | | | | | | tokens stack on v3 |
| E091 | 11-03 | subset, 5-fold, e5 | BEST HP: prop@g1 / **propf@g1** / propxf@g1 | 0.1958 / **0.1947** / 0.1947 | .250 | .206 | .196 | .186 | .177 | .167 | **new local best** |
| E092 | 11-03 | subset, 5-fold, e5 | BEST HP + both token types: propf@g1 / propxf@g1 | **0.1946** / 0.1946 | .250 | .206 | .196 | .186 | .176 | .167 | text tokens ≈ neutral on top; x neutral → v4 |
| T020/T021 | 11-03 | subset, 5-fold, e5, γ=1 | fine random search around v3 (82 configs, no tokens) | best 0.1961 vs. 0.1964 | | | | | | | flat: HP space exhausted |
| X7 | 11-03 | subset, folds 0–1 | + template tokens (first 4 / last 3 words) on top of both token types | 0.1999 → **0.1984** | .247 | .210 | .199 | .191 | .182 | .173 | templated benchmarks share task-type difficulty |
| X8 | 11-03 | subset, folds 0–1 | + subject×token effects g (s_g 0.15 / 0.2 / **0.3** / 0.6) | 0.1980 / 0.1978 / **0.1976** / 0.1986 | | | | | | | Riiid-style user×tag skill |
| E093 | 11-03 | subset, 5-fold, e5 | **full stack**: BEST HP + propagation + feature/interactor/text/template tokens (s_f 0.7) + subject×token (s_g 0.3), γ=1 | **0.1941** | .250 | .206 | .196 | .185 | .175 | .166 | **new local best → v5** |
| E094 | 11-03 | same | same with s_f 0.5 | 0.1941 | | | | | | | insensitive to s_f |
| S001 | 11-03 | subset, 5-fold, local | LightGBM stacker on online-model features (cross-fold) | 0.2015 vs. 0.2003 base | .254 | .214 | .202 | .193 | .185 | .175 | **negative**: overfits benchmark differences, worse budget 0 |
| E070 | 11-03 | subset, folds 0–1, local | ability-weighted item residual (y term) | 0.2033 vs. 0.2036 | | | | | | | marginal (−0.0003) |
| E080 | 11-03 | subset, folds 0–1, local | low-rank subject skills (K=3) + per-benchmark loadings λ_b, s_L sweep | 0.2035–0.2049 vs. 0.2036 | | | | | | | **negative**: overfits at s_L 0.5; no gain when shrunk |

Best HP (T001): `a=0.6, s_u=1.0, s_t=0.5, s_v=0.5, s_w=1.2, c=0.2, n_cap=2`.

## Notes

### E000–E003: synthetic sanity check (`paiec/test_synthetic.py`)
- Validates the simulator and the online model before real data. Not evidence about real performance.
- The Empirical Mean is worse than 0.5 at budget 1: one label gives a prediction of exactly 0 or 1.

### E010–E030: first real-data results
- The online model's gain comes from pooling labels across subjects (benchmark offset) and from shrinkage. **Budget 0 is still ≈ 0.25 for every method without text**, because the unseen benchmark's base rate is unknown. The text prior (E040) targets this.
- The name prior is useless or harmful on its own at budget 0, and neutral inside the online model. Subject ability matters less than benchmark and item effects, consistent with the variance decomposition (logbook 03: 19% benchmark, 9% subject, 72% item level).
- HP search is flat (0.2014 → 0.2003), so the gains must come from structure (text prior, item propagation) and acquisition.

### Engineering fixes
- Simulator: rebuilding the `labeled` DataFrame at every acquisition step was O(n²). `LabeledView` builds it lazily, and acquisition runs went from more than 15 min to 13 s per fold. Policy predictions are vectorised per pair and refit.
- The first acquisition rule (a band of width floor + gain·rate) was broken: with 31 labels over a 200-item pool the band exceeded 0.5, so it accepted everything (equivalent to taking the first items). It was replaced by the rate-matched rule `1 − 4p(1−p) ≤ γ · labels_remaining / items_remaining`.
- Kaggle embedding kernels: (1) the first runs failed at startup because the datasets were still processing, so they were re-pushed; (2) the `sentence-transformers` multi-GPU pool needs an `if __name__ == "__main__"` guard. Kaggle assigned T4×2.
- E040 v1 crashed at budget 0: the empty `labeled` frame has object dtype for `sid`, so indexing failed. Now cast to int64.
- A scripted logbook edit truncated this file (Windows cp1252 default encoding). It was rewritten. All scripted edits now use `encoding="utf-8"`.

### E040–E041: text features
- **Item difficulty from text does not transfer across benchmarks.** With leave-one-benchmark-out calibration, the kNN easiness explains only 0.3–2% of item-easiness variance on unseen benchmarks. The uncalibrated prior was harmful (budget-0 Brier 0.268). This is consistent with PredictaBoard's finding that out-of-distribution assessors are weak, and it is an important negative result for the report.
- **Within-benchmark text propagation works.** Once some items of a test benchmark are labelled (by any subject), their fitted difficulty offsets spread to textually similar unlabelled items: 0.2003 → 0.1988.
- Budget 0 stays at ≈ 0.250. With a 10% weight, it now limits the achievable gain less than the budget-1/3 region does.

### Round 2 (11-03) notes
- Research round 2 (`logbooks/04_research_round2.md`) ranked low-rank loadings and subject-similar item offsets first. Both were implemented and gave ≤ 0.0003 locally (E070, E080). The subject × benchmark component is already captured by the pair offset learned from the pair's own labels, and many test subjects (RL checkpoints, fine-tuned classifiers) have no training history.
- GBM stacking (S001) failed: the stacker learns benchmark-specific patterns that do not transfer to held-out benchmarks.
- The useful gains came from tuning the text propagation (sharper kernels) and from the subject-residual term: 0.1983 → **0.1964**.
- An encoder comparison showed no gain from Qwen3-0.6B over e5-small, so e5 stays (smaller, safer at scoring time).

### Round 3 (11-03) notes: depth and breadth
- **Headroom analysis** (`paiec/oracle.py`): knowing each pair's true rate gives 0.181; knowing item difficulty as well gives 0.127. We are at 0.167 at budget 31, so the remaining headroom is almost entirely **within-benchmark item difficulty**.
- **Breadth (budget 0) is closed:** ridge on e5/Qwen embeddings and hand-built format features has leave-one-benchmark-out R² ≤ 0.04 for both item easiness and benchmark base rate. This agrees with Contextual MIRT (arXiv 2608.22295): cross-benchmark difficulty transfer from text is an open problem.
- **Depth wins:** within-benchmark categorical channels. `item_features` tokens explain about 17% of within-benchmark difficulty variance (in-sample). As random effects they give −0.003, and derived text tokens a further −0.0007. Tokens are additive with text propagation (E091: 0.1958 → 0.1947).
- `interactors` (attacker/opponent, 16 benchmarks) is now part of the item key and the tokens in the submission. It cannot be validated in the simulator because cells are aggregated across interactors.

### Simulator vs. dict-API smoke tests (11-03): conflicting evidence on tokens
Smoke tests run the actual packages through the competition's dict API (`paiec/test_submission.py`, live acquisition), on a random set of benchmarks drawn from all items.

| Package (same 68 pairs, 12 benchmarks, seed 11) | Dict-API ALC | 5-fold simulator ALC |
|---|---|---|
| v3 (v3 HP + residual, no tokens, with text) | **0.1789** | 0.1964 |
| A: v3 HP + all tokens (with text) | 0.1818 | not run |
| B: BEST HP, no tokens (with text) | 0.1807 | 0.1958 |
| v5 (BEST HP + all tokens, with text) | 0.1820 | **0.1941** |
| v3 HP, no text, no tokens (local) | 0.1810 | |
| v3 HP, no text, + tokens (local) | 0.1808 | |

- In the dict API, tokens are neutral without text propagation and hurt with it (+0.003). In the simulator they help (−0.0045 over ~145 benchmarks).
- Per-benchmark differences in the smoke test swing ±0.03 (e.g. 0.141 → 0.130 and 0.228 → 0.236), so with 12 benchmarks the standard error is about 0.008: **the smoke test is underpowered**. The simulator's token gain is about 2σ.
- Possible systematic differences: (1) the simulator restricts items to the embedded subset and caps subjects at 20; (2) the competition defines subjects by visible attributes, so training subjects that differ only in `subject_features_extra` may be merged in the hidden data.
- **Decision rule:** Codabench v3 vs. v4 (v4 = tokens + BEST HP) decides whether tokens go into the final package.

### Round 4 (11-03 evening): formative-like simulator, noise analysis, coverage acquisition
- **Formative mode** (`build_pairs_formative`, `evaluate.py/tune.py --formative K`): per held-out benchmark, about 1000/K subjects that all answered the same K items (K = 100 → about 10 subjects × 100 items, 50/50 split per pair). This mimics a ≤ 1000-row Codabench sample with **full item overlap across pairs**. It gives 891 pairs on 113 benchmarks over 5 folds. No-embedding runs take seconds per fold locally.
- **F-series results (formative, K = 100, γ = 1):**

| ID | Setup | ALC | Curve B0..B31 | Note |
|---|---|---|---|---|
| F-l1 | no emb, v2 HP: online0 / name_t | 0.1913 / 0.1910 | .250 .210 .196 .183 .168 .149 | |
| F-l2 | no emb, v3 HP: online0 / name_t | 0.1885 / 0.1883 | .250 .208 .195 .181 .162 .143 | |
| F-l3 | no emb, v3 HP, n_cap 1 | 0.1886 | | n_cap is irrelevant without repeats |
| F-l4 | no emb, v4 HP (s_v 1, n_cap 1) | **0.1878** | .250 .208 .194 .179 .161 .143 | |
| F-l5 | v4 HP + y term (4 settings) | 0.1878–0.1880 | | item-discrimination residual: neutral (only 2–3 other-subject labels per item) |
| F-l6 | v4 HP, coverage acquisition β = 0.3 / **1** / 3 / −0.3 | 0.1876 / **0.1869** / 0.1871 / 0.1884 | β=1: .250 .208 .193 .178 .159 .144 | **coverage helps (−0.0009)**; preferring shared items hurts; B31 slightly worse |
| F002Q | Kaggle, Qwen emb, v3 config (online0_propx@g1) | 0.1879 | .250 .208 .194 .179 .162 .143 | emp_mean 0.2416 |
| F003Q | Kaggle, Qwen emb, v4 config (online0_propf@g1, both tokens) | **0.1861** | .250 .207 .192 .176 .159 .142 | formative sim **also prefers v4** (−0.0018) |
| F004Q | Kaggle, Qwen emb, v4 config + coverage β=1 / β=2 faded | **0.1851** / 0.1853 | β=1: .250 .207 .191 .175 .157 .143 | **coverage −0.0010 on the full model**; fading not needed |
| S-l1 | standard sim, no emb, v4 HP: @g1 / c1 / c2f | 0.1980 / **0.1976** / 0.1976 | | coverage also helps with low overlap (−0.0004) |
| S004Q | Kaggle, standard sim, Qwen emb, v4 config: @g1 / c1 / c2f | 0.1949 / **0.1947** / 0.1947 | c1: .250 .208 .196 .186 .176 .166 | coverage −0.0002 on the full model with low overlap |

- **Noise analysis (formative, v3 vs. v4 HP, 891 pairs):** per-pair SD of ALC is 0.063, and the SD of per-benchmark mean ALC is 0.049 (range 0.05–0.26, median 0.20). The per-pair SD of a config *difference* is 0.006, and per benchmark 0.0035.
  - If Codabench **draws a new ≤ 1000-row sample per run** (1–2 benchmarks), its absolute score has SE ≈ 0.03–0.05. Then v1/v2/v3/v4 (0.198/0.195/0.170/0.186) are **within noise of each other**.
  - If the **sample is fixed**, config differences have SE ≈ 0.002–0.004. Then the v3–v4 gap (0.016) is real and all simulators miss something.
  - **D0 (identical v3) decides which.**
- Ported to `submission/labeling.py`: config `acq.beta` and `acq.fade`, with incremental item counts over `labeled`. Packages `dist/paiec_v6.zip` (v5 + β = 1) and `dist/paiec_v3c.zip` (v3 + β = 1) built; checker OK.
- Coverage-aware acquisition: `acq.py accept(..., cover)` scales the uncertainty score by exp(−β·min(n_item, 3)), where n_item counts labels already revealed on that item by any subject. Selected with the `@g1c{β}` suffix in `evaluate.py`.

### Codabench resubmissions (11-03): formative scores are resampled noise
- Resubmitting identical packages gave **v3: 0.1697 → 0.1920** and **v4: 0.1862 → 0.1853**.
- The competition page confirms the cause. Quotes: "Each formative evaluation uses a newly sampled subset of the hidden test data"; "The final evaluation uses a common test subset for all teams"; reproducibility checks allow for "expected variation from sampling, randomized algorithms, and API outputs".
- All six formative scores so far (0.1978, 0.1948, 0.1697, 0.1862, 0.1920, 0.1853) have mean **0.1876** and SD **0.0100**. That is consistent with pure sampling noise around about 0.188, which is also what the formative simulator predicts for these configs (0.186–0.188).
- **Consequences:**
  1. A single formative score cannot rank configs that differ by < 0.02. The v2 → v3 "jump" and the v3 > v4 "reversal" were noise.
  2. The diagnostic packages D0–D4 are cancelled (each would need about 25+ runs per arm to resolve a 0.003 effect).
  3. Codabench is now used only as a crash, runtime and sanity check.
  4. Decisions come from the large paired simulators (standard + formative, ~900–1700 pairs, where the SE of a difference is about 0.0002–0.0005) and the large dict-API A/B test.
  5. The "simulators mispredict Codabench" conclusion above is withdrawn: they were never contradicted beyond noise.
- Repeats: "a subject that attempted an item three times contributes three separate targets", and repeated identical inputs reuse one prediction. So n > 1 cells exist in evaluation, as in our cell Brier.

### Codabench round 3 (11-03/04): 13 runs, all consistent with one level plus sampling noise
| Package | Runs |
|---|---|
| v1 | 0.1978 |
| v2 | 0.1948 |
| v3 | 0.1697, 0.1920, 0.1763, 0.1999 (D0 = identical v3) |
| v4 | 0.1862, 0.1853 |
| D1 v3 + tokens | 0.2044 |
| D2 v3 n_cap 1 | 0.1864 |
| D3 v3 + name prior | 0.1807 |
| D4 v3 − x term | 0.1458 |
| v6 (v5 + coverage) | 0.2086 |

- All 13 runs: mean **0.1868**, SD **0.0166**. The four identical v3 runs alone: mean 0.1845, SD 0.0139, range 0.170–0.200.
- No package differs from the pooled mean by more than 2.5 SE of a single run. D4 (z = −2.5) is the most extreme of 9 groups, which is expected by chance with this many comparisons. The simulators say the x term is worth about +0.0007, not −0.04.
- Implied sample breadth: the per-benchmark SD in the formative simulator is 0.049, so (0.049 / 0.0166)² ≈ **9 benchmarks per formative sample**.
- Detecting a 0.005 difference at 95% would need about 85 runs per arm. **Codabench cannot rank our variants**; it only confirms that packages run and land in the expected range. v6 ran successfully with the encoder.
- The large dict-API A/B kernels (`paiec-subtest-big-v3/-d1`) hit Kaggle's 12-hour limit and were cancelled without output. The wrapper only printed at the end. Not rerun: the paired simulators already have about 4× the pairs, and Codabench confirms the real-API path works.
- **Final choice rule:** choose by paired simulator evidence (both regimes agree). v6 = full stack + coverage is best in both: formative 0.1851 vs. v3-config 0.1879; standard 0.1947 vs. 0.1964.

### Public Codabench leaderboard snapshot (2026-11-04)
Saved to `logbooks/runs/codabench_leaderboard_20261104.json`, from the public API: `api/phases/29785/get_leaderboard/`, titled "Provisional development scores — not oral or poster selections".
- Competition: 253 participants, 6,345 submissions, 79 leaderboard entries. Phase ends 2026-11-07 11:59 UTC; limit 50 submissions/day, 1,000 total.
- Brier ALC quantiles 0 / 10 / 25 / 50 / 75 / 90 / 100%: 0.117 / 0.136 / 0.151 / 0.172 / 0.187 / 0.210 / 0.255. The organizer entry "AIMS Foundations" scores 0.1801.
- **The leaderboard shows one (best or selected) formative run per user**, and every run is a fresh noisy sample (SD ≈ 0.017). Expected best-of-N for our mean 0.187: N = 1 → 0.187, 5 → 0.168, 13 → 0.159, 25 → 0.154, 50 → 0.150, 100 → 0.145, 500 → 0.136. Our actual best of 13 is 0.146, which would rank 16th of 80.
- **Reading:**
  1. Our mean (0.187) is a fair estimate of the final score scale. The leaderboard values are optimistic by 0.02–0.05 for heavy submitters.
  2. The top entries (0.117–0.13) would need a mean of about 0.165 or lower even with 500 tries. So some teams are probably genuinely better than us by about 0.02 or more. The page allows LLM APIs at scoring time (participants' own keys), and the organizer BLE baseline is LLM-based, so LLM priors are the likely source.
  3. Our mean is close to the "AIMS Foundations" entry (0.180, number of runs unknown).

### Cross-subject pooling ablation (11-04, local, no embeddings, v5 random-effect HP, γ = 1)
| Variant | Formative ALC | Standard ALC |
|---|---|---|
| full pooling (benchmark u, item w shared across subjects) | **0.1878** | **0.1980** |
| no item sharing (s_w → 0.001) | 0.2022 | 0.2026 |
| own labels only (s_u, s_w → 0.001) | 0.2101 | 0.2112 |

Using other subjects' labels (which the protocol reveals) is the **largest single component**: −0.022 formative, −0.013 standard. Item sharing accounts for most of it. Files: `logbooks/runs/eval_L{F,S}_{pool,own,noitem}.json` and the matching `pairs_*.npz`.

### Final ablation for the paper (11-04, Kaggle VN1, e5, v5 HP + β, `--tok both --template`): A1 standard (1,738 pairs), A2 formative-like (891 pairs)
Δ is the paired difference to the final system (online0_propf@g1c1 = v6), ± paired SE.

| Method | A1 ALC | Δ ± SE | A2 ALC | Δ ± SE |
|---|---|---|---|---|
| const | 0.2500 | +0.0558 ± .0016 | 0.2500 | +0.0653 ± .0022 |
| emp_mean | 0.2438 | +0.0497 ± .0014 | 0.2416 | +0.0570 ± .0021 |
| online0 (random acq.) | 0.1988 | +0.0046 ± .0005 | 0.1895 | +0.0048 ± .0006 |
| + uncertainty acq. | 0.1980 | +0.0039 ± .0003 | 0.1878 | +0.0031 ± .0004 |
| + propagation | 0.1958 | +0.0017 ± .0002 | 0.1872 | +0.0025 ± .0004 |
| + tokens, s_g | 0.1941 | −0.0001 ± .0001 | 0.1857 | +0.0011 ± .0003 |
| + coverage = **final v6** | **0.1942** | — | **0.1847** | — |
| − propagation | 0.1940 | −0.0002 ± .0002 | 0.1846 | −0.0001 ± .0001 |
| − tokens | 0.1960 | +0.0019 ± .0002 | 0.1862 | +0.0015 ± .0002 |
| − active acquisition | 0.1949 | +0.0008 ± .0004 | 0.1870 | +0.0023 ± .0006 |
| + x residual | 0.1941 | −0.0001 ± .0001 | 0.1846 | −0.0001 ± .0000 |

**Reading:**
- Coverage helps only with item overlap (A2 −0.0010; A1 neutral).
- Propagation is strong without tokens (A1 −0.0022) but redundant once tokens are present.
- v6 is best in A2 and tied best in A1.
- Paper figures and tables are generated from `logbooks/runs/kaggle/a{1,2}/pairs_*.npz`.

### Round 5 (11-04): generalisation-oriented improvements
- **Name effect n_{name,b}** (`s_n`, configurations of one model share benchmark ability): formative 0.1845/0.1846/0.1848 for s_n 0.3/0.6/1.0 vs. 0.1846; standard (s_n 0.3) 0.1940 vs. 0.1940. **Neutral.** Same-model subjects rarely co-occur in the sampled pairs.
- **Name-based ability prior, split by how widely the model is evaluated** (`paiec/name_subgroup.py`, online_name_f@g1c1 vs. online0_f@g1c1, paired per pair):

| Train benchmarks containing the model name | Formative Δ (n) | Standard Δ (n) |
|---|---|---|
| 0 (unnamed or unseen) | 0.0000 (464) | 0.0000 (943) |
| 1–4 | −0.0009 ± .0004 (98) | 0.0000 ± .0004 (219) |
| 5–19 | −0.0006 ± .0003 (265) | −0.0005 ± .0004 (454) |
| 20+ | −0.0003 ± .0004 (64) | 0.0000 ± .0005 (122) |
| **all named** | **−0.0006 ± .0002 (427)** | −0.0003 ± .0003 (795) |

  On aggregate the name prior looked neutral, but that average was diluted by the many unnamed checkpoint subjects in the public data. For named models it gives a small, consistent gain and never hurts on average. Hidden subjects are likely named API models, so **adopt the name prior in the final package** (more general). Name pooling of t stays off.
- **Robustness to pool size** (formative-like, v5 HP, local):

| Regime | online0@g1 | + tokens (online0_f@g1) | + coverage β=1 | β=2 |
|---|---|---|---|---|
| K = 50 items × 20 subjects | 0.1754 | **0.1734** | 0.1734 | 0.1734 |
| K = 100 × 10 | 0.1878 | 0.1857* | **0.1846** | |
| K = 200 × 5 | 0.2009 | 0.1976 | **0.1974** | 0.1974 |

  (*K = 100 taken from A2, with embeddings.) Tokens help in every regime (−0.002 to −0.003). Coverage is never harmful. At K = 50, each pair's acquisition pool (25 items) is smaller than the budget, so every item is taken and acquisition is irrelevant. The v6 design choices are robust across pool sizes; β = 1 vs. 2 makes no difference.
- Ensemble bug fixed (`_online` for the hp2 model); rerunning.

### Round 6 (11-06): sparse hidden-like regime, name-prior strength, value of LLM difficulty ratings
- **v7** = v6 + name prior (known names only; `provider_prior: false`, no name pooling). Simulator formative 0.1842 vs. 0.1846. Dict-API, same 19 pairs (seed 5): **0.1786 vs. 0.1814**. Package `dist/paiec_v7.zip`, checker OK.
- tune_full (202 configs, both regimes): best average 0.1894 vs. v6 0.1898. This is within best-of-200 selection noise, so the HP are unchanged. The 2-config ensemble (+e) is worse: 0.18486 vs. 0.1846.
- Name-prior strength a (online_name_f@g1c1): formative 0.1843–0.1844 for a = 0.2–1.0; standard 0.1939 for a ≤ 0.6, then 0.1943–0.1944 for a = 0.8–1.0. Keep a = 0.4.
- **Sparse regime** (`--formative K --rows R`; one or two pairs per benchmark, as competitors report for hidden draws):

| Regime | emp_mean | online0@g1 | + tokens | + coverage | + name prior (v7) |
|---|---|---|---|---|---|
| K=100, 1 pair/bench | 0.2476 | 0.2144 | 0.2122 | 0.2122 | 0.2121 |
| K=100, 2 pairs/bench | 0.2345 | 0.1999 | 0.1962 | 0.1962 | **0.1958** |
| K=200, 1 pair/bench | 0.2425 | 0.2127 | 0.2100 | 0.2100 | 0.2102 |
| K=200, 2 pairs/bench | 0.2356 | 0.2022 | 0.1989 | 0.1989 | **0.1985** |

  Tokens help everywhere; coverage is neutral (no partners to share with); the name prior is neutral or slightly positive. Isolated pairs score about 0.21, so hidden draws with mostly isolated pairs are harder than our formative-like regime.
- **Value of an item-difficulty rating** (`paiec/llm_value.py`): simulated ratings correlated ρ with true within-benchmark easiness are used as the item prior mean. This first run used a = 1 on μ, which inflates B0 (0.283), so read the differences from ρ = 0:
  - sparse 1 pair/bench: ρ 0 / 0.3 / 0.5 / 0.7 → 0.2193 / 0.2122 / 0.1996 / 0.1811
  - formative: 0.1876 / 0.1837 / 0.1769 / 0.1669

  At ρ ≥ 0.5 the gain would be −0.01 to −0.03, the largest lever left.
- **Real LLM ratings** (`paiec/llm_rate.py`, NVIDIA API, 450 items over 9 public benchmarks with ≥15 subjects). Within-benchmark Spearman(rating, difficulty):
  - nemotron-3.5-bolt-e4b (4B, shippable): mean **0.06**. Useless.
  - nemotron-3-ultra-550b: pooled within-benchmark Spearman **−0.001** on 72 rated items.
  - mistral-large-4-1t: 0.41 on only 8 items (not interpretable).
  - The free NVIDIA tier is heavily rate-limited: about 80 of 450 calls succeed per 20 minutes.
  - Sampled benchmarks: afrieval, agi_elo, alignment_faking, bridging_gap, cruxeval, editbench, legalbench, mmbench_v11, sib200. Several are not "difficulty" tasks (RL checkpoints, classifiers, image items without the image), so text ratings carry no signal there. Competitor Claude ratings reached 0.28–0.82, but only on 4 math/code benchmarks.
  - **Conclusion:** without an approved-provider key the ratings cannot be deployed. The open models we can reach do not rate well on this heterogeneous mix. v7 stays final.
- The NVIDIA API is not an approved scoring-time provider, so it is used only for these offline studies.

### Round 7–8 (11-06/07): codex-review experiments (one-response reveals, cheap baselines, overlap dose-response, gate)
All runs reveal ONE binary response per acquisition (Bernoulli(k/n), seed 910+fold), are text-free proxies, use --no_provider, and share identical pairs, streams and draws. Intervals: benchmark bootstrap, 10,000 draws. Source: `logbooks/runs/r7_summary.json`, generated by `paper/scripts/analyze_r7.py`.

| Method | Standard | High overlap | Sparse 2 sys. | Sparse 1 sys. |
|---|---|---|---|---|
| Empirical Mean | 0.2455 | 0.2434 | 0.2383 | 0.2472 |
| Beta(1,1) pair mean | 0.2119 | 0.2105 | 0.2067 | 0.2140 |
| pair mean shrunk to benchmark rate | 0.2077 | 0.2068 | 0.2070 | 0.2178 |
| hierarchical, random acq. | 0.1991 | 0.1901 | 0.2019 | 0.2143 |
| + uncertainty acq. | 0.1984 | 0.1884 | 0.2006 | 0.2146 |
| + tokens | 0.1950 | 0.1862 | 0.1970 | 0.2125 |
| tokens, random acq. | 0.1957 | 0.1877 | 0.1988 | 0.2118 |
| + coverage (v6) | 0.1945 | 0.1853 | 0.1969 | 0.2125 |
| + known-name prior (**v7**) | 0.1945 | 0.1850 | 0.1965 | 0.2123 |

| Contrast (Δ ALC [95% CI]) | Standard | High overlap | Sparse 2 | Sparse 1 |
|---|---|---|---|---|
| Unc | -0.0007 [-0.0015, +0.0002] | -0.0017 [-0.0028, -0.0007] | -0.0014 [-0.0034, +0.0007] | +0.0002 [-0.0019, +0.0023] |
| Tokens | -0.0034 [-0.0046, -0.0023] | -0.0022 [-0.0031, -0.0013] | -0.0035 [-0.0051, -0.0021] | -0.0021 [-0.0039, -0.0004] |
| Cov | -0.0005 [-0.0008, -0.0002] | -0.0009 [-0.0018, -0.0001] | -0.0001 [-0.0005, +0.0003] | +0.0000 [+0.0000, +0.0000] |
| Name | +0.0000 [-0.0003, +0.0004] | -0.0003 [-0.0006, -0.0000] | -0.0004 [-0.0010, +0.0002] | -0.0002 [-0.0008, +0.0004] |
| NameKnown | -0.0001 [-0.0008, +0.0005] | -0.0007 [-0.0014, -0.0002] | -0.0007 [-0.0019, +0.0006] | -0.0004 [-0.0016, +0.0008] |
| Act | -0.0013 [-0.0022, -0.0003] | -0.0024 [-0.0037, -0.0011] | -0.0018 [-0.0043, +0.0006] | +0.0007 [-0.0019, +0.0033] |
| Gate | -0.0003 [-0.0007, +0.0001] | +0.0002 [-0.0003, +0.0007] | +0.0013 [-0.0002, +0.0028] | -0.0007 [-0.0034, +0.0019] |
| VsBeta | -0.0174 [-0.0198, -0.0153] | -0.0255 [-0.0294, -0.0216] | -0.0102 [-0.0133, -0.0071] | -0.0017 [-0.0047, +0.0013] |

**Overlap dose–response** (`paiec/overlap.py`, 65 benchmarks where 10 systems share 100 items; focal pair fixed; random acquisition):
- 1 systems, realized overlap 0%: full sharing 0.2124, own labels 0.2134, gain 0.0010 [0.0002, 0.0017]
- 2 systems, realized overlap 31%: full sharing 0.2065, own labels 0.2134, gain 0.0069 [0.0036, 0.0102]
- 4 systems, realized overlap 67%: full sharing 0.2007, own labels 0.2134, gain 0.0126 [0.0078, 0.0173]
- 10 systems, realized overlap 96%: full sharing 0.1889, own labels 0.2134, gain 0.0244 [0.0179, 0.0311]

**Reading:**
- The cross-system benefit scales with overlap: 0 → 0.0244 ALC.
- For isolated pairs, v7 is tied with the Beta pair mean, and active acquisition does not help (+0.0007, CI includes 0).
- Tokens help in every regime. Coverage helps only with partners.
- The known-name prior is significant for known names under high overlap (−0.0007 [−0.0014, −0.0002]) and is exactly neutral for unknown names.
- The overlap gate (`@g1c1o`) is neutral in all regimes (every CI includes 0), so it is not adopted. **v7 stays final.**
- The whole-cell vs. one-response mismatch moves the Empirical Mean by 0.002–0.003, larger than several component effects.
- Paper rewritten per `report/codex_review.md`: new title, mechanism-first structure, Figure 1 redrawn as an evidence matrix, an overlap figure, all numbers from `paper/results.tex`, and no placeholders.

### Round 9 (11-07): codex review round 2 — repaired overlap study, final paper edits
- `paiec/overlap.py` rewritten. The base logit is fitted on the other benchmark folds (held out). Predictors are the SAME model given (a) all labels, (b) all labels with item effects suppressed, and (c) only the focal pair's labels. Random acquisition, one-response draws, 65 benchmarks. Two selection seeds.
- Seed 0: partner-evidence gain 0.0000 (1 system), 0.0062 (2, coverage 31%), 0.0117 [0.0071, 0.0162] (4, 67%), **0.0234 [0.0172, 0.0297]** (10, 96%). Of this, shared item effects: 0.0014, 0.0042, **0.0131 [0.0075, 0.0187]**. Seed 1 replicate at 10 systems: 0.0251 [0.0194, 0.0310], item 0.0147 [0.0101, 0.0195]. Outputs: `logbooks/runs/overlap_R9_s{0,1}.json`.
- Paper changes:
  - Abstract follows the review: embedding-free proxy wording, Beta and Empirical Mean improvements, partner-evidence and item-effect split.
  - Results lead with the controlled study.
  - Strongest-baseline comparison added: v6 proxy vs. benchmark-shrunk mean, 0.0133 standard and 0.0215 high overlap.
  - Sparse-one result marked inconclusive. v7 is exploratory; the gate is tested on the v7 proxy.
  - Corrected: unseen-name effect, noise calculation (30 runs per mean, 60 per package for a difference).
  - PAIEC entries and IRT-reliability work cited.
  - `MANIFEST.md` added; checklist removed; title provisionally from the review's options.

## Codabench (official formative feedback)

| Submission ID | Package | Date | Brier ALC (hidden) | Local estimate | Gap |
|---|---|---|---|---|---|
| 1028212 | `paiec_v1.zip` (no text) | 2026-11-02 21:45 | **0.197811** | 0.1995 (E031 online_name_t@g1) | −0.0017 |
| 1028213 | `paiec_v2.zip` (+ e5 text propagation) | 2026-11-02 21:45 | **0.194774** | ≈ 0.1983 (E041 online0_prop@g1) | −0.0035 |
| 1028569 | `paiec_v3.zip` (tuned prop + residual x, const prior, n_cap 4) | 2026-11-03 01:01 | **0.169708** | 0.1964 (E061) | **−0.0267** |
| 1028570 | `paiec_v4.zip` (BEST HP n_cap 1 + tokens) | 2026-11-03 01:01 | 0.186192 | 0.1946 (E092) | −0.0084 |

**Analysis of v3/v4 (11-03):**
- v3 is far better than v2 (−0.025) and than v4 (−0.0165). The original-subject simulator predicted small changes and the wrong v3/v4 order. The dict-API smoke test (68 pairs) predicted v3 > v4 correctly.
- **The simulator ran in the wrong regime.** The competition defines subjects as (model, harness, harness_version, reasoning_effort). Re-keying the training data this way merges 14,518 subjects into 1,932, raises cells with ≥ 5 responses from 0.4% to 12.2%, and raises the 99th-percentile n from 3 to 49. `data.load_cells(merged=True)` and `evaluate.py --merged` now simulate this.
- n_cap ordering (v4 1 < v2 2 < v3 4) matches the Codabench ordering. In the merged simulator, n_cap alone moves ALC only 0.0005 without text (M001: 1 → 0.2039, 4 → 0.2034, 1000 → 0.2036), so it is not the whole story.
- The formative set is small (≤ 1,000 subject–item pairs, probably 10–20 pairs), so a few benchmarks dominate its score.
- **Merged-simulator check (M010–M012, 5-fold, e5):** v2 config 0.2032, v3 0.2019, v4 0.2008, v3+tokens 0.1991. The merged simulator **also fails** to reproduce Codabench (it still prefers tokens and v4). Also, the hidden input carries `subject_features_extra`, so hidden subjects may not be merged at all. The subject-definition hypothesis is unconfirmed.
- **Working conclusion:** the simulators are good for coarse effects (v1 → v2 direction, online vs. baseline) but not for these second-order choices. **v3 is the base**, and additions must win on real-API evidence: Codabench diagnostics D0–D4 (`dist/diag/`) and a large dict-API A/B test (30 benchmarks × 8 subjects, seed 21: `paiec-subtest-big-v3` vs. `-d1`).

**Takeaways:** (1) the local simulator is a reliable proxy: its estimates are within 0.002–0.004 and slightly pessimistic; (2) text propagation helps more on the hidden set than locally (−0.0030 vs. −0.0012); (3) v2 is the current best submission.

## Submission packages
| Package | Contents | Check |
|---|---|---|
| `dist/paiec_v0.zip` | online model (default HP) + name prior, random acquisition, no text | official `check_submission_zip.py`: OK. End-to-end dict-API test (`paiec/test_submission.py`, 24 pairs): runs, valid probabilities, curve 0.246 → 0.183 |
| **`dist/paiec_v1.zip`** | online model (T001 best HP) + name prior + name-pooled subject offset + **uncertainty acquisition γ=1**, no text, no external model (48 KB) | checker OK; end-to-end with live acquisition: ALC 0.1947 on 24 pairs, 157 s. **First Codabench test candidate.** |
| **`dist/paiec_v5.zip`** | v4 + template tokens + subject×token effects (s_g 0.3); the E093 config (local **0.1941**) | checker OK; Kaggle end-to-end (24 pairs): 0.1948, 93 s |
| **`dist/paiec_v4.zip`** | BEST HP (E060) + **item_features/interactors tokens (s_f 0.7) + derived text tokens** + text propagation + constant prior, γ=1, no x term; the E092 `online0_propf@g1` config (local **0.1946**) | checker OK; Kaggle end-to-end (24 pairs, 4 benchmarks): 0.1952, 79 s; interactors now in item key and tokens |
| **`dist/paiec_v3.zip`** | v2 + tuned propagation (τ_w 0.02, κ_w 1, s_w 2.2) + subject-residual term (x_scale 1, τ_x 0.03, κ_x 10) + constant prior (no name prior/pooling), γ=1; exactly the E061 `online0_propx@g1` config (local 0.1964) | checker OK; Kaggle end-to-end (same 24 pairs): **0.1916** vs. v2 0.1932, v1 0.1947 |
| **`dist/paiec_v2.zip`** | v1 + **test-time text propagation** with `intfloat/multilingual-e5-small` declared in `models.txt` (falls back to v1 behaviour if the encoder cannot load) | checker OK; end-to-end on Kaggle T4 (kernel `paiec-subtest-v2`, same 24 pairs): **ALC 0.1932 vs. 0.1947 for v1**, 68 s; encoder loaded |

## Next experiments
1. E040: kNN text prior (e5-small) ± test-time propagation ± name pooling ± acquisition.
2. E041: the same with Qwen3-Embedding-0.6B (encoder comparison).
3. Tune kNN (k, τ, κ) and propagation (τ_w, κ_w).
4. Subject × item-cluster residual (subject-specific propagation), the Riiid-style "user × question-cluster accuracy" idea.
5. Final package: best config, text encoder via `models.txt`, smoke test, checker.
