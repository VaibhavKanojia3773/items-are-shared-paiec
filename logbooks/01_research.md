# Logbook 01: Research notes and design rationale

Started 2026-11-02. Records what we read, what it implies for PAIEC, and which ideas we adopted or rejected.

## 1. Facts about the competition that drive the design

Source: competition page (aims.stanford.edu/competition), `paiec_baseline` repo (`tools/streaming_ingestion.py`, `tools/prepare_data.py`).

| Fact | Source | Design consequence |
|---|---|---|
| Score = Brier ALC, weights 0.1/0.2/0.2/0.2/0.2/0.1 at budgets 0/1/3/7/15/31 | page | 90% of the weight is at budgets ≥ 1, so the adaptive update matters more than pure budget-0 prediction. Budget 0 is still 10% and anchors everything. |
| Each subject–benchmark pair gets equal weight | page | Small benchmarks and rare subjects count as much as large ones. Validate with per-pair averaging. |
| `labeled` spans **all sampled pairs**, not just the target pair | page (verbatim: "across sampled subject–benchmark pairs") | **At test time we see other subjects' labels on the same benchmark.** Benchmark base rate and item difficulty can be estimated online, which is test-time IRT / collaborative filtering. |
| Acquisition/evaluation split is **per pair**, 50/50, pairs need ≥ 80 items; all responses for an item stay together | page | An item in subject A's eval pool can be in subject B's acquisition pool, so **its label from B may be revealed**. Item difficulty for eval items can be learned from other subjects. |
| Benchmarks are randomly assigned to train/test pools; `benchmark_id` is anonymous | page | Budget 0 = cold start on an unseen benchmark. Benchmark identity can't be looked up. |
| Subjects are identified by attributes (`normalized_name`, provider, harness, reasoning effort, dates) | page | Subjects **likely overlap** with training subjects (same models on other benchmarks), but "coverage may differ between the pools". A training-derived subject ability prior is very valuable when it matches, and needs a fallback from metadata when it doesn't. |
| Streaming acquisition: items arrive one at a time and we answer yes/no; `context` has `labels_remaining`, `items_remaining` | `streaming_ingestion.py` | An online (secretary-style) selection rule. Always spend the full budget: accept everything once `items_remaining ≤ labels_remaining`. |
| Prediction workers are **recreated with frozen evidence at each checkpoint** | `streaming_ingestion.py` docstring | No state carries between checkpoints. Any fit on `labeled` must be recomputed per checkpoint. A module-level cache keyed on a hash of `labeled` is fine within one checkpoint. |
| Eligible benchmarks: `response_type == "binary"` and `granularity == "item"` | `prepare_data.py` | Filter training data the same way. |
| Runtime 8 h incl. setup; ≤ 5 HF model repos via `models.txt`; network only to listed LLM APIs; extra pip installs gated | page | Inference must be cheap. Any text encoder must be bundled or prefetched, with a pure-numpy fallback if `torch` or `sentence-transformers` is missing. |
| Public precomputed item embeddings exist (`aims-foundations/measurement-db-embed`, OpenAI `text-embedding-3-small`, 1536-d) | `prepare_data.py` | Useful only if the OpenAI embedding API can be called at scoring time (we have no key). **Rejected for now.** We embed with an open model that we can also run at scoring time. |
| Deadline | page says 30 Oct, extended to 6 Nov 2026 AoE; one AIMS news page says the phase is open until 20 Nov 2026 | Conflicting. Plan for 6 Nov, and check Codabench. |

## 2. Literature: what is relevant and what we take from it

### IRT and amortized difficulty (the core modelling family)
- **Truong et al., "Reliable and Efficient Amortized Model-based Evaluation", ICML 2025** (same Stanford group as the organizers). Rasch/IRT separates item difficulty from model ability, and *amortized calibration* predicts item difficulty from item content. Over 22 benchmarks and 172–183 LMs, adaptive testing roughly halves the query count. **Adopted**: logistic ability-minus-difficulty structure, plus a content-based difficulty prior for unseen items. https://arxiv.org/abs/2503.13335
- **tinyBenchmarks (Polo et al., 2024)**: about 100 IRT-selected items estimate MMLU accuracy. Supports the idea that a few well-chosen labels pin down ability. https://arxiv.org/abs/2402.14992
- **Fluid Benchmarking (AI2, COLM 2025)**: 2PL IRT plus computerized adaptive testing, picking items by Fisher information. Beats random sampling on validity and variance. **Adopted for acquisition**: Fisher information p(1−p) under the current posterior. https://arxiv.org/abs/2509.11106
- **Krumdick et al., "Cost-Efficient Estimation of General Abilities Across Benchmarks", arXiv 2604.01418 (Apr 2026)**: multidimensional IRT with optimal-design item selection predicts held-out task performance with < 7% MAE after only 16 items. Multidimensional IRT beats unidimensional IRT on held-out tasks. **Adopted**: a low-rank (multidimensional) subject ability, with the benchmark's position in ability space inferred from item content. https://arxiv.org/abs/2604.01418
- **"Beyond Scores: Diagnostic LLM Evaluation via Fine-Grained Abilities", arXiv 2604.12191 (2026)**: MIRT with neutral defaults for unseen items reaches within-benchmark AUC > 0.81 and cross-benchmark AUC 0.77–0.81. This is a reference point for item-level predictability. https://arxiv.org/abs/2604.12191

### Benchmark-level structure (subject priors)
- **Observational scaling laws (Ruan, Maddison, Hashimoto, NeurIPS 2024)**: about 3 principal components of benchmark scores predict complex capabilities. **Adopted**: low-dimensional subject ability vectors learned from training benchmarks. https://arxiv.org/abs/2405.10938
- **Cai et al., latent-variable scaling laws, arXiv 2512.06553 (Dec 2025)**: a shared low-dimensional ability vector with prediction intervals. Supports Bayesian uncertainty on abilities.
- **BenchPress (2026, secondary summary)**: logit-space ALS matrix completion; rank 2 explains > 90% of variance in benchmark scores. Suggests a very low rank is enough at benchmark level.
- **Zhao et al., "Can We Predict Performance of Large Models across Vision-Language Tasks?", ICML 2025**: probabilistic matrix factorization with uncertainty-driven acquisition. Warns that PMF collapses to the mean with few observations. **Lesson**: priors must come from side information (metadata and text), not only from factors.
- **Ruan et al. 2023 (BIG-bench predictability)**: GBMs and MLPs on (model, task) features predict scores well (R² > 0.95). **Adopted**: a GBM "assessor" as the budget-0 prior.

### Instance-level assessors
- **PredictaBoard (Pacchiardi et al., ACL Findings 2025)**: assessors such as logistic regression and XGBoost on prompt embeddings predict per-instance success. Out-of-distribution prediction is hard. **Lesson**: expect weak budget-0 discrimination on unseen benchmarks, so calibration matters more than ranking. https://arxiv.org/abs/2502.14445
- **Item-difficulty-from-text (BEA/AIME-CON 2026, SMART 2025)**: text features predict difficulty moderately (R² ≈ 0.5 in-domain). LLM embeddings give modest gains over simpler features. Single train/test splits overestimate accuracy. **Lesson**: use group (benchmark-level) cross-validation only.

### Competition writeups with the same structure
- **Riiid Answer Correctness Prediction (Kaggle 2020–21, 3,395 teams)** is the closest Kaggle analogue: predict P(correct) for a (student, question) pair given that student's history. Winners used transformers over the interaction history (1st: Last Query Transformer RNN, arXiv 2102.05038; 2nd: LSTM-SAKT, arXiv 2102.00845). Strong GBM solutions relied on **target-encoded question difficulty, user running accuracy and the user×question-cluster accuracy**. **Adopted**: explicit "running accuracy" features at test time (subject on benchmark, item across subjects, subject on similar items) inside a principled shrinkage model. Not adopted: sequence transformers, because we have ≤ 31 labels per pair and no long histories.
- **NeurIPS 2020 Education Challenge (Eedi)**: Task 1 (correctness prediction) was won by feature engineering plus GBM ensembles. Task 4 was personalized question selection, which is the analogue of our acquisition function. Overview: arXiv 2104.04034.
- **Active testing / Active Surrogate Estimators (Kossen et al., NeurIPS 2022)**: label-efficient evaluation with a surrogate model. The XWED acquisition targets the final estimation task, not raw uncertainty. **Lesson**: acquisition should target the *evaluation loss on the eval pool*. Fisher information for the subject's ability is a good proxy.

## 3. The gaps we see (where we expect to beat the baselines)

1. **The baselines ignore cross-pair labels.** The Empirical Mean only uses the target pair's own labels and returns 0.5 at budget 0. Because `labeled` spans all pairs, a model that pools labels across subjects (benchmark base rate, item difficulty) and across benchmarks (subject ability) should beat it by a wide margin at every budget.
2. **Budget 0 should not be 0.5.** Subject ability from training data plus a text-based benchmark/item difficulty prior gives a calibrated prior. Even a well-calibrated benchmark-level guess cuts Brier below 0.25.
3. **Shrinkage rather than raw means.** With 1–3 labels the empirical mean is 0 or 1, which gives catastrophic Brier scores. Bayesian shrinkage toward the prior fixes the low-budget region, which carries 40% of the weight (budgets 1 and 3).
4. **Item-level transfer.** Eval items may have labels from other subjects. A 1PL/2PL online fit recovers their difficulty directly, and unseen items borrow from embedding neighbours.
5. **Acquisition.** Random acquisition wastes labels on items whose outcome is already near-certain. Fisher-information acquisition plus a streaming threshold should help most at budgets 1–7.
6. **The BLE baseline needs a paid LLM API per prediction.** It is expensive and slow, and its calibration is unknown. Our approach is API-free, which also makes it reproducible (graded manually).

## 4. Chosen architecture (v1 plan)

```
logit p(s, i | L) = prior_logit(s, i)            # offline assessor, budget-0 prediction
                  + u_b                           # benchmark offset, learned online from all labels on b
                  + v_{s,b}                       # subject×benchmark offset, from s's labels on b
                  − w_i                           # item offset, from other subjects' labels on i (+ embedding neighbours)
```
- `prior_logit` comes from an offline model trained on public data: subject ability features (subject name/provider/date/harness/effort, plus a learned per-subject global ability where the name matches a training subject) × item features (text embedding, item_features, benchmark-level mean embedding).
- `u, v, w` are Gaussian random effects with variances fitted by empirical Bayes on simulated test episodes, and a MAP/Laplace fit at each checkpoint (a few Newton steps, which is cheap).
- Acquisition: accept when Fisher information p(1−p) under the current posterior exceeds an adaptive threshold set by `labels_remaining / items_remaining`; always spend the full budget.
- Final temperature/calibration step tuned on simulated episodes.

Validation: an exact simulator of the streaming protocol on public benchmarks with **benchmark-level holdout** (GroupKFold by benchmark), per-pair equal weighting and the Brier-ALC weights. Everything is tuned against this number.
