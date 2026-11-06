# PAIEC: Task Analysis and Feasibility

Written for: you and your team, to decide what to build in the time left.
Sources: the competition page (fetched from the Stanford AIMS site, which redirects to the NeurIPS 2026 PAIEC page), the `measurement-db` Hugging Face dataset page, and the `paiec_baseline` GitHub repo. Your `info.md` is empty, so this is based on the web pages only.

---

## 1. The most important fact: the deadline

| Milestone | Date (AoE) |
|---|---|
| Final codebase + technical report | 30 Oct 2026, 23:59 |
| **Extended deadline** | **6 Nov 2026, 23:59** |
| NeurIPS session (Sydney) | 11 Dec 2026 |

Today is **2 Nov 2026**. The extended deadline is about **4 days away**. The page says submissions need the Codabench submission ID plus a technical report on OpenReview, and the report must be reproducible.

Confirm on Codabench (competition 17828) that submissions are still open. If they are not, the realistic goal becomes a documented entry, not a leaderboard placement.

---

## 2. What the task is

**Predict the probability (0 to 1) that an AI "subject" answers a benchmark item correctly, without running the subject on that item.**

- **Subject** = model + harness + harness version + reasoning effort. The same model under two harnesses is two subjects.
- **Target** = one response (scored 1 or 0). Each subject–item run is a separate target.
- **Cold start**: held-out items come from benchmarks in a private test pool, so the predictor must generalize to unseen benchmarks.

### Interface you must implement

`model.py` (required):
```python
def predict(input: list[dict], labeled: list) -> float
```
- `input[0]` = subject fields: `normalized_name`, `provider`, `release_date`, `access_date`, `harness`, `harness_version`, `reasoning_effort`, `subject_features_extra`
- `input[1]` = item fields: `item_content`, `item_features`, `interactors`, `benchmark_id`
- All values are strings (missing = `""`).
- `labeled` = revealed `[[subject, item], label]` pairs at the current budget (empty at budget 0).
- Return a finite float in [0, 1]. NaN, inf, or out-of-range values are rejected.

`labeling.py` (optional): `acquisition_function(input, prediction, labeled, context) -> bool`
- Returns `True` to reveal this item's label, `False` to skip it permanently.
- `context` gives `labels_remaining` (out of 31), `max_labels`, `items_remaining`, and so on.
- Without it, labels are chosen at random.

Submission = a ZIP with `model.py` at the root (plus optional `labeling.py`, `models.txt`, `requirements.txt`).

### How it is scored

- **Primary metric: Brier ALC** (lower is better). It is the area under the Brier learning curve at label budgets **0, 1, 3, 7, 15, 31** per subject–benchmark pair.
  - Weights: 0.1·B₀ + 0.2·B₁ + 0.2·B₃ + 0.2·B₇ + 0.2·B₁₅ + 0.1·B₃₁
  - Each subject–benchmark pair gets equal weight.
  - Predicting 0.5 always gives 0.25. Perfect is 0.
- ECE is diagnostic only.
- Final score = automatic Brier ALC on a common hidden subset **plus manual grading** of the codebase and technical report (reproduction check included).

So a good predictor needs to be **accurate at budget 0** (features only), **improve quickly with a few labels**, and be **well calibrated**. The adaptive part (which items to reveal) is part of the score, not an extra.

---

## 3. Dataset size

| Item | Size |
|---|---|
| Training data | `aims-foundations/measurement-db` on Hugging Face |
| Recorded observations | **571,921** (from the schema v3 migration page) |
| Benchmarks in the archive | **249** validated benchmark migrations |
| Items / subjects / models counts | **Not stated** on the page |
| Raw source archives (6 benchmarks) | 1,191 files, **10.4 GB** |
| Total repo size listed | **740 GB** |
| Test data | Private, same format, not downloadable |
| Formative feedback | Up to 1,000 subject–item pairs per evaluation |

Notes:
- The HF page's size category says "1M–10M", which does not match the 571,921 figure. Check the real row count after you download the Parquet files.
- **Do not download all 740 GB.** Kaggle notebooks have limited disk (roughly 20 GB, check current limits). You only need the `formatted_tables/` Parquet files (`items.parquet`, `responses.parquet`), and only for the benchmarks you use.
- In memory, 571k rows of responses is small (a few hundred MB). Storage is not the real constraint; download size and disk are.
- Training is limited to sources in the benchmark inventory and measurement-db. Non-measurement-db data must be released with your code.
- License: CC BY-SA 4.0 for AIMS curation, with each benchmark's own upstream license applying.

---

## 4. Compute limits: where your Kaggle setup fits

**Important:** your GPUs are for **development**, not for scoring. At scoring time:

| Limit | Value |
|---|---|
| Runtime per run | 8 hours (includes setup) |
| Models | up to 5 repos, each up to 300B params / 1,000 GB; 1,024 GB total |
| GPU | allocated by organizers (type and memory unspecified) |
| Downloads at runtime | **not allowed**; bundle the model or list it in `models.txt` |
| Extra pip installs | disabled by default |
| Network | only inference APIs (Anthropic, OpenAI, Google, xAI, DeepSeek, Mistral, Together, Groq). No web search or browsing. |

### Your setup

- Each Kaggle GPU session = **2× T4 (16 GB each, 32 GB total)**.
- 2 parallel sessions per account × 6 accounts = **12 sessions at once**.
- Kaggle weekly GPU quota is documented as roughly **30 h/week per account** (verify on your account page). That gives roughly **180 session-hours/week across 6 accounts**, which is about 360 T4-hours.
  - Running 2 sessions in parallel uses the quota twice as fast. Effective time is still about 30 h per account per week.
  - Each session is capped at about 12 h, so plan work in 9–11 h chunks with checkpoints.

**Feasibility verdict on compute:**
- **Feasible** for everything except large local LLMs. The core predictors (matrix factorization, GBM, logistic/IRT-style models, small text embeddings) are light and train in minutes on one T4.
- **Local LLMs**: a 7–8B model in fp16 fits on one T4 (about 16 GB); a 2-T4 session can hold ~14B–30B with 4-bit quantization. Generation is slow on T4, so use local LLMs only for offline feature extraction (e.g. item embeddings), not per-prediction at scoring time.
- **Scoring-time LLM calls** go through the approved APIs. Each run has 8 h, so this is possible, but cost and latency must be checked. Do not use search or browsing.

---

## 5. Baseline

From `paiec_baseline` (GitHub):
- **Empirical Mean**: predicts the mean acquired response for the subject–benchmark pair, or 0.5 if there are no labels. Uses the Python standard library only. No data download, no API calls, no GPU. **This is the fastest valid submission.**
- **BLE (Bayesian Linguistic Evaluator)**: an LLM retrieves public evidence and revises a belief state. Uses an LLM, model not named.
- **Empirical Mean + BLE acquisition**: same predictor, but labels chosen by BLE.
- No baseline score is published, so you cannot tell how strong these are.
- The submission tool uses `requests` only.

Quick start:
```
python tools/build_mean_zip.py
python check_submission_zip.py dist/empirical_mean.zip
python submit_api.py --login
```

---

## 6. Recommended direction

Given 4 days, prioritize a **valid, reproducible submission first**, then improve.

### Step 0 (today): get a valid submission in
1. Run the Empirical Mean baseline locally with `check_submission_zip.py` and `tools/run_smoke_test.py`.
2. Submit it to Codabench. This secures a score and a submission ID.

### Step 1: small local evaluation harness
- Load only `formatted_tables/*.parquet` for a few benchmarks.
- Build the subject × item response matrix per subject–benchmark pair.
- Reproduce the **budget protocol**: split each pair's items 50/50 into acquisition and evaluation pools (pairs need ≥80 items), reveal labels at budgets 0/1/3/7/15/31, and compute Brier ALC with the given weights.
- Benchmark-level holdout: hold out whole benchmarks to mimic the cold-start test.

### Step 2: predictor (budget 0 matters most)
- Features: provider, release date, harness, reasoning effort, plus item text embeddings from a small embedding model (fits on a T4).
- Models to try in order of effort:
  1. Global mean per subject and per benchmark (shrinkage).
  2. Logistic regression / GBM on the features above.
  3. IRT-style model: `p = sigmoid(subject_ability − item_difficulty)`, with item difficulty predicted from embeddings for unseen items.
- Clip to [0.001, 0.999] to avoid large Brier losses.

### Step 3: adapt with labels (budgets 1–31)
- Update the subject's ability (or a per-pair offset) from revealed labels, shrinking toward the Step 2 prior.
- Use a Bayesian or shrinkage update so a single label does not overfit.

### Step 4: acquisition (`labeling.py`)
- Reveal items with the highest predictive uncertainty (p near 0.5) or the largest expected information gain.
- Diversify by embedding cluster so the labels cover the item space.
- Skip near-duplicate items.

### Step 5: calibration and report
- Check ECE and use isotonic or Platt scaling on your local holdout.
- Write the technical report (NeurIPS template, 4–8 pages main text) with: method, budget curve plot, ablations (features vs. labels vs. acquisition), disclosed pretrained models and LLM coding assistants, and reproduction steps.

---

## 7. Risks and things to check

1. **Deadline** (section 1): verify on Codabench that submissions are still open. If not, the report and code are still worth finishing for the record.
2. **Team rules and multiple accounts**: each participant may join only one team, and teams may have up to six members. Using six Kaggle accounts is only legitimate if they belong to six different people in one team. If one person runs all six, check Kaggle's terms (one account per person) and the competition rules before relying on them.
3. **Quota math is an estimate**: confirm your real weekly GPU hours on Kaggle.
4. **Internet on Kaggle**: downloading from Hugging Face needs internet enabled in the notebook (phone verification may be required). Accept the dataset conditions on HF first.
5. **Private test set**: do not try to identify subjects or benchmarks behind anonymous feedback IDs. This is a stated rule.
6. **Unknown facts**: number of benchmarks, items, and subjects; the prize structure (not stated); and the GPU type at scoring time.
7. **Local LLMs at scoring time**: bundle the model or list it in `models.txt`, and stay under the 1,024 GB total limit.

---

## 8. Links

- Competition and starting kit: https://codabench.org/competitions/17828
- Training data (measurement-db): https://huggingface.co/datasets/aims-foundations/measurement-db
- Baseline code: https://github.com/aims-foundations/paiec_baseline
- Technical report: OpenReview, NeurIPS.cc/2026/Workshop/PAIEC
- Contact: aimslab@cs.stanford.edu (office hours by appointment)
- Community: Discord (linked from the competition page)
- Announcement: https://aims.stanford.edu/news/predictive-evaluation-challenge-opens

---

## 9. How we are scored, our local estimate, and what is rewarded (added 2026-11-02)

### 9.1 The score
- **Brier ALC**, lower is better. For each subject–benchmark pair, the Brier score (mean squared error between predicted probability and the 0/1 outcome) is computed on that pair's evaluation items at label budgets 0, 1, 3, 7, 15 and 31.
- ALC = 0.1·B₀ + 0.2·B₁ + 0.2·B₃ + 0.2·B₇ + 0.2·B₁₅ + 0.1·B₃₁, averaged with **equal weight per pair**.
- Reference points: constant 0.5 = **0.250**; perfect = 0. A Brier of about 0.18 means the predictor knows each pair's exact success rate but nothing about individual items (the within-pair variance in the public data is 0.177).

### 9.2 Our local estimate (benchmark-held-out simulation of the official protocol)
`paiec/sim.py` reproduces the protocol: 50/50 pool split per pair, streaming acquisition, `labeled` spanning all pairs, the six checkpoints and per-pair weighting. Test benchmarks are never seen by the prior (5-fold split by benchmark). Full details are in `logbooks/02_experiments.md`.

| Predictor | Local Brier ALC | Note |
|---|---|---|
| Constant 0.5 | 0.250 | reference |
| **Official Empirical Mean baseline** (logic re-implemented) | **0.242** (5 folds) | 0.353 at budget 1: worse than 0.5, because with one label it predicts exactly 0 or 1 |
| Our online hierarchical model (no text) | **0.200** (5 folds) | about 17% better than the official baseline |
| + uncertainty acquisition | ≈ 0.208 vs. 0.209 on fold 0 | small gain; 5-fold run in progress |
| + text propagation within benchmark (v2) | 0.198 | text-based difficulty does *not* transfer across benchmarks (R² < 0.02), but it propagates well within one |

**Codabench formative results (2026-11-02):** v1 = **0.1978**, v2 = **0.1948** (submission IDs 1028212 / 1028213). The local estimates were 0.1995 / 0.1983, so the simulator tracks the official score within about 0.003.

**Caveats.** The real test uses a private benchmark pool, and the platform chooses subjects and pairs. Subject overlap with the public data is unknown. Our simulation caps subjects per benchmark at 20 and pool sizes at 200 acquisition / 100 evaluation items. Treat local numbers as a *relative* guide (method A vs. method B), not a leaderboard prediction. The official baselines have no published scores, so the Codabench formative feedback is the only absolute calibration.

### 9.3 How they evaluate
1. **Formative (development):** each Codabench submission is run on a sample of the hidden test data (up to 1,000 subject–item pairs per evaluation, split across acquisition and evaluation pools). It returns Brier ALC plus ECE as a diagnostic. Leaderboard visibility is "announced separately".
2. **Summative (final):**
   - an **automatic** Brier ALC on a *common hidden test subset* (the same for all teams), using the Codabench submission ID entered on OpenReview;
   - **manual grading of the codebase and technical report**, including a **reproduction check**: they rerun the code.
3. Runtime rules: 8 h per run, ≤ 5 HF model repos via `models.txt`, network only to listed LLM APIs, no runtime downloads.

### 9.4 What is rewarded
- **Selection for presentation** at the NeurIPS 2026 competition session (Fri 11 Dec 2026, Sydney, in person): **5 oral presentations** plus a poster session.
- Selection criteria (from the page): technical report quality, predictive performance, scientific rigour, and contributions to data curation and the community. Novelty is secondary. Technical soundness, clarity, reproducibility and experimental analysis are emphasized.
- **Prizes:** the current competition page states **no prize**. An older version of the AIMS site described a two-track design (AUC-ROC and Kendall's τ) with "performance-based prizes"; that design was replaced, so do not count on money.

### 9.5 The paper (technical report)
- NeurIPS template, about **4–8 pages** of main text, unlimited references and appendix. Submitted on **OpenReview: NeurIPS.cc/2026/Workshop/PAIEC**. Not anonymous.
- Must allow reruns: release the code and any non-measurement-db training data.
- Must disclose: pretrained models used (for us, the text encoder) and **LLM coding assistants (Claude Code)**.
- No formal peer review is planned. Graded on soundness, clarity, reproducibility and experimental analysis.
- Our planned structure: (1) the problem as test-time IRT; (2) the protocol simulator as the validation method; (3) the model (offline prior + hierarchical random effects + text propagation + Fisher acquisition); (4) budget-curve results against the official baselines; (5) ablations; (6) data findings (92.6M responses, a variance decomposition where 72% is item-level, subject matching by name). The logbooks in `logbooks/` are the raw material.

### 9.6 Deadline (unresolved)
The page says 30 Oct, extended to **6 Nov 2026 23:59 AoE**. One AIMS news page says the phase runs until 20 Nov 2026. **Plan for 6 Nov.** Team registration and a Codabench account are needed to submit; their status is unknown to us.

**Update 2026-11-03: formative scores are resampled.** Each Codabench run scores a *new* random sample of up to 1,000 unique subject–item pairs. Only the final evaluation uses a common subset for all teams. Resubmitting identical zips gave v3 0.1697 then 0.1920, and v4 0.1862 then 0.1853. All six runs so far have mean 0.188 and SD 0.010. A single formative score is therefore only a sanity check (± about 0.02), not a way to rank close variants. We choose configs with large paired simulations instead.

**Update 2026-11-04: public leaderboard and selection.**
- Codabench has a public "Provisional development scores — not oral or poster selections" leaderboard: 79 entries; median 0.172; top 0.117; organizer entry 0.180. Each entry is one chosen formative run, so heavy submitters look 0.02–0.05 better than their true mean.
- Final grading is (a) the Brier ALC of ONE chosen Codabench submission ID, re-scored on a common hidden subset, plus (b) manual review of the report and code, including a rerun. The page names no "winner" and no prizes.
- 5 oral talks plus a poster session are selected by report quality (stated as the important factor), predictive performance, scientific rigour, and data or community contributions.
- Deadlines: report and code on OpenReview by 6 Nov 23:59 AoE. The Codabench phase closes 7 Nov 11:59 UTC.
