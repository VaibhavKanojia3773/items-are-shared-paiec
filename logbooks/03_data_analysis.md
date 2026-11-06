# Logbook 03: Dataset analysis (measurement-db, downloaded 2026-11-02)

Script: `scripts/build_tables.py` (log: `data/build_tables.log`). Only core tables were downloaded: responses, subjects, benchmarks for all 281 benchmarks, plus items for all except `cseo_bench`. That is 8.5 GB on disk, within the agreed 14.8 GB.

## Size: the HF card's "571,921 observations" is far too low

| Quantity | Value |
|---|---|
| Response rows (all 281 benchmarks) | **92,580,425** |
| Benchmarks eligible (binary + item granularity, as in the official `prepare_data.py`) | **159** of 281 |
| Skipped benchmarks | 118 (49 mixed, 17 continuous_bounded, 14 continuous, 14 ordinal, 8 fraction, …) |
| Binary responses kept | 68,631,488 (2.76M non-binary values dropped) |
| Aggregated subject×item cells | **31,750,924** (9.9% have n > 1; max n = 10,000, from `monkey_power_laws`) |
| Distinct items with responses | 3,679,304 |
| Subject rows / unique `subject_id` / unique `normalized_name` | 14,836 / 14,239 / **674** |
| Subject–benchmark pairs / eval-eligible (≥ 80 items) | 14,518 / **13,023** over 145 benchmarks |
| Global success rate | 0.349 per response, 0.515 per cell |

The largest responses tables are `monkey_power_laws` (28.8M rows: repeated sampling), `llm_uncertainty_bench` (5.1M), `wmt_mqm` (4.9M), `bridging_gap` (4.7M) and `mmbench_v11` (3.3M).

## Aggregation is lossless for Brier
Brier over the n repeated responses of a (subject, item) cell with k successes is (n p² − 2pk + k) / n. It depends only on (n, k). `test_condition` is not part of the competition input, so it is pooled. This reduced 68.6M rows to 31.75M cells and made everything fit in 16 GB RAM.

## Subjects
- `subject_id` is almost always benchmark-specific (only 3.4% appear in more than one benchmark), so **subjects must be matched across benchmarks by `normalized_name`**.
- 354 names appear in more than one benchmark, 66 in at least 5, and 11 in at least 20. The most widespread are GPT-4o (44), GPT-4o mini (32), Claude 3.7 Sonnet (30), Claude 3.5 Sonnet (27), Gemini 2.5 Pro (26), GPT-5 (22) and DeepSeek R1 (22).
- Very heterogeneous: RL checkpoints (`alignment_faking`: 4,278 subjects "Agent: igg-baseline" at different training steps), fine-tuned classifiers (`sib200`: 3,341), human judge studies (`haiid`), agent scaffolds, and frontier APIs.
- Field coverage: harness 95%, `subject_features_extra` 97% (key=value strings such as `training_step=14`), provider 46%, release_date 32%, harness_version 3.4%, reasoning_effort 0.2%, access_date 1.1%.

## Benchmarks
- Per-benchmark mean success: min 0.0, quartiles 0.35 / 0.53 / 0.69, max 0.96. A wide spread, so the benchmark base rate is a large unknown at budget 0.
- Subjects per benchmark: median 13, 75th percentile 31, 90th percentile 96, max 4,278. Items per benchmark: median 1,400, 75th percentile 11k, max 571k.
- Eval-eligible pairs per benchmark: median 13. A few benchmarks have thousands of pairs, so simulations cap subjects per benchmark (20) to avoid one benchmark dominating the equal-per-pair average.

## Variance decomposition (eligible cells, success rate)
| Component | Variance | Share |
|---|---|---|
| Between benchmarks | 0.0464 | 19% |
| Subject within benchmark | 0.0217 | 9% |
| **Within pair (item level)** | 0.1774 | **72%** |

**Implication:** item-level variation (difficulty, shared across subjects) dominates. The largest gains should come from (1) cross-subject labels on the same items, which the protocol reveals, and (2) item-text-based difficulty transfer. Benchmark-level and subject-level effects matter, but they are smaller.

## Items
- Item columns: `item_id, benchmark_id, raw_item_id, content, asset_manifest, grading_criterion, verifier, content_hash, item_features`.
- **`grading_criterion` / `verifier` are reference answers and not part of the competition input. Never used.**
- Content length (chars): median 749, 90th percentile 3,239, 99th percentile 11,241. Stored truncated to 3000 head + 500 tail chars (`data/proc/items_all.parquet`, 397 MB).
- Embedding subset: at most 10k items per benchmark (597,441 items, `scripts/subset.py`, deterministic and identical in the Kaggle kernels).

## Final EDA on the raw tables (2026-11-03, `scripts/final_eda.py`)
Per-benchmark output: `logbooks/runs/final_eda_per_bench.csv` (159 eligible benchmarks, 68.6M binary responses; log `logbooks/runs/final_eda.log`). This is the last pass that needs the raw `data/mdb` download. Means below are over benchmarks unless marked pair-weighted (eval-eligible pairs, as the score weights them).

### Subject identity (competition subject = the full subject dict)
- Keying by the **full dict** (name, provider, release/access date, harness, version, reasoning effort, `subject_features_extra`) gives 14,529 subjects from 14,580 subject rows. Only 5 benchmarks have exact duplicate dicts (ceval, devbench, elicitation_game, emoji_attack, vl_rewardbench).
- Keying by the **4-field** definition (name, harness, version, effort) gives only **1,948**. It fuses ≥ 50% of subjects in **72 of 159** benchmarks. Example: MMLU's 150 subjects have empty name, harness, version and effort, and are identified only by `subject_features_extra`.
- **Consequence:** the 4-field "merged" simulator (M001–M012) was invalid, which explains why it failed. The submission correctly keys subjects by the full dict (`_subject_key`).

### Repeats and noise floor (targets = responses; item = item_id + interactors)
- Responses with trial > 1: 9.8% on average; present in 44% of benchmarks.
- Share of targets in cells with n > 1 (full-dict subjects): mean 20%, pair-weighted **7.8%**. Repeat-heavy benchmarks (≈ 100%): adaptivestep, naep_llm_students, monkey_power_laws, pku_saferlhf, revolve, mathconstraint, critic_discernment_game, devbench, swe_smith, cruxeval, swe_rebench, vl_rewardbench.
- Irreducible Brier floor Σk(n−k)/n / Σn: mean 0.014, pair-weighted **0.004**; up to 0.21 (devbench) and 0.10 (vl_rewardbench).
- So `n_cap` matters only on a minority of benchmarks. This is consistent with n_cap being neutral in the simulators.

### Interactors and test conditions
- `interactors` are non-empty in only **6 benchmarks**, all attack or agent settings: advprompter (1 value), emoji_attack (11), haiid (2, on 50% of responses), harmbench (20), jailbreakbench (5), tau2_bench (5). They are part of the item identity in the submission, which is correct.
- `test_condition` is set in 58% of benchmarks. It is not a competition input, so it is pooled (it becomes unexplained within-item variance).

### Items
- Duplicated `content_hash` within a benchmark: 42% of benchmarks, and 38 have > 10% duplicates. Usually the duplicates differ in `item_features` (beavertails: 140 prompts × 3 feature variants = 420 items), so they stay distinct items.
- Some benchmarks have **identical competition inputs under different item_ids**: babilong 526k items but 16.9k unique content + features; felm 4,412 → 845; bridging_gap 132k → 111k. The evaluator reuses one prediction for identical inputs, and our model pools their labels into one cell. The item_id-keyed simulator slightly under-represents this pooling.
- Items with an image or asset manifest: 20% on average; > 50% of items in 31 benchmarks. The text-only input misses the image, so text propagation is weaker there.
- `item_features` are present in 77% of benchmarks.

### Pair structure and overlap (eval-eligible pairs, full-dict subjects)
- 145 benchmarks, 12,972 eligible pairs. Items per eligible pair (per-benchmark median): quantiles 10/25/50/75/90% = 200 / 385 / 1,000 / 5,323 / 21,063.
- **Subject × item matrix density** of eligible pairs: benchmark median **0.92** (pair-weighted 0.33, dominated by a few huge sparse benchmarks).
- **Item overlap:** the share of a pair's items answered by ≥ 5 other eligible subjects is benchmark median 0.83, **pair-weighted 0.96**. Cross-subject item information is available almost everywhere. This supports the formative-like simulator (full overlap) and coverage-aware acquisition.
- Pair base rates: pair-weighted 15% of pairs are near-deterministic (rate < 0.1 or > 0.9). The pair-rate-oracle Brier mean of r(1−r) is 0.174 pair-weighted (benchmark median 0.197).
