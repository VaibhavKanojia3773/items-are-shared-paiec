# measurement-db: which files to download

Written for: you and your team, to decide what to download.
Source: Hugging Face file tree for `aims-foundations/measurement-db`, read through the HF API on 2 Nov 2026. Sizes are from the API, not downloaded. The repo is gated, so the actual download needs an HF login and accepted conditions.

## Repo structure

Each of the 281 benchmark folders contains:

```
<benchmark>/
  formatted_tables/   <- the data you need
    items.parquet
    responses.parquet
    subjects.parquet
    benchmarks.parquet
    traces.parquet    <- reasoning traces (large)
    assets.parquet    <- images/media (large)
  raw/                <- original source archives (not needed)
  characterization.yaml, curation_record.md   <- metadata (small)
```

## Sizes by file type (all 281 benchmarks)

| File | Total size | Needed? |
|---|---|---|
| `responses.parquet` | **2.71 GB** | Yes, the labels |
| `items.parquet` | **12.07 GB** | Yes, item text for features/embeddings |
| `subjects.parquet` | < 0.01 GB | Yes, subject attributes |
| `benchmarks.parquet` | < 0.01 GB | Yes, benchmark attributes |
| `traces.parquet` | 107.78 GB | No (skip for first pass) |
| `assets.parquet` | 154.75 GB | No (images/media) |
| `raw/` folders | not counted | No |

**Core total: 14.79 GB** for all 281 benchmarks.
**Total `formatted_tables`: 277 GB**, so the full dataset is far beyond what you need.

## Per-benchmark core size

- Median benchmark: **2.3 MB**.
- 196 of 281 benchmarks are under 10 MB.
- 258 of 281 are under 100 MB.
- 1 benchmark is over 1 GB: **`cseo_bench`**, with 5.7 GB of items and only 10 MB of responses. It contributes 38% of the core size, so skip it in the first pass.

Largest core sizes (items + responses), for reference:

| Benchmark | Core size | Items | Responses |
|---|---|---|---|
| cseo_bench | 5,696 MB | 5,686 MB | 10 MB |
| scale_mrt | 835 MB | 833 MB | 2 MB |
| monkey_power_laws | 709 MB | 1 MB | 708 MB |
| qatch | 708 MB | 705 MB | 3 MB |
| mind2web | 523 MB | 522 MB | 1 MB |
| naturalreasoning | 422 MB | 379 MB | 43 MB |

## Recommended download tiers

**Tier 1: smoke test and baseline (~2.7 GB).**
Download `responses.parquet`, `subjects.parquet` and `benchmarks.parquet` for all 281 benchmarks. This is enough for the Empirical Mean baseline, a local budget-protocol harness, and matrix-factorization predictors with no text features.

**Tier 2: main work (~9 GB).**
Add `items.parquet` for every benchmark except `cseo_bench`. This enables item text embeddings, the feature-based budget-0 predictor and the IRT-style model.

**Tier 3: optional (~14.8 GB total).**
Add `cseo_bench` items only if you need it. Its responses are tiny, so it adds little signal for the cost.

**Skip:** `traces.parquet` (108 GB) and `assets.parquet` (155 GB). Reasoning traces are not needed to predict responses unless you add trace-based features, and that would be a later step.

## Fit with Kaggle

- Kaggle notebook disk is limited (roughly 20 GB; check the current limit on your account). Tier 2 (~9 GB) fits with room for working files. Tier 3 (~14.8 GB) is tight.
- Work per account: split the benchmarks between the 2 accounts (for example by alphabetical half), so each account only downloads and holds its share.
- Save the processed outputs (e.g. the subject × item matrices, embeddings) as small files and share those between accounts instead of re-downloading the raw Parquet.

## Caveats

- The `responses.parquet` total (2.71 GB) is much smaller than the item total, which is consistent with 571,921 observations being compact in Parquet. The exact row count has not been verified; check after download.
- Item and subject counts are still unknown. The Parquet files will give the real numbers.
- The sizes are the sums of files in each folder as the API reports them. Git LFS storage may differ slightly.

## Accounts and credentials (before using your brother's Kaggle accounts)

- Keep the legacy `kaggle.json` keys inside the `.kaggle` folder. Do not paste them into chat or commit them to git.
- Your brother should agree to his account being used, and the notebook should run under his name.
- The team rules let each member join one team with their own account. Each account should belong to a different person in the team.
