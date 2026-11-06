# Logbook 05: Competitor solutions (public GitHub repos, read 2026-11-06)

Read via web fetch; numbers are as reported by each team. They are formative single draws unless noted.

| Team / repo | Method | Acquisition | Reported scores |
|---|---|---|---|
| **stanley1208/paiec-2026** (V5) | Rasch with partner item pooling; **Claude rates item difficulty 1–10** (prior on z_i, trust 0.5, gated); **Claude guesses the subject's success rate on the benchmark** from its fields plus one item (calibrated by OLS on public pairs, trust 0.75). API called at scoring time, cached | (not found) | Offline: ratings **−0.0144**, guesses a further **−0.0135**; B0 0.250 → 0.204. Formative: V3 0.156, V4 0.170, **V5 0.141 (B0 0.180)**. Rating–Rasch Spearman 0.28–0.82 |
| **rishabhsinha17/brier-rabbit** | Hierarchical logit: mu_b + λθ_s(name prior; attribute ridge for unknown) + φ_s + δ_sb + feature group + item-text effect; damped Newton + probit moderation; "cavity" pooling of same-draw partner labels; pair effect split by shared name/harness | default random | Selected run 0.190 (9 pairs); 4-run mean 0.186. Same zip run 103 times: **SD 0.021**. Level prior m0 −2.0, τ_μ 2.0 helped live |
| **doctawho42/PAIEC** (hier-ship) | Rasch family with an MC guess floor, attribute-ridge subject prior, Gauss–Hermite item integration; level prior mu0 −2.5, σ_μ 2.5 | default random | Formative 0.1926, 0.1817 (17 pairs: 0.1868) |
| **IvanYamasaki** (J2) | 12-line beta-binomial: benchmark rate from other subjects → pair shrinkage | default | 39 runs mean **0.1892** [0.1846, 0.1938]; official empirical mean 0.2161 (20 runs) |
| **Agnuxo1** | Beta-binomial + same-item peer residual | none | 0.2084, 0.1851 (one run each) |
| **fukushima2940acsl** | Empirical-Bayes grid prior | none | 3 runs 0.204–0.207; mean-match variant 0.168–0.195 |
| **vivamichu** | Claude Haiku per-item p0 + shrinkage + sibling residual | none | not reported |

## Facts about the hidden evaluation (from teams that logged hundreds of formative runs)
- About **8–9 pairs per draw**; evaluation pools **44–113 items**. Splits are persistent per pair: revealed labels and evaluation items are identical across submissions (only the draw changes).
- **Sparse:** 16 of 26 hidden appearances were alone on their benchmark (doctawho42); about 38% of appearances share a benchmark with another pair (Brier Rabbit). Our formative-like regime (≈10 subjects per benchmark) overstates cross-subject sharing. We added a sparse regime (`--formative K --rows K` = one pair per benchmark).
- Eligible test subjects are reportedly base models released in 2025–2026, so many names are probably unseen in the public data.
- **Budget-0 metadata priors hurt on the hidden set for three teams** (B0 0.28–0.32). LLM-elicited guesses (Stanley V5) were the only budget-0 prior that beat 0.25 there.
- Formative noise: SD 0.015–0.026 per run for an identical zip (our 0.017 agrees). The mean over many runs of a simple model is ≈0.189; ours is 0.187.

## Consequences for us
1. v7 name prior: keep only for known names; **provider fallback disabled** (`provider_prior: false`).
2. Re-validate v6/v7 in the sparse regime.
3. **Largest remaining lever: LLM priors** (item-difficulty ratings plus subject success guesses), worth about −0.014 and −0.013 offline for Stanley. Needs an approved-provider API key at scoring time.
