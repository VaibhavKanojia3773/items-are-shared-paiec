# Logbook 04: Research round 2 (2026-11-02)

Scope: web research only, focused on what is NEW relative to `01_research.md` (which already covers REEVAL, tinyBenchmarks, Fluid Benchmarking, Krumdick 2604.01418, observational scaling laws, PredictaBoard, Riiid, Kossen active testing).

Context that frames everything below: hidden-set Brier ALC 0.1948 (baseline about 0.244). Variance split: 19% between benchmarks, 9% subject-within-benchmark, 72% within pair. Text-embedding difficulty does not transfer across benchmarks (LOBO R^2 < 0.02). Budget-0 Brier is about 0.25. Random-effect variance tuning is flat. **Reading:** the model is now limited by its structure, not its hyperparameters. The biggest remaining levers are (a) borrowing strength across subjects inside a new benchmark (low-rank structure) and (b) item-level personalisation, not better text priors.

Legend: Value = expected impact on our ALC (H/M/L). Cost = rough implementation hours. [UNVERIFIED] = only an abstract or secondary summary was seen.

---

## 1. Cold start: predicting a new benchmark's base rate or difficulty

- **Anticipatory Evaluation / EvalCast** (Park, Mendes, Stanovsky, Ritter), arXiv 2509.20645 (v4, Oct 2026). https://arxiv.org/abs/2509.20645
  Frontier LLMs predict benchmark scores from a redacted task description, using 2,290 description-score pairs. Absolute regression "remains challenging". *Pairwise* "which benchmark is harder" judgments recover relative ordering.
  *How we could use it:* do not expect an LLM to give an absolute budget-0 base rate. At most, use an LLM's pairwise rank of the new benchmark against training benchmarks as one weak feature for the benchmark offset. We currently have no API key, so this needs a local model. Value L. Cost 6-10 h.

- **ADeLe: "General Scales Unlock AI Evaluation with Explanatory and Predictive Power"** (Zhou, Pacchiardi, Hernandez-Orallo et al.), *Nature* 2026 (arXiv 2503.06378). https://arxiv.org/abs/2503.06378, project page https://kinds-of-intelligence-cfi.github.io/ADELE/
  An LLM annotates each item on 18 non-saturating "demand" rubrics (reasoning, knowledge, metacognition, etc.). A random forest on the demand vector reaches AUROC about 0.84 in-distribution with near-perfect ECE. It reportedly beats embedding and fine-tuned-LLaMA assessors *out of distribution*: about 88% accuracy or AUROC up to 0.88 on new tasks and benchmarks. [UNVERIFIED: the OOD numbers come from an MSR blog. I did not see the paper's OOD table.]
  *How we could use it:* this is the one line of evidence that *rubric-level* item descriptors transfer across benchmarks better than raw embeddings, which is exactly where we failed. Test it cheaply first: annotate about 30 items per training benchmark with a local LLM (demand levels 0-5), then check LOBO R^2 of the *benchmark mean* logit against the demand means. Only build it out if R^2 > 0.1. Their annotation rubrics are public on the project page. Value M (budget 0 plus priors for budgets 1-3). Cost 10-15 h, including local-LLM annotation time.

- **Contextual MIRT for unseen questions** (Shang, Tang, He), arXiv 2608.22295 (Aug 2026). https://arxiv.org/abs/2608.22295
  MIRT with question-text-informed item parameters. It helps within scenario, but the authors call *cross-scenario generalisation "a central open challenge"*.
  *How we could use it:* independent confirmation of our LOBO R^2 < 0.02 finding. Cite it in the tech report and stop investing in embedding-to-difficulty regressors across benchmarks. Value: saves time. Cost 0.

- **"No Answer Needed": question-only linear probes** (Moreno Cencerrado, ..., Pacchiardi), ICLR 2026 workshop (arXiv 2509.10625). https://arxiv.org/abs/2509.10625
  A linear probe on an LLM's mid-layer activations, taken after reading the question and before answering, predicts that LLM's correctness. A probe trained on trivia generalises to OOD knowledge datasets but weakens on math reasoning.
  *How we could use it:* use a local open model's mid-layer activations (not sentence-embedding vectors) as item features, with a probe trained on *pooled* P(correct) across training subjects. Evaluate with LOBO. This is the most plausible "difficulty from text" signal left untried, because it encodes "does a model know this" rather than topic. Value L-M. Cost 8-12 h, plus GPU time.

- **Generalized Correctness Models** (Xiao, Patil, Lee, Stengel-Eskin, Bansal), ICML 2026 (arXiv 2509.24988). https://arxiv.org/abs/2509.24988
  LLMs have no privileged self-knowledge of their own correctness. A *stronger* model predicts another model's correctness better, and correctness predictors trained on many models' histories transfer across datasets and models. Post-hoc calibration adds further gains.
  *How we could use it:* supports a "proxy subject" feature: a strong local open model's own answer correctness or log-prob on the item (only possible if item content includes the answer key, which we should check in measurement-db). It also supports in-context history, i.e. our item offsets from other subjects. Value M if answer keys are present, otherwise L. Cost 8-15 h.

- **"100 instances is all you need"** (Pacchiardi, Cheke, Hernandez-Orallo), arXiv 2409.03563 (2024). https://arxiv.org/abs/2409.03563
  A "generic assessor" takes instance features plus a new LLM's results on a small reference set and matches LLM-specific assessors in-distribution. It degrades OOD.
  *How we could use it:* conceptually what we already do. Mainly confirms that OOD (new-benchmark) item-level prediction is hard. Value L.

- **LLM-simulated respondents for difficulty** (arXiv 2407.10899, and 2607.26317, Jul 2026). https://arxiv.org/abs/2407.10899, https://arxiv.org/abs/2607.26317
  GPT-class respondents give difficulty that correlates r > 0.7 with human IRT in some domains. Simulated populations are "too accurate and too uniform", so easy-item difficulty is unreliable.
  *How we could use it:* if we run several local models as proxy subjects, prefer a spread of small and weak models over one strong model, so that easy items are discriminated. Value L. Cost: high, because of compute.

- **Active Evaluation Acquisition** (Li, Ma, Ballesteros, Benajiba, Horwood), arXiv 2410.05952 (2024). https://arxiv.org/abs/2410.05952
  Models dependencies between prompts, uses them to predict unselected prompts' outcomes, and learns an RL acquisition policy. Handles new prompts with no scores (semi-supervised). [UNVERIFIED details]
  *How we could use it:* idea only, i.e. a learned acquisition policy. See BOBCAT in Section 3. Value L.

## 2. Online ability estimation, MIRT, low-rank interactions

- **Krumdick et al., "Cost-Efficient Estimation of General Abilities Across Benchmarks"**, arXiv 2604.01418 (v2, Aug 2026). https://arxiv.org/abs/2604.01418. Details new since round 1:
  (i) **Bifactor MIRT**: logit = general ability + K=3 multidimensional term + difficulty. (ii) An **auxiliary loss** forces the general-factor-only logit to fit the data, which stabilises low-sample estimates. (iii) An L1 penalty on loadings. (iv) An **empirical prior from training abilities** at test time. (v) **V-optimal** selection (minimise predictive variance over target items) beats 2PL Fisher selection and random. (vi) K was chosen by parallel analysis, where 1 factor explains about 70%. (vii) Plain linear regression wins only at about 1,000+ items.
  *How we could use it:* replace our independent subject x benchmark random effect with a **low-rank loading model**. Learn offline subject vectors u_s (general + K=2-3) on the 159 benchmarks. At test time, estimate the new benchmark's loading vector v_b and intercept from *all subjects' pooled labels*. Pooling over, say, 30 subjects x 3 labels pins v_b far better than 3 labels pin a per-pair offset. Value **H** (attacks the 9% subject-within-benchmark component at budgets 1-7). Cost 8-12 h.

- **BenchMIRT** (Ai2, 1 Sep 2026). Blog https://allenai.org/blog/benchmirt, code https://github.com/allenai/BenchMIRT, data in the HF collection allenai/benchmirt.
  MIRT on 100 LLMs x 16 benchmarks x 34K items. Held-out-item correctness accuracy is 79% vs 70% for the model's overall benchmark score. It found 2 dominant dimensions (safety, general reasoning).
  *How we could use it:* evidence that 2-3 dimensions suffice and that item-level MIRT beats the subject mean by a large margin on held-out items *within* a benchmark. Their code is a ready reference for the item-loading fit in the next item. Value M (supporting). Cost: reading 1-2 h.

- **Item-level personalisation via item loadings** (follows from MIRT; also EmbedLLM, ICLR 2025, https://arxiv.org/abs/2410.02223).
  EmbedLLM learns compact model embeddings that predict per-question correctness and transfer to routing and benchmark forecasting.
  *How we could use it:* at test time, give each item with >= m labels from other subjects a loading vector a_i (strong shrinkage to the benchmark's v_b), so P(correct) = sigma(u_s . a_i - d_i). This turns "item offset" into "item offset for subjects like this one". Value **H** (72% of variance is within pair, and our item offset is currently subject-agnostic). Cost 6-10 h, shared with the previous item.

- **"Great Models Think Alike" (CAPA error similarity)** (Goel et al.), ICML 2025. https://arxiv.org/abs/2502.04313, package `lm-sim`.
  Chance-adjusted error agreement between 130+ LMs. Errors grow *more* correlated as capability rises, and same-family models share mistakes.
  *How we could use it:* a cheap alternative to item loadings. Precompute subject-subject CAPA similarity from training data (and from metadata such as family, provider and size for unseen subjects). When estimating item i's offset for target subject s, weight other subjects' labels by similarity to s. Value **H** relative to its cost. Cost 3-5 h.

- **Sloth: skills scaling laws across families** (Maia Polo et al.), NeurIPS 2025. https://arxiv.org/abs/2412.06540
  Benchmark scores are driven by a few latent skills. Each family turns compute into skills with family-specific efficiency, which predicts unseen family members within about 3.5 pp.
  *How we could use it:* for subjects absent from training, build the ability prior as family offset + size/date regression on the low-rank skills, rather than a single scalar. Value M (only for unseen subjects. Check how many test subjects are unseen). Cost 4-6 h.

- **CollabEval** (Fisch et al., Google), arXiv 2607.05046 (Jul 2026). https://arxiv.org/abs/2607.05046
  Low-rank completion of the model x prompt matrix, used as control variates in prediction-powered inference. [Abstract only]
  *How we could use it:* confirms that low-rank completion is the right backbone. PPI targets aggregate estimates, not per-item Brier, so it is not directly useful. Value L.

- **IRT-Router** (arXiv 2506.01048, 2025). https://arxiv.org/abs/2506.01048
  For query cold start, blends a new query's embedding with its nearest training neighbours' embeddings before mapping to IRT parameters.
  *How we could use it:* we already propagate within a benchmark by kNN. Low marginal value. Value L.

## 3. Acquisition: objectives better than uncertainty sampling, and streaming thresholds

Fact that drives the objective: eval items are a *disjoint* half of each pair. A label on item j helps the eval pool mainly through (a) the subject's offset or loadings and (b) item j's neighbours. Pure p(1-p) on the item ignores how *well-identified* item j's difficulty is. A label on an item whose difficulty is already pinned by other subjects is more informative about the subject than one on an unknown-difficulty item.

- **EPIG: Prediction-Oriented Bayesian Active Learning** (Bickford Smith, Kirsch, Farquhar, Gal, Foster, Rainforth), AISTATS 2023. https://proceedings.mlr.press/v206/bickfordsmith23a.html
  Scores candidates by mutual information with predictions on a *target input distribution*. It is a drop-in replacement that beats BALD.
  *How we could use it:* our target distribution is the pair's eval pool. Approximate it with the benchmark's items seen so far, or with the acquisition stream. With Laplace or Gaussian posteriors over (subject offset, loadings), EPIG reduces to a closed-form variance-reduction score. See V-optimality below. Value M. Cost 4-6 h.

- **V-optimal design** (as used in Krumdick 2604.01418): choose the item that minimises summed posterior predictive variance over target items. In a linearised logistic model the gain from labelling j is roughly `w_j * (g_j' S g_target)^2 / (1 + w_j g_j' S g_j)`, with w_j = p_j(1-p_j) and S = posterior covariance of the subject-specific parameters. *How we could use it:* replace `1-4p(1-p)` in our threshold rule with the normalised V-gain. This automatically down-weights items with uncertain difficulty, by including item-offset variance in the noise term. Value M. Cost 3-5 h.

- **Scaling Up Active Testing to LLMs** (Berrada, Kossen, Bickford Smith, Razzak, Gal, Rainforth), NeurIPS 2025. https://arxiv.org/abs/2508.09093
  Cheap in-context surrogates for LURE-weighted active testing, plus a bootstrap estimate of run quality. It targets the *aggregate* metric, not per-item predictions.
  *How we could use it:* not directly applicable, because our loss is per-item Brier on held-out items, so importance-weighted estimators are the wrong objective. Value L.

- **BOBCAT: Bilevel Optimization-Based Computerized Adaptive Testing** (Ghosh, Lan), IJCAI 2021 (the winning approach for NeurIPS 2020 Eedi Task 4). https://arxiv.org/abs/2108.07386
  Meta-learns the response model *and* the question-selection policy end to end. The outer loss is BCE on held-out "meta" questions after adapting on the selected ones, and it beats Fisher-information CAT.
  *How we could use it:* a light version is to **learn the acceptance threshold gamma(labels_remaining, items_remaining, budget checkpoint) by simulating our exact streaming protocol on LOBO training benchmarks**, scoring Brier ALC on a held-out half. A full learned policy is too much work before the deadline. Value M. Cost 4-6 h (simulation harness).

- **Budgeted stream-based active learning** (Fujii, Kashima), NeurIPS 2016 (https://papers.nips.cc/paper_files/paper/2016/file/07cdfd23373b17c6b337251c22b7ea57-Paper.pdf), and **multi-secretary with uniformly bounded regret** (arXiv 1710.07719, https://arxiv.org/abs/1710.07719).
  For a fixed budget over a stream, the Bellman-optimal policy is a **threshold that depends only on residual budget and residual horizon**.
  *How we could use it:* validates our rule family. Fit thresholds as *quantiles* of the score distribution: accept if score >= quantile_{1 - L/N} of recently seen scores, a "rank-based" threshold that is robust to scale drift. Note also that the ALC checkpoints (1,3,7,15,31) reward front-loading. Accepting the first plausible item quickly is worth more than waiting for the perfect one, so simulate a stricter threshold early vs late. Value M. Cost 2-3 h.

## 4. Knowledge-tracing and Kaggle lessons that transfer

- **NeurIPS 2020 Education Challenge overview** (Wang et al.), arXiv 2104.04034. https://arxiv.org/abs/2104.04034. I read pages 10-12 directly.
  *Task 1 winner (Aidemy, 77.29% accuracy):* **target encoding** of correctness over many segment definitions, user-history counts, subject features, and **SVD vectors of the student x question matrix** as features. Three-stage stacking: per-model top-100 feature selection, then meta-features, then an ensemble regressor.
  *Task 4 winner (Ghosh, BOBCAT):* meta-learning that *trains under the few-shot test protocol*, with separate train/local-test student splits and per-student answer splits.
  *How we could use it:* (a) Add a **GBM stacker** on top of the hierarchical model. Inputs: our posterior logit and sd, pair running accuracy, item mean across subjects (shrunk), similarity-weighted item mean (Section 2, CAPA), benchmark pooled accuracy, budget, label counts, and subject SVD factors. Train it on simulated LOBO episodes at each budget. (b) Generate all training examples *episodically* under the exact streaming protocol. Value M-H (stacking a GBM on IRT features was the recurring winning pattern). Cost 6-8 h.

- **Riiid lessons not covered in round 1** (4th place, arXiv 2102.04250; 2nd place, arXiv 2102.00845).
  Hand-crafted cumulative features were needed because sequence models cannot see beyond their window. Time-aware attention and continuous-feature embeddings mattered. [Specific feature rankings not found online.]
  *How we could use it:* the transferable analogue of "lag/elapsed time" is **subject release date relative to the benchmark's era** (newer subjects dominate older benchmarks), along with harness and reasoning effort. Make sure these enter as interactions with the benchmark loading, not only as a main effect. Value L-M. Cost 2 h.

## 5. Calibration under few labels

- **Temperature scaling** (Guo et al., ICML 2017) and **beta calibration** (Kull, Silva Filho, Flach, AISTATS 2017; https://proceedings.mlr.press/v54/kull17a.html). Standard references, not newly verified.
  *How we could use it:* fit calibration **per budget checkpoint and per "information regime"**, e.g. buckets of #labels on the item from other subjects (0 / 1-2 / 3+). Miscalibration is likely to differ between items with propagated offsets and items with directly observed offsets. Beta calibration (3 parameters) can fix asymmetric over-confidence near 0 and 1, which plain temperature cannot. Value M. Cost 2-3 h.

- **Stacked recalibration** (from Section 4a). A monotone GBM or logistic regression on [logit, posterior sd, budget, n_pair, n_item_other, n_bench] trained on LOBO out-of-fold episodes directly minimises Brier. It subsumes per-budget temperature and the probit sd-correction. Prefer this over hand-tuned Laplace variance inflation. Value M. Cost: covered by 4a.

- **gp-IRT blending** (tinyBenchmarks, Polo et al., ICML 2024): mixes observed outcomes with IRT predictions according to the observed fraction. Not useful for per-item eval predictions, because eval items are disjoint. Listed only to rule it out.

- **GCM finding** (Section 1): post-hoc calibration adds gains *on top of* history-based correctness models. This is consistent with keeping a final calibration layer even after better structure. Value: supporting.

## 6. PAIEC-specific public material

- **Competition page** (https://aims.stanford.edu/competition, fetched 2026-11-02). New facts and ones to re-check:
  - Deadline: 30 Oct 2026, **extended to 6 Nov 2026 23:59 AoE**. NeurIPS session on 11 Dec 2026 in Sydney.
  - Scoring network access is allowed *only* to Anthropic, OpenAI, Google, xAI, DeepSeek, Mistral, Together and Groq inference APIs.
  - Up to 5 HF model repos, each up to 300B parameters, 1,024 GB total. 8 h limit per run.
  - Selection for oral or poster weighs report quality, rigour, and "community contributions" as well as score, and organizers rerun the code. Reports go to OpenReview `NeurIPS.cc/2026/Workshop/PAIEC` (4-8 pages, NeurIPS template, must disclose training sources and LLM coding-assistant use).
  - The page shows **no real leaderboard**. Its example Brier values (such as 0.140) are labelled illustrative.
- **Baseline repo** https://github.com/aims-foundations/paiec_baseline: the BLE baseline uses `openai/gpt-5.6-luna` (medium reasoning) with up to 10 tool turns over keyword, table and semantic search of the public DB. It uses only same-subject revealed labels, clamps to [0.02, 0.98], applies **no numerical Bayesian update**, and acquires when |p - 0.5| < 0.15. No scores are reported. *Use:* our report can contrast our numerical hierarchical posterior with the LLM belief-state approach, and note that BLE ignores cross-subject labels.
- **No public participant writeups, Discord summaries, or OpenReview PAIEC submissions found** (searches on "PAIEC", the OpenReview venue, and GitHub returned nothing). The Codabench page (https://www.codabench.org/competitions/17828/) did not render for the fetcher, so leaderboard standings are unknown. The FAQ URL returns 404. The Discord invites (discord.gg/W2rmjJdcm, discord.gg/DQZgXCEJjc) were not accessible to me.
- **Older AIMS pages describe a different format** (two tracks: matrix completion and contamination; CSV, Docker or "LLM Featurizer" submissions; 20 Nov deadline). These look like a superseded challenge version. Treat them as stale.
- **torch-measure** (https://torch-measure.readthedocs.io): the organizers' PyTorch IRT library (Rasch, 2PL, 3PL, Beta-IRT, multifacet, factor models, CAT). It could serve as an off-the-shelf MIRT fitter for Section 2. Check that it can be pip-installed in the scoring environment.

---

## Ranked top-5 next experiments

1. **Low-rank subject x benchmark structure with pooled online loadings** (Sections 2, Krumdick + BenchMIRT). Learn bifactor subject vectors (general + K=2) offline. At test time, fit the new benchmark's loading vector and intercept on *all subjects' labels*, replacing the independent subject x benchmark offset. Use the auxiliary general-factor loss or strong shrinkage for stability. *Expected:* the largest gain at budgets 1-7. Value H. Cost 8-12 h.
2. **Subject-similar item offsets** (Section 2, CAPA / item loadings). Weight other subjects' labels on item i by error-agreement similarity (precomputed from training, with a metadata fallback), or fit per-item loadings shrunk to v_b. *Expected:* targets the 72% within-pair variance on eval items that other subjects have labelled. Value H. Cost 3-5 h for the similarity weighting (try first), 6-10 h for loadings.
3. **Episodic LOBO simulator plus GBM stacker/recalibrator** (Sections 4a, 5). Simulate the exact streaming protocol on held-out training benchmarks, generate (features, label) rows per budget, and train a monotone GBM on top of the hierarchical posterior. Calibrate per budget and per information regime. This also provides the harness that items 4 and 5 need. Value M-H. Cost 6-8 h.
4. **V-optimal (EPIG-style) acquisition with learned residual-budget thresholds** (Section 3). Score = predictive-variance reduction over the pair's target pool, with item-offset uncertainty in the noise term. Accept by a rank or quantile threshold tuned in the simulator, testing front-loading for the checkpoints at 1 and 3. Value M. Cost 5-8 h.
5. **One cheap test of transferable item or benchmark descriptors before giving up on budget 0** (Section 1). Pick ONE of: (a) ADeLe-style demand rubrics from a local LLM on about 30 items per benchmark, or (b) mid-layer question-only probe features from a local open model. Gate on LOBO R^2 of the *benchmark-mean* logit > 0.1. If neither clears the gate, freeze budget 0 at a calibrated subject-prior-only prediction and document it as a negative result (Contextual-MIRT 2608.22295 corroborates). Value L-M. Cost 8-12 h.

Uncertainty notes: ADeLe OOD numbers, Active Evaluation Acquisition details and CollabEval results are abstract or secondary-source only. No PAIEC leaderboard information could be verified. Whether item content contains answer keys (needed for proxy-subject features) has not been checked in measurement-db.
