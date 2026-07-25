# Target-vector estimation on this corpus — measurement results

Measured 2026-07-25 over `corpus/manifest.jsonl` **train split only** (264
human / 230 llm documents), using the vendored `biber.py` 18-feature
extractor. Holdout and dev were never opened.

The prototype derives its target as `v0 / published_ratio`, using GPT-4o
ratios from Reinhart et al. (2025). This measures the corpus directly
instead. The headline result is that the published ratios do not transfer,
and that two of the five proposed homing features are unusable here.

## Bottom line

Three of the five homing features survive. They are the three the five
implemented transducers already target, so nothing implemented is wasted —
but the plan's proposed transducers for phrasal coordination and
`that`-subject extraposition target features that are dead on this corpus and
should not be built.

| feature | verdict |
|---|---|
| `present_participial` | **keep** — 2.01× over-used, d = 1.13, all 6 models |
| `nominalization` | **keep** — 1.42× over-used, d = 0.77, all 6 models |
| `agentless_passive` | **keep** — 0.62× under-used, d = −0.71, all 6 models |
| `phrasal_coord` | **drop** — ratio 1.00, d = 0.00. Does not replicate. |
| `that_subj` | **drop** — zero variance. Degenerate, not merely weak. |

## Published ratios do not transfer

Corpus LLM documents are local Ollama models (gemma2:9b, gemma3:4b,
granite4.1:3b, llama3.1:8b, llama3.2:3b, qwen2.5:7b-instruct), not GPT-4o, so
this is a directional check rather than a replication.

| feature | human | llm | measured ratio | published (GPT-4o) |
|---|---|---|---|---|
| `present_participial` | 4.69 | 9.47 | 2.01× | 5.3× |
| `nominalization` | 25.11 | 35.54 | 1.42× | 2.1× |
| `phrasal_coord` | 20.90 | 20.92 | 1.00× | 1.9× |
| `that_subj` | 0.009 | 0.000 | — | 2.6× |
| `agentless_passive` | 9.61 | 5.92 | 0.62× | 0.5× |

Direction replicates for three features; magnitude does not, in every case.
`present_participial` is over-used by half the published factor. Bootstrapping
`μ = v0 / 5.3` would therefore have aimed at roughly half the human rate —
overshooting the target rather than reaching it, which the objective would
have reported as success.

## `agentless_passive` is the most robust signal

The feature the module rewrites *upward* has the most model-independent
evidence of the five:

| model | `agentless_passive` ratio |
|---|---|
| gemma2:9b | 0.46× |
| gemma3:4b | 0.64× |
| granite4.1:3b | 0.63× |
| llama3.1:8b | 0.58× |
| llama3.2:3b | 0.72× |
| qwen2.5:7b-instruct | 0.69× |

All six under-use it. The effect is strongest in `docs` (d = −1.18) — the
genre this module is scoped to. `present_participial` (1.6–2.6×) and
`nominalization` (1.3–1.7×) are likewise consistent across all six.

`phrasal_coord` is near 1× for five of six models and is the one published
direction that does not survive contact with this corpus.

## Σ is not estimable at 18 features

Per-genre sample covariance, human train:

| genre | n | cond(Σ) | effective rank (of 18) |
|---|---|---|---|
| blog | 68 | ∞ (singular) | 2.73 |
| docs | 58 | 1.36e5 | 3.58 |
| email | 30 | ∞ (singular) | 1.38 |
| forum | 51 | ∞ (singular) | 4.15 |
| readme | 57 | 3.93e5 | 2.18 |

The three singular genres are singular because `that_subj` has **exactly zero
variance** there — 0 of 68, 30, and 51 documents contain one. A zero-variance
column makes the sample covariance exactly singular at any n. In the two
genres where it is technically invertible, effective rank is 2–4 of 18.

Stable estimation wants n ≳ 5–10p, i.e. 90–180 documents per genre at p = 18.
The largest genre here has 68.

Diagonal-only does not route around this: the `that_subj` weight is `1/0` in
three genres.

## What does work: 4 features with shrinkage, per genre

Dropping `that_subj` and applying Ledoit-Wolf:

| genre | n | raw cond | shrinkage δ | shrunk cond |
|---|---|---|---|---|
| blog | 68 | 37.3 | 0.148 | 9.4 |
| docs | 58 | 30.2 | 0.269 | 5.4 |
| email | 30 | 16.4 | 0.389 | 3.6 |
| forum | 51 | 47.4 | 0.989 | 1.0 |
| readme | 57 | 79.2 | 0.247 | 9.8 |

This is the only estimator/subset combination in the sweep that lands in a
usable range. Note forum's δ = 0.989: the estimator is reporting that the
off-diagonal structure is essentially all noise at n = 51. Even four features
strain this sample size.

**Pooling Σ across genres is rejected.** Box's M on the four non-degenerate
features: M = 146.6, χ² = 141.3, df = 40, **p = 3.1×10⁻¹³**. Genres differ in
how features co-vary, not only in their means. Per-genre Σ is required, which
is exactly where small n hurts most.

## μ is stable for the surviving features

Bootstrap, 2000 resamples, seeded. Ten (genre, feature) pairs have a 95% CI
spanning ≥2× the point estimate. **None of the three surviving homing
features appears among them** — all have CI spans under 2× in every genre.

The flagged features are sparse lexical counts: `downtoners` (flagged in 4 of
5 genres), `first_person`, `hedges`, `demon_sent_initial`, and `that_subj`
again. These are reportable as diagnostics but are not usable point targets
at this sample size.

Split-half agreement per genre is at chance level (0–1 of 18 features
disagreeing at α = 0.05), consistent with stable μ. Email (n = 30, so 15 per
half) is underpowered and its pass is weak evidence.

## `docs` is a mixture

All 58 human `docs` train documents come from distinct repositories, spanning
conceptual guides, API reference pages, tutorials, and build-tool authoring
guides — different sub-registers under one label. Syntactic density features
are *less* variable within `docs` than corpus-wide (`nouns` CV 0.14 vs 0.26),
but stance features are *more* variable (`first_person` CV 1.33 vs 1.22,
`hedges` 1.16 vs 1.04) — the tutorial-voice/reference-voice split appearing
exactly where it should.

A single per-genre μ remains defensible (Q3 holds), but genre-level Σ likely
overstates true within-sub-register variance, which makes the small-n problem
worse rather than better.

## Consequences

1. The homing set is **three features**, not five.
2. `μ` must be **measured**, not derived from published ratios.
3. `Σ` must be **per-genre and shrunk**. It cannot be pooled and cannot be raw.
4. At three homing dimensions with condition numbers of 1–10, the full
   Mahalanobis/Lagrangian apparatus is likely over-engineered relative to what
   the data supports. Three rate targets plus the χ² shell criterion plus a
   diagonal drift penalty over the measured-only features probably captures
   everything the covariance would buy. This should be decided with evidence
   when the objective is specified, not assumed either way.
5. Transducers targeting `phrasal_coord` or `that_subj` should not be built.
6. `email` (n = 30) is undersized for every method applied here and should not
   carry a shipped target without more documents.

## Not measured

Formal sub-clustering within `docs`/`blog`/`forum` (inferred from provenance
and coefficient of variation, not verified). Reduced-dimension Σ via
PCA/factor analysis. Anything from dev or holdout.

Cached vectors and analysis scripts were left in the session scratchpad and
are not vendored; the numbers above are reproducible from the train split with
`biber.py`, Ledoit-Wolf from scikit-learn, and a seeded bootstrap.
