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
features appears among them**: all have CI spans under 2× in every genre.

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

## `docs` and `readme` do not pool

Scoping the module to technical documentation and dropping the genre
parameter makes `docs` + `readme` the natural target register — 115 human
train documents instead of 58. They cannot be pooled.

| feature | docs (n=58) | readme (n=57) | CIs overlap? |
|---|---|---|---|
| `present_participial` | 5.06 [4.40, 5.79] | 4.46 [3.64, 5.32] | yes |
| `nominalization` | 29.77 [26.44, 33.37] | 22.46 [19.74, 25.20] | **no** |
| `agentless_passive` | 12.50 [10.80, 14.33] | 8.38 [7.22, 9.61] | **no** |

Two of three homing features differ with non-overlapping bootstrap CIs. Box's
M rejects equal covariance (M = 16.60, χ² = 16.12, df = 6, p = 0.0131), and
the `present_participial`↔`nominalization` correlation *flips sign* between
them: r = −0.041 in docs, r = +0.263 in readme. Formal technical prose and
project-README voice are different registers, and a single pooled μ is wrong
for both.

Pooling does not even buy conditioning. Pooled Ledoit-Wolf at n = 115 gives
δ = 0.055 and condition 12.79, against docs-only at 11.97 and readme-only at
9.16 — slightly *worse*, because the two correlation structures partly cancel
rather than reinforce. Doubling n lowers the required shrinkage; it does not
improve the estimate.

**LLM output collapses the distinction human writers maintain.** On the
machine side, Box's M gives p = 0.669 — no covariance difference at all — and
only `nominalization` differs by sub-genre (d = 0.45, p = 0.034). Human
writers adapt register between reference documentation and READMEs; these
models largely do not. That is a document-level property rather than a
span-level one, so it is a diagnostic rather than something the four
operations or the transducers can act on.

**Decision:** use the `docs` target. README-style text is then served by a
target that is measurably too formal for it, which is a known cost rather
than an oversight. At the operating threshold chosen below the practical
effect is small (98.2% of human README documents fall inside the shell and
are never touched), but it is a real limitation and it is the reason a
sub-register signal would be worth having if one ever becomes available
cheaply.

## The χ² termination criterion is the wrong shape

The plan terminates rewriting at `D² ≤ χ²(k, 0.5)`. Two problems, and the
second is fatal to that specific formula.

**The distribution is not χ².** Human D² over docs+readme (n = 115): min
0.107, median 1.974, 90th percentile 5.062, max 13.389, against χ²(3)'s
median 2.366 and 90th percentile 6.251. KS test versus χ²(3): D = 0.173,
**p = 0.0018: reject**. Docs-only alone also rejects (D = 0.210, p = 0.0099).
The empirical distribution sits below the χ²(3) curve through most of its
range, so χ²-derived thresholds systematically over-admit: at nominal
q = 0.50, 60% of human documents fall inside, not 50%.

**The median is untenable with the module on by default.** Even reading the
nominal quantile generously, a median shell edits roughly 40% of genuine human
technical documentation, 45.6% of README-genre documents specifically.

Empirical-quantile thresholds, taken from the human D² sample directly:

| q | threshold | human inside | llm inside | separation |
|---|---|---|---|---|
| 0.50 | 1.974 | 50.4% | 23.9% | 26.5 |
| 0.75 | 2.898 | 74.8% | 38.0% | **36.7** |
| **0.90** | **5.062** | **89.6%** | **65.2%** | 24.3 |
| 0.95 | 6.051 | 94.8% | 72.8% | 22.0 |
| 0.99 | 11.328 | 98.3% | 91.3% | 7.0 |

**Decision: empirical q = 0.90, not `chi2.ppf(0.5, 3)`.** 89.6% of human
documents pass through untouched; 34.8% of machine documents are caught. q =
0.75 separates the classes best in raw terms but accepts a 25% false-edit rate
on real human documentation, which an always-on module cannot justify. q =
0.99 is useless — separation collapses to 7 points.

**The honest limitation:** no quantile achieves human-inside above 95% and
llm-inside below 50% at the same time. The three-feature vector is not a clean
document-level discriminator; the class distributions genuinely overlap. The
shell is a reasonable termination criterion and a weak trigger, and it should
not be described as the latter.

**Open, and not measurable until the transducers exist:** how many edits a
false-positive human document actually receives. A document just outside the
shell needs few edits to get inside, so a 10.4% false-positive *rate* may
still carry a small false-positive *magnitude*. Which is what the near-no-op
guarantee is actually stated in (edits per 1000 words, not documents
touched). This must be measured against the existing calibration before the
module ships on by default.

## The bands actually shipped, and what they imply

Only two of the three surviving features have a transducer that can move
them. The participial transducers were dropped because they select on `acl`
and `advcl`, which the shipped parser resolves at 52–58% F1 — they would fire
wrongly about half the time. `present_participial` therefore stays measured
and reported, but nothing acts on it.

For the two that remain, per-document rate quantiles over the 58 human `docs`
train documents:

| feature | p10 | p50 | p90 | LLM p50 | direction |
|---|---|---|---|---|---|
| `nominalization` | 15.47 | 28.21 | 50.56 | 32.29 | reduce |
| `agentless_passive` | 5.43 | 11.73 | 21.49 | **4.35** | increase |
| `present_participial` | 2.00 | 4.86 | 8.91 | 7.89 | *(no transducer)* |

The band is `[p10, p90]`, and a document inside it is done — the shell, not the
centroid. Optimizing to the median would land every document on the same
coordinates, which is a tell in its own right.

**This is where the module's value actually is, and it is lopsided:**

| feature | machine documents already inside the band |
|---|---|
| `nominalization` | **37 of 46** |
| `agentless_passive` | **19 of 46** |

The human `nominalization` band is wide enough (15 to 51) that most machine
documents already qualify, so the unpacking transducer has little to do. The
machine `agentless_passive` median sits *below* the human 10th percentile, so
27 of 46 documents need the passive transducer.

The value concentrates almost entirely in the one transducer that fires
*upward*. Which is also the feature with the strongest cross-model evidence,
under-used by all six models and most sharply in this genre. That is a
coherent outcome rather than a lucky one: an under-use tell is exactly what a
purely subtractive engine cannot address, and it is the reason this module
justifies its own contract.

## Not measured

Formal sub-clustering within `docs`/`blog`/`forum` (inferred from provenance
and coefficient of variation, not verified). Reduced-dimension Σ via
PCA/factor analysis. Anything from dev or holdout.

Cached vectors and analysis scripts were left in the session scratchpad and
are not vendored; the numbers above are reproducible from the train split with
`biber.py`, Ledoit-Wolf from scikit-learn, and a seeded bootstrap.
