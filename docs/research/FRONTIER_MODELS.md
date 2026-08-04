# Making friction work on frontier-model text (Sonnet/Opus/GPT-5 class)

Research notes, 2026-07-26. No implementation: this documents why the current
engine underperforms on frontier-model output, what that output actually looks
like, and the corpus/mining options for closing the gap. The fixer constraint is
treated as first-class throughout: friction rewrites, so every candidate pattern
is assessed by whether a gate-passing deterministic edit exists, not just
whether the pattern is detectable.

## 1. Why the current engine skews toward small models

Three separate causes, in decreasing order of impact:

1. **The DMS channel has no frontier index.** `dms-index-en-v1.toml` carries token
   streams for four families only (qwen 38 docs / 21,578 tokens; gemma 81 /
   47,126; llama 75 / 38,654; granite 36 / 21,858). The v4 evidence ledger
   already records the consequence: Claude-register text evaded the
   qwen/gemma/llama indexes. The DMS machinery itself is model-agnostic
   (matched-length differential against a human automaton): it is purely a
   data gap, not an algorithm gap.
2. **The literal inventory was mined from small-model habits.** The
   `mined`/`mined-paired` entries come from six local model families plus the
   gemma stock-vs-antislop pair. Frontier models emit far less of that crude
   lexical slop. Their tells moved up a level (see §2). The `seed` entries
   ("in order to", "prior to", LVC pairs) remain generic and still fire.
3. **The paired-mining trick doesn't transfer as-is.** The gemma pair worked
   because an antislop *finetune* answers the same prompt with the same content
   in a de-slopped register (style isolated, topic cancelled). There is no
   antislop finetune of Sonnet or Opus, and prompting is not an equivalent
   substitute (§3.2).

## 2. What frontier LLM-speak looks like

### 2.1 Lexical drift — old lists are partially stale

- Kobak et al. (*Science Advances* 2025, `berenslab/llm-excess-vocab`) quantify
  the 2023-era vocabulary: "delve" ~28× baseline, plus tapestry / testament /
  boasts / meticulous / underscore / intricate / pivotal etc. The full set is
  900 annotated words with frequency ratios in `results/excess_words.csv`.
- Drift is **word-specific and non-monotonic** (COLING 2025, "Why Does ChatGPT
  'Delve' So Much?"): "boasts" faded in GPT-4o-mini while "underscore" *rose*.
  RLHF is the best-supported cause. So era-1 words should be demoted to a
  lower-confidence tier, not deleted.
- A 34-language study (arXiv 2605.25358) finds GPT-family models highly
  correlated with each other on overused words (ρ 0.86–0.96), moderately with
  Claude Haiku (ρ ≈ 0.85), weakly with Gemini (ρ ≈ 0.46). A single "frontier"
  wordlist is defensible for GPT+Claude. Gemini needs its own index.
- Rising era-2 markers: sentence-initial "Additionally", emphasize/highlight
  verbs, importance nouns (significance, priority), innovation adjectives.

### 2.2 The durable signal is constructional, not lexical

Frontier tells that survive across model generations:

| Pattern | Example | Evidence |
|---|---|---|
| Negated contrast frame | "It's not just X — it's Y" | 25% of EQ-Bench's slop score by itself; friction already measures `not_just_but_rate` |
| Tricolon fragments | "Fast. Simple. Effective." | Wikipedia Signs-of-AI-writing; EQ-Bench |
| Participial tail | "…, underscoring its importance for…" | friction's `participial_closer_rate`; Wikipedia guide |
| Hedge stacking | "Generally speaking, in most cases…" | Wikipedia guide; stop-slop |
| Em-dash density | Claude ≈ 1.0–1.3 per 100 words | context-link.ai; multiple analyses |
| Bold-lead-in bullets, header bloat | "**Term:** explanation" lists | Wikipedia AI-cleanup; Claude system prompt *tells* it not to do this in prose contexts |
| Syntactic templates | repeated POS 4–8-gram skeletons | arXiv 2407.00211: 83–97% of LLM outputs template-saturated vs 36–46% of human text |
| Noun-heavy informational density | nominalization-heavy register | Reinhart et al., PNAS 2025 (arXiv 2410.16107) — persists even when prompted to write informally |

This is good news for friction specifically: `corpus/SEPARATION.md` already
computes several of these as deterministic Biber-style features, and the
register-band operation (`register-en-v1.toml`) is exactly the right shape for
density-type tics (edit while outside the human band, stop at the band edge).

## 3. Corpus strategy — the "better training corpus" answer

### 3.1 There is nothing to download

No public dataset pairs Claude output with human technical writing. RAID,
MAGE-OOD and M4GT-Bench contain GPT-4 (2023 checkpoints) but no Claude and no
technical-documentation genre. HC3/Ghostbuster are GPT-3.5-era. The public sets
are useful only as out-of-domain generalization checks. The frontier side must
be **generated via API**, and that is cheap:

> 500 prompts × 3 samples × ~800 output tokens ≈ 1.2M output tokens per model
> ≈ tens of dollars per model at current Sonnet/Opus pricing. EQ-Bench's slop
> lists were powered by ~32 prompts × 3 iterations per model, and friction's
> existing per-family DMS streams are 21k–47k tokens: the same scale costs
> single-digit dollars.

### 3.2 Paired-corpus replacement: three designs

1. **Frontier-vs-human on the existing genre battery (primary).** Generate
   Sonnet/Opus/GPT-5 documents against the same docs/readme/blog/email/forum
   prompt distribution that shaped `corpus/human`, and run the *unchanged*
   pipeline: `corpus-tool index` (new DMS families `claude`, `gpt`, optionally
   `gemini`), `mine`/`mine-inventory` (Monroe et al. log-odds, already the
   statistically stronger method: slop-forensics' own formula is a cruder
   weighted-frequency heuristic), `separation`, `envelope`. The antislop pair
   was always a proxy for "human-like." The human corpus is the ground truth
   and it already exists.
2. **Prompted pairs (default vs anti-slop system prompt) as a priority filter,
   not a mining source.** The delta isolates *instruction-following*, not
   latent habit. PNAS 2025 shows the noun-heavy register persists under
   prompting, and detector-evasion studies show prompted "de-AI-ing" degrades
   register rather than normalizing it. Use: tics that **survive** an anti-slop
   prompt are load-bearing → high-priority hard rules. Tics that vanish under
   prompting are low-priority. Bonus for the fixer: the suppressed rendering of
   the same prompt is a source of candidate *replacements*, which then must
   independently pass the existing human-train attestation gate (closure rule
   unchanged: model output never becomes replacement text without human-corpus
   attestation).
3. **Generation-over-generation contrast (secondary).** Sonnet 4 vs Sonnet 5 vs
   Opus on identical prompts, same three-way framing as `FINGERPRINT.md`, with
   model version as the axis. Tracks drift. Tells you which pack entries are
   going stale.

### 3.3 Structural mining — one genuinely new lens

Add syntactic-template mining (arXiv 2407.00211): POS-abstracted 4–8-gram
skeletons above a frequency threshold τ, mined from the frontier corpus and
scored against human-train the same way literal n-grams are. Deterministic,
no classifier, and it targets exactly the layer where frontier slop lives.
friction already builds POS-skeleton n-gram sets for attestation, so the
representation exists. This reuses it for *detection*.

### 3.4 External data worth harvesting into the literal inventory

- `berenslab/llm-excess-vocab` → `excess_words.csv` (900 words, frequency
  ratios, quantified confidence tiers for free).
- `sam-paech/antislop-sampler` → `slop_phrase_prob_adjustments.json`
  (phrase + weight pairs).
- `sam-paech/slop-forensics` → per-model profile pipeline. Run it against
  Claude/GPT via API to get per-model profiles the repo doesn't ship.
- Wikipedia "Signs of AI writing" + WikiProject AI Cleanup guide → best
  taxonomy for hand-conversion to rules (not machine-readable).
- `hardikpandya/stop-slop` → closest existing thing to friction-style
  deterministic rules, worth a diff against the inventory.
- Caveat: `AlpinDale/gptslop`'s `claudeslop.yaml` is roleplay/fiction tics,
  irrelevant to technical docs.

All imports get `source = "external"`-style provenance with the origin cited in
`notes`, then must clear the same mining cross-check against human-train before
becoming fix-tier (a word common in human technical writing must not be
substituted just because a public list names it).

## 4. Fix-side feasibility (friction rewrites, it doesn't just flag)

Mapping frontier tics onto the five existing operations:

**Directly extensible (existing operations, new data):**
- `ritual.delete`: sycophantic/validating openers, "In conclusion/Overall"
  closers, aphoristic kicker sentences.
- `span.delete` (+ seam attestation): hedge stacks, throat-clearing openers
  ("It's worth noting that", "At its core"), sentence-initial "Additionally,".
- `sub.apply`: era-2 lexical substitutions mined per §3, replacements attested
  in human train as today.
- `pivot.lvc`: already generic. PNAS's noun-density finding says frontier text
  is *richer* in nominalizations, so the existing 31 pairs likely fire more,
  not less, worth mining for new licensed pairs in the frontier corpus.
- Register bands (`transduce`): the natural home for density tics. New
  band-driven features to calibrate from human corpus: **em-dash rate**,
  discourse-marker density, possibly sentence-length CV. Same contract as
  nominalization/passive today: edit only while outside the human band, stop
  at the edge.

**Fixable with care (new templates, gated hard):**
- Participial tails: trailing `, VBG …` adjunct after a complete main clause is
  a deletable span. Clause-completeness and seam-attestation gates already
  express the safety condition. The tail restates rather than adds content, so
  deletion preserves meaning in the common case.
- Contrast frame: for the negated-contrast sentences described in §2.2,
  removing the rhetorical negation and keeping only the affirmative clause is
  a deterministic template, but content loss when the negated half carries
  real information makes this Suggest-tier first, promoted only after
  fixture evidence.

**Flag-only (Suggest tier / `check` report, no deterministic fix):**
- Tricolon fragments (rewrite requires synthesis, banned by the closure rule).
- Low burstiness / uniform sentence length (document-level, no local edit).
- Structural markdown bloat (bold-lead-in bullets, header shape), detectable
  cheaply from the existing block tree, but the fix is a rewrite-to-prose,
  which friction by design does not synthesize.

## 5. Recommended sequence (when development resumes)

1. **Frontier DMS index**: generate Claude/GPT corpora on the existing genre
   battery, `corpus-tool index` with new families. Highest use, zero new
   code, single-digit-dollar token cost. Immediately un-blinds `friction check
   --family claude`.
2. **Re-mine** (`mine`, `mine-inventory`) frontier-vs-human, curate new
   inventory entries with existing provenance discipline, and import external
   lists (§3.4) through the same cross-check.
3. **Prompted-suppression pass** to rank rules (survives-prompting = hard rule,
   prompt-suppressible = low tier) and to propose replacement candidates for
   attestation.
4. **New register bands** (em-dash rate first, cheap, well-evidenced,
   Claude-specific) and the participial-tail deletion template.
5. **POS-template miner** for the structural layer, plus contrast-frame as
   Suggest-tier finding.
6. **Drift maintenance**: pin each pack to the generating model version (the
   packs already embed corpus manifest hashes), re-run the battery per major
   model release, and keep era-1 words at reduced confidence rather than
   removing.

Fixture discipline is unchanged and is the safety rail for all of this: every
new rule lands with accept/reject fixtures, and no existing fixture flips.

## 6. Empirical results from the first claude-family corpus (2026-07-26)

The plan in §5 was executed the same day for the `claude` family: 180
Opus-generated documents (claude-opus-5 via Claude Code agents, default
register, prompts 1–36 of the existing five-genre battery), ingested as
train/dev only (sealed holdout untouched), indexed as a fifth DMS stream.

Findings, in order of confidence:

- **Attribution works, but sign-classification lags.** With 90 train docs the
  claude index became the best-matching family on 30/30 dev docs (from
  12/15 at 45 docs), and the median dev differential improved from −0.24
  to −0.08. But mean-match sign (differential > 0) still classified only a
  minority of claude dev docs as machine: frontier text genuinely sits
  closer to the human automaton, confirming the §1 evasion finding is
  about register distance, not just missing data. Corpus growth moves the
  needle roughly linearly. Expectations should stay below the
  llama-level 94% calibration.
- **The constructional hypothesis (§2.2) reproduced in-house.** A
  Monroe-log-odds pass of claude-train vs human-train with a dispersion
  filter (n-gram in ≥8 docs across ≥3 genres, to exclude topic words),
  cross-checked against the pooled other-llm rate: the top claude-favored
  items are `rather than` (z=11.8, 10× the other-llm rate), `not` /
  `not a` / `more than` / `did not` (the negated-contrast frame),
  spelled-out numerals `one/two/three/four` and `two or three`
  (enumeration/tricolon), totality markers `every` / `anything` /
  `the whole`, and the casual connective `so` / `so you can`. Almost none
  of these appear in small-model slop lists, and none are safely
  fixable as literal inventory entries (they are ordinary English, merely
  over-represented), so they stay out of the fix tier and instead
  validate the DMS channel plus the envelope's `not_just_but_rate` /
  `triad_rate` / `em_dash_density` metrics as the right detectors.
- **Band metrics fire as predicted.** On dev docs, `em_dash_density`
  exceeded the human band (e.g. 6.8 vs ≤4.24 per 1k words) while
  `llm_favored_phrase_rate` fell *below* its band: Opus does not use the
  small-model slop vocabulary at all. An em-dash register band remains
  the most promising new fix-side operation (§4).

## Sources

- Kobak et al., *Science Advances* 2025 — https://arxiv.org/abs/2406.07016 /
  https://github.com/berenslab/llm-excess-vocab
- "Why Does ChatGPT 'Delve' So Much?", COLING 2025 —
  https://arxiv.org/html/2412.11385v1
- AI-associated lexical shifts across 34 languages —
  https://arxiv.org/html/2605.25358
- Reinhart et al., "Do LLMs write like humans?", PNAS 2025 —
  https://arxiv.org/abs/2410.16107
- Syntactic templates in generated text — https://arxiv.org/html/2407.00211v2
- Monroe, Colaresi & Quinn, "Fightin' Words" (2008) —
  https://languagelog.ldc.upenn.edu/myl/Monroe.pdf
- EQ-Bench slop score — https://eqbench.com/slop-score.html /
  https://github.com/sam-paech/slop-score
- antislop-sampler — https://github.com/sam-paech/antislop-sampler (formalized
  in https://arxiv.org/pdf/2510.15061)
- slop-forensics — https://github.com/sam-paech/slop-forensics
- Wikipedia, Signs of AI writing —
  https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing
- WikiProject AI Cleanup guide —
  https://en.wikipedia.org/wiki/Wikipedia:WikiProject_AI_Cleanup/Guide
- stop-slop — https://github.com/hardikpandya/stop-slop
- RAID benchmark: https://arxiv.org/abs/2405.07940 ; MAGE (Li et al.) —
  https://arxiv.org/abs/2305.13242 ; M4 — https://arxiv.org/abs/2305.14902 ;
  M4GT-Bench — https://arxiv.org/abs/2402.11175
- Prompted detector-evasion trade-off —
  https://www.scitepress.org/Papers/2025/133575/133575.pdf
- Claude em-dash analysis — https://www.context-link.ai/blog/claude-em-dash-remover
- Claude 4 system prompt notes —
  https://simonwillison.net/2025/May/25/claude-4-system-prompt/
