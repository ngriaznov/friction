# Beyond the closure rule: deterministic synthesis and jargon attestation

Research notes, 2026-07-27. Nothing here is implemented. Companion to
[FRONTIER_MODELS.md](FRONTIER_MODELS.md), which established that the
strongest frontier-model tells (contrast frames, tricolon, uniform sentence
length, pseudo-jargon) are currently detect-only. These notes ask how far
deterministic, meaning-preserving repair can reach into that territory, and
they revise an earlier call: "flag-only forever" was too strong for three of
the four.

## 0. The closure rule is a budget, not a ban

friction already synthesizes, but only from two closed vocabularies: the
inventory pack's `closure_function_word_allowance`
(`a/an/the/to/of/and/or`) and the register pass's `PERMITTED_FUNCTION_WORDS`
(`was/were/is/are`). Every "synthesis" proposed below widens that budget
under new gates. It is not a new kind of machinery. The published work
validates exactly this shape: every mature rule-based rewriting system
inserts connectives **only from a closed list, chosen deterministically by
the discourse relation being preserved** (Siddharthan 2003/2006; DisSim,
Niklaus et al. ACL 2019; DiscoFuse, Geva et al. NAACL 2019: its published
connective inventory is ~50 entries, partitioned by syntactic role,
composable only in pre-approved combinations).

A realistic ceiling to plan around: the best deterministic systems are
judged correct 82–86% of the time overall by human raters, but do sharply
better on specific construct classes (DisSim: **99.1% on coordinate-clause
splits**). That ceiling argues for adopting per-construct operations in
descending safety order, never a general rewriter.

## 1. What needs no new synthesis at all (do first)

- **Contrast-frame de-fanging: deleting "just".** "is X *just* A, or B?" →
  "is X A, or B?" Deleting the dismissive particle kills the rhetorical
  framing while preserving the genuine disjunctive question. This reuses
  the machinery that already deletes spans (the same class as `simply`).
  No new machinery needed.
- **Tricolon fragment merge.** "Fast. Simple. Effective." → "Fast, simple,
  effective." (punctuation-only, all content stays). Same family as the ops
  that touch only em dashes and semicolons, driven by a fragment-run
  detector (2+ consecutive verbless single-phrase sentences).
- **Contrast-frame Suggest finding.** Detecting this pattern is unusually
  tractable on machine text specifically: Boggia (arXiv 2607.21498, 2026)
  studies the "not X, it's Y" figure as classical *epanorthosis* and
  measures detection precision at 0.82 on LLM output vs 0.17 on human text:
  LLM instances are template-canonical. A deterministic template detector
  (lexical frame + parse check) feeding fix's existing paraphrase report
  clears friction's precision bar for a Suggest tier on exactly the text
  friction targets.

## 2. Sentence splitting and fusion (widen the budget, gated)

The burstiness tell (uniform sentence length) has no direct repair prior
art, but both component operations do, with published safety conditions:

- **Split** (long uniform sentences): DisSim's 35-rule catalog covers ~10
  construct classes. Adopt only the safest class first: coordinate-clause
  splits (99.1%), which is structurally the semicolon op generalized to
  `", and"`/`", but"`/`", so"` seams: both conjuncts must be independent
  clauses (the existing subtree-anchored subject check), no shared
  dependents crossing the seam (the existing conflict machinery), sentence
  break + recapitalization (the existing closure-invisible mechanics).
  Siddharthan contributes the cohesion warning: a split must not orphan the
  discourse relation the conjunction encoded: for `and` (pure addition),
  deleting it is safe. For `but`/`so`, the relation must be re-encoded by a
  sentence-initial connective from a closed cue list (`However,` / `So,`)
  chosen 1:1 from the deleted conjunction. That cue list is the
  closure-budget widening, and it is exactly the shape the published work
  licenses.
- **Fuse** (runs of short choppy sentences, the opposite problem):
  DiscoFuse's nine rule-based fusion phenomena with closed connective
  lists. Lower priority: LLM text errs long, not short.
- **The controller is novel**: drive both from the existing
  `sentence_length_cv` band (already measured per genre in envelope-v2,
  would need a register-pack band measured like em_dash/semicolon),
  stopping at the band edge exactly like every register feature. Nobody in
  the published work has aimed split/fuse at length variance. The parts
  are proven, the objective is new.
- **Gates worth importing**: SAMSA's criterion (each output sentence equals
  one proposition with participants intact, per Sulem et al. NAACL 2018) is
  reference-free and could be approximated with friction's own dependency
  parser as a structural gate. MacCartney & Manning's natural-logic
  relations give the formal target: an edit is licensed only if it composes
  to *equivalence*, not mere entailment. PPDB 2.0's entailment-labeled,
  quality-scored paraphrase rules (Pavlick et al. 2015) are a candidate
  audited-inventory source (license to be verified before any use).

## 3. Contrast-frame foil deletion (friction would be first)

The full repair ("is X just A, or B?" → "B?" and "It's not just X — it's
Y." → "It's Y.") only deletes text. Nothing is synthesized. It still drops
the rhetorical foil, though, which names the rejected alternative. Boggia's
paper confirms no published deterministic repair exists. All known
mitigations are training-time (fine-tuning, steering, prompting). Proposed
path: ship as a Suggest-tier finding first (the 0.82-on-machine-text
detector), gather fixture evidence on which epanorthosis subtypes lose real
content when the foil is deleted (emphatic-correction usually loses
nothing, while properly-corrective usually does), then promote the safe
subtype into a gated fix that deletes it. This mirrors how every friction
operation earned fix-tier status.

## 4. Pseudo-jargon: the attestation trick generalizes

"semantic wells", "cross-domain resonance", "centroid-drift detection" have
a precise statistical pattern: **every head word frequent, the compound
unattested**. This is the core idea behind weirdness-ratio and termhood
work (Frantzi & Ananiadou's C-value; Ahmad's weirdness ratio), and
web-scale attestation licensing already runs at production precision
elsewhere (real-word spell correction over Google Web 1T: ~99% on non-word,
~70% on context errors). No one has packaged it as an LLM pseudo-jargon
flagger. Published work on hallucination covers citations, not vocabulary.

Design (detection-only, Suggest tier, no auto-fix, since nothing
deterministic can replace an invented term):

1. **Extracting candidates**: noun-noun / adj-noun compounds in prose
   scope, after syntactic excludes: capitalized (product names), backticked
   or code-formatted, hyphenated project identifiers.
2. **Attestation gauntlet**, all compiled offline into a pack (friction
   runtime stays offline and deterministic):
   - curated allowlist: Wikipedia titles + redirects (CC BY-SA, ~6.8M),
     OpenAlex Topics (CC0, ~4.5k), ACM CCS. Stack Overflow tags are the
     best software-vocabulary source (~65k with usage counts) but the dump
     has been login-gated since 2024 with use restrictions: verify before
     including it, or use the StackLite mirror instead;
   - web-scale n-gram zero-check: infini-gram (exact counts over 5T tokens)
     queried at *table-compile time*, never at runtime.
   - Pack-size note: a Bloom filter over attested compounds fits the
     embedded-pack model, and it fails safe: a collision reads as
     "attested", suppressing a flag rather than inventing one.
3. **When to flag**: both heads sit above a frequency floor in the
   reference corpus, and the compound is absent from every table. An
   optional second signal: a small static list of metaphor-borrowed
   modifiers (`well`, `resonance`, `drift`, `soup`, `symphony`, `fabric`)
   that raises confidence when paired with a technical head.
4. **Precision target**: ≥90–95% (Google's Tricorder bar for analyzers
   that developers don't disable). Published work says this is reachable
   only on the clean cases (hard-zero web attestation plus a missing
   curated-list entry), and borderline coinages must still pass silently.
5. **Honest framing** (the same honesty as the seam-attestation tables):
   this detects "vocabulary unattested as of table-compile date," not
   "hallucinated jargon." New, real terminology inside the table
   lag window is the systematic false positive, but the excludes and
   shipping it at Suggest tier keep that in check.

## 5. Recommended sequence

1. Deleting "just" in contrast frames, plus the tricolon fragment merge
   (existing machinery, small inventory/detector additions, fixtures).
2. The contrast-frame Suggest finding (epanorthosis template detector →
   paraphrase report).
3. A coordinate-clause split behind a `sentence_length_cv` register band,
   with the closed cue-word mapping (`but`→`However,`, `so`→`So,`) as the
   first deliberate closure-budget widening.
4. The pseudo-jargon attestation pack plus Suggest findings (compiling the
   tables is the bulk of the work. Running them is just a lookup).
5. Promoting the foil-delete fix (after fixture evidence), and fusion, if
   short-run choppy sentences ever turn up in the measured corpus (they
   currently don't).

Every step keeps the fixture discipline: no existing fixture flips, and
each new op lands with its own accept/reject fixtures.

## Sources

- DisSim: https://aclanthology.org/P19-1333/ /
  https://github.com/Lambda-3/DiscourseSimplification
- Siddharthan, syntactic simplification & text cohesion:
  https://link.springer.com/article/10.1007/s11168-006-9011-1
- Split-and-Rephrase: https://aclanthology.org/D17-1064/
- BiSECT: https://aclanthology.org/2021.emnlp-main.500/
- DiscoFuse: https://aclanthology.org/N19-1348/
- SAMSA: https://aclanthology.org/N18-1063/
- Natural logic: https://nlp.stanford.edu/pubs/natlog-wtep07.pdf
- PPDB 2.0: https://aclanthology.org/P15-2070.pdf
- Simple PPDB: https://aclanthology.org/P16-2024.pdf
- Epanorthosis in LLMs (Boggia 2026): https://arxiv.org/abs/2607.21498
- Rhetorical parallelism detection:
  https://aclanthology.org/2023.emnlp-main.305/
- ATE survey: https://arxiv.org/abs/2301.06767
- C/NC-value: Frantzi & Ananiadou
- weirdness ratio: Ahmad et al. 1999
- Novel noun-noun compound plausibility: https://arxiv.org/pdf/1906.03634
- Web 1T real-word correction: https://arxiv.org/pdf/1204.5852
- infini-gram: https://infini-gram.io/
- OpenAlex Topics: https://docs.openalex.org/api-entities/topics
- Wikipedia dumps: https://dumps.wikimedia.org/
- StackLite: https://github.com/dgrtwo/StackLite
- Tricorder (linter precision bar):
  https://research.google.com/pubs/archive/43322.pdf
