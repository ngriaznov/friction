# Register-vector prototype — vendored reference

Vendored **unmodified** from the research phase. These files are a reference
implementation: nothing in the workspace depends on them, builds against
them, or runs them at build or test time.

They play the same role for the register module that `../ref/` plays for the
four-operation engine: the behaviour a Rust port is checked against.

| file | what it is |
|---|---|
| `biber.py` | 18-feature register-vector extractor over spaCy. Explicitly *not* a faithful `pseudobibeR` port; its own header names that as the largest source of error. |
| `rewrite.py` | Five rewrite transducers with their licensing conditions, feature deltas, and inflection tables. |
| `home.py` | Target-vector derivation, log-space distance, greedy selection with re-evaluation. |
| `run.py` | Driver: extract, propose, select, apply, then re-extract and check predicted deltas against measured ones. |
| `output_rewritten.md` | A rewrite the prototype actually produced. |
| `feature_parity.json` | Generated, not vendored — see below. |
| `TARGET_ESTIMATION.md` | What the corpus actually supports, measured. |

## `feature_parity.json`

188 sentences drawn from the docs and readme train split, each carrying the
reference dependency parse *and* the per-feature counts the reference
extractor produced from it. Tokens are `[surface, penn_pos, head_index,
relation]`, `head_index` of `-1` marking the root.

Shipping the parse with the counts is the point. A port can be fed the
parse from this file and checked on counts alone, which separates an
extractor bug from a parser bug. Without separating them, a parity mismatch
only shows that something broke, which is exactly how the prototype ended
up silently miscounting a feature.

Sentences are selected by stratifying over both features and relations
rather than at random, so rare items show up often enough to be usable:
`agent` and `csubj` each appear 7 times here, against roughly 7 occurrences
in an entire 20% test split of a random 1,900-sentence sample.

Two things it pins that are easy to get wrong:

- `agent` **suppresses** `agentless_passive`. One fixture sentence has
  `auxpass: is` and a count of 1; another has `agent: by` and a count of 0.
- Complementizer *that* must not count as a demonstrative: that's the bug
  the prototype's delta-validation step caught.

Sentences are segmented from raw Markdown rather than by
extracting prose the way friction itself does, so a few carry heading text
that the prose-blocks-only gate would never pass to the real pipeline. This
does not affect parity (both sides see identical input), but the sentences
are not uniformly representative of real-world input.

## Reading notes

Two known defects are visible in `output_rewritten.md` and are useful as
test material rather than embarrassments:

- *"An explicit schema … was registered and eliminated the fallback"*: the
  coordinated clauses ended up with mismatched voice. Two edits on disjoint
  spans still interacted, because non-overlapping spans are not enough to
  detect a conflict between a passive rewrite and a coordinate predicate.
- Six sentences open with `This`. Each construction has only one way to be
  repaired, so the selector can only trade one marked feature for another.

The delta-validation step in `run.py` is the load-bearing check: it caught
a bug in how a feature was defined (complementizer *that* counted as a
demonstrative) that was invisible in the output text.

`home.py` derives its target as `v0 / published_ratio`, which assumes the
input is typical GPT-4o output. friction has a human corpus and does not
need to assume that.
