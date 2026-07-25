# Register-vector prototype — vendored reference

Vendored **unmodified** from the research phase. These files are a reference
implementation, not a dependency: nothing in the workspace builds against
them, and nothing here runs at build or test time.

They play the same role for the register module that `../ref/` plays for the
four-operation engine — the behaviour a Rust port is checked against.

| file | what it is |
|---|---|
| `biber.py` | 18-feature register-vector extractor over spaCy. Explicitly *not* a faithful `pseudobibeR` port; its own header names that as the largest source of error. |
| `rewrite.py` | Five rewrite transducers with their licensing conditions, feature deltas, and inflection tables. |
| `home.py` | Target-vector derivation, log-space distance, greedy selection with re-evaluation. |
| `run.py` | Driver: extract, propose, select, apply, then re-extract and check predicted deltas against measured ones. |
| `output_rewritten.md` | A rewrite the prototype actually produced. |

## Reading notes

Two known defects are visible in `output_rewritten.md` and are useful as
test material rather than embarrassments:

- *"An explicit schema … was registered and eliminated the fallback"* —
  voice-mismatched coordination. Two edits on disjoint spans still
  interacted, because span non-overlap is not sufficient conflict detection
  between a passivisation and a coordinate predicate.
- Six sentences open with `This`. There is one repair realization per
  construction, so the selector can only trade one marked feature for
  another.

The delta-validation step in `run.py` is the load-bearing check: it caught a
feature-definition bug (complementizer *that* counted as a demonstrative)
that was invisible in the output text.

`home.py` derives its target as `v0 / published_ratio`, which assumes the
input is typical GPT-4o output. friction has a human corpus and does not need
that assumption.
