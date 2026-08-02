# friction v4.1 handoff bundle

For the coding agent implementing the friction rewrite. Read in this order:

1. friction-build-spec.md (sibling file, one level up) — the contract: scope,
   operations, gates, crate mapping, invariants, milestones.
2. ALGORITHMS.md — exact procedures for everything the spec names: DMS,
   mining, the four operations, every gate, with measured numbers and the
   byte-span production note (NF-5).
3. fixtures.json — literal accept/reject/rank cases from the validation
   session. Wire these into the M0 harness FIRST, before any engine code.
   If a change makes a reject-fixture pass, the change is wrong.
4. ref/ — runnable Python prototypes, the exact code the evidence ledger
   cites. ref_dms.py (detection), ref_pivot.py (operation 4),
   ref_engine.py (combined pipeline). Run against the repo corpus:
   `python3 ref/ref_dms.py /path/to/friction/corpus`. They are string-level
   references; production emits byte-span patches.
5. samples/: chatspeak.md and good_docs.md, the rank-fixture documents.

The one sentence that governs everything: friction deletes, substitutes from a
mined inventory, or derivationally pivots words already in the text; it never
searches for words to insert, and where no gate-passing edit exists it reports
instead of editing.
