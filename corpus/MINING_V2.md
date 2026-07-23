# Inventory mining and DMS index — methodology, curation, and results

This report documents a second, separate data-tooling pass over the corpus (the
existing `corpus/MINING.md` and its discriminative n-gram mining is untouched
and unrelated). Two new `corpus-tool` subcommands were run for real against the
committed corpus, their output was mined and hand-curated into a new pack, and
both a determinism check and the full test suite were re-run against the
result.

## 1. What was actually run

Both commands were run against the release binary (`cargo build --release -p
corpus-tool`), train split only.

```
$ cargo build --release -p corpus-tool
    Finished `release` profile [optimized] target(s) in 10.41s

$ ./target/release/corpus-tool index \
    --corpus-dir corpus \
    --pack crates/friction-packs/packs/dms-index-v1.toml \
    --calibrate
index: wrote 18335 vocab token(s), 4 family stream(s), 264 human doc(s) to crates/friction-packs/packs/dms-index-v1.toml
dev calibration (not holdout):
  qwen: 58/67 dev doc(s) classified correctly
  gemma: 66/71 dev doc(s) classified correctly
  llama: 69/73 dev doc(s) classified correctly
  granite: 58/71 dev doc(s) classified correctly
# wall time: ~1.2s

$ ./target/release/corpus-tool mine-inventory \
    --corpus-dir corpus \
    --report corpus/MINE_INVENTORY.md
mine-inventory: wrote report to corpus/MINE_INVENTORY.md
# wall time: ~68s (dominated by NlpruleTagger model load + the POS-skeleton
# and LVC passes tagging every sentence in the corpus)
```

Both commands read `corpus/manifest.jsonl`, filter to
`record.split == Some(Split::Train)` (confirmed directly in source:
`crates/corpus-tool/src/commands/index.rs:147` and
`crates/corpus-tool/src/commands/mine_inventory.rs:121`), and never touch
`corpus/holdout.lock`. `index`'s `--calibrate` block is the one explicitly
labeled exception that reads `dev` (never `holdout`), per its own
`.filter(|r| r.split == Some(Split::Dev))` at `index.rs:388`, printed under an
explicit "dev calibration (not holdout)" banner as designed.

`index` produced `crates/friction-packs/packs/dms-index-v1.toml`: 18,335 vocab
tokens, one shared vocabulary across five streams (qwen/gemma/llama/granite/
human), matching the manifest's real per-family doc counts (qwen 38, gemma 81,
llama 75, granite 36, plus 264 human docs = 230 llm + 264 human = 494 train
docs total, consistent with both this run and `mine-inventory`'s own header).
The `--calibrate` dev-split sanity check classified 58-69 of ~70 dev docs
correctly per family (not part of the pack, a regression signal only).

`mine-inventory` produced `corpus/MINE_INVENTORY.md`: literal 2-/3-/4-gram
ratio mining, the POS-skeleton pass, block-position-conditioned mining
(preview-frame and ritual/closing-frame candidates), and the LVC seed-pair
rate table plus new-candidate discovery, all at the tool's default
thresholds (eps=0.4, min-human-token-freq=25, min-machine-count 8/5/4 for
n=2/3/4, top=50, min-lvc-count=2, skeleton pass on).

## 2. Determinism

Both subcommands were run a second time and diffed byte-for-byte against the
first run's output.

- `mine-inventory`, run twice independently (including a full re-load of the
  tagger model both times): identical sha256
  `86f5e9ffba644b95ac65d138f31ad923e1633190bebfd39c63f36e4ee448443e` for both
  copies of `MINE_INVENTORY.md`. `diff` between the two files: empty.
- `index`, run twice independently: identical sha256
  `3a6f36760358be28498230069733930ed8e7ecf9dd24d38f1afec30dcc50746b` for both
  copies of `dms-index-v1.toml`. `diff` between the two files: empty.

**Both subcommands are deterministic: two independent runs against the same
corpus produce byte-identical output.**

## 3. Inventory pack v1 curation

`crates/friction-packs/packs/inventory-v1.toml` was hand-curated from
`corpus/MINE_INVENTORY.md` plus the pre-existing runtime inventory
(`friction-harness::tellspan`'s RITUAL/SUBS/SPANS statics and
`friction-nlp::lvc`'s `DERIVATIONAL_LEXICON`). Nothing in this pack was
machine-generated; the mining tool only ever writes the markdown report.

### 3.1 Sections and counts

| section | entries | seed | mined | curated-seed |
|---|---|---|---|---|
| `deletion_spans` | 8 | 8 | 0 | 0 |
| `substitution_pairs` | 9 | 6 | 0 | 3 |
| `ritual_frames` | 8 | 4 | 3 | 1 |
| `preview_frame_rules` | 0 | - | - | - |
| `lvc_pairs` | 31 | 31 | 0 | 0 |
| `guard_tokens` | 4 lists, 34 tokens | (hand-transcribed closed classes, not mined) | | |

One correction against the design brief this was built from: `friction_nlp::lvc::DERIVATIONAL_LEXICON`
actually holds **31** entries, not 30 (the table's own doc comment and the
brief both say "30-entry" / "all 30 seed entries" — both undercount by one;
verified by direct enumeration of the source `BTreeMap::from([...])`
literal). All 31 real entries are transcribed into the pack; the miscount is
noted here rather than silently perpetuated.

### 3.2 Admitted: genuinely stylistic tells

Beyond the pre-existing seed inventory, three new entries were mined and
admitted this round, all whole-sentence, all 0 human-side occurrences on the
train split:

| id | pattern (essence) | machine count | human count | evidentiary tier |
|---|---|---|---|---|
| `ritual.hope_this_email_finds_you_well` | "I hope this email/message finds you well" | 16 | 0 | formally gated: "this email" 27/0 at 2-gram (gate needs >=8), "hope this email" 12/0 at 3-gram (gate needs >=5) |
| `ritual.would_you_like_me_to_elaborate` | "would you like me to <verb>" | 9 | 0 | formally gated: "would you like me"/"you like me to" both 11/0 at 4-gram (gate needs >=4), "like me to" 11/0 at 3-gram |
| `ritual.please_replace_bracketed_placeholders` | "(please) replace the bracketed placeholders..." | 4 | 0 | ungated — surfaced only in the block-position closing-frame raw counts; "bracketed"/"placeholders" are too rare in the human corpus to clear the per-token frequency gate, which is exactly why this one needed a curator's eye rather than the formal pipeline |

The first two passed the mining pipeline's own formal ratio-threshold gate
outright. The third is admitted despite thin evidence (n=4) because it is
qualitatively unambiguous: it is literal unfilled-template instruction text
leaking into a finished document, not topic vocabulary that happens to be
rare — the same category of judgment call the three frontier phrases below
also required, just via the block-position pass instead of curated-seed.

Two designated frontier phrases were curated in as `curated-seed`
substitution pairs (`sub.delve_into` / `sub.we_will_delve_into` for "delve
into" / "we'll delve into", and `sub.our_development_journey`), each with the
exact failing counts recorded in its own `notes` field (see §4).

### 3.3 Rejected: topic/content n-grams

Inspecting the top of `corpus/MINE_INVENTORY.md`'s formally-gated literal
machine-favored tables (2-/3-/4-gram, all passing the eps-ratio threshold
with 0 human occurrences) turned up a long run of entries that are specific
to this corpus's generation prompts, not general llm-vs-human register.
None of these were admitted:

| n-gram | machine count | human count | why rejected |
|---|---|---|---|
| memory file | 37 | 0 | topic: specific to prompts about a memory/journal file feature |
| github pull (requests) | 35 | 0 | topic: specific to prompts about a GitHub PR workflow |
| multi stage / stage builds / docker multi stage | 24 / 21 / 12 | 0 | topic: Docker multi-stage build docs |
| monitoring tools / our monitoring | 22 / 11 | 0 | topic: a specific monitoring-setup prompt |
| free list / the free list | 20 / 8 | 0 | topic: memory-allocator internals prompt |
| leave policy | 19 | 0 | topic: HR-policy document prompt |
| memory management | 17 | 0 | topic: memory-allocator internals prompt |
| advanced features | 15 | 0 | generic marketing filler, but too thin/ambiguous to separate from topic framing |
| migration script / migration process / database migration / the migration script / the migration process | 15 / 14 / 11 / 14 / 12 | 0 | topic: a specific database-migration prompt |
| two person / a two person / as a two person | 15 / 12 / 5 | 0 | topic: pair-programming-policy prompt |
| level keys / top level keys | 14 / 14 | 0 | topic: a YAML/config-format prompt |
| specific needs / your specific needs | 14 / 4 | 8 (block-position) | considered for the closing-hedge family (conceptually close to the seed `span.to_suit_your_needs_trailing`), rejected: no safe rewrite that doesn't risk deleting real content in some sentences, and it would duplicate an already-seeded hedge family without adding coverage |
| image size / instance type | 13 / 13 | 0 | topic: container/cloud-provisioning prompt |
| web root | 13 | 0 | topic: web-server-configuration prompt |
| job processing / job status | 12 / 11 | 0 | topic: a background-job-system prompt |
| storage space | 12 | 0 | topic: disk/storage prompt |
| api key | 11 | 0 | topic: authentication/API prompt |
| of pair (programming) | 11 / 7 | 0 / 2 | topic: pair-programming prompt |
| command line tool (+ "for"/"in"/"designed"/"written") | 37 / 9 / 9 / 8 / 5 | 0-1 | topic: the `readme`/`docs` genres were prompted to describe CLI tools, so this is a topic artifact of the prompt set, not general register |
| full text search | 10 | 0 | topic: a search-feature prompt |
| multiple log files | 10 | 0 | topic: a logging prompt |

Three additional mined candidates were evaluated and explicitly rejected for
being insufficiently clean, despite superficially promising raw counts:

| candidate | machine count | human count | why rejected |
|---|---|---|---|
| "feel free to `<verb>`" | 30 | 8 | too common in ordinary human writing too (8 human occurrences on this corpus alone) — not a clean machine/human split |
| "you should be able to" / "you'll be able to" | 11 | 3 | same problem: a normal hedge/encouragement phrase humans also use; not distinctively machine-register |
| "please refer to the official `<X>` documentation" | 2 | 0 | direction is right (0 human) but the count is too thin (n=2) to trust even as a curated-seed entry |

### 3.4 Preview-frame family: measured, not admitted

`preview_frame_rules` ships empty this round. The block-position pass found
249 first-paragraph-after-H1 candidates (33 from llm train docs, 216 from
human train docs — llm-generated docs use an H1-then-paragraph shape far
less often than the human corpus's blogs/READMEs do). Header-content-lemma
overlap ratio (candidate lemmas also present in the document's own header
text) averaged 0.222 on the machine side vs. 0.150 on the human side; using a
>=0.2-overlap threshold as the operationalization of "content lemmas covered
by adjacent structure," 13/33 (39%) of machine candidates cross it vs. 67/216
(31%) of human candidates. That is a real, correctly-signed but modest
effect (~1.26x), not the kind of clean separation the other admitted entries
show (all 0-on-the-human-side). No entry was curated in; the measurement and
its rejection are recorded here so the next `mine-inventory` run against a
larger corpus has a baseline to compare against.

## 4. Frontier phrase coverage

All three phrases named in the design brief were checked against the actual,
current train split (fresh grep-based measurement, not a re-quote of any
prior figure):

| phrase | count_M | count_H | key blocking constituent | route |
|---|---|---|---|---|
| "delve into" | 5 | 0 | `delve`: 0/264 human train docs | curated-seed substitution (`sub.delve_into` → `cover`) |
| "we'll delve into" | 3 | 0 | `delve`: 0 | curated-seed substitution (`sub.we_will_delve_into` → `we cover`) |
| "our development journey" | 1 | 0 | `journey`: 9/264 human train docs (gate needs >=25) | curated-seed substitution (`sub.our_development_journey` → `development`) |
| "fortunate enough to have had the opportunity to" | 1 | 0 | `fortunate`: 2 human occurrences; `opportunity`: 23 (gate needs >=25, just under) | curated-seed, `repair = "diagnostic_only"` (no safe closed-set rewrite — see rationale in the pack's own `notes` field) |

Per-token human-corpus frequencies measured directly this run (train split,
`[a-z']+` tokenization, matching the mining tool's own regex):
`delve`=0, `journey`=9, `fortunate`=2, `opportunity`=23, `into`=286 (so it is
specifically the rare content word in each phrase, not general phrase
rarity, that fails the gate). All three phrases are covered by the pack, via
the mining-vs-curation route the design predicted: none of them clear the
literal-mining thresholds on this corpus, and all three enter through
`curated-seed` with their exact failing counts recorded in the pack's own
`notes` field rather than the mining pipeline.

Replacement tokens for the two executable substitutions were spot-checked
against the human train corpus directly:

- `"cover"` / `"we cover"`: `cover` occurs 9 times and `covers` 6 times in the
  human train corpus as an intro-framing verb (e.g. "this guide covers the
  steps required," "this document will cover general methods," "we will
  cover every little detail") — the same slot the pre-existing
  `sub.will_walk_you_through` seed entry already uses.
- `"development"`: occurs 100 times in the human train corpus as a plain
  content noun.

## 5. Pack artifact registry

`crates/friction-packs/src/registry.toml` is reserved specifically for
downloadable, externally-sourced NLP artifacts (`friction setup`'s cache
mechanism) — its own header comment explains why it does not, and should
not, track locally-generated pack files, and it was left untouched. There is
no other pre-existing mechanism in this codebase for recording produced-pack
checksums, so the sha256 of every pack artifact touched or produced this
round is recorded here instead, alongside each pack's own
`corpus_manifest_sha256` provenance field:

| pack file | sha256 | corpus_manifest_sha256 |
|---|---|---|
| `crates/friction-packs/packs/inventory-v1.toml` | `69f81e7a1b7e6e0eed280f6ae75b29a8f31464cc89b9151b081294a6a4f8c4f5` | `001d33df8d362ee94ffd2d8e0fcdf811fd4ce1de34cd909a7d852793a34b6969` |
| `crates/friction-packs/packs/dms-index-v1.toml` | `3a6f36760358be28498230069733930ed8e7ecf9dd24d38f1afec30dcc50746b` | `001d33df8d362ee94ffd2d8e0fcdf811fd4ce1de34cd909a7d852793a34b6969` |
| `crates/friction-packs/packs/envelope-v2.toml` (unchanged, re-verified) | `cb10f536fefad0610d8523d0d00625cce9f76a78434fcd66a14bc48c73d3aee7` | `001d33df8d362ee94ffd2d8e0fcdf811fd4ce1de34cd909a7d852793a34b6969` |
| `crates/friction-packs/packs/mined-ngrams-v1.toml` (unchanged, re-verified) | `ba1be507d439f513828b039aedcb236e5cca4f2b8a94e701367270ec1de5b9d2` | (pre-existing pack, own provenance) |
| `crates/friction-packs/packs/envelope-v1.toml` (unchanged, re-verified) | `802033d18018a148ece0c9cc68348b283878867003e208bb0be6835906fcb1d6` | (pre-existing pack, own provenance) |

`corpus/manifest.jsonl` itself hashes to
`001d33df8d362ee94ffd2d8e0fcdf811fd4ce1de34cd909a7d852793a34b6969` at the
time both new packs were built, matching the two pre-existing packs
(`envelope-v2.toml`, `dms-index-v1.toml`) that also embed it — all four packs
were built against the identical corpus snapshot.

## 6. Test status

`cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D
warnings`, and `cargo test --workspace` were all re-run with both real packs
present on disk, after this milestone's work (including the new
`inventory-v1.toml` and `dms-index-v1.toml` pack files) was in place:

- `cargo fmt --check`: clean (exit 0).
- `cargo clippy --workspace --all-targets -- -D warnings`: clean (exit 0).
- `cargo test --workspace`: 60 `test result: ok` blocks, **958 tests passed,
  0 failed**, 0 unexpected. Total wall time ~4m30s.

## 7. What this milestone deliberately does not do

Nothing in `friction-harness` reads either new pack this round — the runtime
still uses its own hardcoded RITUAL/SUBS/SPANS statics and the licensed
`DERIVATIONAL_LEXICON`, unchanged. `preview_frame_rules` ships empty pending
either a larger corpus or a different operationalization of the header-
overlap signal. `ritual.fortunate_enough_to_have_had_the_opportunity_to`
ships `diagnostic_only`: recorded, never executed as a rewrite, because no
closed-set operation in this system safely collapses it.

## 8. Paired-recipe mining

A third, separate pass, run after the paired stock/antislop corpus
(`corpus/paired/`) was generated. Unlike §1–§7's `mine-inventory` (pools all
`corpus/llm` models against `corpus/human`, so topic and register are
confounded whenever a prompt's subject matter happens to correlate with
which model answered it), `mine-paired` compares the *same* base model
(`gemma3:12b`) against its antislop-tuned counterpart answering the
*identical* prompt under the *identical* derived seed. Topic cancels by
construction at the per-prompt level, so whatever survives at a clean ratio
is a register difference the tuning specifically targeted, not a subject
one model happened to be asked about more.

### 8.1 What was run

```
$ cargo build --release -p corpus-tool
    Finished `release` profile [optimized] target(s) in 9.40s

$ ./target/release/corpus-tool mine-paired \
    --corpus-dir corpus --paired-dir corpus/paired \
    --report corpus/MINE_PAIRED.md
mine-paired: wrote report to corpus/MINE_PAIRED.md
```

Read `corpus/paired/manifest.jsonl` (55 pairs / 110 docs: 55 stock,
55 antislop — confirmed directly against the header line of
`corpus/MINE_PAIRED.md`), pooled sentences across all five genres, at the
tool's default thresholds (eps=0.4, min_antislop_token_freq=8,
min_stock_count n2/n3/n4=8/5/4, top=50). The human-train cross-check pass
separately reads `corpus/manifest.jsonl`, restricted to
`Class::Human && Split::Train` (264 docs) — confirmed directly in source at
`crates/corpus-tool/src/commands/mine_paired.rs:236-240` — and never touches
`corpus/llm` or `corpus/holdout.lock`. `corpus/paired/` itself is never
manifest-tracked, never split, and was untouched by this pass (read-only).

### 8.2 Determinism

Run twice independently, full report diffed byte-for-byte:

- Run 1 sha256: `64bac35477f80adf898d85985a3099fc24891a810cafd7ba49e10d2834644752`
- Run 2 sha256: `64bac35477f80adf898d85985a3099fc24891a810cafd7ba49e10d2834644752`
- `diff` between the two copies of `corpus/MINE_PAIRED.md`: empty.

**`mine-paired` is deterministic: two independent runs against the same
paired corpus produce a byte-identical report.**

### 8.3 Admitted: genuinely stylistic machine-register entries

`corpus/MINE_PAIRED.md`'s stock-favored tables (the only eligible direction
— see the antislop-favored banner rendered in the report itself, and §8.4
below) hold 121 rows total across the three n-gram orders (50 + 50 + 21).
Every row was inspected. Three families were admitted, all present across
multiple, topically unrelated paired genres (so not one prompt's subject
matter), all with a `human_train_count` of 0 or 1 out of 264 train docs, and
— the key test this round adds — **none of the three appear anywhere in
`corpus/MINE_INVENTORY.md`'s formally-gated literal tables**, confirming
they are exactly the "topic-diluted in the pooled mixed-model corpus, but
clean once topic is cancelled per-prompt" case `mine-paired` exists to
catch:

| id | pattern (essence) | stock | antislop | human_train | mine-inventory coverage |
|---|---|---|---|---|---|
| `sub.crucial_for` | "crucial for" / "is crucial for" | 15 / 9 | 3 / 1 | 1 / 0 | absent (0 rows) |
| `sub.is_crucial` | "is crucial" / "this is crucial" | 16 / 8 | 8 / 2 | 1 / 0 | absent (0 rows) |
| `ritual.let_me_know_if_you_would_like` | "let me know if you'd like..." (raw 3-gram "me know if") | 11 | 3 | 1 | one thin, ungated candidate only (n=1) |

- **`crucial` family** (`sub.crucial_for`, `sub.is_crucial`): the standalone
  word `crucial` is already flagged as an llm-favored intensifier by the
  unrelated `mine`/`mined-ngrams-v1.toml` pipeline (z=6.4922) — so the
  *word* was already known to skew machine — but no phrase built on it ever
  cleared `mine-inventory`'s literal 2-/3-/4-gram gate on the pooled
  corpus (spot-checked: `corpus/MINE_INVENTORY.md` contains a single
  "a crucial task to" candidate at n=1, nowhere near the gate). Paired
  mining surfaces "crucial for" / "is crucial for" / "is crucial" / "this is
  crucial" at a clean stock/antislop ratio (scores 2.0–6.95), spread across
  blog (8 stock docs), docs (7), email (8), forum (8), and readme (2) —
  every genre, not a topic. This is now an *executable* pair, not just a
  metric marker: replacement "important for" is grep-verified against the
  human train corpus (9 occurrences). Replacement "matters" occurs 12 times
  in the human train corpus, but only 8 of those are the plain
  predicate-verb sense this substitution needs ("this matters", "order
  matters"); the other 4 are the unrelated noun sense "matters of X" /
  "such matters" and do not support the verb-slot claim — see
  `sub.is_crucial`'s own `notes` in `inventory-v1.toml` for the count
  breakdown.
- **`ritual.let_me_know_if_you_would_like`**: an assistant-voice closing
  continuation offer ("Let me know if you'd like any refinements/
  adjustments/modifications!"), the same family as the pre-existing
  `ritual.would_you_like_me_to_elaborate` seed. The raw gated 3-gram "me
  know if" (stock=11, antislop=3, human_train=1) is spread across blog (6),
  email (3), forum (2) — genre-conditioned (a closing-offer only makes
  sense where the model is handing over a drafted artifact) but not
  topic-bound. `mine-inventory` only ever surfaced a single, thin, ungated
  "let me know if" candidate (n=1) on the pooled corpus. The bare phrase
  "let me know if" is broader than the tell, though: it also matches
  ordinary human continuation requests with no chatbot register at all
  (a genuine hit in `corpus/quarantine/forum/a53feca0624ff561.md`, a
  human/train doc: "Good luck, and please let me know if you find that
  \"magic bullet.\""). The admitted pattern is narrowed, the same way
  `ritual.would_you_like_me_to_elaborate` narrows its own raw n-gram, to
  the specific continuation-offer construction ("let me know if you'd
  like" / "let me know if you would like") the id names — curator-refined
  counts against `corpus/paired/`: stock=9, antislop=1, human_train=0 (the
  one human hit above does not match the narrowed form). See the entry's
  own `notes` in `inventory-v1.toml` for the full count breakdown.

### 8.4 Rejected

Every other stock-favored row was inspected and rejected. Grouped by reason
(counts are `stock` / `antislop` / `human_train` unless noted):

**Topical / genre-task artifacts** — specific to one prompt's subject
matter, or to what a genre's prompts literally ask the model to produce
(mirrors the exact discipline §3.3 already applied to `mine-inventory`'s
output):

| n-gram(s) | counts | why rejected |
|---|---|---|
| `move cursor` | 12/2/0 | topic: a text-editor-feature prompt |
| `free list` | 13/6/0 | topic: memory-allocator prompt — literal repeat of §3.3's exact prior rejection ("free list / the free list", 20/8/0) |
| `front matter` | 14/8/2 | topic: a Jekyll/Hugo blog-authoring prompt |
| `multi stage builds` | 6/4/0 | topic: Docker multi-stage build docs — repeat of §3.3's prior rejection |
| `notes at` / `notes at the` / `the notes at` / `read the notes` / `please read the notes` / `read the notes at` / `notes at the end` | 12/7/0, 12/7/0, 7/5/0, 7/5/0, 7/5/0, 7/5/0, 4/2/0 | genre-task artifact: whenever the stock model is asked to draft an email/blog post/README on someone's behalf, it tends to append a "please read the notes at the end/bottom" pointer; direction is also inconsistent across n-gram orders (the closely related 4-gram "notes at the very" is actually antislop-favored, 4/5) — not a clean, direction-consistent register tell, just genre-conditioned verbosity around the drafting task itself |
| `blog post draft` / `a blog post draft` / `here's a draft` / `here's a blog` | 9/4/0, 9/4/0, 9/8/0, 9/9/0 | genre-task artifact of the blog genre's own prompt framing ("draft a blog post..."), the same category §3.3 already used to reject "command line tool" for the readme/docs genres; the last two ("here's a draft"/"here's a blog") are also near-1:1 ratios (9 vs 8, 9 vs 9) — essentially no separation |
| `a draft email` / `the email` | 5/2/0, 10/1/0 | genre-task artifact of the email genre — both models describe their own output as "the email" when explaining it |
| `the path to your` / `path to your` / `your photo library` | 4/0/0, 6/2/0, 5/3/0 | topic: a specific photo-library-path prompt |
| `food bank name` | 9/7/0 | topic: a food-bank-newsletter prompt with a literal placeholder field |
| `the worker` | 8/4/3 | topic: a background-worker-process prompt |
| `i've aimed for` / `i've aimed for a` / `i've aimed for clarity` / `i've included some` | 15/12/0, 6/1/0, 4/3/0, 6/1/0 | weak, near-1:1 separation (15 vs 12, 6 vs 1 is thin, 4 vs 3 is thin) on a *shared* base-model habit — grep-verified directly: the parent construction "Okay, here's a ... draft" occurs at **exactly 28/28** stock/antislop, i.e. completely uncancelled by the antislop tuning. The mild skew on "aimed for"/"included" reads as sampling noise on a habit both models share equally, not something antislop specifically targeted |

**Generic / common-in-human-writing** — high `human_train_count` (roughly
≥20, often far higher), so common in ordinary human prose that a machine/
human split isn't clean regardless of the stock/antislop ratio (mirrors
§3.3's exact "feel free to" / "you'll be able to" precedent, which is
itself repeated verbatim here at nearly the same counts):

`or a` (37), `but the` (45), `it's not` (33), `all the` (104), `feel free`
/ `free to` / `feel free to` (8/12/8 — literal repeat of §3.3's prior
rejection), `may be` (71), `as well` (96), `can also` (63), `we have` (94),
`allow(s) you (to)` (13/23/11), `the data` (42), `built in` (35), `to do`
(93), `you need` (61), `they are` (104), `into the` (94), `the end` (45),
`the problem` (28), `here's how` (8), `you to` (72), `with the` (320),
`us to` (37), `one of` (115), `the same` (251), `number of` (109), `a full`
(13), `over the` (51), `the next` (45), `need to be` (16), `with the
following` (10), `so you can` (19), `at the end` (14), `you want to` (85),
`you have a` (28), `if you have` (64), `this is a` (54), `is designed to`
(8), `a list of` (23), `we want to` (26), `the number of` (22),
`designed to be` (7). None of these show a clean split from human register;
several (`with the`, `the same`, `they are`, `all the`) have human counts in
the hundreds.

**Thin / dubious / ambiguous** — surfaced but not admitted for insufficient
evidence or ambiguous meaning:

| n-gram | counts | why rejected |
|---|---|---|
| `like any` | 9/1/3 | too ambiguous a fragment (could be "not like any other...", a comparative, or several other constructions) to write one safe rule around |
| `building a` | 10/3/4 | generic gerund opener, no clear register signal beyond raw frequency |
| `the root` | 10/4/12 | ambiguous across senses ("web root" / "root cause" / "root directory") — likely topic-conditioned per sense, not a clean register marker |
| `is a common` / `thank you for` / `wanted to share` / `when dealing with` | 5/3/3, 5/3/3, 5/3/1, 7/5/1 | thin counts (n≤5 antislop-side) on otherwise-common courtesy/hedge phrasing; not clean enough to trust |
| `your specific` / `on your specific` / `to your specific` / `based on your specific` | 19/8/1, 7/1/0, 5/4/0, 5/0/0 | genuinely topic-cancelling (spans blog/docs/email/forum on entirely unrelated subjects: environment variables, database choice, regex patterns, backups, drive models, photo-library characteristics) and directionally very clean (0-1 human), but — re-examined against the actual sentences containing it — too structurally variable for a safe closed-set rewrite: it appears as a sentence-initial subject ("Your specific implementation will vary..."), as an object of a preposition mid-clause ("adjust to match your specific recipe"), and as a bare hedge tail ("...depending on your specific environment"); a trailing-deletion rule would sometimes strip real content and sometimes leave a dangling clause. This is the same structural risk §3.3 already flagged when rejecting "specific needs / your specific needs" (14/4/8 there) and would also duplicate the pack's existing `span.to_suit_your_needs_trailing` hedge family without adding safe coverage. Left rejected, now with much stronger topic-cancelled evidence recorded here should a future milestone find a safer operationalization (e.g. once the DMS automaton mentioned in `ritual.fortunate_enough_to_have_had_the_opportunity_to`'s notes is wired in, which could match the span without needing a literal, position-blind regex) |

The antislop-favored tables (diagnostic-only per the report's own banner —
`crucial`'s antislop-side counterpart shows up there too, as `is critical`/
`this is critical`, meaning the tuning appears to have specifically
targeted the *word* "crucial" rather than the underlying "X is
{intensifier}" construction) were read for context but never treated as a
substitution-pair source, per the pack's own curation convention and the
project rule that the human side of every pair must come from real human
text.

### 8.5 Frontier phrase cross-check against the paired tables

All three frontier phrases already curated into the pack (`§3`/`§4` above)
were checked directly against `corpus/paired/` (55 pairs, both sides) —
grep-verified, not a re-quote of any prior figure:

| phrase | stock | antislop | human_train | paired mining confirms? |
|---|---|---|---|---|
| "delve into" | 1 | 0 | 0 | No — directionally consistent (1 vs 0) but far below any threshold; a 110-doc corpus is too small to expect this phrase to recur |
| "we'll delve into" / "we will delve into" | 0 | 0 | 0 | No — 0 occurrences on either side |
| "our development journey" | 0 | 0 | 0 | No — 0 occurrences on either side |
| "fortunate enough to have had the opportunity to" | 0 | 0 | 0 | No — 0 occurrences on either side |

None of the four are confirmed by paired mining with real counts; all four
remain `curated-seed` (`ritual.fortunate_enough_to_have_had_the_opportunity_to`
also remains `diagnostic_only`) on the strength of the train-split
measurements already recorded in `§3`/`§4`. This is expected, not a
weakness in the paired corpus: at only 55 prompts, a specific 2-4-word
phrase that occurred once or a handful of times across the full 494-doc
train split has no real chance of recurring in a corpus roughly a fifth the
size — the paired corpus's job is topic cancellation for phrases common
enough to show up at all in 55 prompts, not a substitute for the larger
corpus's raw statistical power. The cross-check is recorded here, in each
entry's own `notes` field in `inventory-v1.toml`, and is not itself grounds
to change any of the four entries' `source` or `repair`.

### 8.6 Pack changes summary

`crates/friction-packs/packs/inventory-v1.toml`:

- Added `mine_paired_report_path = "corpus/MINE_PAIRED.md"` and
  `pack.paired_manifest_sha256` (sha256 of `corpus/paired/manifest.jsonl`
  at the time of this pass) alongside the existing
  `mine_inventory_report_path` / `pack.corpus_manifest_sha256` provenance
  fields.
- Documented `"mined-paired"` as a fourth `source` value in the header
  comment, alongside `"seed"` / `"mined"` / `"curated-seed"`.
- Added 2 `substitution_pairs` entries (`sub.crucial_for`, `sub.is_crucial`)
  and 1 `ritual_frames` entry (`ritual.let_me_know_if_you_would_like`), all
  `source = "mined-paired"`, all with real stock/antislop/human_train counts
  recorded in `notes` (and, for the ritual frame, in explicit
  `train_stock_count` / `train_antislop_count` / `train_human_count`
  fields, mirroring `mine`-sourced ritual entries' existing
  `train_machine_count` / `train_human_count` convention).
- Appended a paired-mining cross-check paragraph to each of the four
  existing frontier-phrase entries' `notes` (`sub.delve_into`,
  `sub.we_will_delve_into`, `sub.our_development_journey`,
  `ritual.fortunate_enough_to_have_had_the_opportunity_to`) recording the
  §8.5 result — no `source`/`repair` on any of the four changed.
- `substitution_pairs`: 11 → 13 entries. `ritual_frames`: 8 → 9 entries.
  `deletion_spans` (8) and `lvc_pairs` (31) untouched this round.
- `corpus/manifest.jsonl` itself is unchanged
  (`001d33df8d362ee94ffd2d8e0fcdf811fd4ce1de34cd909a7d852793a34b6969`,
  matching every prior round); `corpus/paired/manifest.jsonl` hashes to
  `c67c42b15e19bf6a152fefd9e835694ae509b95585591c26cffe238668a0c9fa` at the
  time this pass was built.

Nothing in `friction-harness` reads this pack yet (see §7) — this round adds
entries to the same not-yet-wired pack, it does not change that.
