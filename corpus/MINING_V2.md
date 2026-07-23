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
