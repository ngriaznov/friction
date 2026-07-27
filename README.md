# friction

A deterministic engine that removes the machine layer from LLM-generated technical
documentation — the ritual closers, filler spans, hedge phrases, and light-verb
constructions that make text read as machine-written — without touching the
information.

**English technical documentation only.** Both constraints are real, not
aspirational, and neither is checked at runtime:

- **English.** Every shipped artifact is English-trained — the tagger, the
  dependency parser, the sentence segmenter, and every phrase in the tell
  inventory. There is no language detection and no other language pack. On
  non-English prose friction is inert rather than wrong (measured: zero edits on
  German, Spanish, Dutch, Italian and Portuguese), but it will repair English
  sentences embedded in another language. See the limits below.
- **Technical documentation.** Reference docs, READMEs, design notes,
  postmortems, migration guides. The inventory was mined from that kind of
  writing, and the register pass is calibrated against it specifically. It is
  not a general-purpose editor and not for prose whose voice is the point —
  fiction, marketing, personal writing.

friction never invents content. Every content word it emits is already present in
the input or derived from one through a static table ("performs validation of" →
"validates"). It may introduce function words, but only from a fixed set declared
per operation — `was`, `were`, `is`, `are`, and nothing else — never a word chosen
by searching for one that fits. Where no safe edit exists, it reports instead of
rewriting. No model runs at fix time; everything is offline, table-driven, and
byte-deterministic.

## What it does

Six operations edit text. Nothing else does.

1. **Ritual deletion** — sentences matching ritual frames are removed whole:
   *"If you have any questions or require further assistance, please reach out
   to our support team."* → gone.
2. **Gated span deletion** — a detected filler span is removed only if the
   resulting seam is attested in a human-written corpus or falls at a sentence
   start, and the sentence still has a finite verb afterward:
   *"It is important to note that the agent…"* → *"The agent…"*.
3. **Paired substitution** — a slop frame becomes its plain-register counterpart
   from a versioned, audited inventory: *"will walk you through"* → *"covers"*,
   *"in order to"* → *"to"*, *"leverages"* → *"uses"*, *"is crucial for"* →
   *"is important for"*.
4. **Derivational pivot** — a licensed light-verb construction collapses to its
   root verb, inheriting tense and agreement: *"the agent performs validation of
   the config file"* → *"the agent validates the config file"*, *"we made the
   decision to switch"* → *"we decided to switch"*.
5. **Frame-gated `just`-deletion** — inside a detected dismissive-foil question
   (*"is X just A, or B?"*), the marker is deleted, never the question itself:
   *"is provenance just frontmatter metadata, or does every derived graph edge
   retain immutable lineage?"* → *"is provenance frontmatter metadata, or does
   every derived graph edge retain immutable lineage?"*. Fires only for
   `just`/`merely`/`simply` — never `only`, which often carries real quantity
   meaning — and only when the marker sits strictly between the question's
   auxiliary and its coordinating `or`, so a genuine either-or question is
   never touched.
6. **Register rephrasing** — the only operation that can *raise* a construction
   rather than remove one. Language models under-produce the agentless passive
   relative to human technical writing, consistently and across every model
   measured, and no amount of deleting fixes an under-use. So a clause with a
   recoverable agent may be demoted: *"as you make changes"* → *"as changes are
   made"*, *"how we handle code reviews"* → *"how code reviews are handled"*. A
   nominalization may also be unpacked: *"the integration of SQS"* →
   *"integrating SQS"*. A third feature only ever removes: the em dash, which
   Claude-family output uses heavily and human technical documentation almost
   never does, is rewritten toward the punctuation the surrounding clauses
   already license — *"the queue holds items — 3 workers drain it"* → *"the
   queue holds items; 3 workers drain it"*, *"reached over the network — it is
   never called directly"* → *"reached over the network. It is never called
   directly"*. A fourth homes toward a band that is genuinely nonzero, not to
   zero: the semicolon, which Claude-family output also uses well past the
   human rate, is split into a sentence break only when it joins two
   independent clauses — *"the job reads from the queue; it commits offsets
   only after the batch is durably written"* → *"the job reads from the
   queue. It commits offsets only after the batch is durably written"*. A
   semicolon that is part of a serial list, or that has no independent clause
   on either side, is left alone; so is one inside the human range.

   It fires only while the document sits outside a band measured from human
   documents, and stops at the band's edge rather than its centre — every
   document landing on the same coordinates would be a tell of its own. On the
   corpus it edits roughly one machine document in ten.

Every edit passes a stack of hard gates: closure as described above — content
words input-derived, function words only from the declared set — is checked on
every candidate before it is applied. Beyond that, the sentence must stay
clause-complete, edits never touch code spans,
links, numbers, identifiers, negation, quantifiers, modals, or logical
connectives, quoted text is left alone because it is someone's example rather
than the author's own register, and edits fire only inside prose blocks — never
in headings, code, tables, or link text. On human-written text the whole engine
is calibrated to be a near-no-op. When a gate says no, the candidate is kept
verbatim and reported as a suggestion — doing nothing is the designed fallback,
not a failure mode.

Most of what the gates encode was learned by running the engine over the corpus
and reading what it produced. Passivizing across a preposition turned *"as we
continue down this path"* into *"as this path is continued"*; promoting a
post-modified object turned *"inspected each board for knots and defects"* into
*"each board for knots and defects was inspected"*, which no longer says what
the inspection was for. Neither is caught by a grammar check — both are
meaning changes that read fine. The guards that refuse them are in the source
with the sentence that motivated each one.

Detection (what finds the candidates) runs six channels. Five are span-level: a
mined literal inventory, a shallow tag-pattern scan for light-verb
constructions, a differential matching-statistics profile computed against
per-generator-family suffix-automaton indexes — which also powers the
document-level report in `friction check` — a deterministic contrast-frame
template scan: `frame.contrast.question` (the dismissive-foil interrogative
above) and `frame.contrast.correction` (declarative epanorthosis, *"not just
X — it's Y"*), both detect-only in `check` and, for `fix`, reported in the
paraphrase list alongside DMS (differential matching statistics — friction's
own name for its corpus-differential detection channel, built on the
matching-statistics literature; see `docs/research/ALGORITHMS.md` §1)
candidates whenever no gated edit applies (an
`only`-marked question, or any correction span) — and `jargon.metaphor`, a
tag-gated scan over a curated list of physical/aesthetic metaphor nouns
("resonance", "tapestry", "well", "soup"…), flagged only when one heads a noun
compound as its rightmost, tagged-noun word, immediately preceded by at least
one noun/adjective modifier: *"semantic wells"*, *"cross-domain resonance"*, *"a
rich tapestry of services"*. The same word as a bare noun (*"the well is
deep"*) or as a modifier (*"soup kitchen"*) never matches, and a compound with
any mid-sentence capitalized word is treated as a possible product name and
declined. The web-scale attestation design `docs/research/SYNTHESIS.md` §4
describes — every head word frequent, but the compound itself unattested
anywhere — is the exemption mechanism: a compound is checked against
`jargon-attest-v1`, a `BinaryFuse8` filter over ~2M normalized Wikipedia-title
and OpenAlex-topic keys built offline and embedded in the binary (*"data
fabric"*, *"primordial soup"*, *"color harmony"*, *"resonance frequency"*, and
every other real title, are attested and never flagged), OR against a small
hand-curated TOML exception list that only carries what the filter still
misses (*"service fabric"*, *"test well"*), each with its own stated reason.
This is a deliberately narrow, high-precision slice of pseudo-jargon — a
curated head-word lexeme (not an open-vocabulary jargon detector) paired with
a real attestation table, not a hand-picked exceptions list alone — and it is
detection-only like the contrast-frame templates: there is no deterministic
true replacement for an invented term, so a flagged span is reported in
`check` and unioned into `fix`'s paraphrase list, never rewritten.
The sixth channel is document-level: a count of register-marking constructions
over a dependency parse, compared against the human band. It answers a
question the others cannot, because a construction that is *missing* has no
span to detect.

## Install

### npm

```bash
npm install -g friction-cli          # global, on your PATH as `friction`
npm install --save-dev friction-cli  # project-local, for CI and npm scripts
npx friction-cli fix draft.md        # no install at all
```

Project-local is usually what you want for a repo, since it pins the version in
your lockfile:

```json
{
  "scripts": {
    "lint:docs": "friction fix README.md | cmp -s - README.md"
  },
  "devDependencies": {
    "friction-cli": "0.2.0"
  }
}
```

`friction-cli` is the only package you name. The binary for your platform
arrives as an optional dependency — one of `@mhriaznov/friction-darwin-arm64`,
`-darwin-x64`, `-linux-x64`, `-linux-arm64`, `-win32-x64` — and npm picks it from
each one's `os`/`cpu` fields, skipping the rest.

There is **no postinstall script and nothing is downloaded at install time**, so
lockfiles pin it, npm verifies its integrity, and `--ignore-scripts`, offline and
air-gapped installs all work. Linux x64 gets the statically linked build, so
Alpine and Debian are the same package.

**It does not update itself, by design.** A tool your package manager installed
should be updated by that package manager — a self-updating binary would fight
npm over a file npm owns, break integrity checks, and fail outright on a
read-only install. Pick whichever of these you want:

| you want | do this |
|---|---|
| always the latest, no install | `npx friction-cli@latest fix draft.md` |
| update on demand | `npm update -g friction-cli` |
| automatic PRs on new versions | Dependabot or Renovate against your `package.json` |
| pinned, reproducible | commit the lockfile; nothing moves until you bump |

For CI, pin it. A linter that silently changes version changes your build's
output, and you want that to be a commit.

### Prebuilt binary

Download for your platform from [Releases](../../releases), extract, and put
`friction` on your `PATH`. No toolchain needed.

| platform | archive |
|---|---|
| macOS, Apple silicon | `aarch64-apple-darwin` |
| macOS, Intel | `x86_64-apple-darwin` |
| Linux x86-64 | `x86_64-unknown-linux-gnu` |
| Linux x86-64, static | `x86_64-unknown-linux-musl` |
| Linux arm64 | `aarch64-unknown-linux-gnu` |
| Windows x86-64 | `x86_64-pc-windows-msvc` |

Prefer the **musl** build for containers and CI images: it is statically linked
and does not care about the host's glibc. Each archive ships a `.sha256`.

The macOS binaries are **unsigned**, so Gatekeeper refuses the first run. Either
`xattr -cr ./friction` or right-click → Open once.

### From source

```bash
cargo build --release -p friction-cli
```

The binary lands at `target/release/friction`; MSRV is 1.96.

Either way the result is self-contained: the part-of-speech tagger (1.4 MB) and
dependency parser (6.5 MB) are vendored weight artifacts compiled into the
binary, and nothing is downloaded at build or run time. Both are fixed tables,
so the runtime stays deterministic — there is no inference engine here, only
lookups. That accounts for the binary being around 16 MB.

## Usage

### `friction fix` — repair a document

Fixed text goes to **stdout**, the summary to **stderr**. They are separate
streams, so no flag is needed to keep them apart — redirect stderr and you have
clean text.

```bash
friction fix draft.md                  # fixed text to stdout, summary to stderr
friction fix draft.md 2>/dev/null      # fixed text only
friction fix draft.md > fixed.md       # original untouched
friction fix draft.md --in-place       # rewrite the file atomically
friction fix draft.md --suggest        # also list what was detected but held
friction fix draft.md --format json    # summary as JSON instead of a table
```

#### Reading from stdin

Pass `-` as the path:

```bash
echo "$text" | friction fix -
cat draft.md  | friction fix - 2>/dev/null > fixed.md
OUT=$(printf '%s' "$text" | friction fix - 2>/dev/null)

friction fix - 2>/dev/null <<'EOF'
This guide will walk you through deploying the collector.
EOF
```

`--in-place` is rejected with `-`: there is no file to write back to.

#### Four one-liners, verbatim

```bash
$ echo 'We will walk you through it. It is crucial that you utilize the debug flag.' \
    | friction fix - 2>/dev/null
We cover it. It matters that the debug flag is used.
```

Three substitutions and a passivization in one sentence pair.

```bash
$ echo 'The style guide bans "will walk you through" outright. This guide will walk you through the rest.' \
    | friction fix - 2>/dev/null
The style guide bans "will walk you through" outright. This guide covers the rest.
```

The quoted phrase is a *mention* — someone's example, not the author's register —
so it survives while the same words unquoted get fixed.

```bash
$ echo 'Set `leverage` in the config. We leverage the cache heavily.' \
    | friction fix - 2>/dev/null
Set `leverage` in the config. We use the cache heavily.
```

Input is parsed as Markdown, so the code span is out of scope and the prose is not.

```bash
$ friction fix - 2>/dev/null <<'EOF'
This guide will walk you through deploying the collector.

If you have any questions or require further assistance, please reach out to our support team.
EOF
This guide covers deploying the collector.
```

The ritual closer is removed whole.

#### A longer document

Input:

```text
This guide will walk you through configuring the backup agent for your staging
environment. It is important to note that the agent performs validation of the
configuration file before each run. Once validation succeeds, the agent
conducts an analysis of the snapshot catalog and simply uploads any missing
segments. By following these steps, you can quickly and easily verify that
your backups are consistent. If you have any questions or require further
assistance, please reach out to our support team.
```

Output:

```text
This guide covers configuring the backup agent for your staging
environment. The agent validates the
configuration file before each run. Once validation succeeds, the agent
conducts an analysis of the snapshot catalog and simply uploads any missing
segments. You can verify that
your backups are consistent.
```

```
friction fix: 3 pass(es), 8 patch(es) applied
  edit.recapitalize: 2
  pivot.lvc: 1
  ritual.delete: 1
  span.delete: 3
  sub.apply: 1
  suggest: 0 finding(s) remain
  paraphrase: 0 span(s) flagged for manual rewrite
```

That is the engine's real output, line breaks included: friction edits bytes
and leaves the input's own wrapping alone rather than reflowing paragraphs it
touched.

Note what it kept: "conducts an analysis of" was a valid second pivot but the
per-document budget held it, and "simply" stayed because deleting it would
create a word seam unattested in human writing. Neither is forced, and both are
named — by `friction explain`, which lists every hold with the gate that
declined it. The `suggest:` line above counts a different thing: findings the
detector surfaced that no operation claimed at all.

The `paraphrase:` line counts a third, unrelated thing: after fixing, `fix`
also scans its own output with the DMS statistical channel against every
generator family the embedded index covers, and with the contrast-frame
template scan, and reports how many spans either flagged. It never touches
them — there is no licensed rewrite for a DMS tell or for an `only`-marked
question or a declarative correction frame, only a fixed set of literal
edits, so a flagged span is left exactly as written for a human to
paraphrase. `--suggest` lists each one on stderr: location, generator family
(or frame id), score, and the flagged snippet.

Clean text passes through byte-identical:

```
$ friction fix clean.md
friction fix: 2 pass(es), 0 patch(es) applied
  suggest: 0 finding(s) remain
  paraphrase: 0 span(s) flagged for manual rewrite
```

### `friction check` — detect and measure, change nothing

```
friction check draft.md --family qwen --genre blog
friction check draft.md --family gemma --format sarif > report.sarif
```

Reports detected spans with byte-exact locations, tell counts per family,
distribution metrics against the genre envelope, and the document-level
matching-statistics differential. The SARIF output validates against the SARIF
2.1.0 schema.

`--family` is required and matters: detection indexes are specific to the
generator family they were mined from (`qwen`, `gemma`, `llama`, `granite`).
Text from a model family you have no index for will mostly evade the
statistical channel — see limits below.

**`check` is a report, not a gate.** Its exit code is `0` only when *every*
metric sits inside its envelope and no span was detected, and the envelope is
two-sided — a document is flagged for falling outside it in either direction,
including for being more human-favoured than the human band's upper bound.
Measured over 25 human documentation files from the corpus, **24 exit non-zero**.
That is the metric layer working as designed (it describes a distribution, it
does not classify a document), but it means `check` will fail almost any real
document and should not be wired to a build.

Gate on `fix` instead — see below.

### `friction explain` — why did (or didn't) it edit?

```
$ friction explain draft.md
pass 1: 8 operation(s) fired
  sub.apply                0..32  -> "This guide covers"
  span.delete              92..121  (deleted)
  edit.recapitalize        121..122  -> "T"
  pivot.lvc                131..153  -> "validates"
  span.delete              316..342  (deleted)
  edit.recapitalize        342..343  -> "Y"
  span.delete              350..369  (deleted)
  ritual.delete            409..504  (deleted)
  2 held:
    pivot.lvc              194..315  KEPT (pivot held: near-no-op budget exhausted)
    span.delete            194..315  KEPT (deletion span.simply held: SeamNotAttested)
pass 2: 0 operation(s) fired — converged
  2 held:
    pivot.lvc              137..258  KEPT (pivot held: near-no-op budget exhausted)
    span.delete            137..258  KEPT (deletion span.simply held: SeamNotAttested)
pass 3: 0 operation(s) fired — converged
```

Every range is a byte offset into your original file, and every hold names the
gate that declined. Deleting a span can leave a lowercase word opening a
sentence, which is why `edit.recapitalize` follows each one. The same two holds
reappear in pass 2 at shifted offsets: the earlier deletions moved the bytes,
and the gates decline again on the same grounds.

## Scripting it — pipelines, CI, agents

Everything needed is already there: stdin in, stdout out, structured output on
demand, meaningful exit codes, no interactive prompts, no state, no network. The
packs and models are compiled into the binary, so per-invocation cost is model
load rather than I/O.

### Exit codes

| command | `0` | `1` | `2` |
|---|---|---|---|
| `fix` | always, when it ran | — | error (unreadable input, bad UTF-8) |
| `explain` | always, when it ran | — | error |
| `check` | every metric in envelope **and** no spans | otherwise | error |

`fix` does not signal "I changed something" through its exit code. To detect
that, compare bytes or read the JSON summary:

```bash
# did anything change?
friction fix draft.md 2>/dev/null | cmp -s - draft.md || echo "friction would edit this"

# how much, structured
friction fix draft.md --format json 2>&1 >/dev/null
# {"passes":3,"patches_applied":4,"patches_by_rule":{"register.passivize":1,"sub.apply":3},"suggest_count":0}
```

Note the redirect order: the summary is on **stderr**, so `2>&1 >/dev/null`
sends the summary to stdout and discards the fixed text. Reversing them gives
you the opposite.

### A CI gate that works

Fail the build when friction would still change a committed document — i.e.
treat "already clean" as the invariant:

```bash
#!/bin/sh
# fails if any tracked Markdown file is not already friction-clean
rc=0
for f in $(git ls-files '*.md'); do
  if ! friction fix "$f" 2>/dev/null | cmp -s - "$f"; then
    echo "not friction-clean: $f"
    rc=1
  fi
done
exit $rc
```

Two practical notes. Use `rc`, not `status`: the latter is a read-only special
variable in zsh and the loop aborts on assignment. And scope the glob to
documents you actually author — pointing it at every tracked `*.md` will also
catch generated reports and vendored third-party text, which are not yours to
rewrite and will fail the gate legitimately.

This repository's own README satisfies that check, which is why the examples
above can quote machine-register phrases without the tool rewriting its own
exhibits.

### Every edit, machine-readably

`explain --format json` gives each edit's rule, byte range, and replacement —
enough to build a diff, apply a subset, or show a human what would change
before changing it:

```bash
$ friction explain draft.md --format json
{
  "passes": [
    {
      "pass": 1,
      "fired": [
        { "rule": "sub.apply", "start": 0,  "end": 24, "replacement": "We cover" },
        { "rule": "sub.apply", "start": 32, "end": 42, "replacement": "matters" },
        { "rule": "sub.apply", "start": 52, "end": 59, "replacement": "use" }
      ],
      "held": []
    }
  ]
}
```

Ranges index the **original** bytes, so `text[start..end]` is exactly what the
replacement replaces. Held candidates appear alongside with the gate that
declined them, which is the part worth surfacing to a reviewer: it says what
friction noticed and chose not to touch.

### Filtering to the operations you trust

There is no flag to disable an individual operation. If you only want the five
closed operations and not register rephrasing, read the edits from `explain
--format json`, drop the `register.*` rules, and apply the rest yourself —
the ranges are byte-exact against your input, so that is a mechanical splice.

## Guarantees

- **Deterministic** — same input, same pack, same bytes. Always.
- **Idempotent** — a second pass may clean up after first-pass deletions; a
  third pass changes nothing (enforced by a CI canary over fixtures and real
  corpus documents).
- **Near-no-op on human text** — edits per document on curated human writing
  stay under a corpus-calibrated threshold (1.87 edits per 1000 words, ceiling),
  recalibrated whenever the packs are rebuilt and cross-checked against a
  held-out split that never feeds back into the threshold.
- **Span-honest** — every reported range slices your original bytes to exactly
  the text the finding is about.
- **Closed** — no code path can insert a *searched-for* word. Content words are
  always input-derived. Function words may be introduced only from a fixed set
  declared per operation, checked on every candidate before it is applied, so
  the guarantee is enforced structurally rather than asserted. The approaches
  that would break it (bridge insertion, corpus-path synthesis) were tried
  during research, failed, and are pinned as red tests so they cannot come
  back.

## What it deliberately won't do

friction removes machine framing surgically. It does not add ideas, voice, or
fluency — its ceiling is a careful copy editor with a narrow brief, and an empty
source stays empty. It is not for fiction. It is not a detector-beater. And
detection packs are specific to the model family they were built from: the
shipped indexes cover qwen/gemma/llama/granite-style output, so prose from an
unindexed family will largely pass the statistical channel untouched (the
literal inventory still applies). Growing coverage means growing the data — see
below — not cleverer search.

Three limits worth knowing before you rely on it.

**It is English-only, and there is no language detection.** The tagger, the
dependency parser, the segmentation rules and every phrase in the inventory are
English; given another language, friction tags it as English anyway.

What that means in practice, measured rather than assumed. On German, Spanish,
Dutch, Italian and Portuguese paragraphs it is **completely inert** — zero
patches, output byte-identical. The tells it looks for are English phrases and
English dependency structures, and neither is there to find, so nothing fires.
That is the safe failure, and it is the one you get.

It does, however, edit **English sentences embedded in another language**:

```bash
$ echo 'Ce document décrit le setup. It is important to note that we utilize the cache.' \
    | friction fix - 2>/dev/null
Ce document décrit le setup. The cache is used.
```

The French is untouched and the English sentence is repaired correctly. Whether
that is what you want in a mixed-language document is your call — friction has
no way to ask.

Supporting another language is not a code change: it needs its own tagger,
parser, segmentation rules and mined inventory. The engine is language-agnostic;
all of the knowledge is in the data.

Register rephrasing is scoped to **technical documentation**, and that scope is
narrower than it sounds. The band was measured on reference documentation.
READMEs measured as a *different* register — two of the three features differ
with non-overlapping confidence intervals, and their correlation structure
differs too — so pooling the two was rejected and the documentation target is
used for both. Running it on prose from another genre aims at the wrong target,
and nothing at runtime will stop you.

Separately, the gates check whether an edit preserves meaning and grammar — not
whether the result reads *better*. A rewrite can be licensed, correct, and
still flatter than what it replaced. Only a reader catches that.

## The data behind it

The packs under `crates/friction-packs/packs/` are versioned, sha256-recorded,
and audited at load time (a substitution whose replacement re-triggers any
detection frame, or introduces an unattested content word, fails the build):

- `inventory-v1.toml` — the curated tell inventory: deletion spans,
  substitution pairs, ritual frames, licensed light-verb pairs, guard-token
  classes. Hand-reviewed; every mined entry carries its corpus counts.
- `dms-index-v1.toml` — per-family machine/human token streams for the
  matching-statistics channel.
- `attestation-v1.toml` — human-corpus bigram seams, tag-skeleton sets, and the
  near-no-op calibration.
- `register-v1.toml` — the per-feature bands register homes toward, as the 10th
  and 90th percentiles of the per-document rate across 58 human documentation
  files. The band, not its centre, is the target.
- `jargon-v1.toml` — the curated metaphor-lexeme list `jargon.metaphor`
  matches against, and a small attested-exceptions allowlist of compounds it
  never flags regardless of what the filter below says. Each lexeme is
  `mined` (measured directly against this corpus) or `external` (documented
  LLM metaphor vocabulary); every entry, and every exception, carries its own
  provenance note.
- `jargon-attest-v1.bin` + `jargon-attest-v1.toml` — the web-scale
  attestation oracle behind `jargon.metaphor`'s exception check: a
  `BinaryFuse8` filter (`xorf`) over ~2M normalized, multiword compound keys
  mined from Wikipedia article titles and OpenAlex Topics display names,
  built offline by `corpus-tool jargon-attest` and embedded in the binary.
  A compound in the filter is attested and never flagged; the filter's
  false-positive direction is the safe one (~0.4%, no false negatives) — a
  spurious "attested" only ever suppresses a flag, never invents one. The
  TOML exception list above is now a near-empty override layer for what the
  filter misses, not the primary mechanism. Wikipedia titles are
  CC BY-SA 4.0 and GFDL (Wikipedia contributors,
  <https://en.wikipedia.org/>); OpenAlex Topics are CC0
  (<https://openalex.org/>). Full provenance — input sha256s, normalization
  and hash spec, build date — is in the `.toml` sidecar.

The contrast-frame templates (`frame.contrast.question`,
`frame.contrast.correction`) are the one detection channel that is not
pack data: both are a fixed, closed-vocabulary lexical frame (eight
auxiliaries, four markers, a handful of connectives) compiled directly into
`friction-match`, with no corpus-mined pattern to version.

They are produced by `corpus-tool` (`mine-inventory`, `mine-paired`, `index`,
`attest`) from the repository's document corpus — machine text from six local
models plus a topic-matched stock-vs-antislop paired corpus that cancels topic
so mining isolates pure style; the human side of every pair comes exclusively
from license-vetted, pre-2022 human documents. All tuning uses the train split;
a sealed holdout is guarded by CI and never touched.

### Evaluation

Measurements live next to the data they describe, and each was produced by a
command recorded in the file itself:

- [`corpus/HOLDOUT_REPORT.md`](corpus/HOLDOUT_REPORT.md) — the sealed-holdout
  evaluation. Read once, report-only: no threshold, envelope, or rule was
  changed in response to anything on that page.
- [`corpus/SEPARATION.md`](corpus/SEPARATION.md) — per-metric human-vs-machine
  AUC on the dev split, with the combined score's own AUC.
- [`corpus/NEARNOOP.md`](corpus/NEARNOOP.md) — what fraction of human corpus
  sentences receive any edit, per genre.
- [`corpus/MEANING_AUDIT.md`](corpus/MEANING_AUDIT.md) — a deterministic
  50-document sample checked for meaning preservation.
- [`corpus/MINE_INVENTORY.md`](corpus/MINE_INVENTORY.md),
  [`corpus/MINE_PAIRED.md`](corpus/MINE_PAIRED.md) and
  [`corpus/OUTPUT_BANDS.md`](corpus/OUTPUT_BANDS.md) — the mining provenance the
  inventory pack's own entries cite for their corpus counts.
- [`corpus/STATS.md`](corpus/STATS.md) — corpus composition by class, genre and
  split.

The tagger and parser are trained from the same corpus, by a pipeline that
drafts annotations with an offline tool and corrects them mechanically. That
gold data is **not** committed: it is derived, no build reads it, and every
retrain would add megabytes to history permanently. `tools/requirements.txt`
pins the environment that reproduces it byte-for-byte — the pin matters, since
a different model version parses differently and would silently produce
different gold while every check still passed.

For the algorithms themselves — the matching-statistics construction, mining
thresholds, gate definitions, and the validated reference prototypes — see
[docs/research/ALGORITHMS.md](docs/research/ALGORITHMS.md) and
[docs/research/ref/](docs/research/ref/). For how the register targets were
measured, including what the corpus turned out not to support, see
[docs/research/regvec/TARGET_ESTIMATION.md](docs/research/regvec/TARGET_ESTIMATION.md).

## Releasing

The version in `crates/friction-cli/Cargo.toml` is the single source of truth.
To cut a release: **bump it and push to `main`.** CI derives the tag, builds all
six targets, publishes the GitHub release with checksums, then generates and
publishes the npm packages at the same version.

npm publishing needs an `NPM_TOKEN` repository secret. Without it that job
reports green and does nothing, so the binary release is never held up by
credentials that are not configured. The platform packages publish before the
wrapper, since the wrapper pins them by exact version and the reverse order
would leave a window where `npm install friction-cli` resolves a wrapper whose
binaries are not on the registry yet.

A push whose version is already tagged is a no-op that still reports green, so
ordinary commits neither fail the workflow nor publish duplicates. Nothing is
tagged by hand — tagging is the pipeline's job, because a step a human has to
remember is a step that gets forgotten.

## Development

Rust 2024 workspace, MSRV 1.96. `cargo fmt --check`, `cargo clippy --workspace
--all-targets -- -D warnings`, and `cargo test --workspace` must all pass; the
test suite includes the literal accept/reject/rank fixtures from the validation
research (`docs/research/fixtures.json`) — if a change makes a reject fixture
pass or an accept fixture fail, the change is wrong, not the fixture.

Two invariants are worth knowing before changing anything. A **sealed holdout
split** is verified by CI and must never be read; all tuning uses the train
split. And the register feature extractor is pinned against a **reference
parity fixture** that carries both the reference parse and the counts derived
from it, so a failure tells you whether the counting or the parsing broke —
without that separation, a miscounted feature is invisible in the output.

## License

MIT ([LICENSE-MIT](LICENSE-MIT)) or Apache-2.0 ([LICENSE-APACHE](LICENSE-APACHE)),
at your option.

The vendored corpus under `corpus/` is third-party material under its own
licenses; every document's provenance and license is recorded in
`corpus/manifest.jsonl`.
