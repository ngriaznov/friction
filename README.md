# friction

A deterministic engine that removes the machine layer from LLM-generated technical
documentation (the ritual closers, filler spans, hedge phrases, and light-verb
constructions that make text read as machine-written) without touching the
information.

**English technical documentation only.** Both constraints are real, not
aspirational, and neither is checked at runtime:

- **English.** Every shipped artifact is English-trained: the tagger, the
  dependency parser, the sentence segmenter, and every phrase in the tell
  inventory. There is no language detection and no other language pack. On
  non-English prose friction is inert rather than wrong (measured: zero edits on
  German, Spanish, Dutch, Italian and Portuguese), but it will repair English
  sentences embedded in another language. See the limits below.
- **Technical documentation.** Reference docs, READMEs, design notes,
  postmortems, migration guides. The inventory was mined from that kind of
  writing, and the register pass is calibrated against it specifically. It is
  not a general-purpose editor and not for prose whose voice is the point:
  fiction, marketing, personal writing.

friction never invents content. Every content word it emits is already present in
the input or derived from one through a static table ("performs validation of" →
"validates"). It may introduce function words, but only from a fixed set declared
per operation (`was`, `were`, `is`, `are`, and nothing else): never a word chosen
by searching for one that fits. Where no safe edit exists, it reports instead of
rewriting. No model runs at fix time; everything is offline, table-driven, and
byte-deterministic.

## What it does

Seven operations edit text. Nothing else does.

1. **Ritual deletion**: boilerplate sentences removed whole: *"If you have
   any questions or require further assistance, please reach out to our
   support team."* → gone.
2. **Gated span deletion**: *"It is important to note that the agent…"* →
   *"The agent…"*.
3. **Paired substitution**: *"will walk you through"* → *"covers"*,
   *"leverages"* → *"uses"*, *"in order to"* → *"to"*.
4. **Derivational pivot**: *"performs validation of the config file"* →
   *"validates the config file"*.
5. **Frame-gated `just`-deletion**: *"is provenance just metadata, or…?"* →
   *"is provenance metadata, or…?"*; the question itself is never touched.
6. **Frame rewriting**: 913 corpus-adjudicated rules with measured rates:
   *"We utilized the cache"* → *"We used the cache"*, *"worth confirming"* →
   *"worth checking"*, *"Honestly,"* → deleted.
7. **Register rephrasing**: homes punctuation and constructions the models
   over- or under-use toward bands measured from human writing: em dashes,
   semicolons, the agentless passive.

Every candidate passes hard gates: meaning preserved, clauses complete, code
spans, links, numbers, and quotes untouched, near-no-op on human text. When a
gate says no, friction reports instead of rewriting, and seven detection
channels only ever report. [docs/OPERATIONS.md](docs/OPERATIONS.md) covers
each operation, gate, and channel in depth.

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
    "friction-cli": "0.5.2"
  }
}
```

`friction-cli` is the only package you name. The binary for your platform
arrives as an optional dependency (one of `@mhriaznov/friction-darwin-arm64`,
`-darwin-x64`, `-linux-x64`, `-linux-arm64`, `-win32-x64`) and npm picks it from
each one's `os`/`cpu` fields, skipping the rest.

There is **no postinstall script and nothing is downloaded at install time**, so
lockfiles pin it, npm verifies its integrity, and `--ignore-scripts`, offline and
air-gapped installs all work. Linux x64 gets the statically linked build, so
Alpine and Debian are the same package.

**It does not update itself, by design.** A tool your package manager installed
should be updated by that package manager. A self-updating binary would fight
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

Either way the result is self-contained: the part-of-speech tagger (1.4 MB),
the dependency parser (6.5 MB), the matching-statistics automata
(dms-index, 25.8 MB: per-family streams plus the pooled machine automaton
the runtime scans), the compound-attestation filter (2.3 MB), and the
evidence packs are all compiled into the binary, and nothing is downloaded
at build or run time. Everything is fixed tables, so the runtime stays
deterministic. There is no inference engine here, only lookups. That
accounts for the binary being around 59 MB (about a 27 MB compressed
download).

A typical README (3–7 KB) runs through `friction fix` in about 30 ms; a
10 MB document takes about 2.4 s. <sub>Measured on an M-series MacBook
Pro, process start included.</sub>

## Usage

### `friction fix` — repair a document

**`fix` never modifies your file unless you pass `--in-place`.** The default
run prints the fixed text to **stdout** and the summary to **stderr**, so the
summary can report applied patches while the file on disk stays byte-identical.
That default is what makes piping safe and an original impossible to destroy
unasked; when you want the file itself repaired, say so:

```bash
friction fix draft.md --in-place
```

The two streams are separate, so no flag is needed to keep them apart:
redirect stderr and you have clean text.

```bash
friction fix draft.md                  # fixed text to stdout, summary to stderr
friction fix draft.md 2>/dev/null      # fixed text only
friction fix draft.md > fixed.md       # original untouched
friction fix draft.md --in-place       # rewrite the file atomically
friction fix draft.md --suggest        # also list what was detected but held
friction fix draft.md --format json    # summary as JSON instead of a table
friction fix page.html                 # static HTML: text nodes edited, markup untouched
```


#### Static HTML

A `.html`/`.htm` file (or stdin starting with a doctype or `<html>`) is read
as a static HTML page instead of markdown: text runs between tags become the
prose, and everything else (tags, attributes, comments, entities, and the
entire contents of `head`, `script`, `style`, `pre`, and `code`) is
untouchable by construction, so the output differs from the input only inside
text nodes. Headings and table cells follow the same rules as in markdown
(detected against, never edited).

JSON *data blocks* are prose too: a `<script>` whose `type` is
`application/json` or any `+json` variant is inert data, not code, so the
string **values** inside it (slide notes, embedded content) are extracted and
edited: keys, structure, and every escape sequence stay untouchable, which
keeps the document valid JSON by construction. Executable script content is
never touched: prose inside real JavaScript string literals is code, and the
right place for friction there is the source the page is generated from.

Template expressions are another processor's bytes, not prose: `{{ ... }}`,
`{% ... %}`, and `<% ... %>` (mustache, Jinja, Liquid, Hugo, ERB, EJS) are
run boundaries wherever they appear. A `.html` file is often really a
template, and no edit can alter or splice across an expression another tool
will evaluate.

One boundary to know: a sentence interrupted by inline markup is analyzed as
fragments on either side of the tag, so a tell spanning an `<em>` boundary is
invisible to the span channels.

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

The quoted phrase is a *mention* (someone's example, not the author's register) so it survives while the same words unquoted get fixed.

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
named by `friction explain`, which lists every hold with the gate that
declined it. The `suggest:` line above counts a different thing: findings the
detector surfaced that no operation claimed at all.

The `paraphrase:` line counts a third, unrelated thing: after fixing, `fix`
also scans its own output with the DMS statistical channel against the
pooled machine automaton, and with the contrast-frame
template scan, and reports how many spans either flagged. It never touches
them: there is no licensed rewrite for a DMS tell or for an `only`-marked
question or a declarative correction frame, only a fixed set of literal
edits, so a flagged span is left exactly as written for a human to
paraphrase. `--suggest` lists each one on stderr: location, channel id, score, and
the flagged snippet.

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
friction check draft.md --family qwen --residual
```

Reports detected spans with byte-exact locations, tell counts,
distribution metrics against the genre envelope, and the document-level
matching-statistics differential. The SARIF output validates against the SARIF
2.1.0 schema.

`--residual` appends the spans the statistical channel flags that no
compiled frame rule covers. The machine tells the rule set cannot yet
explain or rewrite, which is exactly the evidence queue the next
rule-generation batch should start from.

`--family` is still required by the interface but no longer selects an
index: since 0.5.0 every scan runs against one pooled machine automaton
built from all mined family corpora (`qwen`, `gemma`, `llama`, `granite`,
`claude`), and the flag only labels the report. Text from a generator
whose register none of the mined corpora resemble can still evade the
statistical channel. See limits below.

**`check` is a report, not a gate.** Its exit code is `0` only when *every*
metric sits inside its envelope and no span was detected, and the envelope is
two-sided. A document is flagged for falling outside it in either direction,
including for being more human-favoured than the human band's upper bound.
Measured over 25 human documentation files from the corpus, **24 exit non-zero**.
That is the metric layer working as designed (it describes a distribution, it
does not classify a document), but it means `check` will fail almost any real
document and should not be wired to a build.

Gate on `fix` instead: see below.

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

Fail the build when friction would still change a committed document (i.e.
treat "already clean" as the invariant):

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
documents you actually author: pointing it at every tracked `*.md` will also
catch generated reports and vendored third-party text, which are not yours to
rewrite and will fail the gate legitimately.

This repository's own README satisfies that check, which is why the examples
above can quote machine-register phrases without the tool rewriting its own
exhibits.

### Every edit, machine-readably

`explain --format json` gives each edit's rule, byte range, and replacement
(enough to build a diff, apply a subset, or show a human what would change
before changing it):

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
replacement replaces. Held candidates appear with the gate that declined
them, and that's what's worth showing a reviewer: it says what
friction noticed and chose not to touch.

### Filtering to the operations you trust

There is no flag to disable an individual operation. To keep just the five
closed operations and skip register rephrasing, read the edits from `explain
--format json`, drop the `register.*` rules, and apply the rest yourself: the ranges are byte-exact against your input, so that is a mechanical splice.

## Guarantees

- **Deterministic**: same input, same pack, same bytes. Always.
- **Idempotent**: a second pass may clean up after first-pass deletions; a
  third pass changes nothing (enforced by a CI canary over fixtures and real
  corpus documents).
- **Near-no-op on human text**: edits per document on curated human writing
  stay under a corpus-calibrated threshold (1.87 edits per 1000 words, ceiling),
  recalibrated whenever the packs are rebuilt and cross-checked against a
  held-out split that never feeds back into the threshold.
- **Span-honest**: every reported range slices your original bytes to exactly
  the text the finding is about.
- **Closed**: no code path can insert a *searched-for* word. Content words are
  always input-derived. Function words may be introduced only from a fixed set
  declared per operation, checked on every candidate before it is applied, so
  the guarantee is enforced by construction rather than asserted. The approaches
  that would break it (bridge insertion, corpus-path synthesis) were tried
  during research, failed, and are pinned as red tests so they cannot come
  back.

## What it deliberately won't do

friction removes machine framing surgically. It does not add ideas, voice, or
fluency. Its ceiling is a careful copy editor with a narrow brief, and an empty
source stays empty. It has no place in fiction, and it won't help you beat a
detector. And the statistical channel only knows the model families its
corpus was mined from: the pooled index covers
qwen/gemma/llama/granite/claude-style output, so
prose from an unindexed family will largely pass that channel untouched (the
literal inventory still applies). Growing coverage means growing the data, see
below, not cleverer search.

Three limits worth knowing before you rely on it.

**It is English-only, and there is no language detection.** The tagger, the
dependency parser, the segmentation rules and every phrase in the inventory are
English; given another language, friction tags it as English anyway.

What that means in practice, measured rather than assumed. On German, Spanish,
Dutch, Italian and Portuguese paragraphs it is **completely inert**: zero
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
that is what you want in a mixed-language document is your call. friction has
no way to ask.

Supporting another language is not a code change: it needs its own tagger,
parser, segmentation rules and mined inventory. The engine is language-agnostic;
all of the knowledge is in the data.

Register rephrasing is scoped to **technical documentation**, and that scope is
narrower than it sounds. The band was measured on reference documentation.
READMEs measured as a *different* register (two of the three features differ
with non-overlapping confidence intervals, and their correlation structure
differs too), so pooling the two was rejected and the documentation target is
used for both. Running it on prose from another genre aims at the wrong target,
and nothing at runtime will stop you.

Separately, the gates check whether an edit preserves meaning and grammar, not
whether the result reads *better*. A rewrite can be licensed, correct, and
still flatter than what it replaced. Only a reader catches that.

## FAQ

**Is this an AI detector?**
No. friction never answers "did a model write this" and has no authorship
score. It edits register, and register is a property of the text, not of the
author: people write in machine register (that is where the models learned
it), and models can write clean. Using it to accuse anyone of anything is
using it wrong.

**Then what does `friction check` measure?**
The density of machine-register constructions in the document in front of it,
against bands measured from human writing. That is a statement about the
prose, useful for deciding whether to run `fix` or to gate a pipeline. It is
not evidence about who typed it.

**How is this different from a slop word list?**
A word list encodes someone's taste. Every friction rule ships with measured
per-million rates from a machine corpus on one side and 49.7 million words of
pre-ChatGPT human code review on the other, and it compiles into an edit only
while the numbers hold. "Thank you for your understanding" looks
machine-flavored, but people write it constantly, so friction refuses to touch
it. The evidence can also expire: rebuild the packs against new corpora, and a
rule whose numbers stopped holding stops compiling.

**Why repair instead of detect?**
A detection verdict leaves you nothing to do with it except distrust the
author. A repair is an artifact you can diff, review, and gate CI on: same
input, same pack, same bytes, every time.

**Will it eat my voice?**
It is scoped to technical documentation and calibrated to be a near-no-op on
human writing (at most 1.87 edits per 1000 words on the human corpus, enforced
in CI). Fiction, marketing, and personal writing are out of scope on purpose.

**Does my text leave the machine?**
No. No network, no telemetry, no model call at fix time. One offline binary
with the packs compiled in.

## The data behind it

The packs under `crates/friction-packs/packs/` are versioned, sha256-recorded,
and audited at load time (a substitution whose replacement re-triggers any
detection frame, or introduces an unattested content word, fails the build):

- `inventory-en-v1.toml`: the curated tell inventory of deletion spans,
  substitution pairs, ritual frames, licensed light-verb pairs, guard-token
  classes. Hand-reviewed; every mined entry carries its corpus counts.
- `dms-index-en-v1.toml`: per-family machine/human token streams for the
  matching-statistics channel. The runtime embeds `dms-index-en-v1.bin`
  next to it: the same streams with their suffix automata pre-built by
  `corpus-tool dms-pack` and serialized flat, loaded as a zero-copy view
  so process start pays no parse or automaton construction (a test
  re-packs the TOML and fails if the two ever diverge).
- `attestation-en-v1.toml`: human-corpus bigram seams, tag-skeleton sets, and the
  near-no-op calibration.
- `register-en-v1.toml`: the per-feature bands register homes toward, as the 10th
  and 90th percentiles of the per-document rate across 58 human documentation
  files. The band, not its centre, is the target.
- `frame-rules-en-v1.toml`: the adjudicated frame-rewrite rule set: 3,388
  rules in seven evidence buckets (the delivered 3,333 plus corpus-mined
  arrivals), of which only the corpus-confirmed buckets ever compile;
  the rest are staged evidence for `corpus-tool adjudicate`, the referee
  that re-derives every verdict from the corpora, from the DMS streams
  first, then from the review-register evidence pair for rules whose
  register the DMS corpora do not carry. The runtime embeds
  `frame-pack-en-v1.bin` (134 KB): the surviving rule program (913 edits
  and guards plus 127 report-only rules after the rejection gauntlet),
  serialized flat by `corpus-tool frame-pack` and loaded zero-copy,
  with a drift test that recompiles the TOML and fails on any
  divergence.
- `human-evidence-en-v1.bin` + `human-evidence-en-v1.toml`: external
  human-corpus evidence pooled into the frame-rewrite compile fences:
  unigram rates and per-word burst envelopes (the densest any single
  human document used a word, which is what arms the `overuse.word` channel)
  over ~50M tokens of pre-2022, human-written code-review prose, plus
  occurrence counts for every frame rule's literal probes,
  built offline by `corpus-tool human-evidence` from locally staged
  corpora (the raw text never enters this repository, only these
  aggregate counts). External evidence feeds the one-sided fences only:
  target-word attestation (best single-corpus rate, so one register
  cannot dilute another's words) and the human-rate ceiling.
  Direction verdicts stay register-matched on the DMS streams. A
  machine-vs-human ratio over mismatched registers would measure
  register difference, not machine-ness. Shipped-pack inputs: the Code
  Review Stack Exchange data dump of 2022-03-07 (CC BY-SA,
  <https://codereview.stackexchange.com/>) and the natural-language
  review comments of Microsoft's CodeReviewer dataset (CC BY 4.0,
  <https://doi.org/10.5281/zenodo.6900648>), both predating ChatGPT's
  release by construction. Input sha256s and per-bucket totals are in
  the `.toml` sidecar.
- `machine-evidence-en-v1.bin` + `machine-evidence-en-v1.toml`: the machine
  half of the review-register evidence pair: the same tables, built by
  the same command over the 150 committed machine-written review
  documents in `corpus/review/machine/`. Register-matched against the
  human pack, it gives the two-sided fences (direction, guard
  confirmation) and the adjudication referee a review-register
  measurement the DMS streams cannot provide.
- `jargon-en-v1.toml`: the curated metaphor-lexeme list `jargon.metaphor`
  matches against, and a small attested-exceptions allowlist of compounds it
  never flags regardless of what the filter below says. Each lexeme is
  `mined` (measured directly against this corpus) or `external` (documented
  LLM metaphor vocabulary); every entry, and every exception, carries its own
  provenance note.
- `jargon-attest-en-v1.bin` + `jargon-attest-en-v1.toml`: the web-scale
  attestation oracle behind `jargon.metaphor`'s exception check: a
  `BinaryFuse8` filter (`xorf`) over ~2M normalized, multiword compound keys
  mined from Wikipedia article titles and OpenAlex Topics display names,
  built offline by `corpus-tool jargon-attest` and embedded in the binary.
  A compound in the filter is attested and never flagged; the filter's
  false-positive direction is the safe one (~0.4%, no false negatives). A
  spurious "attested" only ever suppresses a flag, never invents one. The
  TOML exception list above is now a near-empty override layer for what the
  filter misses, not the primary mechanism. Wikipedia titles are
  CC BY-SA 4.0 and GFDL (Wikipedia contributors,
  <https://en.wikipedia.org/>); OpenAlex Topics are CC0
  (<https://openalex.org/>). Full provenance (input sha256s, normalization
  and hash spec, build date) is in the `.toml` sidecar.

The contrast-frame templates (`frame.contrast.question`,
`frame.contrast.correction`) are the one detection channel that is not
pack data: both are a fixed, closed-vocabulary lexical frame (eight
auxiliaries, four markers, a handful of connectives) compiled directly into
`friction-match`, with no corpus-mined pattern to version.

They are produced by `corpus-tool` (`mine-inventory`, `mine-paired`, `index`,
`attest`) from the repository's document corpus: machine text from six local
models plus a topic-matched stock-vs-antislop paired corpus that cancels topic
so mining isolates pure style; the human side of every pair comes exclusively
from license-vetted, pre-2022 human documents. All tuning uses the train split;
a sealed holdout is guarded by CI and never touched.

### Evaluation

Measurements live next to the data they describe, and each was produced by a
command recorded in the file itself:

- [`corpus/HOLDOUT_REPORT.md`](corpus/HOLDOUT_REPORT.md): the sealed-holdout
  evaluation. Read once, report-only: no threshold, envelope, or rule was
  changed in response to anything on that page.
- [`corpus/SEPARATION.md`](corpus/SEPARATION.md): per-metric human-vs-machine
  AUC on the dev split, with the combined score's own AUC.
- [`corpus/NEARNOOP.md`](corpus/NEARNOOP.md): what fraction of human corpus
  sentences receive any edit, per genre.
- [`corpus/MEANING_AUDIT.md`](corpus/MEANING_AUDIT.md): a deterministic
  50-document sample checked for meaning preservation.
- [`corpus/MINE_INVENTORY.md`](corpus/MINE_INVENTORY.md),
  [`corpus/MINE_PAIRED.md`](corpus/MINE_PAIRED.md) and
  [`corpus/OUTPUT_BANDS.md`](corpus/OUTPUT_BANDS.md): the mining provenance the
  inventory pack's own entries cite for their corpus counts.
- [`corpus/STATS.md`](corpus/STATS.md): corpus composition by class, genre and
  split.

The tagger and parser are trained from the same corpus, by a pipeline that
drafts annotations with an offline tool and corrects them mechanically. That
gold data is **not** committed: it is derived, no build reads it, and every
retrain would add megabytes to history permanently. `tools/requirements.txt`
pins the environment that reproduces it byte-for-byte. The pin matters, since
a different model version parses differently and would silently produce
different gold while every check still passed.

For the algorithms themselves (the matching-statistics construction, mining
thresholds, gate definitions, and the validated reference prototypes), see
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
tagged by hand. Tagging is the pipeline's job, because a step a human has to
remember is a step that gets forgotten.

## Development

Rust 2024 workspace, MSRV 1.96. `cargo fmt --check`, `cargo clippy --workspace
--all-targets -- -D warnings`, and `cargo test --workspace` must all pass; the
test suite includes the literal accept/reject/rank fixtures from the validation
research (`docs/research/fixtures.json`): if a change makes a reject fixture
pass or an accept fixture fail, the change is wrong, not the fixture.

Two invariants are worth knowing before changing anything. A **sealed holdout
split** is verified by CI and must never be read; all tuning uses the train
split. And the register feature extractor is pinned against a **reference
parity fixture** that carries both the reference parse and the counts derived
from it, so a failure tells you whether the counting or the parsing broke.
Without that separation, a miscounted feature is invisible in the output.

## License

MIT ([LICENSE-MIT](LICENSE-MIT)) or Apache-2.0 ([LICENSE-APACHE](LICENSE-APACHE)),
at your option.

The vendored corpus under `corpus/` is third-party material under its own
licenses; every document's provenance and license is recorded in
`corpus/manifest.jsonl`.
