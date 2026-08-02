# friction v4.1 — algorithm reference

Companion to the build spec. The spec says what and why. This file says exactly how.
Every procedure here is transcribed from the prototype that validated it (July 2026
session, friction corpus). The reference scripts in `ref/` are those prototypes,
runnable against the repo corpus. Where the prototype and production must differ,
it is flagged with PRODUCTION NOTE.

## 0. Conventions shared by everything

Cleaning (corpus and input, before analysis only; the original bytes are kept for
patching): normalize curly quotes to straight; strip fenced code, inline code;
replace links by their anchor text; strip markdown syntax characters `#>*_|`.

Tokenization: `re.findall(r"[a-z']+|[.,;:!?]", text.lower())` for analysis. For
edit application, capture token byte spans with `re.finditer` over the ORIGINAL
text so every token index maps to a byte range.

PRODUCTION NOTE (NF-5): all prototypes rebuild sentences from token strings. The
engine must not. Every operation below identifies a token index range; translate
it through the finditer span table into a byte range on the original source and
emit a `Patch { range, replacement }`. Whitespace between retained tokens is
preserved by construction because deletions and substitutions splice at token
boundaries. The only synthesized whitespace is the single space around a
substituted or pivoted token.

Coarse POS tag: full Penn tag from the tagger, truncated to its first two
characters; punctuation tags kept verbatim. Sentence tag sequences are wrapped
with `<S>` and `<E>` sentinels.

## 1. DMS — differential matching statistics (detection)

### 1.1 Index construction (offline, per model family)

For each class (machine, human): concatenate cleaned, tokenized documents,
inserting one reserved separator token between documents (prevents cross-document
matches). Map tokens to integer ids via a shared vocabulary built over both
corpora. Build one suffix automaton per class over the id stream.

Suffix automaton, standard online construction. State arrays: `next` (map token
id -> state), `link` (suffix link), `len` (longest string length for the state).
See `ref/ref_dms.py::SAM.extend` for the exact clone-handling transcription.
Sizes measured: 106k-token corpus -> 139k states; 277k -> 348k. Build time in
Python: 1.4 s for both.

### 1.2 Matching statistics (runtime, linear)

Query tokens map through the shared vocabulary; unknown tokens become id -1.
Walk the automaton with state v (start root) and length l (start 0); for each
query id c:

    if c != -1 and next[v] has c:  v = next[v][c]; l += 1
    else:
        while v != root and (c == -1 or c not in next[v]):
            v = link[v]; l = len[v]
        if c != -1 and c in next[v]: v = next[v][c]; l += 1
        else: v, l = root, 0
    emit l            # longest match ENDING at this position

Run once against each automaton: curves mM, mH.

### 1.3 Differential profile and spans

The raw curve marks match ends. For visualization/spans, spread each match back
over the tokens it covers: `dM[k] = max(dM[k], mM[i]) for k in [i-mM[i]+1 .. i]`,
same for dH; then `d = dM - dH`.

Span extraction (raw-curve variant used in the validated run): a span starts
where `d[i] >= 3`, continues while `d[j] >= 2`, requires length >= 2 tokens,
is extended left by `mM[start]-1` tokens to cover the matched phrase, and is
scored by `sum(d)` over the run. Thresholds are dev-calibrated; these values
produced zero false-positive spans on a 1,414-token human README.

Document-level: `mean(mM) - mean(mH)`. Sign alone classified 16/16 held-out
docs (8 machine, 8 human) with the human index 2.6x larger, which biases
against the machine side.

Measured limits, to preserve: detection is generator-family-specific (a
Claude-register text scored negative against a Qwen/Gemma/Llama index; the
tells "worth noting"/"pain point" occur 0 times in that machine corpus). DMS is
NOT a per-edit judge: it scored an obviously improved paragraph as more
machine-like because the fix's residue matched the machine corpus. Use DMS for
span finding and document reporting only.

Throughput measured: ~0.5M tokens/s single-core Python including both walks.

## 2. Literal automaton (detection of short tells)

Aho-Corasick over the mined literal inventory, leftmost-longest match policy,
case-insensitive over the cleaned stream. Covers single words and short phrases
that carry no match-length signal.

## 3. Mining (offline)

N-gram ratio mining between machine corpus M and human corpus H, n = 2..4:

    score(g) = (count_M(g)/|M| + eps/|M|) / (count_H(g)/|H| + eps/|H|),  eps = 0.4

Constraints that matter (both were validated the hard way): restrict to n-grams
whose every token has human-corpus frequency >= 25 (drops topic vocabulary;
without it the top hits are content like "handheld transceiver"), and prefer
n >= 3 (higher-order phrases are stylistic). Minimum machine count
8/5/4 for n = 2/3/4. POS-skeleton mining (content nouns abstracted to slots)
generalizes across topics; block-position-conditioned mining (what appears in
the first block after H1, etc.) yields the ritual and preview frames.

LVC pair licensing: emit (light_verb, nominalization) pairs where the machine
corpus prefers the construction and the human corpus prefers the derived verb.
Measured rate on the friction corpus: licensed LVC instances 1.8x more frequent
per token in machine text.

## 4. The four operations

Pipeline per sentence, in order: ritual check, paired substitutions, derivational
pivot (loop, max 2), gated span deletions. Two engine passes maximum; a third
pass must be a no-op (CI canary). Idempotence follows from the pack disjointness
check: no operation output matches any detection frame.

### 4.1 Ritual deletion

If a sentence matches a ritual frame (mined; e.g. `if you have any
questions.{0,60}(reach out|contact|let us know)`, `^congratulations`,
`we hope (this|you)`) delete the whole sentence. A block-level ritual (preview
paragraph) is deletable only when its content lemmas are covered by adjacent
structure (headers, TOC).

### 4.2 Gated span deletion

For a matched slop span with left token L (or sentence start) and right token R
(or terminal punctuation): deletion is allowed iff

    (span is sentence-initial) OR (R is terminal punct) OR (R in bigram[L])

where `bigram` is the human-corpus bigram table with `<s>` sentence-start
tokens, AND the skeleton gate passes, AND the clause gate passes. No bridge of
any kind: if plain deletion fails the gates, the span is KEPT and reported.
(Bridge insertion was tried in three vocabulary regimes and failed in all
three. The failures are fixtures.)

### 4.3 Paired substitution

Regex or frame match -> fixed replacement from the pack ("this guide will walk
you through" -> "this guide covers"; "in order to" -> "to"). Apply, then clause
gate. Replacement text comes only from the pack, checked at pack build for
closure and for disjointness with detection frames.

### 4.4 Derivational pivot (LVC collapse)

Pattern over the tagged token stream (no dependency parser):

    LV  in licensed light-verb forms      (perform/conduct/make/do inflections)
    DET optional: a | an | the
    NOM next token, lowercased, present in the licensed derivational lexicon
    OF  optional: literal "of"

Gates, checked in order; any failure = no edit:
- passive: LV is a past form preceded by a form of "be" -> abort
- modified nominal: token after DET is tagged JJ -> abort (also catches
  quantifiers like "several")
- plural nominal: NOM ends in s and its stem is in the lexicon -> abort
- licensing: the (LV, NOM) pair must be in the pack, not the cross product;
  subject-genitive nominals (approval, decision-of-agent readings) are simply
  never licensed with an of-phrase
- prose block only (gate 7)

Rewrite: let v = derive(NOM) inflected to LV's form. Form map: base -> v;
3sg -> v+"s" (y -> "ies"); past -> v+"d" if v ends in e else v+"ed";
gerund -> drop trailing e, +"ing". Capitalize if sentence-initial. Splice:

    with OF:    tokens[:i] + [v] + tokens[of_index+1:]   # of-NP promoted to object
    without OF: tokens[:i] + [v] + tokens[nom_index+1:]

Tense/agreement inheritance is exactly the form map; nothing else changes.

### 4.5 Shared gates

Skeleton gate: tag the candidate sentence, wrap with sentinels, and require
every coarse-tag 5-gram in the window [edit_start-3, edit_end+3] to be in the
human skeleton set, falling back to its 4-gram prefix. Measured: 90% of
held-out human tag-5-grams are attested by a 277k-token corpus; density rises
with corpus size. Known limitation: tag n-grams cannot see clause completeness
(a verbless fragment passed), hence the separate clause gate.

Clause gate: after the edit the sentence must contain a finite verb (full Penn
tag in {VBZ, VBP, VBD, MD}) or begin with VB (imperative). Enforced only when
the original sentence had one.

Guard tokens: code spans, identifiers, links, numbers, named entities,
negation, quantifiers, modals, logical connectives. No edit may add, remove,
or cross them. Modals and connectives entered this list after measured meaning
flips ("can", "because").

Near-no-op: edits per human-corpus document must stay under the calibrated
threshold. The pivot is a rate tell (humans use LVCs at ~0.55x the machine
rate), so this gate is its overfire brake.

## 5. Performance targets

Python prototype measurements: DMS build 1.4 s (both indexes), DMS streaming
~0.5M tokens/s, full four-operation paragraph pass 5-19 ms including NLTK
tagging. Rust targets: index build offline into the pack; fix-time budget
sub-10 ms per KB end to end; DMS walk is automaton traversal and should reach
tens of MB/s per core.

## 6. What not to rebuild

Banned by experiment, with fixtures: search-chosen insertion of any kind (word
salad; meaning flips; broken function words), metric-centroid objectives
(chat-speak), aggressive collapse to imperatives (dumb-human), DMS as per-edit
judge, document-level embedding checks. ternlight is offline-mining-only
(template clustering). If a change makes any fixture pass that should fail,
the change is wrong, not the fixture.
