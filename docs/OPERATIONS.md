# The operations in detail

The eight editing operations, the gate stack, and the detection channels,
in depth. The README's [What it does](../README.md#what-it-does) section is
the short form of this file.

1. **Ritual deletion**: sentences matching ritual frames are removed whole:
   *"If you have any questions or require further assistance, please reach out
   to our support team."* → gone.
2. **Gated span deletion**: a detected filler span is removed only if the
   resulting seam is attested in a human-written corpus or falls at a sentence
   start, and the sentence still has a finite verb afterward:
   *"It is important to note that the agent…"* → *"The agent…"*.
3. **Paired substitution**: a slop frame becomes its plain-register counterpart
   from a versioned, audited inventory: *"will walk you through"* → *"covers"*,
   *"in order to"* → *"to"*, *"leverages"* → *"uses"*, *"is crucial for"* →
   *"is important for"*.
4. **Derivational pivot**: a licensed light-verb construction collapses to its
   root verb, inheriting tense and agreement: *"the agent performs validation of
   the config file"* → *"the agent validates the config file"*, *"we made the
   decision to switch"* → *"we decided to switch"*.
5. **Frame-gated `just`-deletion**: inside a detected dismissive-foil question
   (*"is X just A, or B?"*), the marker is deleted, never the question itself:
   *"is provenance just frontmatter metadata, or does every derived graph edge
   retain immutable lineage?"* → *"is provenance frontmatter metadata, or does
   every derived graph edge retain immutable lineage?"*. Fires only for
   `just`/`merely`/`simply` (never `only`, which often carries real quantity
   meaning), and only when the marker sits strictly between the question's
   auxiliary and its coordinating `or`, so a genuine either-or question is
   never touched.
6. **Frame rewriting**: a compiled rule program of corpus-adjudicated
   lexical frames: *"We utilized the cache"* → *"We used the cache"*,
   *"It showcases the results"* → *"It shows the results"*, *"alongside the
   dashboard"* → *"with the dashboard"*. Every rule carries measured
   per-million frequencies from the machine and human corpora, and the pack
   compiler enforces the evidence at build time: a rule whose target word
   is unattested in human text, whose trigger measures human-tilted, or
   whose trigger is too common in human prose to auto-edit
   (100+/M) never compiles as an edit: it demotes to a report-only
   finding, and `--suggest` shows it with its measured rates. Rewrites
   realize through the inflection tables, so tense and agreement survive
   the swap, and every candidate still passes the same seam, clause, and
   skeleton gates as a deletion.
7. **Clause restructuring**: parse-level, per-instance rewrites of
   constructions the flat frame grammar structurally cannot express. Two
   ship today. The `ensure that` collapse takes an embedded bare-BE
   passive and promotes its participle to the main verb: *"Review the
   config to ensure that logging is enabled"* → *"Review the config to
   enable logging"*, in imperative, infinitival, and finite-subject
   frames (the last with person/number/tense re-inflection). The
   transitive substitution rewrites a verb only when the parse proves a
   direct object: *"Profiling surfaced a bottleneck"* → *"Profiling
   found a bottleneck"*, while intransitive uses (*"surfaced as a
   contributor"*) keep their report-only finding. Every candidate
   passes the same seam/skeleton/clause gates as a frame rewrite, and
   the decline set is the design: negated complements (two independent
   guards), adverbs between auxiliary and participle, modal or active
   complements, and any parse anomaly all decline with a named reason
   rather than guess. Unlike every rule above, these are licensed per
   instance with no band and no rate: a correct rewrite applies no
   matter what the rest of the document looks like.
8. **Register rephrasing**: the only operation that can *raise* a construction
   rather than remove one. Language models under-produce the agentless passive
   relative to human technical writing, consistently and across every model
   measured, and no amount of deleting fixes an under-use. So a clause with a
   recoverable agent may be demoted: *"as you make changes"* → *"as changes are
   made"*, *"how we handle code reviews"* → *"how code reviews are handled"*. A
   nominalization may also be unpacked: *"the integration of SQS"* →
   *"integrating SQS"*. A third feature only ever removes: the em dash, which
   Claude-family output uses heavily and human technical documentation almost
   never does, is rewritten toward the punctuation the surrounding clauses
   already license: *"the queue holds items — 3 workers drain it"* → *"the
   queue holds items; 3 workers drain it"*, *"reached over the network — it is
   never called directly"* → *"reached over the network. It is never called
   directly"*. A fourth homes toward a band that is nonzero, not to
   zero: the semicolon, which Claude-family output also uses well past the
   human rate, is split into a sentence break only when it joins two
   independent clauses: *"the job reads from the queue; it commits offsets
   only after the batch is durably written"* → *"the job reads from the
   queue. It commits offsets only after the batch is durably written"*. A
   semicolon that is part of a serial list, or that has no independent clause
   on either side, is left alone; so is one inside the human range.

   It fires only while the document sits outside a band measured from human
   documents, and stops at the band's edge rather than its centre. Every
   document landing on the same coordinates would be a tell of its own. The
   band comparison uses a Wilson confidence bound, not the raw rate, so the
   evidence has to be worth the sample: a 58-word comment with four
   nominalizations reads 69 per 1000 words (above the band), but a sample
   that small can't support the claim, and the feature stays quiet; the same
   density sustained over 700 words arms it. Short texts are never refused,
   just held to a higher evidentiary bar (a single em dash still arms its
   zero band at any length). On the corpus it edits roughly one machine
   document in ten.

Every edit passes a stack of hard gates: closure as described above (content
words input-derived, function words only from the declared set) is checked on
every candidate before it is applied. Beyond that, the sentence must stay
clause-complete, edits never touch code spans,
links, numbers, identifiers, negation, quantifiers, modals, or logical
connectives, quoted text is left alone because it is someone's example rather
than the author's own register, and edits fire only inside prose blocks: never
in headings, code, tables, or link text. On human-written text the whole engine
is calibrated to be a near-no-op. When a gate says no, the candidate is kept
verbatim and reported as a suggestion: doing nothing is the designed fallback,
not a failure mode.

Most of what the gates encode was learned by running the engine over the corpus
and reading what it produced. Passivizing across a preposition turned *"as we
continue down this path"* into *"as this path is continued"*; promoting a
post-modified object turned *"inspected each board for knots and defects"* into
*"each board for knots and defects was inspected"*, which no longer says what
the inspection was for. Neither is caught by a grammar check. Both are
meaning changes that read fine. The guards that refuse them are in the source
with the sentence that motivated each one.

Detection (what finds the candidates) runs six channels. Five are span-level: a
mined literal inventory, a shallow tag-pattern scan for light-verb
constructions, a differential matching-statistics profile computed against one pooled
machine suffix automaton (every mined generator family's corpus in a
single index, which also powers the document-level report in
`friction check`), a deterministic contrast-frame
template scan: `frame.contrast.question` (the dismissive-foil interrogative
above) and `frame.contrast.correction` (declarative epanorthosis, *"not just
X — it's Y"*), both detect-only in `check` and, for `fix`, reported in the
paraphrase list with DMS (differential matching statistics, friction's
own name for its corpus-differential detection channel, built on the
matching-statistics literature; see `docs/research/ALGORITHMS.md` §1)
candidates whenever no gated edit applies (an
`only`-marked question, or any correction span), and `jargon.metaphor`, a
tag-gated scan over a curated list of physical/aesthetic metaphor nouns
("resonance", "tapestry", "well", "soup"…), flagged only when one heads a noun
compound as its rightmost, tagged-noun word, immediately preceded by at least
one noun/adjective modifier: *"semantic wells"*, *"cross-domain resonance"*, *"a
rich tapestry of services"*. The same word as a bare noun (*"the well is
deep"*) or as a modifier (*"soup kitchen"*) never matches, and a compound with
any mid-sentence capitalized word is treated as a possible product name and
declined. The web-scale attestation design `docs/research/SYNTHESIS.md` §4
describes (every head word frequent, but the compound itself unattested
anywhere) is the exemption mechanism: a compound is checked against
`jargon-attest-v1`, a `BinaryFuse8` filter over ~2M normalized Wikipedia-title
and OpenAlex-topic keys built offline and embedded in the binary (*"data
fabric"*, *"primordial soup"*, *"color harmony"*, *"resonance frequency"*, and
every other real title, are attested and never flagged), OR against a small
hand-curated TOML exception list that only carries what the filter still
misses (*"service fabric"*, *"test well"*), each with its own stated reason.
This is a deliberately narrow, high-precision slice of pseudo-jargon (a
curated head-word lexeme, not an open-vocabulary jargon detector, paired with
a real attestation table, not a hand-picked exceptions list alone), and it is
detection-only like the contrast-frame templates: there is no deterministic
true replacement for an invented term, so a flagged span is reported in
`check` and unioned into `fix`'s paraphrase list, never rewritten.
The sixth channel is document-level: a count of register-marking constructions
over a dependency parse, compared against the human band. It answers a
question the others cannot, because a construction that is *missing* has no
span to detect. A seventh, `overuse.word`, is also document-level and also
detect-only: a finite verb or adverb repeated in one document more densely
than ANY single human document in the reference corpus ever used it (each
word's ceiling, its burst envelope, ships in `human-evidence-v1`). Nouns
and adjectives are exempt on purpose: a document about Haskell is allowed to
say "haskell", and the topic words a document is entitled to are exactly what
per-word envelopes measured from real human documents encode.
