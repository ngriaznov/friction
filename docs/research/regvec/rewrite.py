"""
Rewrite transducers. Each candidate carries an exact feature delta (counts,
not rates) because Biber features are countable.

Precision discipline: a transducer fires only when its licensing condition
holds. No condition, no candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from biber import RESULT_VBG, is_nominalization

# Fixed discourse-marker participials. These are formulaic adverbials, not
# real participial clauses; rewriting them produces "We moves forward".
DISCOURSE_VBG = {"moving", "having", "speaking", "given", "considering",
                 "regarding", "concerning", "assuming", "supposing",
                 "following", "excluding", "including", "depending", "starting"}


def agrees_plural(subj) -> bool:
    """Subject takes a base-form present verb (we/they/plural nouns)."""
    if subj is None:
        return False
    if subj.lower_ in ("we", "they", "i", "you"):
        return True
    return subj.tag_ in ("NNS", "NNPS")


def finite_present(lemma: str, subj) -> str:
    return lemma if agrees_plural(subj) else third_sg(lemma)


def has_conj_participial(tok) -> bool:
    """', enriching X and writing Y' -- converting only the first conjunct
    leaves the second stranded. The coordinated VBG is not always a direct
    child (it may attach to the object or to a nested verb), so scan the
    subtree for any later VBG reachable through a conj edge."""
    for d in tok.subtree:
        if d is tok:
            continue
        if d.tag_ == "VBG" and d.dep_ == "conj":
            return True
        if d.tag_ == "VBG" and any(c.lower_ in ("and", "or") for c in d.lefts):
            return True
    return False

IRREGULAR_PAST = {
    "be": "was", "have": "had", "do": "did", "make": "made", "take": "took",
    "give": "gave", "find": "found", "run": "ran", "hold": "held",
    "lead": "led", "keep": "kept", "leave": "left", "mean": "meant",
    "send": "sent", "build": "built", "bring": "brought", "buy": "bought",
    "catch": "caught", "cut": "cut", "put": "put", "set": "set", "let": "let",
    "read": "read", "write": "wrote", "drive": "drove", "rise": "rose",
    "fall": "fell", "grow": "grew", "show": "showed", "see": "saw",
    "get": "got", "go": "went", "come": "came", "become": "became",
    "begin": "began", "choose": "chose", "draw": "drew", "feed": "fed",
    "meet": "met", "pay": "paid", "sell": "sold", "spend": "spent",
    "lose": "lost", "win": "won", "think": "thought", "teach": "taught",
}
IRREGULAR_PP = {
    "be": "been", "have": "had", "do": "done", "make": "made", "take": "taken",
    "give": "given", "find": "found", "run": "run", "hold": "held",
    "lead": "led", "keep": "kept", "leave": "left", "mean": "meant",
    "send": "sent", "build": "built", "bring": "brought", "buy": "bought",
    "catch": "caught", "cut": "cut", "put": "put", "set": "set", "let": "let",
    "read": "read", "write": "written", "drive": "driven", "rise": "risen",
    "fall": "fallen", "grow": "grown", "show": "shown", "see": "seen",
    "get": "got", "go": "gone", "come": "come", "become": "become",
    "begin": "begun", "choose": "chosen", "draw": "drawn", "feed": "fed",
    "meet": "met", "pay": "paid", "sell": "sold", "spend": "spent",
    "lose": "lost", "win": "won", "think": "thought", "teach": "taught",
}
SIBILANT = ("s", "x", "z", "ch", "sh", "o")


def third_sg(lemma: str) -> str:
    if lemma.endswith(SIBILANT):
        return lemma + "es"
    if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
        return lemma[:-1] + "ies"
    return lemma + "s"


def past(lemma: str) -> str:
    if lemma in IRREGULAR_PAST:
        return IRREGULAR_PAST[lemma]
    if lemma.endswith("e"):
        return lemma + "d"
    if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
        return lemma[:-1] + "ied"
    return lemma + "ed"


def past_participle(lemma: str) -> str:
    return IRREGULAR_PP.get(lemma, past(lemma))


@dataclass
class Candidate:
    kind: str
    start: int            # char offset in source
    end: int
    replacement: str
    delta: dict[str, int] = field(default_factory=dict)
    original: str = ""
    conf: float = 1.0


def _subtree_span(tok):
    toks = sorted(tok.subtree, key=lambda t: t.i)
    return toks[0], toks[-1]


def _clause_of(tok):
    """Walk up to the governing finite verb."""
    cur = tok
    while cur.head is not cur and cur.dep_ not in ("ROOT",):
        cur = cur.head
    return cur


def _is_past_tense(verb) -> bool:
    if verb.tag_ in ("VBD",):
        return True
    return any(c.tag_ == "VBD" for c in verb.children if c.dep_ in ("aux", "auxpass"))


# ---------------------------------------------------------------------------
# T1  trailing participial adverbial  ->  finite clause
#     ", ensuring throughput is maintained"  ->  ". This ensures ..."
#     ", using a new schema"                 ->  " and used a new schema"
# ---------------------------------------------------------------------------
def t1_trailing_participial(doc) -> list[Candidate]:
    out = []
    for t in doc:
        if t.tag_ != "VBG" or t.dep_ != "advcl":
            continue
        if any(c.dep_ in ("aux", "auxpass") for c in t.children):
            continue
        if t.lower_ in DISCOURSE_VBG or has_conj_participial(t):
            continue
        first, last = _subtree_span(t)
        if first is not t:            # only handle VBG-initial adjuncts
            continue
        if t.i == 0 or doc[t.i - 1].text != ",":
            continue                  # require the comma juncture
        comma = doc[t.i - 1]
        main = t.head
        if main.pos_ not in ("VERB", "AUX"):
            continue
        body = doc[t.i + 1: last.i + 1].text
        start = comma.idx
        end = last.idx + len(last.text)

        if t.lower_ in RESULT_VBG:
            # implicit subject is the preceding EVENT -> new sentence with "This"
            verb = third_sg(t.lemma_)
            rep = f". This {verb} {body}".rstrip()
            delta = {"present_participial": -1, "demonstratives": +1}
            conf = 0.9
        else:
            # implicit subject is the main-clause subject -> coordinate clause
            msubj = next((c for c in main.children if c.dep_ in ("nsubj", "nsubjpass")), None)
            verb = past(t.lemma_) if _is_past_tense(main) else finite_present(t.lemma_, msubj)
            rep = f" and {verb} {body}".rstrip()
            delta = {"present_participial": -1, "clausal_coord": +1}
            conf = 0.75
        out.append(Candidate("participial_trailing", start, end, rep, delta,
                             doc.text[start:end], conf))
    return out


# ---------------------------------------------------------------------------
# T2  postnominal participial  ->  finite relative clause
#     "a cache holding the dataset" -> "a cache that holds the dataset"
# ---------------------------------------------------------------------------
def t2_postnominal_participial(doc) -> list[Candidate]:
    out = []
    for t in doc:
        if t.tag_ != "VBG" or t.dep_ != "acl":
            continue
        if t.head.pos_ not in ("NOUN", "PROPN"):
            continue
        if any(c.dep_ in ("aux", "auxpass") for c in t.children):
            continue
        first, _ = _subtree_span(t)
        if first is not t:
            continue
        if t.i > 0 and doc[t.i - 1].text == ",":
            continue                  # non-restrictive; leave alone
        head_noun = t.head
        plural = head_noun.tag_ in ("NNS", "NNPS")
        verb = t.lemma_ if plural else third_sg(t.lemma_)
        start, end = t.idx, t.idx + len(t.text)
        out.append(Candidate("participial_postnominal", start, end,
                             f"that {verb}",
                             {"present_participial": -1},
                             t.text, 0.85))
    return out


# ---------------------------------------------------------------------------
# T3  leading participial  ->  coordinate finite clause
#     "Leveraging X, the system reduces Y" -> "The system leverages X and reduces Y"
# ---------------------------------------------------------------------------
def t3_leading_participial(doc) -> list[Candidate]:
    out = []
    for sent in doc.sents:
        first = sent[0]
        if first.tag_ != "VBG" or first.dep_ != "advcl":
            continue
        if any(c.dep_ in ("aux", "auxpass") for c in first.children):
            continue
        if first.lower_ in DISCOURSE_VBG or has_conj_participial(first):
            continue
        _, last = _subtree_span(first)
        if last.i + 1 >= sent.end or doc[last.i + 1].text != ",":
            continue
        main = first.head
        subj = next((c for c in main.children if c.dep_ in ("nsubj", "nsubjpass")), None)
        if subj is None:
            continue
        s_first, s_last = _subtree_span(subj)
        if s_first.i != last.i + 2:   # subject must directly follow the comma
            continue
        subj_txt = doc[s_first.i: s_last.i + 1].text
        obj_txt = doc[first.i + 1: last.i + 1].text
        verb = past(first.lemma_) if _is_past_tense(main) else finite_present(first.lemma_, subj)
        start = first.idx
        end = s_last.idx + len(s_last.text)
        rep = f"{subj_txt[0].upper()}{subj_txt[1:]} {verb} {obj_txt} and"
        # the main verb follows; "and" chains it
        out.append(Candidate("participial_leading", start, end, rep,
                             {"present_participial": -1, "clausal_coord": +1},
                             doc.text[start:end], 0.7))
    return out


# ---------------------------------------------------------------------------
# T4  active -> agentless passive     (RAISES a feature LLMs under-produce)
#     "We deployed the change" -> "The change was deployed"
# ---------------------------------------------------------------------------
GENERIC_SUBJ = {"we", "i", "you", "they", "one", "the team", "our team"}


def t4_activize_to_passive(doc) -> list[Candidate]:
    out = []
    for t in doc:
        if t.pos_ != "VERB" or t.tag_ not in ("VBD", "VBZ", "VBP"):
            continue
        if any(c.dep_ == "auxpass" for c in t.children):
            continue
        subj = next((c for c in t.children if c.dep_ == "nsubj"), None)
        obj = next((c for c in t.children if c.dep_ == "dobj"), None)
        if subj is None or obj is None:
            continue
        if subj.lower_ not in GENERIC_SUBJ:
            continue                  # only demote recoverable/generic agents
        # "We instrumented X, capturing Y, and discovered Z" -- passivising the
        # first conjunct orphans "and discovered". Skip coordinated predicates.
        if any(c.dep_ == "conj" and c.pos_ == "VERB" for c in t.children):
            continue
        if t.dep_ == "conj":
            continue
        o_first, o_last = _subtree_span(obj)
        if o_last.i < o_first.i:
            continue
        obj_txt = doc[o_first.i: o_last.i + 1].text
        # anything after the object stays put
        plural = obj.tag_ in ("NNS", "NNPS") or obj.lower_ in ("they", "these", "those")
        be = ("were" if plural else "was") if t.tag_ == "VBD" else ("are" if plural else "is")
        s_first, _ = _subtree_span(subj)
        start = s_first.idx
        end = o_last.idx + len(o_last.text)
        rep = f"{obj_txt[0].upper()}{obj_txt[1:]} {be} {past_participle(t.lemma_)}"
        out.append(Candidate("active_to_passive", start, end, rep,
                             {"agentless_passive": +1, "first_person": -1},
                             doc.text[start:end], 0.8))
    return out


# ---------------------------------------------------------------------------
# T5  nominalization unpacking:  "the optimization of X"  ->  "optimizing X"
#     Keeps it verbal without inventing an agent.
# ---------------------------------------------------------------------------
NOMINAL_VERB = {
    "optimization": "optimizing", "reduction": "reducing", "creation": "creating",
    "implementation": "implementing", "integration": "integrating",
    "migration": "migrating", "deployment": "deploying", "improvement": "improving",
    "allocation": "allocating", "utilization": "using", "configuration": "configuring",
    "validation": "validating", "verification": "verifying", "execution": "executing",
    "generation": "generating", "adoption": "adopting", "expansion": "expanding",
    "introduction": "introducing", "elimination": "eliminating",
    "consolidation": "consolidating", "degradation": "degrading",
    "compression": "compressing", "duplication": "duplicating",
}


def t5_nominalization(doc) -> list[Candidate]:
    out = []
    for t in doc:
        if not is_nominalization(t) or t.lower_ not in NOMINAL_VERB:
            continue
        det = next((c for c in t.children if c.dep_ == "det" and c.lower_ == "the"), None)
        of = next((c for c in t.children if c.dep_ == "prep" and c.lower_ == "of"), None)
        if det is None or of is None:
            continue
        pobj = next((c for c in of.children if c.dep_ == "pobj"), None)
        if pobj is None:
            continue
        p_first, p_last = _subtree_span(pobj)
        arg = doc[p_first.i: p_last.i + 1].text
        start = det.idx
        end = p_last.idx + len(p_last.text)
        rep = f"{NOMINAL_VERB[t.lower_]} {arg}"
        if det.is_sent_start:
            rep = rep[0].upper() + rep[1:]
        out.append(Candidate("nominalization_unpack", start, end, rep,
                             {"nominalization": -1, "prepositions": -1},
                             doc.text[start:end], 0.85))
    return out


ALL_TRANSDUCERS = (
    t1_trailing_participial,
    t2_postnominal_participial,
    t3_leading_participial,
    t4_activize_to_passive,
    t5_nominalization,
)


def candidates(doc) -> list[Candidate]:
    out: list[Candidate] = []
    for fn in ALL_TRANSDUCERS:
        out.extend(fn(doc))
    return sorted(out, key=lambda c: (c.start, c.end))


def apply_candidates(text: str, chosen: list[Candidate]) -> str:
    """Apply non-overlapping edits right-to-left."""
    for c in sorted(chosen, key=lambda c: -c.start):
        text = text[:c.start] + c.replacement + text[c.end:]
    return text
