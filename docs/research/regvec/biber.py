"""
Biber-subset feature extractor.

Implements the discriminating features from Reinhart et al. (2025) PNAS
122(8):e2422455122, Figure 3 (top-15 by random-forest importance), plus a few
stance features from Jiang & Hyland (2024).

NOT a faithful pseudobibeR port. That is the production task and any drift in
feature definitions invalidates the target vector. Definitions here are my
reading of Biber's categories via spaCy, and are the POC's largest source of
error.

Rates are per 1000 words.
"""

from __future__ import annotations

from dataclasses import dataclass

import spacy

NLP = spacy.load("en_core_web_sm")

NOMINAL_SUFFIXES = (
    "tion", "sion", "ment", "ness", "ity", "ance", "ence",
    "ism", "ship", "hood", "ency", "ancy", "ure", "al",
)
NOMINAL_EXCEPT = {"animal", "signal", "material", "capital", "total", "final",
                  "several", "nature", "future", "picture", "measure", "figure"}

DOWNTONERS = {"barely", "hardly", "mildly", "nearly", "partially", "partly",
              "practically", "scarcely", "slightly", "somewhat", "almost"}

HEDGES = {"perhaps", "maybe", "possibly", "probably", "apparently", "seemingly",
          "arguably", "presumably", "roughly", "approximately", "likely",
          "might", "could", "may", "suggest", "suggests", "appear", "appears",
          "seem", "seems", "tend", "tends"}

# Present participials that express result/purpose -- their implicit subject is
# the PRECEDING EVENT, not the main-clause subject. Governs rewriting.
RESULT_VBG = {"allowing", "ensuring", "enabling", "resulting", "leading",
              "making", "providing", "giving", "causing", "creating",
              "improving", "reducing", "increasing", "helping", "yielding",
              "offering", "delivering", "producing", "driving", "meaning"}

# --------------------------------------------------------------------------
# Published measurements. GPT-4o rate relative to genre-matched human text.
# Source: Reinhart et al. 2025, PNAS 122(8):e2422455122.
# `d` is the reported paired Cohen's d. None = not published; feature is
# MEASURED ONLY and excluded from the homing objective.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureSpec:
    name: str
    ratio: float | None   # LLM rate / human rate
    d: float | None       # paired Cohen's d
    note: str = ""


SPECS: list[FeatureSpec] = [
    FeatureSpec("present_participial", 5.3, 1.38, "GPT-4o; largest single divergence"),
    FeatureSpec("nominalization",      2.1, 1.23, "GPT-4o"),
    FeatureSpec("phrasal_coord",       1.9, 0.81, "GPT-4o"),
    FeatureSpec("that_subj",           2.6, 0.77, "GPT-4o"),
    FeatureSpec("agentless_passive",   0.5, None, "GPT-4o UNDER-uses; d not published"),
    # measured-only diagnostics (direction known from Fig 3, magnitude not given)
    FeatureSpec("clausal_coord",       None, None, "GPT-4o avoids; Llama over-uses"),
    FeatureSpec("attributive_adj",     None, None, ""),
    FeatureSpec("past_part_postnom",   None, None, ""),
    FeatureSpec("adverbs",             None, None, ""),
    FeatureSpec("downtoners",          None, None, ""),
    FeatureSpec("demonstratives",      None, None, ""),
    FeatureSpec("prepositions",        None, None, ""),
    FeatureSpec("infinitives",         None, None, ""),
    FeatureSpec("nouns",               None, None, ""),
    FeatureSpec("mean_word_length",    None, None, "not a rate; excluded from norm"),
    FeatureSpec("hedges",              None, None, "Jiang & Hyland: LLMs under-use"),
    FeatureSpec("first_person",        None, None, "Jiang & Hyland: fewer human subjects"),
    FeatureSpec("demon_sent_initial",  None, None, "POC-added: positional, not in Biber"),
]

NAMES = [s.name for s in SPECS]
SPEC_BY_NAME = {s.name: s for s in SPECS}
HOMING = [s.name for s in SPECS if s.ratio is not None]


# --------------------------------------------------------------------------
# Feature detectors
# --------------------------------------------------------------------------

def is_nominalization(tok) -> bool:
    if tok.pos_ != "NOUN":
        return False
    low = tok.lower_
    if low in NOMINAL_EXCEPT or len(low) < 6:
        return False
    return low.endswith(NOMINAL_SUFFIXES)


def present_participials(doc) -> list:
    """VBG heading a non-finite clause (acl / advcl), excluding progressives."""
    out = []
    for t in doc:
        if t.tag_ != "VBG":
            continue
        if t.dep_ not in ("acl", "advcl", "xcomp"):
            continue
        # progressive "is running" -> has an aux child
        if any(c.dep_ in ("aux", "auxpass") for c in t.children):
            continue
        # xcomp after a raising verb ("kept running") is progressive-ish; only
        # count xcomp when introduced by a comma (a real participial adjunct)
        if t.dep_ == "xcomp" and t.i > 0 and doc[t.i - 1].text != ",":
            continue
        out.append(t)
    return out


def that_subject_clauses(doc) -> list:
    """'That' clause functioning as clausal subject."""
    return [t for t in doc if t.dep_ == "csubj" and
            any(c.lower_ == "that" and c.dep_ == "mark" for c in t.children)]


def phrasal_coords(doc) -> list:
    """Coordination whose conjuncts are phrases (NP/AdjP/AdvP), not clauses."""
    out = []
    for t in doc:
        if t.dep_ != "conj":
            continue
        if t.pos_ in ("NOUN", "PROPN", "PRON", "ADJ", "ADV", "NUM"):
            if t.head.pos_ in ("NOUN", "PROPN", "PRON", "ADJ", "ADV", "NUM"):
                out.append(t)
    return out


def clausal_coords(doc) -> list:
    return [t for t in doc if t.dep_ == "conj" and t.pos_ in ("VERB", "AUX")]


def agentless_passives(doc) -> list:
    out = []
    for t in doc:
        if not any(c.dep_ == "auxpass" for c in t.children):
            continue
        has_agent = any(
            c.dep_ == "agent" or (c.dep_ == "prep" and c.lower_ == "by")
            for c in t.children
        )
        if not has_agent:
            out.append(t)
    return out


def past_part_postnominal(doc) -> list:
    return [t for t in doc if t.tag_ == "VBN" and t.dep_ == "acl"
            and t.head.pos_ in ("NOUN", "PROPN")]


def infinitives(doc) -> list:
    return [t for t in doc if t.tag_ == "VB"
            and any(c.lower_ == "to" and c.dep_ == "aux" for c in t.children)]


def extract(text: str) -> tuple[dict[str, float], object]:
    doc = NLP(text)
    words = [t for t in doc if not t.is_punct and not t.is_space]
    n = len(words)
    k = 1000.0 / max(n, 1)

    counts = {
        "present_participial": len(present_participials(doc)),
        "nominalization":      sum(1 for t in doc if is_nominalization(t)),
        "phrasal_coord":       len(phrasal_coords(doc)),
        "that_subj":           len(that_subject_clauses(doc)),
        "agentless_passive":   len(agentless_passives(doc)),
        "clausal_coord":       len(clausal_coords(doc)),
        "attributive_adj":     sum(1 for t in doc if t.dep_ == "amod"),
        "past_part_postnom":   len(past_part_postnominal(doc)),
        "adverbs":             sum(1 for t in doc if t.pos_ == "ADV"),
        "downtoners":          sum(1 for t in doc if t.lower_ in DOWNTONERS),
        # Biber's demonstratives are pronouns/determiners. Complementizer
        # "that" (dep=mark) and relativizer "that" must NOT count -- this bug
        # was caught by the delta-validation check, not by inspection.
        "demonstratives":      sum(1 for t in doc if t.lower_ in
                                   ("this", "that", "these", "those")
                                   and t.dep_ not in ("mark",)
                                   and t.pos_ in ("DET", "PRON")),
        "prepositions":        sum(1 for t in doc if t.pos_ == "ADP"),
        "infinitives":         len(infinitives(doc)),
        "nouns":               sum(1 for t in doc if t.pos_ in ("NOUN", "PROPN")),
        "hedges":              sum(1 for t in doc if t.lower_ in HEDGES),
        "first_person":        sum(1 for t in doc if t.lower_ in
                                   ("i", "we", "my", "our", "us", "me")),
        # Positional feature, NOT part of Biber's tagset. Added because the
        # repair for participials ("...  . This ensures X") concentrates
        # demonstratives at sentence-initial position -- a slop signature the
        # count vector alone cannot express.
        "demon_sent_initial":  sum(1 for t in doc if t.is_sent_start
                                   and t.lower_ in ("this", "these", "that")),
    }
    vec = {name: c * k for name, c in counts.items()}
    vec["mean_word_length"] = (sum(len(t.text) for t in words) / max(n, 1))
    vec["_words"] = float(n)
    return vec, doc
