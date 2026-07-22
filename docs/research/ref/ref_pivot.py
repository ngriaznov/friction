#!/usr/bin/env python3
"""Derivational pivot (LVC collapse) reference implementation (validated July 2026).

Reproduces: 5/5 constructed cases, 6/6 traps rejected, real-corpus rewrites.
Requires: nltk with averaged_perceptron_tagger.

STRING-LEVEL REFERENCE: production emits byte-span patches (ALGORITHMS.md 0)
and fires only inside prose blocks (gate 7) — the heading failure fixture
exists because this script has no block tree.
"""
import re
from nltk.tag.perceptron import PerceptronTagger
TAGGER = PerceptronTagger()

# Derivational lexicon: NOMLEX/CatVar-seed style. Production loads this from the
# pack, and an entry is ACTIVE only if a (light_verb, nominalization) pair is
# licensed by mining. Subject-genitive-prone nominals (approval, consent) are
# excluded by never licensing them, not by a runtime heuristic.
DERIV = {"initialization": "initialize", "configuration": "configure",
"installation": "install", "validation": "validate", "optimization": "optimize",
"migration": "migrate", "deletion": "delete", "creation": "create",
"execution": "execute", "analysis": "analyze", "assessment": "assess",
"evaluation": "evaluate", "comparison": "compare", "conversion": "convert",
"integration": "integrate", "implementation": "implement",
"modification": "modify", "verification": "verify",
"authentication": "authenticate", "encryption": "encrypt",
"compression": "compress", "reduction": "reduce", "adjustment": "adjust",
"extraction": "extract", "decision": "decide", "investigation": "investigate",
"inspection": "inspect", "calculation": "calculate",
"transformation": "transform", "aggregation": "aggregate",
"synchronization": "synchronize"}

LV = {"perform": "base", "performs": "3sg", "performed": "past", "performing": "ger",
      "conduct": "base", "conducts": "3sg", "conducted": "past", "conducting": "ger",
      "make": "base", "makes": "3sg", "made": "past", "making": "ger",
      "do": "base", "does": "3sg", "did": "past", "doing": "ger"}
BE = {"is", "are", "was", "were", "been", "being", "be"}

def inflect(v, feat):
    if feat == "base":
        return v
    if feat == "3sg":
        return v[:-1] + "ies" if v.endswith("y") else v + "s"
    if feat == "past":
        return v + "d" if v.endswith("e") else v + "ed"
    return (v[:-1] if v.endswith("e") else v) + "ing"  # gerund

def pivot(sentence):
    """Returns (rewritten_sentence, note) or (None, reject_reason)."""
    words = re.findall(r"[A-Za-z']+|[.,;:!?]", sentence)
    tags = [t for _, t in TAGGER.tag(words)]
    for i, w in enumerate(words):
        lw = w.lower()
        if lw not in LV:
            continue
        feat = LV[lw]
        if feat == "past" and i > 0 and words[i - 1].lower() in BE:
            return None, "REJECT: passive"
        j = i + 1
        if j < len(words) and words[j].lower() in ("a", "an", "the"):
            j += 1
        if j < len(words) and tags[j] == "JJ":
            return None, f"REJECT: modified nominal ('{words[j]}')"
        if j >= len(words):
            continue
        nom = words[j].lower()
        if nom.endswith("s") and nom[:-1] in DERIV:
            return None, "REJECT: plural nominal"
        if nom not in DERIV:
            continue  # unlicensed -> not a match at all
        v = inflect(DERIV[nom], feat)
        if i == 0 and w[0].isupper():
            v = v.capitalize()
        k = j + 1
        if k < len(words) and words[k].lower() == "of":
            out = words[:i] + [v] + words[k + 1:]   # of-NP promoted to object
        else:
            out = words[:i] + [v] + words[j + 1:]
        s = " ".join(out)
        s = re.sub(r"\s+([.,;:!?])", r"\1", s)
        return s, f"'{' '.join(words[i:j+1])}...' -> '{v}'"
    return None, None

if __name__ == "__main__":
    accept = [
        "The system performs an initialization of the database.",
        "The tool performs validation of the configuration file before applying changes.",
        "Before deployment, the script conducts an analysis of the dependency graph.",
        "You should make a decision soon.",
        "The installer performed a comparison of the two versions."]
    traps = [
        "Obtain the approval of the committee before merging.",
        "The migration performs a full initialization of the database.",
        "Create an index of the most common queries.",
        "The wizard facilitates the integration of the plugin with your DAW.",
        "An initialization of the database is performed by the system.",
        "The tool performs several initializations of the database during testing."]
    for c in accept:
        s, why = pivot(c)
        print(("OK   " if s else "MISS ") + (s or c), f"[{why}]")
    for c in traps:
        s, why = pivot(c)
        print(("PASS " if s is None else "FAIL ") + c, f"[{why}]")
