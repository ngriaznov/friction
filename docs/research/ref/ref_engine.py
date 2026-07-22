#!/usr/bin/env python3
"""Combined four-operation engine, reference implementation (validated July 2026).

Pipeline per sentence: ritual deletion -> paired substitution -> derivational
pivot (max 2) -> gated span deletion. Reproduces the composed-paragraph run
(all four operations, 9 ms) and the real-corpus one-edit run.

Usage: ref_engine.py /path/to/friction/corpus
Requires nltk (averaged_perceptron_tagger). Imports pivot from ref_pivot.

STRING-LEVEL REFERENCE: production emits byte-span patches; operates only on
prose blocks from friction-parse's tree (this script trusts its input to be
prose, which is why the heading fixture exists).
"""
import re, glob, sys, random
from collections import Counter, defaultdict
from nltk.tag.perceptron import PerceptronTagger
from ref_pivot import pivot

TAGGER = PerceptronTagger()
CORPUS = sys.argv[1] if len(sys.argv) > 1 else "."

def clean(s):
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    s = re.sub(r"`[^`]*`", " ", s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    return re.sub(r"[#>*_|]", " ", s)

def toks(s):
    return re.findall(r"[a-z']+|[.,;:!?]", s.lower())

def coarse(pt):
    return pt if pt in ",.;:!?" else pt[:2]

# ---- human-corpus bigram table and skeleton sets (the attestation gates) ----
random.seed(42)
BIGRAM = defaultdict(Counter); SKEL5 = set(); SKEL4 = set()
_hum = [f for g in ("docs", "readme", "blog") for f in sorted(glob.glob(f"{CORPUS}/human/{g}/*.md"))]
_hold = set(random.sample(_hum, 8)) if len(_hum) > 8 else set()
for _f in _hum:
    if _f in _hold:
        continue
    for _sent in re.split(r"(?<=[.!?])", clean(open(_f, errors="ignore").read())):
        _ts = toks(_sent)
        if not _ts:
            continue
        _tags = [coarse(t) for _, t in TAGGER.tag(_ts)]
        _seq = ["<s>"] + _ts
        for a, b in zip(_seq, _seq[1:]):
            BIGRAM[a][b] += 1
        _tg = ["<S>"] + _tags + ["<E>"]
        for i in range(len(_tg) - 4):
            SKEL5.add(tuple(_tg[i:i + 5]))
        for i in range(len(_tg) - 3):
            SKEL4.add(tuple(_tg[i:i + 4]))

# ---- inventories (demo seed; production loads the mined, versioned pack) ----
RITUAL = [re.compile(p, re.I) for p in [
    r"if you have any questions.{0,60}(reach out|contact|let us know)",
    r"^congratulations", r"^happy \w+ing", r"we hope (this|you)"]]
SUBS = [(re.compile(p, re.I), r) for p, r in [
    (r"\bthis guide will walk you through\b", "this guide covers"),
    (r"\bwill walk you through\b", "covers"),
    (r"\bin order to\b", "to"), (r"\bprior to\b", "before"),
    (r"\butilizes?\b", "uses"), (r"\bleverages?\b", "uses")]]
SPANS = [re.compile(p, re.I) for p in [
    r"^it is important to note that\s+", r"\bit is important to note that\s+",
    r"^it'?s worth noting that\s+", r"^by following these steps,?\s+",
    r"\bquickly and easily\s+", r"\bsimply\s+", r"\s+to suit your needs\b",
    r"^please note that\s+"]]

# ---- gates ----
def full_tags(ts):
    return [t for _, t in TAGGER.tag(ts)]

def has_finite(ts):
    tg = full_tags(ts)
    return any(t in ("VBZ", "VBP", "VBD", "MD") for t in tg) or (tg and tg[0] == "VB")

def skeleton_ok(ts, lo, hi):
    tg = ["<S>"] + [coarse(t) for _, t in TAGGER.tag(ts)] + ["<E>"]
    a = max(0, lo - 3); b = min(len(tg) - 4, hi + 3)
    for i in range(a, max(a + 1, b)):
        w = tuple(tg[i:i + 5])
        if w not in SKEL5 and tuple(w[:4]) not in SKEL4:
            return False
    return True

def fix_sentence(sent):
    edits = []; s = sent
    for pat in RITUAL:                                   # op 1
        if pat.search(s):
            return "", [("ritual", "SENTENCE DELETED")]
    for pat, rep in SUBS:                                # op 3
        if pat.search(s):
            s2 = pat.sub(rep, s)
            if has_finite(toks(s2)) or not has_finite(toks(sent)):
                edits.append((pat.pattern, f"SUB '{rep}'")); s = s2
    for _ in range(2):                                   # op 4
        p, why = pivot(s)
        if p:
            s = p; edits.append(("LVC", why))
        else:
            break
    for pat in SPANS:                                    # op 2 (deletion only, no bridges)
        m = pat.search(s)
        if not m:
            continue
        pre, post = s[:m.start()], s[m.end():]
        pre_t, post_t = toks(pre), toks(post)
        L = pre_t[-1] if pre_t else "<s>"
        R = post_t[0] if post_t else "."
        seam_ok = (not pre_t) or R == "." or (R in BIGRAM.get(L, {}))
        ct = pre_t + post_t; lo = len(pre_t); hi = lo + 1
        if seam_ok and skeleton_ok(ct, lo, hi) and (has_finite(ct) or not has_finite(toks(sent))):
            s = (pre.rstrip() + " " + post.lstrip()).strip()
            s = re.sub(r"\s+([.,;:!?])", r"\1", s)
            s = re.sub(r"\s{2,}", " ", s)
            edits.append((m.group(0).strip(), "deleted"))
        else:
            edits.append((m.group(0).strip(), "KEPT (gates)"))
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s, edits

def fix_paragraph(para):
    outs, log = [], []
    for sent in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", para).strip()):
        if not sent.strip():
            continue
        s, e = fix_sentence(sent.strip())
        if s:
            outs.append(s)
        log += e
    return " ".join(outs), log

if __name__ == "__main__":
    demo = ("This guide will walk you through configuring the backup agent for your "
            "staging environment. It is important to note that the agent performs "
            "validation of the configuration file before each run. If you have any "
            "questions or require further assistance, please reach out to our support team.")
    out, log = fix_paragraph(demo)
    print("BEFORE:", demo, "\n\nAFTER:", out, "\n")
    for a, b in log:
        print(f"  {b:24s} {a[:60]}")
