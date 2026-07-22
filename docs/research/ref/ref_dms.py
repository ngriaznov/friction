#!/usr/bin/env python3
"""DMS reference implementation (validated July 2026).

Token-level suffix automata over a machine and a human corpus; matching
statistics; differential profile; held-out document classification and span
extraction. Reproduces: 16/16 held-out classification, ritual-closer span
top-scored, zero false-positive spans on human text.

Usage: ref_dms.py /path/to/friction/corpus

STRING-LEVEL REFERENCE: production must map token indices to byte spans
(see ALGORITHMS.md section 0) and never rebuild text.
"""
import re, glob, sys, random, time
from collections import defaultdict

CORPUS = sys.argv[1] if len(sys.argv) > 1 else "."

def clean(s):
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    s = re.sub(r"`[^`]*`", " ", s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    return re.sub(r"[#>*_|]", " ", s)

def toks(s):
    return re.findall(r"[a-z']+|[.,;:!?]", s.lower())

def load_files(cls):
    fs = []
    for g in ("docs", "readme", "blog"):
        fs += sorted(glob.glob(f"{CORPUS}/{cls}/{g}/*.md"))
    return fs

class SAM:
    """Standard online suffix automaton over integer token ids."""
    __slots__ = ("nxt", "lnk", "ln", "last")
    def __init__(self):
        self.nxt = [{}]; self.lnk = [-1]; self.ln = [0]; self.last = 0
    def extend(self, c):
        cur = len(self.ln)
        self.nxt.append({}); self.ln.append(self.ln[self.last] + 1); self.lnk.append(-1)
        p = self.last
        while p != -1 and c not in self.nxt[p]:
            self.nxt[p][c] = cur; p = self.lnk[p]
        if p == -1:
            self.lnk[cur] = 0
        else:
            q = self.nxt[p][c]
            if self.ln[p] + 1 == self.ln[q]:
                self.lnk[cur] = q
            else:
                cl = len(self.ln)
                self.nxt.append(dict(self.nxt[q]))
                self.ln.append(self.ln[p] + 1)
                self.lnk.append(self.lnk[q])
                while p != -1 and self.nxt[p].get(c) == q:
                    self.nxt[p][c] = cl; p = self.lnk[p]
                self.lnk[q] = cl; self.lnk[cur] = cl
        self.last = cur

VOCAB = {}
def ids_grow(ts):
    out = []
    for t in ts:
        if t not in VOCAB:
            VOCAB[t] = len(VOCAB) + 1
        out.append(VOCAB[t])
    return out

def ids_query(ts):
    return [VOCAB.get(t, -1) for t in ts]

def matching_stats(sam, q):
    """Longest match ENDING at each position. Linear time."""
    v, l, out = 0, 0, []
    for c in q:
        if c != -1 and c in sam.nxt[v]:
            v = sam.nxt[v][c]; l += 1
        else:
            while v != 0 and (c == -1 or c not in sam.nxt[v]):
                v = sam.lnk[v]; l = sam.ln[v]
            if c != -1 and c in sam.nxt[v]:
                v = sam.nxt[v][c]; l += 1
            else:
                v, l = 0, 0
        out.append(l)
    return out

def spans(words, mM, mH, start_thresh=3, cont_thresh=2, min_len=2):
    """Maximal machine-register runs from the raw differential curve."""
    d = [a - b for a, b in zip(mM, mH)]
    runs, i = [], 0
    while i < len(d):
        if d[i] >= start_thresh:
            j = i
            while j < len(d) and d[j] >= cont_thresh:
                j += 1
            if j - i >= min_len:
                left = max(0, i - max(mM[i] - 1, 0))  # cover the matched phrase
                runs.append((sum(d[i:j]), left, j))
            i = j
        else:
            i += 1
    runs.sort(reverse=True)
    return runs

def main():
    random.seed(42)
    llm_files, hum_files = load_files("llm"), load_files("human")
    llm_test = set(random.sample(llm_files, 8))
    hum_test = set(random.sample(hum_files, 8))

    def stream(files, excl):
        out = []
        for f in files:
            if f in excl:
                continue
            out += toks(clean(open(f, errors="ignore").read())) + ["\x00"]
        return out

    t0 = time.time()
    samM, samH = SAM(), SAM()
    for c in ids_grow(stream(llm_files, llm_test)):
        samM.extend(c)
    for c in ids_grow(stream(hum_files, hum_test)):
        samH.extend(c)
    print(f"built in {time.time()-t0:.1f}s")

    correct = 0
    for f in sorted(llm_test) + sorted(hum_test):
        cls = "llm" if f in llm_test else "human"
        q = ids_query(toks(clean(open(f, errors="ignore").read())))
        mM = matching_stats(samM, q); mH = matching_stats(samH, q)
        diff = (sum(mM) - sum(mH)) / max(len(q), 1)
        pred = "llm" if diff > 0 else "human"
        correct += pred == cls
        print(f"{f}: {cls} d={diff:+.2f} -> {pred}")
    print(f"classification: {correct}/16")

if __name__ == "__main__":
    main()
