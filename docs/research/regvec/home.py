"""
Register-vector homing.

Target mu_f = v0_f / ratio_f, where v0 is the measured document vector and
ratio_f is the published LLM/human rate ratio. This assumes the input is
typical instruction-tuned LLM output -- the assumption the whole POC rests on.

Objective is log-space because the published divergences are multiplicative
and Reinhart et al. plot them on a log scale:

    D^2(v) = sum_f  w_f * ( ln(v_f + eps) - ln(mu_f + eps) )^2

w_f is the published paired Cohen's d. A true Mahalanobis form needs the
human covariance matrix, which requires a reference corpus. This diagonal
weighting is the POC substitute and is the second-largest source of error
after the feature definitions.
"""

from __future__ import annotations

import math

from biber import HOMING, SPEC_BY_NAME, extract
from rewrite import Candidate

EPS = 0.05
# Cohen's d is unpublished for agentless_passive. Placeholder = median of the
# four published values. Flagged in output; do not treat as measured.
PLACEHOLDER_W = {"agentless_passive": 1.02}


def weight(name: str) -> float:
    d = SPEC_BY_NAME[name].d
    return d if d is not None else PLACEHOLDER_W.get(name, 0.0)


def target(v0: dict[str, float]) -> dict[str, float]:
    return {f: v0[f] / SPEC_BY_NAME[f].ratio for f in HOMING}


# POC FINDING: homing on a subset of features with a DIAGONAL weighting lets
# the unconstrained features drift badly. In the first run, converting
# participials to coordinate clauses drove clausal_coord from 18.7 to 46.6 per
# 1000 -- a feature Reinhart et al. report GPT-4o already *avoids*. Without the
# human covariance matrix there is nothing holding it. This drift term is the
# stand-in: features with no published target are penalised for MOVING AT ALL.
DRIFT_LAMBDA = 0.35
DRIFT_EXCLUDE = {"mean_word_length", "_words"}


def distance(v: dict[str, float], mu: dict[str, float],
             v0: dict[str, float] | None = None) -> float:
    d = sum(
        weight(f) * (math.log(v[f] + EPS) - math.log(mu[f] + EPS)) ** 2
        for f in HOMING
    )
    if v0 is not None:
        for f, val in v.items():
            if f in HOMING or f in DRIFT_EXCLUDE or f.startswith("_"):
                continue
            d += DRIFT_LAMBDA * (math.log(val + EPS) - math.log(v0[f] + EPS)) ** 2
    return d


def predict(v: dict[str, float], deltas: dict[str, int], n_words: float) -> dict[str, float]:
    """Apply count deltas to a rate vector."""
    k = 1000.0 / max(n_words, 1)
    out = dict(v)
    for f, dc in deltas.items():
        if f in out:
            out[f] = max(0.0, out[f] + dc * k)
    return out


def overlaps(a: Candidate, b: Candidate) -> bool:
    return not (a.end <= b.start or b.end <= a.start)


def select(v0: dict[str, float], mu: dict[str, float],
           cands: list[Candidate], min_conf: float = 0.7
           ) -> tuple[list[Candidate], list[tuple[Candidate, float]]]:
    """Greedy with re-evaluation. Production path is Lagrangian relaxation +
    weighted interval scheduling; with this few candidates greedy is adequate
    and the objective is non-linear in log space, so re-evaluation matters."""
    n_words = v0["_words"]
    pool = [c for c in cands if c.conf >= min_conf]
    chosen: list[Candidate] = []
    trace: list[tuple[Candidate, float]] = []
    v = dict(v0)
    cur = distance(v, mu, v0)

    while True:
        best, best_d, best_v = None, cur, None
        for c in pool:
            if any(overlaps(c, x) for x in chosen):
                continue
            cand_v = predict(v, c.delta, n_words)
            d = distance(cand_v, mu, v0)
            if d < best_d - 1e-9:
                best, best_d, best_v = c, d, cand_v
        if best is None:
            break
        chosen.append(best)
        trace.append((best, cur - best_d))
        v, cur = best_v, best_d
    return chosen, trace


def report_vector(v: dict[str, float], mu: dict[str, float] | None = None) -> str:
    rows = []
    for f in [s.name for s in __import__("biber").SPECS]:
        spec = SPEC_BY_NAME[f]
        val = v[f]
        tgt = f"{mu[f]:7.2f}" if (mu and f in mu) else "      -"
        flag = "HOME" if f in HOMING else "    "
        ratio = f"{spec.ratio:4.1f}x" if spec.ratio else "  -  "
        rows.append(f"  {flag} {f:22} {val:7.2f}  target {tgt}  pub {ratio}")
    return "\n".join(rows)
