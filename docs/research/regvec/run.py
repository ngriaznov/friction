import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from biber import extract, SPECS, HOMING
from rewrite import candidates, apply_candidates
from home import target, distance, select, report_vector

src = Path(sys.argv[1] if len(sys.argv) > 1 else "corpus/llm_tech.md").read_text().strip()
v0, doc = extract(src)
mu = target(v0)
d0 = distance(v0, mu, v0)

print("=" * 82)
print(f"BEFORE   {int(v0['_words'])} words   D^2 = {d0:.3f}")
print("=" * 82)
print(report_vector(v0, mu))

cands = candidates(doc)
print(f"\n\nCANDIDATES: {len(cands)}")
for c in cands:
    dl = " ".join(f"{k}{v:+d}" for k, v in c.delta.items())
    print(f"  [{c.kind:24}] conf={c.conf:.2f}  {dl}")
    print(f"      - {c.original[:72]!r}")
    print(f"      + {c.replacement[:72]!r}")

chosen, trace = select(v0, mu, cands)
print(f"\n\nSELECTED {len(chosen)} of {len(cands)}   (greedy, re-evaluated)")
for c, gain in trace:
    print(f"  {c.kind:24} dD^2 = -{gain:.4f}")

out = apply_candidates(src, chosen)
v1, _ = extract(out)
d1 = distance(v1, mu, v0)

print("\n" + "=" * 82)
print(f"AFTER    {int(v1['_words'])} words   D^2 = {d1:.3f}   ({(1-d1/d0)*100:.1f}% closer)")
print("=" * 82)
print(report_vector(v1, mu))

# validation: did predicted deltas match measured?
print("\n\nDELTA VALIDATION (predicted vs actually measured, counts)")
pred = {}
for c in chosen:
    for k, dv in c.delta.items():
        pred[k] = pred.get(k, 0) + dv
k0, k1 = 1000.0 / v0["_words"], 1000.0 / v1["_words"]
for f in sorted(pred):
    actual = round(v1[f] / k1) - round(v0[f] / k0)
    ok = "ok " if actual == pred[f] else "MISMATCH"
    print(f"  {ok} {f:22} predicted {pred[f]:+d}   measured {actual:+d}")

Path("output.md").write_text(out)
print("\n" + "=" * 82 + "\nREWRITTEN TEXT\n" + "=" * 82)
print(out)
