# frontier-2026-09

Measurement-only sample from two current-generation Claude releases,
generated 2026-09-24 through Claude Code subagents (one subagent per
model and genre, default register, no style instructions, temperature
not controllable). `a/` is the Opus-tier release, `b/` the Fable-tier
release. Prompts are `docs-037..042`, `readme-037..042`,
`blog-037..042` and `forum-037..042` from `corpus/prompts/`, which no
manifest document used.

These files are not in `corpus/manifest.jsonl` and feed no pack. They
exist so the rates in `../FRONTIER_MODELS.md` §7 can be re-measured:

```bash
python3 docs/research/frontier-2026-09/measure.py \
    docs/research/frontier-2026-09/patterns.tsv \
    a=docs/research/frontier-2026-09/a b=docs/research/frontier-2026-09/b
```
