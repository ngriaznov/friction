# Salience: deterministic per-line importance over function bodies

A prototype that answers one question for every line of every function:
**does this line carry behavior?**

The answer is a tier and a score, computed from dominance, loop structure and
def-use reachability over an IR. No model runs. Nothing is removed. Every span
carries the reason that produced it.

```
$ salience OrderProcessor.class --format text --annotate OrderProcessor.java

inert   11 |         LOG.info("processing " + prices.size() + " prices");
plumb   13 |         String unused = "this value goes nowhere";
inert   14 |         int inspected = 0;
CORE    16 |         double subtotal = 0.0;
CORE    17 |         for (Double price : prices) {
CORE    18 |             if (price == null || price < 0) {
CORE    21 |             subtotal += price;
inert   22 |             inspected++;
CORE    26 |         if (applyTax) {
CORE    27 |             total = subtotal * (1.0 + taxRate);
inert   30 |         LOG.fine("inspected " + inspected + " entries");
BOUND   32 |         this.runningTotal = total;
BOUND   33 |         return total;
```

Lines 21 and 22 are the point. Both are loop-carried accumulators. Both look
identical. `subtotal` reaches a field write, so it is core; `inspected` reaches
nothing but a log call, so it is inert. No syntactic analysis separates these —
only dependence does.

## Tiers

| tier | meaning |
|---|---|
| `core` | behavior-carrying: branch predicates weighted by what they control-dominate, loop-carried dataflow, statements on def-use chains reaching an effect |
| `boundary` | the frontier where behavior leaves the body: returns, throws, state writes, calls into opaque dependencies |
| `plumbing` | present but not behavior-carrying: local shuffling, results that reach no effect |
| `inert` | denylisted calls (logging, metrics, tracing) and the computation that exists only to feed them |

Alongside the tier, every span carries a score in `0.0..=1.0`. The **tier is for
policy** — an edit-gating hook wants a predicate. The **score is for ranking** —
a weighted call graph, a profiler choosing where to start, or a vulnerability
triage queue wants an ordering, and four buckets throw away the gradient between
a predicate guarding two lines and one guarding forty.

## What it is not

- **Not compression.** No tokens are removed. The artifact is metadata *about*
  source; a consumer that ignores it sees the file unchanged.
- **Not learned.** No inference at build time or query time.
- **Not criterion-anchored.** Classic slicing answers "what affects *this*".
  This answers "what carries behavior at all", unconditionally — so it is
  computed once and cached rather than recomputed per question.
- **Not repo-level ranking.** The unit is the statement inside one body.

## Architecture

```
salience-core   language-neutral. Dominance, post-dominance, control dependence,
                natural loops, reaching definitions, tiering, scoring, projection.
                Knows nothing about any language.
      ^
      |  FunctionIr  (the contract: line, defs, uses, successors, kind)
      |
salience-jvm    .class -> mokapot MokaIR -> FunctionIr
salience-py     .py -> CPython dis -> JSON -> FunctionIr
salience-cli    the `salience` binary
```

The seam is `FunctionIr`. A frontend answers four questions per instruction —
what line, what does it define, what does it use, where can control go — and
gets everything else.

Two properties make that seam hold:

- **The graph is instruction-level, not block-level.** Frontends never have to
  discover basic blocks.
- **Definitions need not be in SSA form.** The core computes reaching
  definitions itself, so the Python frontend over mutable `STORE_FAST` locals is
  exactly as sound as the JVM frontend over an already-SSA IR.

## Why IR and not ASTs

For Java the two agree closely enough that it barely matters. For Kotlin they do
not: a `suspend` function's real control flow is a compiler-generated state
machine, an `inline` function's body is physically copied into each call site,
and a `when` becomes a `tableswitch` or a comparison chain depending on what it
matches. An AST shows the syntax someone wrote; the bytecode shows the control
flow that runs. The `LineNumberTable` maps it back to the lines they will edit.

The same argument holds more weakly for Python — comprehensions are separate
code objects, `and`/`or` are jumps, a `for` loop's real exit test is `FOR_ITER`.

## Adding a language

Implement one function: substrate → `Vec<FunctionIr>`. What a substrate needs:

| language | substrate | line fidelity | status |
|---|---|---|---|
| Java | JVM bytecode via `mokapot` | `LineNumberTable`, needs `javac -g` | **working** |
| Kotlin | same | needs SMAP/JSR-45 handling for `inline` | frontend works; inline attribution unverified |
| Python | CPython bytecode via `dis` | PEP 626 `co_lines()`, exact | **working** |
| Rust | MIR via `rustc_public` | MIR spans | blocked: nightly-only |
| C/C++/Swift | LLVM IR `DILocation` | debug info | not attempted |
| JS/TS | no standard IR; Google's JSIR is the exception | source maps | weak |

## Usage

```bash
salience Foo.class                          # JSON sidecar on stdout
salience foo.py --format text               # one line per span
salience Foo.class --annotate Foo.java      # tiered source view
salience Foo.class --stats                  # histogram and timing
salience Foo.class --inert 'com.acme.Audit' # extend the denylist
salience Foo.class --no-denylist            # treat nothing as inert
```

The artifact:

```json
{
  "schema": "salience-sidecar/v1",
  "generator": "salience-jvm/mokapot",
  "file": "OrderProcessor.java",
  "functions": [{
    "name": "OrderProcessor::process",
    "signature": "(Ljava/util/List;, D, Z) -> double",
    "decl_line": 11,
    "coverage": { "instructions": 55, "with_line": 55 },
    "summary": { "core": 6, "boundary": 4, "plumbing": 1, "inert": 4 },
    "spans": [
      { "start": 21, "end": 21, "tier": "core", "score": 0.8,
        "reasons": ["loop-carried definition at nesting depth 1",
                    "reaches state write OrderProcessor#runningTotal in 1 dependence step(s)"] },
      { "start": 22, "end": 22, "tier": "inert", "score": 0.0,
        "reasons": ["builds arguments for a denylisted call only"] }
    ]
  }]
}
```

`coverage` is the honesty field: when a substrate loses line attribution, a
consumer needs to know it is looking at an incomplete map rather than a body
that genuinely has no core.

## Performance

Release build, 3-method class, 62 instructions:

```
lowering    546µs   (file read + class parse + MokaIR lift)
analysis    202µs   -> 67µs per function
```

Lowering and analysis are reported apart because they are paid at different
times. Lowering happens once per file. Analysis is the part that would run
inside an editor hook, and the part the caching story is about.

Output is byte-identical across runs — every set is a `BTreeSet`, every map a
`BTreeMap`, every worklist drains in index order. That is what makes the
artifact cacheable and diffable.

## The one deliberate soundness trade

The inert rule absorbs opaque calls. A statement is inert when it feeds a
denylisted call *and* reaches no hard effect (return, throw, state write).

`prices.size()` inside `LOG.info("processing " + prices.size())` is an opaque
call, and in principle an opaque call could have side effects — so demoting it
because its result only feeds a log is not conservative. But a denylist that
stops at the logging call and leaves every argument expression tiered as
behavior does not do the job it exists to do. Returns, throws and state writes
are observable without seeing inside any callee, and are never absorbed, so the
trade is bounded to the frontier we already declined to cross.

The rule is backward reachability, not an "every consumer is inert" fixpoint,
because the most valuable case is a dependency *cycle*: a counter incremented
only to be logged reads its own previous value, and a least-fixpoint never
enters that loop.

## Known limitations

- **Kotlin `inline` line attribution is unverified.** The frontend reports
  whether a class carries `SourceDebugExtension` (JSR-45/SMAP) but does not
  resolve it, so lines from an inlined body may point at the wrong file. This is
  the largest open correctness question.
- **SSA renames can vanish.** `double total = subtotal;` compiles to a pure
  rename that MokaIR erases, so that line gets no span.
- **`mokapot`'s MokaIR is behind an `unstable-moka-ir` feature** with no
  stability promise across 0.x. Pinned to `=0.26.0`.
- **Python's stack simulation is exact for ~50 opcodes** and estimates pops from
  `dis.stack_effect` for the rest.
- **Intraprocedural only**, by design. Call graphs and cross-function flow are a
  different problem.
- **No column granularity.** `co_positions()` offers it for Python; the artifact
  records lines.

## Tests

26 tests: 17 in the core over hand-built IR (pinning the algorithm rather than
any frontend's lowering), 4 over real `javac -g` output, 5 over real CPython
bytecode. The JVM and Python suites assert the *same* behavioral claims against
equivalent source, which is the multi-language claim stated as a test.

```bash
cargo test -p salience-core -p salience-jvm -p salience-py
```

JVM and Python tests skip rather than fail when no JDK or interpreter is
present.
