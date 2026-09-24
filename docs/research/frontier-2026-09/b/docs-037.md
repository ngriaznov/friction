# How Kilnstore Decides Whether to Reuse a Cached Build

Kilnstore sits between your build tool and the work it is about to do. Before any compile, test, or bundle step runs, Kilnstore asks a single question: *has this exact piece of work already been done, and can the result be trusted?* If the answer is yes, it restores the stored output instead of running the step. If not, the step runs and its output is recorded for next time.

This page explains the three ideas that make that decision reliable: content-hash keys, invalidation when dependencies change, and sharing a cache across a team.

## Content-hash keys

Every cacheable step in Kilnstore is identified by a **cache key**. The key is not a name or a version number; it is a cryptographic hash of everything that can influence the step's output.

For a typical compile step, the inputs folded into the key are:

- The contents of every source file the step reads (not their paths or modification times).
- The contents of configuration files that affect the step, such as compiler flags or a `tsconfig`.
- The command line, including arguments and any environment variables the step declares as relevant.
- The identity of the tool that runs the step, usually the hash of its binary or a pinned version string.
- The cache keys of any upstream steps whose outputs this step consumes (more on this below).

Kilnstore hashes each input individually, then combines those hashes in a deterministic order into one key. Two machines with byte-identical inputs will always produce the same key, no matter when or where the build runs. That property is what makes the cache safe to share.

Hashing contents rather than timestamps matters. A file that is touched but not changed produces the same hash, so a `git checkout` that rewrites modification times does not throw away a valid cache. Conversely, a one-character change to a file changes its hash, and therefore the key, so stale output is never mistaken for fresh output.

## Cache invalidation on dependency changes

A cache entry never expires because time has passed. It becomes unreachable because its key no longer matches the current inputs. This is the core of Kilnstore's invalidation model: **there is no separate invalidation step.** A changed input simply computes to a different key, and a lookup for that key misses.

The interesting question is how a change *propagates*. Suppose step C depends on the output of step B, which depends on step A. If you edit a source file that only A reads:

1. A's key changes, because one of its file hashes changed.
2. B's key includes A's key as an input, so B's key changes too, even though none of B's own source files were touched.
3. C's key includes B's key, so C's key changes as well.

All three steps miss the cache and run again. Steps elsewhere in the graph that do not depend on A keep their keys and are restored as before.

This chained approach means Kilnstore does not need to understand what your files *mean*. It does not parse imports or track which symbols are used. It only needs the dependency graph between steps, which your build configuration already declares. The trade-off is that invalidation is conservative: if B depends on A, any change to A reruns B, even if the change was in a comment that could not possibly affect B. For most projects this is the right default, because it is impossible to get wrong.

Third-party dependencies are handled the same way. A lockfile is an input to the steps that install packages, and the installed tree is an output that downstream steps consume. Bumping a dependency version changes the lockfile hash, which invalidates the install step, which cascades to everything that uses those packages.

## Remote cache sharing across a team

A local cache saves time when you rebuild the same project on the same machine. A **remote cache** extends that to everyone who works on the project, plus CI.

Kilnstore's remote cache is a content-addressed store: a bucket or server where entries are stored under their cache key. When a step's key is computed, Kilnstore checks in this order:

1. The local cache directory on disk.
2. The configured remote cache, if the local lookup misses.
3. Run the step, then write the output to both caches.

Because keys are derived purely from content, a colleague's build output is just as valid as your own. If they built the `main` branch this morning and you check it out this afternoon, most steps hit the remote cache and your first build finishes in seconds rather than minutes. CI benefits in the same way: a pull-request build only reruns the steps affected by the diff.

Two safeguards keep shared caches trustworthy:

- **Hermetic inputs.** A step that reads something not declared in its key (an unlisted environment variable, the system clock, a file outside the workspace) can produce output that is wrong for other machines. Kilnstore sandboxes steps where the platform allows it and flags steps that read undeclared inputs.
- **Write permissions.** By default only CI is allowed to write to the remote cache; developer machines read from it. This prevents a misconfigured local environment from publishing a bad artifact under a key everyone else will match.

Together, these three mechanisms let Kilnstore answer its one question quickly and correctly: the key says whether the work is the same, chained keys say whether anything upstream changed, and the remote store means someone on the team has probably already done it.
