# How Kilnstore Decides to Reuse a Build Artifact

Every time you run a build, Kilnstore answers one question for each step: *have I already produced this exact output before?* If the answer is yes, it skips the work and restores the stored artifact. If not, it runs the step and saves the result for next time.

This page explains how Kilnstore reaches that answer. It covers the cache key, how changes to dependencies invalidate cached results, and how a team shares one remote cache.

## The core idea: outputs follow from inputs

Kilnstore assumes that build steps are deterministic. Given the same inputs, a step produces the same outputs. When that holds, you don't need to rerun a step if its inputs haven't changed since a previous run. You only need a reliable way to tell whether they have.

Kilnstore doesn't use file timestamps for this. Timestamps change when you check out a branch, copy a directory, or touch a file without editing it, and they mean nothing on another machine. Kilnstore looks at file content instead.

## Content-hash keys

Each build step (Kilnstore calls it an **action**) gets a **cache key**: a cryptographic hash computed from everything that could affect the action's output. The key includes:

- **Source contents.** Kilnstore hashes the bytes of every input file the action declares. File paths are included relative to the workspace root, so the key doesn't depend on where the repository sits on disk.
- **The command.** This is the full command line, including flags and arguments.
- **Environment.** Only the environment variables the action is declared to read are included. Undeclared variables are hidden from the action, so they can't change its output unnoticed.
- **Toolchain identity.** Kilnstore hashes the compiler, interpreter, or tool binary, or a version fingerprint of it, so upgrading a tool produces new keys.
- **Dependency keys.** The keys of every action whose output this action consumes (see below).

Kilnstore combines these into one digest. That digest is the lookup key. If an artifact is stored under it, Kilnstore reuses the artifact. If not, the action runs.

Because the key comes only from content, two developers on different machines who build the same commit with the same toolchain compute identical keys. That property is what makes remote sharing possible.

### Why declared inputs matter

A key can only be as accurate as the list of inputs behind it. If an action reads a file it didn't declare, Kilnstore can't see changes to that file, and it may serve a stale artifact. To prevent this, Kilnstore runs actions in a sandbox that exposes only the declared inputs. An action that tries to read an undeclared file fails loudly. It never quietly produces an output that can't be reproduced.

## Invalidation through the dependency graph

Kilnstore models a build as a directed graph. Each action is a node, and each edge means one action consumes another's output. Invalidation follows this graph.

Say you edit `parser.c`:

1. The key for `compile(parser.c)` changes because one of its source inputs changed.
2. That action has no cached entry under the new key, so it runs and produces a new `parser.o`.
3. `link(app)` includes the key of `compile(parser.c)` among its dependency keys. That dependency key changed, so the link key changes too, and linking runs again.
4. Actions that don't depend on `parser.c`, such as compiling `lexer.c`, keep their keys and hit the cache.

Kilnstore never scans the cache to find entries to delete. Invalidation is a side effect of hashing: when something changes, new keys are computed, and old entries simply stop being looked up. They age out under the eviction policy.

### Early cutoff

Sometimes a changed input produces an identical output. A common case is editing a comment in a source file whose compiled object doesn't change. Kilnstore detects this by hashing each action's **outputs** as well. Downstream keys are built from the output hashes of dependencies, not only their input keys. So if `parser.o` comes out byte-for-byte identical, `link(app)` gets the same key as before and is restored from cache. This "early cutoff" keeps small edits from triggering large rebuilds.

## Remote cache sharing

On its own, Kilnstore keeps a local cache in your user directory. Teams usually add a **remote cache**, a shared store that all developers and CI machines read from and write to.

The lookup order is:

1. **Local cache.** This is the fastest, with no network round trip.
2. **Remote cache.** On a hit, Kilnstore downloads the artifact and also stores it locally.
3. **Execute.** On a miss at both levels, Kilnstore runs the action and uploads the result to the remote cache when permitted.

The payoff is that each unique action runs once for the whole team. If CI builds `main` overnight, a developer who pulls `main` in the morning downloads those outputs rather than rebuilding.

### Trust and write access

A shared cache is only as trustworthy as its writers. If one machine uploads a corrupted or non-reproducible artifact, everyone downstream receives it. The usual setup is:

- **CI writes, developers read.** Only CI agents with a clean, controlled environment can upload. Developer machines are configured as read-only clients.
- **Content verification.** Stored artifacts are addressed by their own content hash, and Kilnstore checks that hash on download, so corruption in transit or at rest is caught.
- **Scoped namespaces.** Separate branches or projects can use separate cache namespaces when they shouldn't share results.

### Keeping keys portable

Remote hits depend on keys matching across machines, so anything machine-specific that leaks into a key reduces the hit rate. Absolute paths, hostnames, timestamps embedded in outputs, and unpinned toolchain versions are the usual causes. Kilnstore's `explain` command compares the key components of two runs and shows which input made them differ. When your remote hit rate is lower than you expect, start there.

## Summary

Kilnstore reuses an artifact when the action's content-derived key matches a stored entry. Changes to any input, whether a source file, flag, tool, or upstream output, produce a new key. Those changes propagate through the dependency graph only as far as they actually change outputs. A shared remote cache extends the same logic across the whole team.
