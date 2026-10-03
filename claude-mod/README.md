# friction for Claude Code

A Claude Code mod that runs `friction fix` on every reply Claude writes,
before the reply is stored. The fixed text is what the transcript file
keeps, what the model reads back on its next turn, and what the screen
shows. Only the text blocks of a reply change: thinking, tool calls, tool
results, and the files Claude writes are left alone. For files, use
[friction-skill](https://github.com/ngriaznov/friction-skill).

```
> Reply with: It is important to note that the agent leverages the cache
  in order to perform validation of the config file.

● The agent uses the cache to validate the config file.
```

## Load it

```bash
claude --plugin-dir ./claude-mod
```

Where no flag can be given (the desktop app, an SDK host), name the folder
in `CLAUDE_CODE_PLUGIN_DIRS`, either in the environment or in the `env`
block of `~/.claude/settings.json`, as an absolute path.

The mod needs friction itself. It uses `friction` from your `PATH` when
there is one (`npm install -g friction-cli`) and falls back to
`npx -y friction-cli@latest`. The fallback costs about 0.8 s per reply
block against about 30 ms for an installed binary.

## Use it

| command | does |
|---|---|
| `/friction` | the mode, the binary, this session's counts, and a word diff of the last change |
| `/friction fix` | rewrite replies (the default) |
| `/friction check` | count what friction would change and change nothing |
| `/friction off` | stop for this session |

The status line keeps a running count: `friction: 12 edits`.

`/friction` shows the last change as a word diff:

```
last change: 5 edits (edit.recapitalize ×1, pivot.lvc ×1, span.delete ×1, sub.apply ×2)
[-It is important to note that the-]{+The+} agent [-leverages-]{+uses+} the cache [-in order-] to [-perform validation of-]{+validate+} the config file.
```

## Options

Set them in `/config`, or under `pluginConfigs.friction.options` in
`settings.json`.

| option | default | meaning |
|---|---|---|
| `mode` | `fix` | `fix`, `check`, or `off` at the start of each session |
| `command` | empty | the command that runs friction, split on spaces (`npx -y friction-cli@0.6.18` pins a version); empty tries `friction`, then npx |
| `subagents` | `false` | also fix subagents' replies, which only the main model reads |

## Limits

- friction's own limits apply: it is trained on English technical
  documentation. Chat replies cover a wider register than READMEs do.
  friction makes almost no edits to human-written text, but `check` mode
  shows what it would change before you let it.
- A reply that quotes friction's phrases in plain prose gets them
  rewritten. Inside backticks they are safe. Use `/friction off` while
  working on friction's own rules.
- When friction would delete a reply block whole (a closer on its own),
  the block stays as written: an empty text block is no reply.
- If friction fails or is missing, the reply is stored as written, and the
  failure shows in the status line and in `/friction`.

## Tests

```bash
claude plugin test claude-mod
claude plugin validate claude-mod
```

The tests stand in for the friction binary, so they need neither friction
nor a network. The function-hooks API they run against is early access
(written against Claude Code 2.1.288) and may change between releases.
