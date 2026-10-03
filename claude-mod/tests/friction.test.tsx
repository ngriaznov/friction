import { describe, expect, test } from 'claude-code/testing'
import type { Engine } from 'claude-code/testing'
import type { Args, CommandRunResult, On, ProcessRunResult } from 'claude-code'

import { wordDiff } from '../hooks/diff'

type Append = Args<'session.append'>
type Block = Append['message']['content'][number]

const RULES: [RegExp, string, string][] = [
  [/\bin order to\b/g, 'to', 'sub.apply'],
  [/\bleverages\b/g, 'uses', 'sub.apply'],
]

const result = (exitCode: number, stdout: string, stderr = ''): { value: ProcessRunResult } => ({
  value: { exitCode, stdout, stderr, isStdoutTruncated: false, isStderrTruncated: false },
})

// Stands in for the friction binary: a couple of its substitutions, its JSON
// summary on stderr, and (like npx) a notice around it. Trims its output so
// the tests see the mod put the block's own edges back.
function fakeFriction(on: On, { fails = false, missing = false } = {}) {
  const calls: string[][] = []
  on('process.run', async ($, e) => {
    calls.push([...e.argv])
    if (missing) throw new Error(`spawn ${e.argv[0]} ENOENT`)
    if (e.argv.includes('--version')) return result(0, 'friction 0.6.18\n')
    if (fails) return result(2, '', 'error: the input failed to parse as markdown\n')
    let text = e.init?.stdin ?? ''
    const byRule: Record<string, number> = {}
    let patches = 0
    for (const [pattern, to, rule] of RULES) {
      text = text.replace(pattern, () => {
        patches++
        byRule[rule] = (byRule[rule] ?? 0) + 1

        return to
      })
    }
    const summary = { passes: 1, patches_applied: patches, patches_by_rule: byRule, suggest_count: 0, paraphrase_count: 0 }

    return result(0, text.trim(), `npm warn exec notice\n${JSON.stringify(summary, null, 2)}\n`)
  })

  return calls
}

// Nothing in the kit stores a row beneath the plugins, so this records each
// row as the mod hands it on (its own `next` then rejects, which the test
// swallows) and answers the status line.
function harness(on: On) {
  const stored: Block[][] = []
  const status: (string | undefined)[] = []
  on('session.append', ($, e, next) => {
    stored.push([...e.message.content])

    return next(e)
  })
  on('ui.status', ($, e) => {
    status.push(e.text)

    return { value: undefined }
  })

  return { stored, status }
}

const reply = (text: string, extra: Partial<Append> = {}): Append => ({
  message: {
    type: 'assistant',
    role: 'assistant',
    content: [
      { type: 'text', text },
      { type: 'tool_use', id: 'toolu_1', name: 'Bash', input: { command: 'ls' } },
    ],
  },
  door: 'response',
  origin: { kind: 'model', model: 'claude-test' },
  uuid: 'row-1',
  ...extra,
})

async function append($: Engine, stored: Block[][], input: Append): Promise<Block[]> {
  const before = stored.length
  await $.session.append(input).catch(() => undefined)
  const row = stored[before]
  if (row === undefined) throw new Error('the row was never handed on')

  return row
}

const friction = ($: Engine, args = ''): Promise<CommandRunResult> =>
  $.command.run({
    command: 'friction',
    args,
    origin: { kind: 'composer' },
    presentation: { isFullscreen: false, columns: 120 },
  })

describe('fix mode', () => {
  test('rewrites the reply text and leaves the tool call alone', async ($, on) => {
    fakeFriction(on)
    const { stored, status } = harness(on)

    const content = await append($, stored, reply('The agent leverages the cache in order to go fast.'))

    expect(content).toEqual([
      { type: 'text', text: 'The agent uses the cache to go fast.' },
      { type: 'tool_use', id: 'toolu_1', name: 'Bash', input: { command: 'ls' } },
    ])
    expect(status.at(-1)).toBe('friction: 2 edits')
  })

  test("keeps the block's leading and trailing whitespace", async ($, on) => {
    fakeFriction(on)
    const { stored } = harness(on)

    const content = await append($, stored, reply('\nThe agent leverages the cache.\n\n'))

    expect(content[0]).toEqual({ type: 'text', text: '\nThe agent uses the cache.\n\n' })
  })

  test('finds friction once, then only fixes', async ($, on) => {
    const calls = fakeFriction(on)
    const { stored } = harness(on)

    await append($, stored, reply('It leverages one.'))
    await append($, stored, reply('It leverages two.', { uuid: 'row-2' }))

    expect(calls).toEqual([
      ['friction', '--version'],
      ['friction', 'fix', '--format', 'json', '-'],
      ['friction', 'fix', '--format', 'json', '-'],
    ])
  })

  test('runs the configured command', { options: { command: 'npx -y friction-cli@0.6.18' } }, async ($, on) => {
    const calls = fakeFriction(on)
    const { stored } = harness(on)

    await append($, stored, reply('It leverages one.'))

    expect(calls[1]).toEqual(['npx', '-y', 'friction-cli@0.6.18', 'fix', '--format', 'json', '-'])
  })

  test('/friction shows what the last reply lost', async ($, on) => {
    fakeFriction(on)
    const { stored } = harness(on)
    await append($, stored, reply('The agent leverages the cache in order to go fast.'))

    const { text } = await friction($)

    expect(text).toContain('friction: fix')
    expect(text).toContain('runs: friction (friction 0.6.18)')
    expect(text).toContain('replies: 1 read, 1 changed, 2 edits')
    expect(text).toContain('last change: 2 edits (sub.apply ×2)')
    expect(text).toContain('The agent [-leverages-]{+uses+} the cache [-in order-] to go fast.')
  })

  test('draws the fixed text where the screen still holds the original', async ($, on) => {
    fakeFriction(on)
    const { stored } = harness(on)
    on('ui.render', { component: 'AssistantMessage' }, ($, e) => {
      const { Text } = $.ui.resolve(e)

      return <Text>{e.props.text}</Text>
    })
    await append($, stored, reply('The agent leverages the cache.'))

    for (const surface of ['terminal', 'desktop'] as const) {
      const ui = await $.ui.mount({
        plugin: 'friction',
        surface,
        component: 'AssistantMessage',
        props: { text: 'The agent leverages the cache.', isFirstOfReply: true },
      })
      expect((await ui.find({ type: 'Text' }))?.text).toBe('The agent uses the cache.')
      await ui.unmount()
    }
  })
})

describe('leaving replies as written', () => {
  test('check mode counts the edits and changes nothing', { options: { mode: 'check' } }, async ($, on) => {
    fakeFriction(on)
    const { stored } = harness(on)

    const content = await append($, stored, reply('It leverages the cache.'))
    const { text } = await friction($)

    expect(content[0]).toEqual({ type: 'text', text: 'It leverages the cache.' })
    expect(text).toContain('replies: 1 read, 1 would change, 1 edits')
    expect(text).toContain('last possible change')
  })

  test('/friction off stops it for the session', async ($, on) => {
    const calls = fakeFriction(on)
    const { stored } = harness(on)

    const { text } = await friction($, 'off')
    const content = await append($, stored, reply('It leverages the cache.'))

    expect(text).toBe('friction is off for this session.')
    expect(content[0]).toEqual({ type: 'text', text: 'It leverages the cache.' })
    expect(calls).toEqual([])
  })

  test("subagents' replies pass through unless asked for", async ($, on) => {
    const calls = fakeFriction(on)
    const { stored } = harness(on)

    const content = await append($, stored, reply('It leverages the cache.', { agentId: 'agent-1' }))

    expect(content[0]).toEqual({ type: 'text', text: 'It leverages the cache.' })
    expect(calls).toEqual([])
  })

  test('a failing run keeps the reply and is counted', async ($, on) => {
    fakeFriction(on, { fails: true })
    const { stored } = harness(on)

    const content = await append($, stored, reply('It leverages the cache.'))
    const { text } = await friction($)

    expect(content[0]).toEqual({ type: 'text', text: 'It leverages the cache.' })
    expect(text).toContain('1 failed')
  })

  test('with no friction installed the reply passes through', async ($, on) => {
    fakeFriction(on, { missing: true })
    const { stored } = harness(on)

    const content = await append($, stored, reply('It leverages the cache.'))
    const { text } = await friction($)

    expect(content[0]).toEqual({ type: 'text', text: 'It leverages the cache.' })
    expect(text).toContain('runs: nothing found')
  })
})

describe('wordDiff', () => {
  test('marks the changed words of the changed lines only', () => {
    expect(wordDiff('a line kept\nWe leverage it in order to win.\n', 'a line kept\nWe use it to win.\n')).toBe(
      'We [-leverage-]{+use+} it [-in order-] to win.',
    )
  })
})
