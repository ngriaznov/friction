import { atom, read, update } from 'claude-code'
import type { Args, EngineInterface, Register } from 'claude-code'

import type { FrictionMode, FrictionRun, FrictionTotals } from '../types'
import { wordDiff } from './diff'

type Append = Args<'session.append'>
type Block = Append['message']['content'][number]
type TextBlock = Block & { type: 'text'; text: string }

type Summary = {
  patches_applied?: number
  patches_by_rule?: Record<string, number>
  suggest_count?: number
}

type Outcome = { text: string; summary: Summary } | { error: string }

const MODES: readonly FrictionMode[] = ['fix', 'check', 'off']
const NPX = ['npx', '-y', 'friction-cli@latest']
const RUNS_KEPT = 30
const FIX_TIMEOUT_MS = 20_000
// The first npx probe may download the package (~27 MB).
const PROBE_TIMEOUT_MS = 180_000

const mode = atom({ plugin: 'friction', key: 'mode' } as const, null)
const runs = atom({ plugin: 'friction', key: 'runs' } as const, [])
const totals = atom({ plugin: 'friction', key: 'totals' } as const, {
  blocks: 0,
  changed: 0,
  patches: 0,
  failures: 0,
} satisfies FrictionTotals)

const isMode = (value: unknown): value is FrictionMode =>
  typeof value === 'string' && (MODES as readonly string[]).includes(value)

// Blocks with no words in them (a lone code fence, a bare path) are not prose.
const isProse = (block: Block): block is TextBlock =>
  block.type === 'text' && typeof block.text === 'string' && /[A-Za-z]{2}/.test(block.text)

const firstLine = (text: string) => text.trim().split('\n')[0]?.slice(0, 120) ?? ''

// friction prints its JSON summary on stderr; npx may print notices around it.
function parseSummary(stderr: string): Summary {
  const start = stderr.indexOf('{')
  const end = stderr.lastIndexOf('}')
  if (start < 0 || end < start) return {}
  try {
    return JSON.parse(stderr.slice(start, end + 1)) as Summary
  } catch {
    return {}
  }
}

// The fixed text keeps the original block's leading and trailing whitespace.
function keepEdges(before: string, after: string): string {
  const lead = before.match(/^\s*/)?.[0] ?? ''
  const trail = before.match(/\s*$/)?.[0] ?? ''
  return lead + after.trim() + trail
}

async function runFriction($: EngineInterface, argv: readonly string[], text: string): Promise<Outcome> {
  try {
    const ran = await $.process.run([...argv, 'fix', '--format', 'json', '-'], {
      stdin: text,
      timeoutMs: FIX_TIMEOUT_MS,
    })
    if (ran.exitCode !== 0) return { error: `exit ${ran.exitCode}: ${firstLine(ran.stderr)}` }
    if (ran.isStdoutTruncated) return { error: 'output truncated' }

    return { text: ran.stdout, summary: parseSummary(ran.stderr) }
  } catch (error) {
    return { error: error instanceof Error ? error.message : String(error) }
  }
}

type Found = { argv: string[]; version: string }

// Which argv runs friction, found once per load: `null` when none does.
let resolving: Promise<Found | null> | undefined

async function probe($: EngineInterface, argv: string[]): Promise<Found | null> {
  try {
    const ran = await $.process.run([...argv, '--version'], { timeoutMs: PROBE_TIMEOUT_MS })
    const version = ran.stdout.trim()

    return ran.exitCode === 0 && version.startsWith('friction') ? { argv, version } : null
  } catch {
    return null
  }
}

async function findFriction($: EngineInterface, configured: string): Promise<Found | null> {
  if (configured) return probe($, configured.split(/\s+/))

  return (await probe($, ['friction'])) ?? (await probe($, NPX))
}

function resolveCommand($: EngineInterface, configured: string): Promise<Found | null> {
  resolving ??= findFriction($, configured)

  return resolving
}

async function currentMode($: EngineInterface, fallback: FrictionMode): Promise<FrictionMode> {
  return (await read($, mode)) ?? fallback
}

async function showStatus($: EngineInterface, fallback: FrictionMode): Promise<void> {
  const now = await currentMode($, fallback)
  const sum = await read($, totals)
  if (now === 'off') return $.ui.status(undefined)
  const edits = `${sum.patches} edit${sum.patches === 1 ? '' : 's'}`
  const failed = sum.failures > 0 ? ` · ${sum.failures} failed` : ''
  $.ui.status(now === 'fix' ? `friction: ${edits}${failed}` : `friction (check): ${edits} possible${failed}`)
}

async function report($: EngineInterface, configured: string, fallback: FrictionMode): Promise<string> {
  let found = await resolveCommand($, configured)
  if (found === null) {
    // Installed since the last look? Probe again.
    resolving = undefined
    found = await resolveCommand($, configured)
  }
  const now = await currentMode($, fallback)
  const sum = await read($, totals)
  const last = (await read($, runs)).at(-1)
  const lines = [
    `friction: ${now}${now === fallback ? '' : ` (this session; the option says ${fallback})`}`,
    found
      ? `runs: ${found.argv.join(' ')} (${found.version})`
      : 'runs: nothing found. Install it with `npm install -g friction-cli`, or set the command option.',
    `replies: ${sum.blocks} read, ${sum.changed} ${now === 'check' ? 'would change' : 'changed'}, ${sum.patches} edits` +
      (sum.failures > 0 ? `, ${sum.failures} failed` : ''),
  ]
  if (last) {
    const rules = Object.entries(last.byRule)
      .map(([rule, n]) => `${rule} ×${n}`)
      .join(', ')
    lines.push(
      '',
      `last ${last.isApplied ? 'change' : 'possible change'}: ${last.patches} edits${rules ? ` (${rules})` : ''}` +
        (last.held > 0 ? `, ${last.held} held` : ''),
      wordDiff(last.before, last.after),
    )
  }

  return lines.join('\n')
}

async function fixReply(
  $: EngineInterface,
  e: Append,
  now: 'fix' | 'check',
  argv: readonly string[],
): Promise<{ content: Block[]; made: FrictionRun[]; scanned: number; failures: number }> {
  const made: FrictionRun[] = []
  let scanned = 0
  let failures = 0
  const content = await Promise.all(
    e.message.content.map(async block => {
      if (!isProse(block)) return block
      scanned++
      const out = await runFriction($, argv, block.text)
      if ('error' in out) {
        failures++

        return block
      }
      const after = keepEdges(block.text, out.text)
      // A block friction would delete whole stays: an empty text block is no reply.
      if (after === block.text || after.trim() === '') return block
      made.push({
        uuid: e.uuid,
        before: block.text,
        after,
        patches: out.summary.patches_applied ?? 0,
        byRule: out.summary.patches_by_rule ?? {},
        held: out.summary.suggest_count ?? 0,
        isApplied: now === 'fix',
      })

      return now === 'fix' ? { ...block, text: after } : block
    }),
  )

  return { content, made, scanned, failures }
}

async function record($: EngineInterface, made: FrictionRun[], scanned: number, failures: number): Promise<void> {
  await update($, totals, sum => ({
    blocks: sum.blocks + scanned,
    changed: sum.changed + made.length,
    patches: sum.patches + made.reduce((n, run) => n + run.patches, 0),
    failures: sum.failures + failures,
  }))
  if (made.length > 0) await update($, runs, list => [...list, ...made].slice(-RUNS_KEPT))
}

type Settings = { configured: string; optionMode: FrictionMode; withSubagents: boolean }

// The reply row with friction's fixes in it, or `undefined` to store it as written.
async function rewrite($: EngineInterface, e: Append, settings: Settings): Promise<Append | undefined> {
  if (e.agentId !== undefined && !settings.withSubagents) return undefined
  const now = await currentMode($, settings.optionMode)
  if (now === 'off' || !e.message.content.some(isProse)) return undefined

  const found = await resolveCommand($, settings.configured)
  if (found === null) {
    $.ui.status('friction: not found, see /friction')

    return undefined
  }

  const { content, made, scanned, failures } = await fixReply($, e, now, found.argv)
  await record($, made, scanned, failures)
  if (made.length > 0 || failures > 0) await showStatus($, settings.optionMode)

  return now === 'fix' && made.length > 0 ? { ...e, message: { ...e.message, content } } : undefined
}

const SWITCHED: Record<FrictionMode, string> = {
  fix: "friction now rewrites Claude's replies.",
  check: "friction now only counts what it would change in Claude's replies.",
  off: 'friction is off for this session.',
}

export const register: Register = (on, options) => {
  const settings: Settings = {
    configured: typeof options.command === 'string' ? options.command.trim() : '',
    optionMode: isMode(options.mode) ? options.mode : 'fix',
    withSubagents: options.subagents === true,
  }
  const { configured, optionMode } = settings
  resolving = undefined

  on('session.start', async ($, e, next) => {
    await $.command.register({
      name: 'friction',
      description: "Show what friction changed in Claude's replies, or switch it: /friction [fix|check|off]",
    })

    return next(e)
  })

  // Friction's work never stands between the reply and its storing: a run that
  // throws leaves the row as the model wrote it.
  on('session.append', { door: 'response' }, async ($, e, next) =>
    next((await rewrite($, e, settings).catch(() => undefined)) ?? e),
  )

  // The stored row carries the fixed text; this keeps the screen on it too,
  // where the transcript drew the reply as it streamed in.
  on('ui.render', { component: 'AssistantMessage' }, async ($, e, next) => {
    const shown = e.props.text.trim()
    const hit = (await read($, runs)).findLast(run => run.isApplied && run.before.trim() === shown)

    return hit ? next({ ...e, props: { ...e.props, text: hit.after.trim() } }) : next(e)
  })

  on('command.run', { command: 'friction' }, async ($, e) => {
    const arg = e.args.trim().toLowerCase()
    if (isMode(arg)) {
      await update($, mode, () => arg)
      await showStatus($, optionMode)

      return { text: SWITCHED[arg] }
    }
    if (arg !== '') return { text: 'Usage: /friction [fix|check|off]' }

    return { text: await report($, configured, optionMode) }
  })
}
