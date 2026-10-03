export type FrictionMode = 'fix' | 'check' | 'off'

/** One reply block friction changed (or, in check mode, would have). */
export type FrictionRun = {
  /** The stored row's id. */
  uuid: string
  before: string
  after: string
  /** `patches_applied` from friction's JSON summary. */
  patches: number
  /** `patches_by_rule`: how many edits each rule made. */
  byRule: Record<string, number>
  /** `suggest_count`: candidates a gate held back, left for a human. */
  held: number
  /** True when the fixed text was stored; false in check mode. */
  isApplied: boolean
}

export type FrictionTotals = {
  /** Reply blocks friction read. */
  blocks: number
  /** Reply blocks it changed. */
  changed: number
  patches: number
  /** Runs that failed and were passed through untouched. */
  failures: number
}

declare module 'claude-code' {
  interface PluginState {
    friction: {
      /** This session's override of the `mode` option; unset follows it. */
      mode: FrictionMode | null
      runs: FrictionRun[]
      totals: FrictionTotals
    }
  }
}
