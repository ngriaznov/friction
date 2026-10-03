const TOKEN_LIMIT = 3_000

// A lone deletion or insertion shows its edge whitespace outside the markers,
// as git does: `it [-in order-] to`, not `it[- in order-] to`.
function outside(chunk: string, open: string, close: string): string {
  const [, lead = '', body = '', trail = ''] = chunk.match(/^(\s*)([\s\S]*?)(\s*)$/) ?? []

  return body ? `${lead}${open}${body}${close}${trail}` : `${open}${chunk}${close}`
}

// A git --word-diff style rendering of the lines a run changed:
// `the agent [-leverages-]{+uses+} the cache`.
export function wordDiff(before: string, after: string): string {
  const a = before.match(/\s+|[^\s]+/g) ?? []
  const b = after.match(/\s+|[^\s]+/g) ?? []
  let head = 0
  while (head < a.length && head < b.length && a[head] === b[head]) head++
  let tail = 0
  while (tail < a.length - head && tail < b.length - head && a[a.length - 1 - tail] === b[b.length - 1 - tail]) {
    tail++
  }
  const x = a.slice(head, a.length - tail)
  const y = b.slice(head, b.length - tail)
  if (x.length > TOKEN_LIMIT || y.length > TOKEN_LIMIT) return '(change too large to show)'

  // Longest common subsequence over the changed middle.
  const width = y.length + 1
  const lcs = new Uint16Array((x.length + 1) * width)
  for (let i = x.length - 1; i >= 0; i--) {
    for (let j = y.length - 1; j >= 0; j--) {
      lcs[i * width + j] =
        x[i] === y[j]
          ? (lcs[(i + 1) * width + j + 1] ?? 0) + 1
          : Math.max(lcs[(i + 1) * width + j] ?? 0, lcs[i * width + j + 1] ?? 0)
    }
  }

  let middle = ''
  let removed = ''
  let added = ''
  const flush = () => {
    if (removed && added) middle += `[-${removed}-]{+${added}+}`
    else if (removed) middle += outside(removed, '[-', '-]')
    else if (added) middle += outside(added, '{+', '+}')
    removed = ''
    added = ''
  }
  let i = 0
  let j = 0
  while (i < x.length || j < y.length) {
    if (i < x.length && j < y.length && x[i] === y[j]) {
      flush()
      middle += x[i]
      i++
      j++
    } else if (j >= y.length || (i < x.length && (lcs[(i + 1) * width + j] ?? 0) >= (lcs[i * width + j + 1] ?? 0))) {
      removed += x[i]
      i++
    } else {
      added += y[j]
      j++
    }
  }
  flush()

  const merged = a.slice(0, head).join('') + middle + a.slice(a.length - tail).join('')

  return merged
    .split('\n')
    .filter(line => line.includes('[-') || line.includes('{+'))
    .join('\n')
}
