/**
 * Safe arithmetic expression evaluator for answer inputs.
 * Supports: numbers, + - * / ^, parentheses, unary minus, sqrt(), abs(), exp(), ln()/log() (natural), log10(), log2(),
 * sin/cos/tan/atan/acos/asin (radians), deg(), pi, e, and optional named variables (e.g. x).
 */
const FUNCS: Record<string, (x: number) => number> = {
  sqrt: Math.sqrt, abs: Math.abs, sin: Math.sin, cos: Math.cos, tan: Math.tan,
  asin: Math.asin, acos: Math.acos, atan: Math.atan, arctan: Math.atan, arccos: Math.acos, arcsin: Math.asin,
  ln: Math.log, log: Math.log, log10: Math.log10, log2: Math.log2, exp: Math.exp,
  sigmoid: (x) => 1 / (1 + Math.exp(-x)), deg: (x) => (x * 180) / Math.PI, rad: (x) => (x * Math.PI) / 180,
}
const CONSTS: Record<string, number> = { pi: Math.PI, e: Math.E }

export function evaluate(input: string, vars: Record<string, number> = {}): number {
  const s = input.replace(/\s+/g, '').replace(/,/g, '.').replace(/π/g, 'pi').replace(/√/g, 'sqrt').replace(/−/g, '-')
  if (!s) throw new Error('empty')
  let i = 0
  const peek = () => s[i]
  const eat = (ch: string) => {
    if (s[i] === ch) {
      i++
      return true
    }
    return false
  }
  function expr(): number {
    let v = term()
    for (;;) {
      if (eat('+')) v += term()
      else if (eat('-')) v -= term()
      else return v
    }
  }
  function term(): number {
    let v = factor()
    for (;;) {
      if (eat('*')) v *= factor()
      else if (eat('/')) v /= factor()
      else return v
    }
  }
  function factor(): number {
    if (eat('-')) return -factor()
    if (eat('+')) return factor()
    const base = atom()
    if (eat('^')) return base ** factor()
    return base
  }
  function atom(): number {
    if (eat('(')) {
      const v = expr()
      if (!eat(')')) throw new Error('missing )')
      return v
    }
    const m = /^(\d+\.?\d*|\.\d+)/.exec(s.slice(i))
    if (m) {
      i += m[0].length
      return parseFloat(m[0])
    }
    const w = /^[a-z][a-z0-9]*/.exec(s.slice(i))
    if (w) {
      i += w[0].length
      if (w[0] in vars) return vars[w[0]]
      if (w[0] in CONSTS) return CONSTS[w[0]]
      if (w[0] in FUNCS && peek() === '(') {
        i++
        const v = expr()
        if (!eat(')')) throw new Error('missing )')
        return FUNCS[w[0]](v)
      }
      throw new Error(`unknown: ${w[0]}`)
    }
    throw new Error(`unexpected: ${peek() ?? 'end'}`)
  }
  const v = expr()
  if (i !== s.length) throw new Error(`unexpected: ${s[i]}`)
  if (!Number.isFinite(v)) throw new Error('not finite')
  return v
}

export function tryEvaluate(input: string): number | null {
  try {
    return evaluate(input)
  } catch {
    return null
  }
}

export const closeEnough = (got: number, want: number, tol: number) =>
  Math.abs(got - want) <= tol * Math.max(1, Math.abs(want))
