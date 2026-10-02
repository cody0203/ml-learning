import type { ConstraintSpec } from '../content/schema'
import {
  approx, asVector, classifySystem, det, dot, EPS, isParallel, isReducedRowEchelon, isRowEchelon, matVec, norm, rank,
  type Mat,
} from './linalg'

export interface CheckResult {
  ok: boolean
  /** Short machine-readable reason, shown next to the constraint. */
  detail?: string
}

const fmt = (x: number) => (Number.isInteger(x) ? String(x) : x.toFixed(3))

export function statOf(kind: 'mean' | 'median' | 'variance' | 'std' | 'sum' | 'range', v: number[], sample = false): number {
  const n = v.length
  const sum = v.reduce((a, b) => a + b, 0)
  const mean = sum / n
  switch (kind) {
    case 'sum':
      return sum
    case 'mean':
      return mean
    case 'range':
      return Math.max(...v) - Math.min(...v)
    case 'median': {
      const s = [...v].sort((a, b) => a - b)
      return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2
    }
    case 'variance':
    case 'std': {
      if (sample && n < 2) throw new Error('need ≥ 2 values')
      const ss = v.reduce((a, x) => a + (x - mean) ** 2, 0) / (sample ? n - 1 : n)
      return kind === 'std' ? Math.sqrt(ss) : ss
    }
  }
}

function needVector(M: Mat) {
  const v = asVector(M)
  if (!v) throw new Error('expected a vector')
  return v
}

function needSquare(M: Mat) {
  if (M.some((r) => r.length !== M.length)) throw new Error('expected a square matrix')
}

export function checkConstraint(c: ConstraintSpec, M: Mat): CheckResult {
  try {
    switch (c.kind) {
      case 'det': {
        needSquare(M)
        const d = det(M)
        return { ok: approx(d, c.value), detail: `det = ${fmt(d)}` }
      }
      case 'singular':
      case 'nonsingular': {
        needSquare(M)
        const d = det(M)
        return { ok: c.kind === 'singular' ? Math.abs(d) < EPS : Math.abs(d) >= EPS, detail: `det = ${fmt(d)}` }
      }
      case 'rank': {
        const r = rank(M)
        return { ok: r === c.value, detail: `rank = ${r}` }
      }
      case 'rowEchelon':
        return { ok: isRowEchelon(M) }
      case 'reducedRowEchelon':
        return { ok: isReducedRowEchelon(M) }
      case 'noZeroEntries':
        return { ok: M.every((r) => r.every((x) => Math.abs(x) > EPS)) }
      case 'integerEntries':
        return { ok: M.every((r) => r.every((x) => Number.isInteger(x))) }
      case 'nonzero':
        return { ok: M.some((r) => r.some((x) => Math.abs(x) > EPS)) }
      case 'notDiagonal':
        return { ok: M.some((r, i) => r.some((x, j) => i !== j && Math.abs(x) > EPS)) }
      case 'entryRange':
        return { ok: M.every((r) => r.every((x) => x >= c.min && x <= c.max)) }
      case 'dot': {
        const d = dot(needVector(M), c.with)
        return { ok: approx(d, c.value), detail: `u·v = ${fmt(d)}` }
      }
      case 'norm': {
        const n = norm(needVector(M), c.p)
        return { ok: approx(n, c.value), detail: `‖u‖ = ${fmt(n)}` }
      }
      case 'solves': {
        const Ax = matVec(c.A, needVector(M))
        return { ok: Ax.every((x, i) => approx(x, c.b[i])), detail: `Ax = (${Ax.map(fmt).join(', ')})` }
      }
      case 'mapsTo': {
        const Ax = matVec(M, c.x)
        return { ok: Ax.every((x, i) => approx(x, c.b[i])), detail: `Ax = (${Ax.map(fmt).join(', ')})` }
      }
      case 'parallelTo':
      case 'notParallelTo': {
        const v = needVector(M)
        const p = isParallel(v, c.of)
        return { ok: c.kind === 'parallelTo' ? p && norm(v) > EPS : !p }
      }
      case 'noSolutionWith':
      case 'infiniteSolutionsWith':
      case 'uniqueSolutionWith': {
        const k = classifySystem(M, c.b)
        const want = { noSolutionWith: 'none', infiniteSolutionsWith: 'infinite', uniqueSolutionWith: 'unique' }[c.kind]
        return { ok: k === want, detail: k }
      }
      case 'mean':
      case 'median':
      case 'variance':
      case 'std':
      case 'sum':
      case 'range': {
        const v = needVector(M)
        const got = statOf(c.kind, v, 'sample' in c ? c.sample : false)
        return { ok: approx(got, c.value), detail: `${c.kind} = ${fmt(got)}` }
      }
      case 'meanGreaterThanMedian':
      case 'meanLessThanMedian': {
        const v = needVector(M)
        const [m, md] = [statOf('mean', v), statOf('median', v)]
        return { ok: c.kind === 'meanGreaterThanMedian' ? m > md + EPS : m < md - EPS, detail: `mean = ${fmt(m)}, median = ${fmt(md)}` }
      }
      case 'nonnegative':
        return { ok: M.every((r) => r.every((x) => x >= -EPS)) }
      case 'distinctValues': {
        const n = new Set(M.flat().map((x) => +x.toFixed(9))).size
        return { ok: n >= c.min, detail: `${n} distinct` }
      }
      case 'probabilityVector': {
        const v = needVector(M)
        const s = v.reduce((a, b) => a + b, 0)
        return { ok: v.every((x) => x >= -EPS) && approx(s, 1), detail: `Σ = ${fmt(s)}` }
      }
      case 'expectation': {
        const p = needVector(M)
        if (p.length !== c.values.length) throw new Error(`expected ${c.values.length} probabilities`)
        const e = p.reduce((s, pi, i) => s + pi * c.values[i], 0)
        return { ok: approx(e, c.value), detail: `E[X] = ${fmt(e)}` }
      }
    }
  } catch (e) {
    return { ok: false, detail: (e as Error).message }
  }
}

export const checkAll = (cs: ConstraintSpec[], M: Mat) => cs.map((c) => checkConstraint(c, M))
