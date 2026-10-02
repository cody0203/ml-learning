export type Vec = number[]
export type Mat = number[][]

export const EPS = 1e-9

export const approx = (a: number, b: number, tol = 1e-6) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b))

export const clone = (A: Mat): Mat => A.map((r) => [...r])

export function det(A: Mat): number {
  const n = A.length
  if (n === 0 || A.some((r) => r.length !== n)) throw new Error('det requires a square matrix')
  const M = clone(A)
  let d = 1
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r
    if (Math.abs(M[p][c]) < EPS) return 0
    if (p !== c) {
      ;[M[p], M[c]] = [M[c], M[p]]
      d = -d
    }
    d *= M[c][c]
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c]
      for (let k = c; k < n; k++) M[r][k] -= f * M[c][k]
    }
  }
  return Math.abs(d) < EPS ? 0 : d
}

export function rref(A: Mat): { R: Mat; pivots: number[] } {
  const M = clone(A)
  const rows = M.length
  const cols = rows ? M[0].length : 0
  const pivots: number[] = []
  let r = 0
  for (let c = 0; c < cols && r < rows; c++) {
    let p = r
    for (let i = r + 1; i < rows; i++) if (Math.abs(M[i][c]) > Math.abs(M[p][c])) p = i
    if (Math.abs(M[p][c]) < EPS) continue
    ;[M[p], M[r]] = [M[r], M[p]]
    const pv = M[r][c]
    for (let k = 0; k < cols; k++) M[r][k] /= pv
    for (let i = 0; i < rows; i++) {
      if (i === r) continue
      const f = M[i][c]
      if (f !== 0) for (let k = 0; k < cols; k++) M[i][k] -= f * M[r][k]
    }
    pivots.push(c)
    r++
  }
  for (const row of M) for (let k = 0; k < cols; k++) if (Math.abs(row[k]) < EPS) row[k] = 0
  return { R: M, pivots }
}

export const rank = (A: Mat) => rref(A).pivots.length

const leading = (row: Vec) => row.findIndex((x) => Math.abs(x) > EPS)

export function isRowEchelon(A: Mat): boolean {
  let last = -1
  let seenZero = false
  for (const row of A) {
    const l = leading(row)
    if (l === -1) {
      seenZero = true
      continue
    }
    if (seenZero || l <= last) return false
    last = l
  }
  return true
}

export function isReducedRowEchelon(A: Mat): boolean {
  if (!isRowEchelon(A)) return false
  for (let r = 0; r < A.length; r++) {
    const l = leading(A[r])
    if (l === -1) continue
    if (!approx(A[r][l], 1)) return false
    for (let i = 0; i < A.length; i++) if (i !== r && Math.abs(A[i][l]) > EPS) return false
  }
  return true
}

export const dot = (u: Vec, v: Vec) => {
  if (u.length !== v.length) throw new Error('dimension mismatch')
  return u.reduce((s, x, i) => s + x * v[i], 0)
}

export function norm(v: Vec, p: 1 | 2 | 'inf' = 2): number {
  if (p === 1) return v.reduce((s, x) => s + Math.abs(x), 0)
  if (p === 'inf') return Math.max(...v.map(Math.abs))
  return Math.sqrt(dot(v, v))
}

export function matVec(A: Mat, x: Vec): Vec {
  return A.map((row) => dot(row, x))
}

export function matMul(A: Mat, B: Mat): Mat {
  if (A[0].length !== B.length) throw new Error('dimension mismatch')
  return A.map((row) => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)))
}

export const transpose = (A: Mat): Mat => A[0].map((_, j) => A.map((r) => r[j]))

export const isParallel = (u: Vec, v: Vec) => rank([u, v]) < 2

export type SolutionKind = 'unique' | 'infinite' | 'none'

export function classifySystem(A: Mat, b: Vec): SolutionKind {
  const rA = rank(A)
  const rAb = rank(A.map((row, i) => [...row, b[i]]))
  if (rA < rAb) return 'none'
  return rA === A[0].length ? 'unique' : 'infinite'
}

/** Flatten a 1×n or n×1 matrix into a vector. */
export function asVector(M: Mat): Vec | null {
  if (M.length === 1) return [...M[0]]
  if (M.every((r) => r.length === 1)) return M.map((r) => r[0])
  return null
}
