import type { ExerciseSpec, Localized, Of } from '../content/schema'
import { det, dot, matMul, matVec, norm, rank, rref, classifySystem, type Mat } from './linalg'
import { pick, randInt, randNonZero, shuffle, type Rng } from './random'

type Concrete = Of<'numeric'> | Of<'matrix'> | Of<'mcq'>
type Params = Record<string, unknown>
type Gen = (rng: Rng, base: Of<'parametric'>, params: Params) => Concrete

const texMat = (M: Mat) => `\\begin{bmatrix}${M.map((r) => r.join(' & ')).join(' \\\\ ')}\\end{bmatrix}`
const texVec = (v: number[]) => texMat(v.map((x) => [x]))
const vars = ['a', 'b', 'c', 'd']
const texEq = (row: number[], rhs: number) => {
  let s = ''
  row.forEach((c, j) => {
    if (c === 0) return
    const sign = c < 0 ? ' - ' : s ? ' + ' : ''
    const abs = Math.abs(c)
    s += `${sign}${abs === 1 ? '' : abs}${vars[j]}`
  })
  return `${s || '0'} = ${rhs}`
}
const texSystem = (A: Mat, b: number[]) => `\\begin{cases} ${A.map((r, i) => texEq(r, b[i])).join(' \\\\ ')} \\end{cases}`
const loc = (vi: string, en: string): Localized => ({ vi, en })
const num = (p: Params, k: string, d: number) => (typeof p[k] === 'number' ? (p[k] as number) : d)

function common(base: Of<'parametric'>) {
  const { id, topic, difficulty, tags, hints, figure } = base
  return { id, topic, difficulty, tags, hints, figure }
}

function randMat(rng: Rng, r: number, c: number, range = 5): Mat {
  return Array.from({ length: r }, () => Array.from({ length: c }, () => randInt(rng, -range, range)))
}

function nonsingularMat(rng: Rng, n: number, range = 5): Mat {
  for (;;) {
    const M = randMat(rng, n, n, range)
    if (Math.abs(det(M)) > 1e-9) return M
  }
}

/** Build a matrix of a given rank as a product of random full-rank factors. */
function matOfRank(rng: Rng, n: number, r: number): Mat {
  for (;;) {
    const B = randMat(rng, n, r, 2)
    const C = randMat(rng, r, n, 2)
    const M = r === 0 ? randMat(rng, n, n, 0) : matMul(B, C)
    if (rank(M) === r && M.every((row) => row.every((x) => Math.abs(x) <= 12))) return M
  }
}

export const generators: Record<string, { describe: Localized; gen: Gen }> = {
  'solve-system': {
    describe: loc('Giải hệ phương trình tuyến tính có nghiệm duy nhất', 'Solve a linear system with a unique solution'),
    gen: (rng, base, p) => {
      const n = num(p, 'size', 2)
      const A = nonsingularMat(rng, n, 4)
      const x = Array.from({ length: n }, () => randInt(rng, -5, 5))
      const b = matVec(A, x)
      return {
        ...common(base), type: 'matrix', tolerance: 1e-3, answer: x.map((v) => [v]),
        prompt: loc(
          `Giải hệ phương trình và nhập nghiệm $(${vars.slice(0, n).join(', ')})$ dưới dạng vector cột:\n\n$$${texSystem(A, b)}$$`,
          `Solve the system and enter the solution $(${vars.slice(0, n).join(', ')})$ as a column vector:\n\n$$${texSystem(A, b)}$$`,
        ),
        explanation: loc(
          `Dùng phép khử (elimination) để được nghiệm $${vars.slice(0, n).map((v, i) => `${v} = ${x[i]}`).join(',\\ ')}$. Kiểm tra lại bằng cách thay vào từng phương trình.`,
          `Using elimination gives $${vars.slice(0, n).map((v, i) => `${v} = ${x[i]}`).join(',\\ ')}$. Verify by substituting into each equation.`,
        ),
      }
    },
  },
  determinant: {
    describe: loc('Tính định thức', 'Compute a determinant'),
    gen: (rng, base, p) => {
      const n = num(p, 'size', 2)
      const M = randMat(rng, n, n, n === 2 ? 6 : 3)
      const d = Math.round(det(M))
      return {
        ...common(base), type: 'numeric', tolerance: 1e-3, answers: [{ value: d }],
        prompt: loc(`Tính $\\det(A)$ với $A = ${texMat(M)}$`, `Compute $\\det(A)$ for $A = ${texMat(M)}$`),
        explanation: n === 2
          ? loc(`$\\det = ${M[0][0]}\\cdot${M[1][1]} - (${M[0][1]})\\cdot(${M[1][0]}) = ${d}$`, `$\\det = ${M[0][0]}\\cdot${M[1][1]} - (${M[0][1]})\\cdot(${M[1][0]}) = ${d}$`)
          : loc(`Dùng quy tắc đường chéo (Sarrus) hoặc khử về tam giác: $\\det = ${d}$.`, `Use the diagonals rule or reduce to triangular form: $\\det = ${d}$.`),
      }
    },
  },
  'classify-system': {
    describe: loc('Phân loại hệ: 1 nghiệm / vô số / vô nghiệm', 'Classify a system: unique / infinite / none'),
    gen: (rng, base) => {
      const kind = pick(rng, ['unique', 'infinite', 'none'] as const)
      let A: Mat
      let b: number[]
      if (kind === 'unique') {
        A = nonsingularMat(rng, 2, 5)
        const x = [randInt(rng, -4, 4), randInt(rng, -4, 4)]
        b = matVec(A, x)
      } else {
        const r0 = [randNonZero(rng, -5, 5), randNonZero(rng, -5, 5)]
        const k = randNonZero(rng, -3, 3)
        const b0 = randInt(rng, -9, 9)
        A = [r0, r0.map((x) => x * k)]
        b = [b0, kind === 'infinite' ? b0 * k : b0 * k + randNonZero(rng, -4, 4)]
      }
      const labels = {
        unique: loc('Đúng 1 nghiệm (non-singular)', 'Exactly one solution (non-singular)'),
        infinite: loc('Vô số nghiệm (singular, redundant)', 'Infinitely many solutions (singular, redundant)'),
        none: loc('Vô nghiệm (singular, contradictory)', 'No solutions (singular, contradictory)'),
      }
      const order = ['unique', 'infinite', 'none'] as const
      const real = classifySystem(A, b)
      return {
        ...common(base), type: 'mcq', shuffle: false,
        options: order.map((k) => labels[k]), answer: [order.indexOf(real)],
        prompt: loc(`Hệ sau có bao nhiêu nghiệm?\n\n$$${texSystem(A, b)}$$`, `How many solutions does this system have?\n\n$$${texSystem(A, b)}$$`),
        explanation: real === 'unique'
          ? loc(`$\\det = ${det(A)} \\neq 0$ nên hệ non-singular, có đúng 1 nghiệm.`, `$\\det = ${det(A)} \\neq 0$, so the system is non-singular with exactly one solution.`)
          : loc(
            `Hàng 2 của ma trận hệ số là bội của hàng 1 (det = 0). ${real === 'infinite' ? 'Vế phải cũng cùng tỉ lệ ⇒ hai đường trùng nhau ⇒ vô số nghiệm.' : 'Nhưng vế phải không cùng tỉ lệ ⇒ hai đường song song ⇒ vô nghiệm.'}`,
            `Row 2 of the coefficient matrix is a multiple of row 1 (det = 0). ${real === 'infinite' ? 'The constants scale the same way ⇒ the lines coincide ⇒ infinitely many solutions.' : 'The constants do not scale the same way ⇒ parallel lines ⇒ no solution.'}`,
          ),
      }
    },
  },
  rank: {
    describe: loc('Tìm hạng (rank) của ma trận', 'Find the rank of a matrix'),
    gen: (rng, base, p) => {
      const n = num(p, 'size', 3)
      const r = randInt(rng, 1, n)
      const M = matOfRank(rng, n, r)
      return {
        ...common(base), type: 'numeric', tolerance: 0, answers: [{ value: r }],
        prompt: loc(`Tìm rank của $A = ${texMat(M)}$`, `Find the rank of $A = ${texMat(M)}$`),
        explanation: loc(
          `Đưa về dạng bậc thang (row echelon): số pivot = số hàng khác 0 = ${r}.`,
          `Reduce to row echelon form: number of pivots = number of non-zero rows = ${r}.`,
        ),
      }
    },
  },
  rref: {
    describe: loc('Đưa ma trận về dạng bậc thang rút gọn', 'Reduce a matrix to reduced row echelon form'),
    gen: (rng, base, p) => {
      const n = num(p, 'size', 3)
      const singular = rng() < 0.4
      const M = singular ? matOfRank(rng, n, n - 1) : nonsingularMat(rng, n, 3)
      const R = rref(M).R.map((r) => r.map((x) => Math.round(x * 1000) / 1000))
      return {
        ...common(base), type: 'matrix', tolerance: 1e-2, answer: R,
        prompt: loc(`Tìm dạng bậc thang rút gọn (RREF) của $A = ${texMat(M)}$`, `Find the reduced row echelon form of $A = ${texMat(M)}$`),
        explanation: loc(
          `Kết quả: $${texMat(R)}$. ${singular ? 'Ma trận singular nên có hàng toàn 0.' : 'Ma trận non-singular nên RREF là ma trận đơn vị.'}`,
          `Result: $${texMat(R)}$. ${singular ? 'The matrix is singular, so a zero row appears.' : 'The matrix is non-singular, so the RREF is the identity.'}`,
        ),
      }
    },
  },
  'dot-product': {
    describe: loc('Tích vô hướng', 'Dot product'),
    gen: (rng, base, p) => {
      const n = num(p, 'size', 3)
      const u = Array.from({ length: n }, () => randInt(rng, -6, 6))
      const v = Array.from({ length: n }, () => randInt(rng, -6, 6))
      const d = dot(u, v)
      return {
        ...common(base), type: 'numeric', tolerance: 0, answers: [{ value: d }],
        prompt: loc(`Tính $u \\cdot v$ với $u = ${texVec(u)}$, $v = ${texVec(v)}$`, `Compute $u \\cdot v$ for $u = ${texVec(u)}$, $v = ${texVec(v)}$`),
        explanation: loc(
          `$${u.map((x, i) => `(${x})(${v[i]})`).join(' + ')} = ${d}$${d === 0 ? ' ⇒ hai vector vuông góc.' : ''}`,
          `$${u.map((x, i) => `(${x})(${v[i]})`).join(' + ')} = ${d}$${d === 0 ? ' ⇒ the vectors are orthogonal.' : ''}`,
        ),
      }
    },
  },
  norm: {
    describe: loc('Chuẩn L1 / L2 của vector', 'L1 / L2 norm of a vector'),
    gen: (rng, base) => {
      const p = pick(rng, [1, 2] as const)
      const triples = [[3, 4], [5, 12], [6, 8], [8, 15], [1, 2], [2, 3], [-3, 4], [2, -1, 2], [1, 2, 2], [4, -4, 2]]
      const v = shuffle(pick(rng, triples), rng).map((x) => x * (rng() < 0.5 ? -1 : 1))
      const val = norm(v, p)
      return {
        ...common(base), type: 'numeric', tolerance: 1e-3, answers: [{ value: val }],
        prompt: loc(`Tính chuẩn L${p} của $v = ${texVec(v)}$ (có thể nhập \`sqrt(5)\`)`, `Compute the L${p}-norm of $v = ${texVec(v)}$ (you may type \`sqrt(5)\`)`),
        explanation: p === 1
          ? loc(`$\\|v\\|_1 = ${v.map((x) => `|${x}|`).join(' + ')} = ${val}$`, `$\\|v\\|_1 = ${v.map((x) => `|${x}|`).join(' + ')} = ${val}$`)
          : loc(`$\\|v\\|_2 = \\sqrt{${v.map((x) => `${x}^2`).join(' + ')}} = ${+val.toFixed(4)}$`, `$\\|v\\|_2 = \\sqrt{${v.map((x) => `${x}^2`).join(' + ')}} = ${+val.toFixed(4)}$`),
      }
    },
  },
  'vector-combination': {
    describe: loc('Tổ hợp tuyến tính của vector', 'Linear combination of vectors'),
    gen: (rng, base) => {
      const n = pick(rng, [2, 3])
      const u = Array.from({ length: n }, () => randInt(rng, -5, 5))
      const v = Array.from({ length: n }, () => randInt(rng, -5, 5))
      const a = randNonZero(rng, -3, 3)
      const b = randNonZero(rng, -3, 3)
      const w = u.map((x, i) => a * x + b * v[i])
      return {
        ...common(base), type: 'matrix', tolerance: 0, answer: w.map((x) => [x]),
        prompt: loc(`Tính $${a}u ${b < 0 ? '-' : '+'} ${Math.abs(b)}v$ với $u = ${texVec(u)}$, $v = ${texVec(v)}$`, `Compute $${a}u ${b < 0 ? '-' : '+'} ${Math.abs(b)}v$ for $u = ${texVec(u)}$, $v = ${texVec(v)}$`),
        explanation: loc(`Nhân từng thành phần rồi cộng: $${texVec(w)}$`, `Scale component-wise then add: $${texVec(w)}$`),
      }
    },
  },
  'matrix-vector': {
    describe: loc('Nhân ma trận với vector', 'Matrix-vector product'),
    gen: (rng, base) => {
      const r = pick(rng, [2, 3])
      const c = pick(rng, [2, 3])
      const A = randMat(rng, r, c, 4)
      const x = Array.from({ length: c }, () => randInt(rng, -3, 3))
      const y = matVec(A, x)
      return {
        ...common(base), type: 'matrix', tolerance: 0, answer: y.map((v) => [v]),
        prompt: loc(`Tính $Ax$ với $A = ${texMat(A)}$, $x = ${texVec(x)}$`, `Compute $Ax$ for $A = ${texMat(A)}$, $x = ${texVec(x)}$`),
        explanation: loc(`Mỗi thành phần là tích vô hướng của một hàng của $A$ với $x$: $${texVec(y)}$`, `Each entry is the dot product of a row of $A$ with $x$: $${texVec(y)}$`),
      }
    },
  },
  'matrix-multiply': {
    describe: loc('Nhân hai ma trận', 'Matrix multiplication'),
    gen: (rng, base) => {
      const [m, k, n] = [pick(rng, [2, 3]), pick(rng, [2, 3]), pick(rng, [2, 3])]
      const A = randMat(rng, m, k, 3)
      const B = randMat(rng, k, n, 3)
      const C = matMul(A, B)
      return {
        ...common(base), type: 'matrix', tolerance: 0, answer: C,
        prompt: loc(`Tính $AB$ với $A = ${texMat(A)}$, $B = ${texMat(B)}$ (kết quả ${m}×${n})`, `Compute $AB$ for $A = ${texMat(A)}$, $B = ${texMat(B)}$ (result is ${m}×${n})`),
        explanation: loc(`Phần tử $(i,j)$ = hàng $i$ của $A$ · cột $j$ của $B$: $${texMat(C)}$`, `Entry $(i,j)$ = row $i$ of $A$ · column $j$ of $B$: $${texMat(C)}$`),
      }
    },
  },
  'vector-angle': {
    describe: loc('Góc giữa hai vector', 'Angle between vectors'),
    gen: (rng, base) => {
      const pairs: [number[], number[], number][] = [
        [[1, 0], [1, 1], 45], [[1, 0], [0, 3], 90], [[2, 2], [-3, 3], 90], [[1, 0], [-1, 1], 135],
        [[3, 0], [-2, 0], 180], [[1, 1], [2, 2], 0], [[1, 0], [1, Math.sqrt(3)], 60], [[0, 2], [1, 1], 45],
      ]
      const [u, v, ang] = pick(rng, pairs)
      const s = randInt(rng, 1, 3)
      const uu = u.map((x) => x * s)
      const fmt = (w: number[]) => w.map((x) => (Number.isInteger(x) ? x : `${x / Math.sqrt(3)}\\sqrt{3}`))
      return {
        ...common(base), type: 'numeric', tolerance: 1e-2, answers: [{ value: ang }],
        prompt: loc(`Tìm góc (độ) giữa $u = (${fmt(uu).join(', ')})$ và $v = (${fmt(v).join(', ')})$`, `Find the angle (degrees) between $u = (${fmt(uu).join(', ')})$ and $v = (${fmt(v).join(', ')})$`),
        explanation: loc(`$\\cos\\theta = \\dfrac{u\\cdot v}{\\|u\\|\\|v\\|}$ ⇒ $\\theta = ${ang}^\\circ$`, `$\\cos\\theta = \\dfrac{u\\cdot v}{\\|u\\|\\|v\\|}$ ⇒ $\\theta = ${ang}^\\circ$`),
      }
    },
  },
}

/* ---------------- calculus & probability generators ---------------- */
const r2 = (x: number) => Math.round(x * 100) / 100
const r4 = (x: number) => Math.round(x * 10000) / 10000
const polyTex = (cs: number[]) => {
  // cs[k] is the coefficient of x^k
  let s = ''
  for (let k = cs.length - 1; k >= 0; k--) {
    const c = cs[k]
    if (!c) continue
    const sign = c < 0 ? ' - ' : s ? ' + ' : ''
    const a = Math.abs(c)
    const coef = a === 1 && k > 0 ? '' : String(a)
    s += `${sign}${coef}${k === 0 ? '' : k === 1 ? 'x' : `x^{${k}}`}`
  }
  return s || '0'
}
const polyEval = (cs: number[], x: number) => cs.reduce((s, c, k) => s + c * x ** k, 0)
const polyDer = (cs: number[]) => cs.slice(1).map((c, k) => c * (k + 1))
const choose = (n: number, k: number) => {
  let r = 1
  for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i
  return r
}

Object.assign(generators, {
  'derivative-poly': {
    describe: loc('Đạo hàm đa thức tại một điểm', 'Derivative of a polynomial at a point'),
    gen: (rng, base) => {
      const deg = pick(rng, [2, 3])
      const cs = Array.from({ length: deg + 1 }, (_, k) => (k === deg ? randNonZero(rng, -3, 3) : randInt(rng, -5, 5)))
      const a = randInt(rng, -3, 3)
      const d = polyDer(cs)
      const val = polyEval(d, a)
      return {
        ...common(base), type: 'numeric', tolerance: 1e-6, answers: [{ value: val }],
        prompt: loc(`Cho $f(x) = ${polyTex(cs)}$. Tính $f'(${a})$.`, `Let $f(x) = ${polyTex(cs)}$. Compute $f'(${a})$.`),
        explanation: loc(
          `Quy tắc lũy thừa (power rule) $\\frac{d}{dx}x^n = nx^{n-1}$: $f'(x) = ${polyTex(d)}$, nên $f'(${a}) = ${val}$.`,
          `Power rule $\\frac{d}{dx}x^n = nx^{n-1}$: $f'(x) = ${polyTex(d)}$, so $f'(${a}) = ${val}$.`,
        ),
      }
    },
  },
  'partial-derivative': {
    describe: loc('Đạo hàm riêng tại một điểm', 'Partial derivative at a point'),
    gen: (rng, base) => {
      const a = randNonZero(rng, -3, 3), m = randInt(rng, 1, 3), n = randInt(rng, 1, 2)
      const b = randInt(rng, -4, 4), c = randInt(rng, -4, 4)
      const x = randInt(rng, -2, 2), y = randInt(rng, -2, 2)
      const wrt = pick(rng, ['x', 'y'] as const)
      const term = `${a === 1 ? '' : a === -1 ? '-' : a}x^{${m}}y^{${n}}`
      const f = `${term}${b ? ` ${b < 0 ? '-' : '+'} ${Math.abs(b)}x` : ''}${c ? ` ${c < 0 ? '-' : '+'} ${Math.abs(c)}y` : ''}`
      const val = wrt === 'x' ? a * m * x ** (m - 1) * y ** n + b : a * n * x ** m * y ** (n - 1) + c
      const der = wrt === 'x' ? `${a * m}x^{${m - 1}}y^{${n}} + (${b})` : `${a * n}x^{${m}}y^{${n - 1}} + (${c})`
      return {
        ...common(base), type: 'numeric', tolerance: 1e-6, answers: [{ value: val }],
        prompt: loc(`Cho $f(x,y) = ${f}$. Tính $\\dfrac{\\partial f}{\\partial ${wrt}}$ tại $(${x}, ${y})$.`, `Let $f(x,y) = ${f}$. Compute $\\dfrac{\\partial f}{\\partial ${wrt}}$ at $(${x}, ${y})$.`),
        explanation: loc(
          `Coi biến còn lại là hằng số: $\\dfrac{\\partial f}{\\partial ${wrt}} = ${der}$. Thay $(${x}, ${y})$: $${val}$.`,
          `Treat the other variable as a constant: $\\dfrac{\\partial f}{\\partial ${wrt}} = ${der}$. At $(${x}, ${y})$: $${val}$.`,
        ),
      }
    },
  },
  'gradient-step': {
    describe: loc('Một bước gradient descent', 'One gradient-descent step'),
    gen: (rng, base) => {
      const a = randInt(rng, 1, 3), b = randInt(rng, 1, 3)
      const cx = randInt(rng, -3, 3), cy = randInt(rng, -3, 3)
      const x0 = randInt(rng, -4, 4), y0 = randInt(rng, -4, 4)
      const lr = pick(rng, [0.1, 0.05, 0.2, 0.25])
      const gx = 2 * a * (x0 - cx), gy = 2 * b * (y0 - cy)
      const x1 = r4(x0 - lr * gx), y1 = r4(y0 - lr * gy)
      const f = `${a === 1 ? '' : a}(x ${cx < 0 ? '+' : '-'} ${Math.abs(cx)})^2 + ${b === 1 ? '' : b}(y ${cy < 0 ? '+' : '-'} ${Math.abs(cy)})^2`
      return {
        ...common(base), type: 'numeric', tolerance: 1e-4, answers: [{ label: 'x_1', value: x1 }, { label: 'y_1', value: y1 }],
        prompt: loc(
          `Cho $f(x,y) = ${f}$, điểm bắt đầu $(x_0, y_0) = (${x0}, ${y0})$ và learning rate $\\alpha = ${lr}$. Thực hiện **một** bước gradient descent.`,
          `Let $f(x,y) = ${f}$, start at $(x_0, y_0) = (${x0}, ${y0})$ with learning rate $\\alpha = ${lr}$. Perform **one** gradient-descent step.`,
        ),
        explanation: loc(
          `$\\nabla f = (${2 * a}(x - ${cx}),\\ ${2 * b}(y - ${cy})) = (${gx}, ${gy})$ tại điểm đầu. $(x_1, y_1) = (x_0, y_0) - \\alpha \\nabla f = (${x1}, ${y1})$.`,
          `$\\nabla f = (${2 * a}(x - ${cx}),\\ ${2 * b}(y - ${cy})) = (${gx}, ${gy})$ at the start. $(x_1, y_1) = (x_0, y_0) - \\alpha \\nabla f = (${x1}, ${y1})$.`,
        ),
      }
    },
  },
  'newton-step': {
    describe: loc('Một bước phương pháp Newton', "One step of Newton's method"),
    gen: (rng, base) => {
      for (;;) {
        const cs = [randInt(rng, -6, 6), randInt(rng, -4, 4), randNonZero(rng, -2, 2)]
        const x0 = randInt(rng, -3, 3)
        const fp = polyEval(polyDer(cs), x0)
        if (fp === 0) continue
        const fx = polyEval(cs, x0)
        const x1 = r4(x0 - fx / fp)
        return {
          ...common(base), type: 'numeric', tolerance: 1e-3, answers: [{ label: 'x_1', value: x1 }],
          prompt: loc(
            `Dùng phương pháp Newton tìm nghiệm của $f(x) = ${polyTex(cs)}$, bắt đầu từ $x_0 = ${x0}$. Tính $x_1$.`,
            `Use Newton's method to find a zero of $f(x) = ${polyTex(cs)}$ starting at $x_0 = ${x0}$. Compute $x_1$.`,
          ),
          explanation: loc(
            `$x_1 = x_0 - \\dfrac{f(x_0)}{f'(x_0)} = ${x0} - \\dfrac{${fx}}{${fp}} = ${x1}$.`,
            `$x_1 = x_0 - \\dfrac{f(x_0)}{f'(x_0)} = ${x0} - \\dfrac{${fx}}{${fp}} = ${x1}$.`,
          ),
        }
      }
    },
  },
  'descriptive-stats': {
    describe: loc('Thống kê mô tả: mean / median / variance', 'Descriptive statistics: mean / median / variance'),
    gen: (rng, base) => {
      const n = randInt(rng, 5, 8)
      const data = Array.from({ length: n }, () => randInt(rng, 0, 12))
      const what = pick(rng, ['mean', 'median', 'pvar', 'svar'] as const)
      const mean = data.reduce((a, b) => a + b, 0) / n
      const s = [...data].sort((a, b) => a - b)
      const med = n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2
      const ss = data.reduce((a, x) => a + (x - mean) ** 2, 0)
      const val = { mean, median: med, pvar: ss / n, svar: ss / (n - 1) }[what]
      const name = {
        mean: loc('trung bình (mean)', 'mean'), median: loc('trung vị (median)', 'median'),
        pvar: loc('phương sai tổng thể (chia $n$)', 'population variance (divide by $n$)'),
        svar: loc('phương sai mẫu (chia $n-1$)', 'sample variance (divide by $n-1$)'),
      }[what]
      return {
        ...common(base), type: 'numeric', tolerance: 1e-3, answers: [{ value: r4(val) }],
        prompt: loc(`Dữ liệu: $${data.join(',\\ ')}$. Tính ${name.vi}.`, `Data: $${data.join(',\\ ')}$. Compute the ${name.en}.`),
        explanation: loc(
          `Sắp xếp: $${s.join(',\\ ')}$; $\\bar x = ${r4(mean)}$, $\\sum (x_i-\\bar x)^2 = ${r4(ss)}$. Kết quả: $${r4(val)}$.`,
          `Sorted: $${s.join(',\\ ')}$; $\\bar x = ${r4(mean)}$, $\\sum (x_i-\\bar x)^2 = ${r4(ss)}$. Result: $${r4(val)}$.`,
        ),
      }
    },
  },
  'binomial-prob': {
    describe: loc('Xác suất nhị thức', 'Binomial probability'),
    gen: (rng, base) => {
      const n = randInt(rng, 3, 8), k = randInt(rng, 0, n)
      const p = pick(rng, [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
      const val = choose(n, k) * p ** k * (1 - p) ** (n - k)
      return {
        ...common(base), type: 'numeric', tolerance: 1e-3, answers: [{ value: r4(val) }],
        prompt: loc(
          `$X \\sim \\text{Binomial}(n = ${n}, p = ${p})$. Tính $P(X = ${k})$ (làm tròn 4 chữ số).`,
          `$X \\sim \\text{Binomial}(n = ${n}, p = ${p})$. Compute $P(X = ${k})$ (4 decimals).`,
        ),
        explanation: loc(
          `$P(X=k) = \\binom{n}{k}p^k(1-p)^{n-k} = ${choose(n, k)}\\cdot ${p}^{${k}}\\cdot ${r2(1 - p)}^{${n - k}} \\approx ${r4(val)}$.`,
          `$P(X=k) = \\binom{n}{k}p^k(1-p)^{n-k} = ${choose(n, k)}\\cdot ${p}^{${k}}\\cdot ${r2(1 - p)}^{${n - k}} \\approx ${r4(val)}$.`,
        ),
      }
    },
  },
  'expected-value': {
    describe: loc('Kỳ vọng và phương sai của biến rời rạc', 'Expectation and variance of a discrete variable'),
    gen: (rng, base) => {
      const k = randInt(rng, 3, 4)
      const xs = shuffle([-2, -1, 0, 1, 2, 3, 4, 5], rng).slice(0, k).sort((a, b) => a - b)
      const raw = Array.from({ length: k }, () => randInt(rng, 1, 5))
      const tot = raw.reduce((a, b) => a + b, 0)
      const ps = raw.map((r) => r / tot)
      const E = xs.reduce((s, x, i) => s + x * ps[i], 0)
      const V = xs.reduce((s, x, i) => s + (x - E) ** 2 * ps[i], 0)
      const frac = raw.map((r) => `\\tfrac{${r}}{${tot}}`)
      return {
        ...common(base), type: 'numeric', tolerance: 1e-3, answers: [{ label: 'E[X]', value: r4(E) }, { label: '\\mathrm{Var}(X)', value: r4(V) }],
        prompt: loc(
          `Biến ngẫu nhiên $X$ có PMF: ${xs.map((x, i) => `$P(X=${x}) = ${frac[i]}$`).join(', ')}. Tính $E[X]$ và $\\mathrm{Var}(X)$.`,
          `Random variable $X$ has PMF: ${xs.map((x, i) => `$P(X=${x}) = ${frac[i]}$`).join(', ')}. Compute $E[X]$ and $\\mathrm{Var}(X)$.`,
        ),
        explanation: loc(
          `$E[X] = \\sum x\\,p(x) = ${r4(E)}$; $\\mathrm{Var}(X) = \\sum (x - E[X])^2 p(x) = ${r4(V)}$ (hoặc $E[X^2] - E[X]^2$).`,
          `$E[X] = \\sum x\\,p(x) = ${r4(E)}$; $\\mathrm{Var}(X) = \\sum (x - E[X])^2 p(x) = ${r4(V)}$ (or $E[X^2] - E[X]^2$).`,
        ),
      }
    },
  },
  bayes: {
    describe: loc('Định lý Bayes (xét nghiệm bệnh)', "Bayes' theorem (medical test)"),
    gen: (rng, base) => {
      const prev = pick(rng, [0.01, 0.02, 0.05, 0.1, 0.2])
      const sens = pick(rng, [0.9, 0.95, 0.99, 0.8])
      const spec = pick(rng, [0.9, 0.95, 0.98, 0.85])
      const pPos = sens * prev + (1 - spec) * (1 - prev)
      const val = (sens * prev) / pPos
      return {
        ...common(base), type: 'numeric', tolerance: 2e-3, answers: [{ value: r4(val) }],
        prompt: loc(
          `Tỉ lệ mắc bệnh $P(D) = ${prev}$. Xét nghiệm có độ nhạy $P(+\\mid D) = ${sens}$ và độ đặc hiệu $P(-\\mid \\neg D) = ${spec}$. Một người có kết quả dương tính. Tính $P(D \\mid +)$.`,
          `Disease prevalence $P(D) = ${prev}$. A test has sensitivity $P(+\\mid D) = ${sens}$ and specificity $P(-\\mid \\neg D) = ${spec}$. A person tests positive. Compute $P(D \\mid +)$.`,
        ),
        explanation: loc(
          `$P(+) = ${sens}\\cdot${prev} + ${r2(1 - spec)}\\cdot${r2(1 - prev)} = ${r4(pPos)}$. $P(D\\mid +) = \\dfrac{P(+\\mid D)P(D)}{P(+)} \\approx ${r4(val)}$ — thấp hơn nhiều so với độ nhạy vì bệnh hiếm.`,
          `$P(+) = ${sens}\\cdot${prev} + ${r2(1 - spec)}\\cdot${r2(1 - prev)} = ${r4(pPos)}$. $P(D\\mid +) = \\dfrac{P(+\\mid D)P(D)}{P(+)} \\approx ${r4(val)}$ — much lower than the sensitivity because the disease is rare.`,
        ),
      }
    },
  },
  'z-score': {
    describe: loc('Chuẩn hoá z-score', 'Standardise with a z-score'),
    gen: (rng, base) => {
      const mu = randInt(rng, 40, 80), sigma = pick(rng, [2, 4, 5, 8, 10]), z = pick(rng, [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5])
      const x = mu + z * sigma
      return {
        ...common(base), type: 'numeric', tolerance: 1e-6, answers: [{ value: z }],
        prompt: loc(`$X \\sim \\mathcal N(\\mu = ${mu}, \\sigma = ${sigma})$. Tính z-score của $x = ${x}$.`, `$X \\sim \\mathcal N(\\mu = ${mu}, \\sigma = ${sigma})$. Compute the z-score of $x = ${x}$.`),
        explanation: loc(`$z = \\dfrac{x - \\mu}{\\sigma} = \\dfrac{${x} - ${mu}}{${sigma}} = ${z}$.`, `$z = \\dfrac{x - \\mu}{\\sigma} = \\dfrac{${x} - ${mu}}{${sigma}} = ${z}$.`),
      }
    },
  },
  'confidence-interval': {
    describe: loc('Khoảng tin cậy cho trung bình (biết σ)', 'Confidence interval for a mean (known σ)'),
    gen: (rng, base) => {
      const [conf, zc] = pick(rng, [[90, 1.645], [95, 1.96], [99, 2.576]] as const)
      const n = pick(rng, [16, 25, 36, 49, 64, 100])
      const sigma = pick(rng, [4, 5, 6, 8, 10, 12])
      const xbar = randInt(rng, 20, 90)
      const me = zc * (sigma / Math.sqrt(n))
      return {
        ...common(base), type: 'numeric', tolerance: 2e-3, answers: [{ label: '\\text{lower}', value: r4(xbar - me) }, { label: '\\text{upper}', value: r4(xbar + me) }],
        prompt: loc(
          `Mẫu cỡ $n = ${n}$ có $\\bar x = ${xbar}$, biết $\\sigma = ${sigma}$. Tìm khoảng tin cậy ${conf}% cho $\\mu$ (dùng $z = ${zc}$).`,
          `A sample of size $n = ${n}$ has $\\bar x = ${xbar}$ with known $\\sigma = ${sigma}$. Find the ${conf}% confidence interval for $\\mu$ (use $z = ${zc}$).`,
        ),
        explanation: loc(
          `Sai số biên (margin of error) $= z\\cdot\\sigma/\\sqrt n = ${zc}\\cdot ${sigma}/${Math.sqrt(n)} = ${r4(me)}$. Khoảng: $${xbar} \\pm ${r4(me)}$.`,
          `Margin of error $= z\\cdot\\sigma/\\sqrt n = ${zc}\\cdot ${sigma}/${Math.sqrt(n)} = ${r4(me)}$. Interval: $${xbar} \\pm ${r4(me)}$.`,
        ),
      }
    },
  },
} satisfies Record<string, { describe: Localized; gen: Gen }>)

export function instantiate(base: Of<'parametric'>, rng: Rng): ExerciseSpec {
  const g = generators[base.generator]
  if (!g) throw new Error(`Unknown generator ${base.generator}`)
  return g.gen(rng, base, base.params)
}
