/* Exercise type modules: each is a controlled view + pure(ish) grader, so multi-step can compose them. */
import { Play, SquareTerminal } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import type { ExerciseSpec, OptionSpec, PartSpec } from '../content/schema'
import { Md } from '../components/Md'
import { Figure } from '../components/Figure'
import { MatrixInput, emptyGrid, gridValues, type Grid } from '../components/MatrixInput'
import { CodeEditor } from '../components/CodeEditor'
import { closeEnough, tryEvaluate } from '../lib/expr'
import { checkAll } from '../lib/checkers'
import { shuffle, type Rng } from '../lib/random'
import { useLang } from '../lib/i18n'
import { py, usePyStatus, type PyResult } from '../lib/py'
import { constraintLabel } from './constraintLabel'
import { Link } from 'react-router-dom'
import { notebookRoute } from '../lib/notebook'

type AnySpec = ExerciseSpec | PartSpec
export interface GradeOut<S> {
  correct: boolean
  state?: S
}
export interface ViewProps<T, S> {
  ex: T
  value: S
  onChange: (s: S) => void
  graded: boolean
  feedback: boolean
}
export interface Module<T, S> {
  init: (ex: T, rng: Rng) => S
  View: (p: ViewProps<T, S>) => ReactNode
  grade: (ex: T, s: S) => GradeOut<S> | Promise<GradeOut<S>>
  ready?: (ex: T, s: S) => boolean
}
type Spec<K extends AnySpec['type']> = Extract<AnySpec, { type: K }>

const range = (n: number) => Array.from({ length: n }, (_, i) => i)
const sameSet = (a: number[], b: number[]) => a.length === b.length && a.every((x) => b.includes(x))

function OptionBody({ o }: { o: OptionSpec }) {
  const { t } = useLang()
  return (
    <span className="option-body">
      <Md inline>{t(o)}</Md>
      {o.figure && <Figure spec={o.figure} size={160} />}
    </span>
  )
}

/* ---------- mcq ---------- */
type McqS = { order: number[]; selected: number[] }
const mcq: Module<Spec<'mcq'>, McqS> = {
  init: (ex, rng) => ({ order: ex.shuffle ? shuffle(range(ex.options.length), rng) : range(ex.options.length), selected: [] }),
  ready: (_, s) => s.selected.length > 0,
  grade: (ex, s) => ({ correct: sameSet(s.selected, ex.answer) }),
  View: ({ ex, value, onChange, graded, feedback }) => {
    const { ui } = useLang()
    const multi = ex.answer.length > 1
    const hasFig = ex.options.some((o) => o.figure)
    return (
      <div>
        {multi && <p className="muted small">{ui('multiSelect')}</p>}
        <div className={`options ${hasFig ? 'options-grid' : ''}`}>
          {value.order.map((i) => {
            const sel = value.selected.includes(i)
            const cls = graded && feedback ? (ex.answer.includes(i) ? 'ok' : sel ? 'bad' : '') : ''
            return (
              <label key={i} className={`option ${sel ? 'selected' : ''} ${cls}`}>
                <input
                  type={multi ? 'checkbox' : 'radio'}
                  checked={sel}
                  disabled={graded}
                  onChange={() => onChange({ ...value, selected: multi ? (sel ? value.selected.filter((x) => x !== i) : [...value.selected, i]) : [i] })}
                />
                <OptionBody o={ex.options[i]} />
              </label>
            )
          })}
        </div>
      </div>
    )
  },
}

/* ---------- true-false ---------- */
type TfS = { value: boolean | null; reason: number | null; reasonOrder: number[] }
const trueFalse: Module<Spec<'true-false'>, TfS> = {
  init: (ex, rng) => ({ value: null, reason: null, reasonOrder: 'reasons' in ex && ex.reasons ? shuffle(range(ex.reasons.length), rng) : [] }),
  ready: (ex, s) => s.value !== null && (!('reasons' in ex && ex.reasons) || s.reason !== null),
  grade: (ex, s) => ({
    correct: s.value === ex.answer && (!('reasons' in ex && ex.reasons) || s.reason === ex.reasonAnswer),
  }),
  View: ({ ex, value, onChange, graded, feedback }) => {
    const { ui, t } = useLang()
    const reasons = 'reasons' in ex ? ex.reasons : undefined
    const btn = (b: boolean) => {
      const cls = graded && feedback ? (ex.answer === b ? 'ok' : value.value === b ? 'bad' : '') : ''
      return (
        <button type="button" className={`tf ${value.value === b ? 'selected' : ''} ${cls}`} disabled={graded} onClick={() => onChange({ ...value, value: b })}>
          {b ? ui('true') : ui('false')}
        </button>
      )
    }
    return (
      <div>
        <div className="tf-row">{btn(true)}{btn(false)}</div>
        {reasons && (
          <div className="reasons">
            <p className="muted small">{ui('why')}</p>
            <div className="options">
              {value.reasonOrder.map((i) => {
                const sel = value.reason === i
                const cls = graded && feedback ? ('reasonAnswer' in ex && ex.reasonAnswer === i ? 'ok' : sel ? 'bad' : '') : ''
                return (
                  <label key={i} className={`option ${sel ? 'selected' : ''} ${cls}`}>
                    <input type="radio" checked={sel} disabled={graded} onChange={() => onChange({ ...value, reason: i })} />
                    <Md inline>{t(reasons[i])}</Md>
                  </label>
                )
              })}
            </div>
          </div>
        )}
      </div>
    )
  },
}

/* ---------- numeric ---------- */
type NumS = { inputs: string[] }
const numeric: Module<Spec<'numeric'>, NumS> = {
  init: (ex) => ({ inputs: ex.answers.map(() => '') }),
  ready: (_, s) => s.inputs.every((x) => tryEvaluate(x) !== null),
  grade: (ex, s) => ({
    correct: ex.answers.every((a, i) => {
      const got = tryEvaluate(s.inputs[i])
      return got !== null && closeEnough(got, a.value, ex.tolerance)
    }),
  }),
  View: ({ ex, value, onChange, graded, feedback }) => {
    const { ui } = useLang()
    return (
      <div className="numeric">
        {ex.answers.map((a, i) => {
          const got = tryEvaluate(value.inputs[i])
          const ok = got !== null && closeEnough(got, a.value, ex.tolerance)
          return (
            <label key={i} className="num-field">
              {a.label && <span className="num-label"><Md inline>{`$${a.label}$`}</Md> =</span>}
              <input
                className={graded && feedback ? (ok ? 'ok' : 'bad') : value.inputs[i] && got === null ? 'invalid' : ''}
                value={value.inputs[i]}
                disabled={graded}
                onChange={(e) => onChange({ inputs: value.inputs.map((x, j) => (j === i ? e.target.value : x)) })}
              />
              {got !== null && value.inputs[i] && !/^-?\d*\.?\d*$/.test(value.inputs[i].trim()) && <span className="muted small">≈ {+got.toFixed(6)}</span>}
              {graded && feedback && !ok && <span className="answer-reveal">= {+a.value.toFixed(6)}</span>}
            </label>
          )
        })}
        <p className="muted small">{ui('exprHint')}</p>
      </div>
    )
  },
}

/* ---------- matrix ---------- */
type GridS = { grid: Grid }
const matrixCellOk = (ex: { answer: number[][]; tolerance: number }, g: Grid, i: number, j: number) => {
  const v = tryEvaluate(g[i][j])
  return v !== null && closeEnough(v, ex.answer[i][j], ex.tolerance)
}
const matrix: Module<Spec<'matrix'>, GridS> = {
  init: (ex) => ({ grid: emptyGrid(ex.answer.length, ex.answer[0].length) }),
  ready: (_, s) => gridValues(s.grid) !== null,
  grade: (ex, s) => ({ correct: ex.answer.every((r, i) => r.every((_, j) => matrixCellOk(ex, s.grid, i, j))) }),
  View: ({ ex, value, onChange, graded, feedback }) => {
    const { ui } = useLang()
    return (
      <div>
        <MatrixInput value={value.grid} onChange={(grid) => onChange({ grid })} disabled={graded} cellState={graded && feedback ? (i, j) => (matrixCellOk(ex, value.grid, i, j) ? 'ok' : 'bad') : undefined} />
        {graded && feedback && (
          <div className="answer-reveal">
            <Md>{`$$\\begin{bmatrix}${ex.answer.map((r) => r.map((x) => +x.toFixed(4)).join(' & ')).join(' \\\\ ')}\\end{bmatrix}$$`}</Md>
          </div>
        )}
        <p className="muted small">{ui('exprHint')}</p>
      </div>
    )
  },
}

/* ---------- construct ---------- */
const construct: Module<Spec<'construct'>, GridS> = {
  init: (ex) => ({ grid: emptyGrid(ex.rows, ex.cols) }),
  ready: (_, s) => gridValues(s.grid) !== null,
  grade: (ex, s) => {
    const M = gridValues(s.grid)
    return { correct: !!M && checkAll(ex.constraints, M).every((r) => r.ok) }
  },
  View: ({ ex, value, onChange, graded, feedback }) => {
    const { t, ui } = useLang()
    const M = gridValues(value.grid)
    const results = M ? checkAll(ex.constraints, M) : null
    return (
      <div>
        <MatrixInput value={value.grid} onChange={(grid) => onChange({ grid })} disabled={graded} />
        <ul className="constraints">
          {ex.constraints.map((c, i) => {
            const r = results?.[i]
            const show = graded && feedback && r
            return (
              <li key={i} className={show ? (r.ok ? 'ok' : 'bad') : ''}>
                {show ? (r.ok ? '✓ ' : '✗ ') : '• '}
                {t(constraintLabel(c))}
                {show && r.detail && !r.ok && <span className="muted small"> — {r.detail}</span>}
              </li>
            )
          })}
        </ul>
        {graded && feedback && (
          <div className="answer-reveal">
            <span className="muted small">{ui('sample')}:</span>
            <Md>{`$$\\begin{bmatrix}${ex.example.map((r) => r.join(' & ')).join(' \\\\ ')}\\end{bmatrix}$$`}</Md>
          </div>
        )}
      </div>
    )
  },
}

/* ---------- find-error ---------- */
type SelS = { selected: number | null }
const findError: Module<Spec<'find-error'>, SelS> = {
  init: () => ({ selected: null }),
  ready: (_, s) => s.selected !== null,
  grade: (ex, s) => ({ correct: s.selected === ex.answer }),
  View: ({ ex, value, onChange, graded, feedback }) => {
    const { t, ui } = useLang()
    return (
      <div>
        <p className="muted small">{ui('selectWrongStep')}</p>
        <ol className="steps">
          {ex.steps.map((s, i) => {
            const cls = graded && feedback ? (i === ex.answer ? 'bad-step' : value.selected === i ? 'selected' : '') : value.selected === i ? 'selected' : ''
            return (
              <li key={i}>
                <button type="button" className={`step ${cls}`} disabled={graded} onClick={() => onChange({ selected: i })}>
                  <Md>{t(s)}</Md>
                </button>
              </li>
            )
          })}
        </ol>
      </div>
    )
  },
}

/* ---------- parsons ---------- */
type OrderS = { order: number[] }
const parsons: Module<Spec<'parsons'>, OrderS> = {
  init: (ex, rng) => {
    let order = shuffle(range(ex.steps.length), rng)
    if (order.every((x, i) => x === i)) order = [...order.slice(1), order[0]]
    return { order }
  },
  grade: (_, s) => ({ correct: s.order.every((x, i) => x === i) }),
  View: ({ ex, value, onChange, graded, feedback }) => {
    const { t, ui } = useLang()
    const move = (i: number, d: number) => {
      const j = i + d
      if (j < 0 || j >= value.order.length) return
      const o = [...value.order]
      ;[o[i], o[j]] = [o[j], o[i]]
      onChange({ order: o })
    }
    return (
      <div>
        <p className="muted small">{ui('dragOrder')}</p>
        <ol className="parsons">
          {value.order.map((k, i) => (
            <li key={k} className={graded && feedback ? (k === i ? 'ok' : 'bad') : ''}>
              <div className="parsons-controls">
                <button type="button" disabled={graded || i === 0} onClick={() => move(i, -1)} aria-label="up">↑</button>
                <button type="button" disabled={graded || i === value.order.length - 1} onClick={() => move(i, 1)} aria-label="down">↓</button>
              </div>
              <Md>{t(ex.steps[k])}</Md>
            </li>
          ))}
        </ol>
        {graded && feedback && !value.order.every((x, i) => x === i) && (
          <ol className="answer-reveal small">
            {ex.steps.map((s, i) => (
              <li key={i}><Md inline>{t(s)}</Md></li>
            ))}
          </ol>
        )}
      </div>
    )
  },
}

/* ---------- matching ---------- */
type MatchS = { rightOrder: number[]; chosen: (number | null)[] }
const matching: Module<Spec<'matching'>, MatchS> = {
  init: (ex, rng) => ({ rightOrder: shuffle(range(ex.pairs.length), rng), chosen: ex.pairs.map(() => null) }),
  ready: (_, s) => s.chosen.every((c) => c !== null),
  grade: (_, s) => ({ correct: s.chosen.every((c, i) => c === i) }),
  View: ({ ex, value, onChange, graded, feedback }) => {
    const { t, ui } = useLang()
    const [active, setActive] = useState<number | null>(null)
    const labels = 'ABCDEFGH'
    const letterOf = (k: number) => labels[value.rightOrder.indexOf(k)]
    const pickRight = (k: number) => {
      if (active === null || graded) return
      const chosen = value.chosen.map((c) => (c === k ? null : c))
      chosen[active] = k
      onChange({ ...value, chosen })
      const nextEmpty = chosen.findIndex((c) => c === null)
      setActive(nextEmpty >= 0 ? nextEmpty : null)
    }
    return (
      <div>
        <p className="muted small">{ui('matchHint')}</p>
        <div className="matching">
          <div className="match-col">
            {ex.pairs.map((p, i) => {
              const c = value.chosen[i]
              const cls = graded && feedback ? (c === i ? 'ok' : 'bad') : active === i ? 'selected' : ''
              return (
                <button type="button" key={i} className={`match-item ${cls}`} disabled={graded} onClick={() => setActive(i)}>
                  <span className="match-slot">{c !== null ? letterOf(c) : '?'}</span>
                  <Md inline>{t(p.left)}</Md>
                  {graded && feedback && c !== i && <span className="answer-reveal small"> → {letterOf(i)}</span>}
                </button>
              )
            })}
          </div>
          <div className="match-col">
            {value.rightOrder.map((k, idx) => (
              <button type="button" key={k} className={`match-item ${value.chosen.includes(k) ? 'used' : ''}`} disabled={graded} onClick={() => pickRight(k)}>
                <span className="match-slot">{labels[idx]}</span>
                <Md inline>{t(ex.pairs[k].right)}</Md>
              </button>
            ))}
          </div>
        </div>
      </div>
    )
  },
}

/* ---------- predict-output ---------- */
type PredS = { order: number[]; selected: number | null }
const predict: Module<Spec<'predict-output'>, PredS> = {
  init: (ex, rng) => ({ order: shuffle(range(ex.options.length), rng), selected: null }),
  ready: (_, s) => s.selected !== null,
  grade: (ex, s) => ({ correct: s.selected === ex.answer }),
  View: ({ ex, value, onChange, graded, feedback }) => (
    <div>
      <pre className="code-block"><code>{ex.code}</code></pre>
      <div className="options">
        {value.order.map((i) => {
          const sel = value.selected === i
          const cls = graded && feedback ? (ex.answer === i ? 'ok' : sel ? 'bad' : '') : ''
          return (
            <label key={i} className={`option ${sel ? 'selected' : ''} ${cls}`}>
              <input type="radio" checked={sel} disabled={graded} onChange={() => onChange({ ...value, selected: i })} />
              <pre className="code-inline">{ex.options[i]}</pre>
            </label>
          )
        })}
      </div>
    </div>
  ),
}

/* ---------- code ---------- */
type CodeS = { code: string; result: PyResult | null; running: boolean; showSolution: boolean }
function TestOutput({ r }: { r: PyResult }) {
  const { ui } = useLang()
  return (
    <div className={`test-output ${r.ok ? 'ok' : 'bad'}`}>
      {r.total !== undefined && r.total > 0 && (
        <div>
          {ui('tests')}: {r.passed}/{r.total} {ui('passed')}
        </div>
      )}
      {r.stdout && <pre className="stdout">{r.stdout}</pre>}
      {r.error && <pre className="stderr">{r.error}</pre>}
    </div>
  )
}
const code: Module<Spec<'code'>, CodeS> = {
  init: (ex) => ({ code: ex.starter, result: null, running: false, showSolution: false }),
  grade: async (ex, s) => {
    const result = await py.test(s.code, ex.tests)
    return { correct: result.ok, state: { ...s, result, running: false } }
  },
  View: ({ ex, value, onChange, graded }) => {
    const { ui } = useLang()
    const status = usePyStatus()
    const runOnly = async () => {
      onChange({ ...value, running: true })
      const result = await py.run(value.code)
      onChange({ ...value, running: false, result })
    }
    return (
      <div className="code-ex">
        <CodeEditor value={value.code} onChange={(c) => onChange({ ...value, code: c })} onRun={runOnly} />
        <div className="row gap">
          <button type="button" onClick={runOnly} disabled={value.running}><Play size={14} /> {ui('run')}</button>
          <button type="button" className="ghost" onClick={() => onChange({ ...value, code: ex.starter, result: null })} disabled={graded}>{ui('reset')}</button>
          {graded && (
            <button type="button" className="ghost" onClick={() => onChange({ ...value, showSolution: !value.showSolution })}>{ui('solution')}</button>
          )}
          <Link className="btn ghost" to={notebookRoute(`exercises/${ex.id}.ipynb`)}><SquareTerminal size={14} /> {ui('openNotebook')}</Link>
          {status === 'loading' && <span className="muted small">{ui('loadingPython')}</span>}
        </div>
        {value.result && <TestOutput r={value.result} />}
        {value.showSolution && <pre className="code-block">{ex.solution}</pre>}
      </div>
    )
  },
}

/* ---------- multi-step ---------- */
type MultiS = { parts: unknown[] }
const partModules = () => ({ mcq, 'true-false': trueFalse, numeric, matrix }) as unknown as Record<string, Module<PartSpec, unknown>>
const multiStep: Module<Spec<'multi-step'>, MultiS> = {
  init: (ex, rng) => ({ parts: ex.parts.map((p) => partModules()[p.type].init(p, rng)) }),
  ready: (ex, s) => ex.parts.every((p, i) => partModules()[p.type].ready?.(p, s.parts[i]) ?? true),
  grade: (ex, s) => ({
    correct: ex.parts.every((p, i) => (partModules()[p.type].grade(p, s.parts[i]) as GradeOut<unknown>).correct),
  }),
  View: ({ ex, value, onChange, graded, feedback }) => {
    const { t } = useLang()
    return (
      <ol className="parts">
        {ex.parts.map((p, i) => {
          const M = partModules()[p.type]
          const ok = graded && (M.grade(p, value.parts[i]) as GradeOut<unknown>).correct
          return (
            <li key={i} className={graded && feedback ? (ok ? 'ok' : 'bad') : ''}>
              <Md>{t(p.prompt)}</Md>
              <M.View ex={p} value={value.parts[i]} graded={graded} feedback={feedback} onChange={(s) => onChange({ parts: value.parts.map((x, j) => (j === i ? s : x)) })} />
              {graded && feedback && p.explanation && (
                <div className="part-expl"><Md>{t(p.explanation)}</Md></div>
              )}
            </li>
          )
        })}
      </ol>
    )
  },
}

export const modules = {
  mcq, 'true-false': trueFalse, numeric, matrix, construct, 'find-error': findError, parsons, matching,
  'predict-output': predict, code, 'multi-step': multiStep,
} as unknown as Record<Exclude<ExerciseSpec['type'], 'parametric'>, Module<ExerciseSpec, unknown>>
