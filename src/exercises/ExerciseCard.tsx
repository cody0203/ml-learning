import { useMemo, useState } from 'react'
import { Check, Lightbulb, RotateCcw, X } from 'lucide-react'
import type { ExerciseSpec, ExerciseType } from '../content/schema'
import { Md } from '../components/Md'
import { Figure } from '../components/Figure'
import { useLang } from '../lib/i18n'
import { mulberry32 } from '../lib/random'
import { instantiate } from '../lib/generators'
import { modules } from './modules'
import { Link } from 'react-router-dom'
import { formulasByExercise } from '../content'

export const TYPE_LABEL: Record<ExerciseType, [string, string]> = {
  mcq: ['Trắc nghiệm', 'Multiple choice'],
  'true-false': ['Đúng / Sai', 'True / False'],
  numeric: ['Điền số', 'Numeric'],
  matrix: ['Ma trận', 'Matrix'],
  construct: ['Tự xây dựng', 'Construct'],
  'find-error': ['Tìm lỗi sai', 'Find the error'],
  parsons: ['Sắp xếp bước', 'Order the steps'],
  matching: ['Ghép cặp', 'Matching'],
  'predict-output': ['Đoán output', 'Predict output'],
  code: ['Lập trình', 'Code'],
  'multi-step': ['Nhiều bước', 'Multi-step'],
  parametric: ['Luyện số ngẫu nhiên', 'Random drill'],
}

interface Props {
  ex: ExerciseSpec
  seed: number
  /** exam: grade silently, no explanation. */
  exam?: boolean
  onGraded?: (correct: boolean) => void
  footer?: React.ReactNode
  context?: React.ReactNode
}

export function ExerciseCard(props: Props) {
  // Re-mount the inner card on retry / new numbers so all state resets.
  const [nonce, setNonce] = useState(0)
  return <Inner key={`${props.ex.id}-${props.seed}-${nonce}`} {...props} seed={props.seed + nonce * 7919} onRetry={() => setNonce((n) => n + 1)} />
}

function Inner({ ex: raw, seed, exam, onGraded, footer, context, onRetry }: Props & { onRetry: () => void }) {
  const { t, ui, lang } = useLang()
  const ex = useMemo(() => (raw.type === 'parametric' ? instantiate(raw, mulberry32(seed)) : raw), [raw, seed])
  const mod = modules[ex.type as keyof typeof modules]
  const [state, setState] = useState(() => mod.init(ex, mulberry32(seed ^ 0x9e3779b9)))
  const [result, setResult] = useState<boolean | null>(null)
  const [checking, setChecking] = useState(false)
  const [hints, setHints] = useState(0)
  const graded = result !== null
  const ready = mod.ready ? mod.ready(ex, state) : true

  const check = async () => {
    setChecking(true)
    try {
      const out = await mod.grade(ex, state)
      if (out.state !== undefined) setState(out.state)
      setResult(out.correct)
      onGraded?.(out.correct)
    } finally {
      setChecking(false)
    }
  }

  const feedback = !exam
  const related = formulasByExercise.get(raw.id) ?? []
  return (
    <article className={`card exercise ${graded && feedback ? (result ? 'is-ok' : 'is-bad') : ''}`}>
      <header className="ex-head">
        <span className="badge">{TYPE_LABEL[raw.type][lang === 'vi' ? 0 : 1]}</span>
        <span className="diff" title={ui('difficulty')}>{'●'.repeat(ex.difficulty)}{'○'.repeat(3 - ex.difficulty)}</span>
        {context}
        <span className="muted small ex-id">{raw.id}</span>
      </header>
      <div className="prompt">
        <Md>{t(ex.prompt)}</Md>
        {ex.figure && <Figure spec={ex.figure} />}
      </div>
      <mod.View ex={ex} value={state} onChange={setState} graded={graded} feedback={feedback} />

      {ex.hints.slice(0, hints).map((h, i) => (
        <div key={i} className="hint"><Lightbulb size={15} /> <Md inline>{t(h)}</Md></div>
      ))}

      <div className="ex-actions">
        {!graded && (
          <>
            <button type="button" className="primary" onClick={check} disabled={!ready || checking}>
              {checking ? '…' : ex.type === 'code' ? ui('runTests') : ui('check')}
            </button>
            {hints < ex.hints.length && !exam && (
              <button type="button" className="ghost" onClick={() => setHints((h) => h + 1)}><Lightbulb size={15} /> {ui('showHint')} ({hints + 1}/{ex.hints.length})</button>
            )}
          </>
        )}
        {graded && feedback && (
          <>
            <span className={`verdict ${result ? 'ok' : 'bad'}`}>{result ? <Check size={16} /> : <X size={16} />} {result ? ui('correct') : ui('incorrect')}</span>
            <button type="button" className="ghost" onClick={onRetry}><RotateCcw size={14} /> {raw.type === 'parametric' ? ui('newDrill') : ui('retry')}</button>
          </>
        )}
        <span className="spacer" />
        {footer}
      </div>

      {graded && feedback && (
        <div className="explanation">
          <h4>{ui('explanation')}</h4>
          <Md>{t(ex.explanation)}</Md>
          {related.length > 0 && (
            <div className="related-formulas">
              <span className="muted small">{ui('relatedFormulas')}:</span>
              {related.map((f) => (
                <Link key={f.id} className="formula-chip" to={`/week/${f.weekKey}?tab=formulas#${f.id}`} title={f.tex}>
                  <Md inline>{`$${f.tex}$`}</Md>
                  <span className="small">{t(f.name)}</span>
                </Link>
              ))}
            </div>
          )}
        </div>
      )}
    </article>
  )
}
