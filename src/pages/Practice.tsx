import { useEffect, useMemo, useState } from 'react'
import { RotateCcw, Timer as TimerIcon } from 'lucide-react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { allWeeks, exerciseById, catalog } from '../content'
import type { ExerciseType } from '../content/schema'
import { ExerciseCard, TYPE_LABEL } from '../exercises/ExerciseCard'
import { useLang } from '../lib/i18n'
import { progress } from '../lib/progress'
import { buildQueue, decodeConfig, encodeConfig, pool, type Mode, type QueueItem, type SessionConfig } from '../lib/session'

const ALL_TYPES = Object.keys(TYPE_LABEL) as ExerciseType[]

function Setup({ initial }: { initial: SessionConfig }) {
  const { t, ui, lang } = useLang()
  const nav = useNavigate()
  const [cfg, setCfg] = useState<SessionConfig>({ ...initial, ids: [] })
  const toggle = <T,>(xs: T[], x: T) => (xs.includes(x) ? xs.filter((y) => y !== x) : [...xs, x])
  const singleWeek = cfg.weeks.length === 1 ? allWeeks.find((w) => w.key === cfg.weeks[0]) : undefined
  const available = pool(cfg).length
  const modes: [Mode, string][] = [['weak', ui('modeWeak')], ['new', ui('modeNew')], ['mixed', ui('modeMixed')], ['exam', ui('modeExam')]]

  return (
    <div className="page narrow">
      <h1>{ui('practice')}</h1>
      <div className="card form">
        <fieldset>
          <legend>{ui('scope')}</legend>
          <div className="chips">
            <button type="button" className={`chip ${cfg.weeks.length === 0 ? 'on' : ''}`} onClick={() => setCfg({ ...cfg, weeks: [], topic: undefined })}>{ui('allWeeks')}</button>
          </div>
          {catalog.courses.map((c) => {
            const keys = c.weeks.map((w) => w.key)
            const allOn = keys.every((k) => cfg.weeks.includes(k))
            return (
              <div key={c.spec.id} className="course-chips">
                <button type="button" className={`chip course ${allOn ? 'on' : ''}`} onClick={() => setCfg({ ...cfg, topic: undefined, weeks: allOn ? cfg.weeks.filter((k) => !keys.includes(k)) : [...new Set([...cfg.weeks, ...keys])] })}>
                  {t(c.spec.title)}
                </button>
                <div className="chips">
                  {c.weeks.map((w) => (
                    <button type="button" key={w.key} className={`chip ${cfg.weeks.includes(w.key) ? 'on' : ''}`} onClick={() => setCfg({ ...cfg, weeks: toggle(cfg.weeks, w.key), topic: undefined })}>
                      {t(w.spec.title)}
                    </button>
                  ))}
                </div>
              </div>
            )
          })}
          {singleWeek && (
            <select value={cfg.topic ?? ''} onChange={(e) => setCfg({ ...cfg, topic: e.target.value || undefined })}>
              <option value="">{ui('allTopics')}</option>
              {singleWeek.spec.topics.map((tp) => (
                <option key={tp.id} value={tp.id}>{t(tp.title)}</option>
              ))}
            </select>
          )}
        </fieldset>
        <fieldset>
          <legend>{ui('types')}</legend>
          <div className="chips">
            <button type="button" className={`chip ${cfg.types.length === 0 ? 'on' : ''}`} onClick={() => setCfg({ ...cfg, types: [] })}>{ui('allTypes')}</button>
            {ALL_TYPES.map((ty) => (
              <button type="button" key={ty} className={`chip ${cfg.types.includes(ty) ? 'on' : ''}`} onClick={() => setCfg({ ...cfg, types: toggle(cfg.types, ty) })}>
                {TYPE_LABEL[ty][lang === 'vi' ? 0 : 1]}
              </button>
            ))}
          </div>
        </fieldset>
        <fieldset>
          <legend>{ui('difficulty')}</legend>
          <div className="chips">
            {[1, 2, 3].map((d) => (
              <button type="button" key={d} className={`chip ${cfg.difficulty?.includes(d) ? 'on' : ''}`} onClick={() => setCfg({ ...cfg, difficulty: toggle(cfg.difficulty ?? [], d) })}>
                {'●'.repeat(d)}
              </button>
            ))}
          </div>
        </fieldset>
        <fieldset>
          <legend>{ui('mode')}</legend>
          <div className="chips">
            {modes.map(([m, label]) => (
              <button type="button" key={m} className={`chip ${cfg.mode === m ? 'on' : ''}`} onClick={() => setCfg({ ...cfg, mode: m })}>{label}</button>
            ))}
          </div>
        </fieldset>
        <fieldset>
          <legend>{ui('count')}</legend>
          <div className="chips">
            {[5, 10, 20, 30].map((n) => (
              <button type="button" key={n} className={`chip ${cfg.n === n ? 'on' : ''}`} onClick={() => setCfg({ ...cfg, n })}>{n}</button>
            ))}
          </div>
        </fieldset>
        <div className="row gap center">
          <button type="button" className="primary" disabled={!available} onClick={() => nav(`/practice?${encodeConfig(cfg)}&go=1`)}>
            {ui('start')} →
          </button>
          <span className="muted small">{available} {ui('exercises')}</span>
        </div>
      </div>
    </div>
  )
}

function Timer({ seconds, onEnd }: { seconds: number; onEnd: () => void }) {
  const [left, setLeft] = useState(seconds)
  useEffect(() => {
    if (left <= 0) {
      onEnd()
      return
    }
    const id = setTimeout(() => setLeft((l) => l - 1), 1000)
    return () => clearTimeout(id)
  }, [left, onEnd])
  return <span className={`timer ${left < 60 ? 'warn' : ''}`}><TimerIcon size={14} /> {Math.floor(left / 60)}:{String(left % 60).padStart(2, '0')}</span>
}

function Session({ cfg }: { cfg: SessionConfig }) {
  const { ui, t } = useLang()
  const nav = useNavigate()
  const [queue] = useState<QueueItem[]>(() => buildQueue(cfg, progress.get()))
  const [i, setI] = useState(0)
  const [results, setResults] = useState<Record<number, boolean>>({})
  const [done, setDone] = useState(false)
  const exam = cfg.mode === 'exam'

  if (!queue.length) return <p className="page">{ui('noExercises')}</p>

  if (done || i >= queue.length) {
    const score = Object.values(results).filter(Boolean).length
    const wrong = queue.filter((_, k) => results[k] === false || results[k] === undefined).map((q) => q.id)
    return (
      <div className="page narrow">
        <h1>{ui('result')}</h1>
        <div className="card score-card">
          <div className="big-score">{score}/{queue.length}</div>
          <p className="muted">{ui('score')}: {Math.round((score / queue.length) * 100)}%</p>
        </div>
        <ol className="review-list">
          {queue.map((q, k) => {
            const e = exerciseById.get(q.id)!
            return (
              <li key={k} className={results[k] ? 'ok' : 'bad'}>
                <span>{results[k] ? '✓' : results[k] === false ? '✗' : '–'}</span>
                <Link to={`/ex/${q.id}`}>{t(e.prompt).replace(/[$#*`>]/g, '').slice(0, 100)}</Link>
              </li>
            )
          })}
        </ol>
        <div className="row gap wrap">
          {wrong.length > 0 && (
            <button type="button" className="primary" onClick={() => nav(`/practice?${encodeConfig({ ...cfg, mode: 'mixed', ids: [...new Set(wrong)] })}&go=1&r=${Date.now()}`)}>
              <RotateCcw size={14} /> {ui('practiceMistakes')} ({wrong.length})
            </button>
          )}
          <button type="button" onClick={() => nav(`/practice?${encodeConfig({ ...cfg, ids: [] })}&go=1&r=${Date.now()}`)}>{ui('retry')}</button>
          <Link className="btn ghost" to="/practice">{ui('practice')}</Link>
        </div>
      </div>
    )
  }

  const item = queue[i]
  const ex = exerciseById.get(item.id)!
  const graded = results[i] !== undefined
  return (
    <div className="page narrow">
      <div className="session-head">
        <span>{i + 1} / {queue.length}</span>
        <div className="bar thin"><div style={{ transform: `scaleX(${i / queue.length})` }} /></div>
        {exam && <Timer seconds={queue.length * 90} onEnd={() => setDone(true)} />}
        <button type="button" className="ghost small" onClick={() => setDone(true)}>{ui('finish')}</button>
      </div>
      <ExerciseCard
        key={i}
        ex={ex}
        seed={item.seed}
        exam={exam}
        onGraded={(ok) => {
          progress.recordAttempt(ex.id, ok)
          setResults((r) => ({ ...r, [i]: ok }))
          if (exam) setTimeout(() => setI((x) => x + 1), 250)
        }}
        footer={
          !exam && (
            <button type="button" className={graded ? 'primary' : 'ghost'} onClick={() => setI(i + 1)}>
              {graded ? ui('next') : ui('skip')} →
            </button>
          )
        }
      />
    </div>
  )
}

export default function Practice() {
  const loc = useLocation()
  const q = useMemo(() => new URLSearchParams(loc.search), [loc.search])
  const cfg = useMemo(() => decodeConfig(q), [q])
  if (q.get('go') === '1') return <Session key={loc.search} cfg={cfg} />
  return <Setup key={loc.search} initial={cfg} />
}
