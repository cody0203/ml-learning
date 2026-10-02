import { useEffect, useRef, useState } from 'react'
import { ChevronRight, FileText } from 'lucide-react'
import { Link, useLocation, useParams } from 'react-router-dom'
import { weekByKey } from '../content'
import { Md } from '../components/Md'
import { Bar } from '../components/Bar'
import { Toc } from '../components/Toc'
import { FormulaList } from '../components/FormulaList'
import { useLang } from '../lib/i18n'
import { mastery, useProgress } from '../lib/progress'
import { encodeConfig } from '../lib/session'

type Tab = 'topics' | 'notes' | 'formulas' | 'cards'

export const topicPath = (weekKey: string, topicId: string) => `/week/${weekKey}/t/${topicId}`

export default function WeekPage() {
  const { course, week } = useParams()
  const w = weekByKey.get(`${course}/${week}`)
  const { t, ui, lang } = useLang()
  const p = useProgress()
  const loc = useLocation()
  const raw = new URLSearchParams(loc.search).get('tab')
  // Old links used ?tab=exercises; exercises now live on topic pages.
  const urlTab: Tab = raw === 'notes' || raw === 'formulas' || raw === 'cards' ? raw : 'topics'
  const [tab, setTab] = useState<Tab>(urlTab)
  const [prevUrl, setPrevUrl] = useState(loc.search + loc.hash)
  if (prevUrl !== loc.search + loc.hash) {
    setPrevUrl(loc.search + loc.hash)
    setTab(urlTab)
  }
  useEffect(() => {
    if (tab !== 'formulas' || !loc.hash) return
    const el = document.getElementById(decodeURIComponent(loc.hash.slice(1)))
    if (!el) return
    el.scrollIntoView({ block: 'center' })
    el.classList.add('flash')
    const id = setTimeout(() => el.classList.remove('flash'), 1600)
    return () => clearTimeout(id)
  }, [tab, loc.hash])
  const notesRef = useRef<HTMLDivElement>(null)
  if (!w) return <p>404</p>

  const m = mastery(p, w.exercises.map((e) => e.id))
  const vi = lang === 'vi'
  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>{t(w.spec.title)}</h1>
          {w.spec.source && <p className="muted small source"><FileText size={13} /> {w.spec.source}</p>}
        </div>
        <div className="row gap wrap">
          <Link className="btn primary" to={`/practice?${encodeConfig({ weeks: [w.key], types: [], mode: 'weak', n: 10 })}&go=1`}>{ui('practiceWeek')}</Link>
          <Link className="btn" to={`/practice?${encodeConfig({ weeks: [w.key], types: [], mode: 'exam', n: 15 })}`}>{ui('modeExam')}</Link>
        </div>
      </header>
      <div className="row gap center mastery-line">
        <Bar value={m.mastery} />
        <span className="small nowrap muted">{ui('mastery')} {Math.round(m.mastery * 100)}% · {m.attempted}/{w.exercises.length} {ui('exercises')}</span>
      </div>

      <div className="tabs" role="tablist">
        <button type="button" role="tab" aria-selected={tab === 'topics'} className={tab === 'topics' ? 'on' : ''} onClick={() => setTab('topics')}>{ui('topics')} ({w.spec.topics.length})</button>
        <button type="button" role="tab" aria-selected={tab === 'notes'} className={tab === 'notes' ? 'on' : ''} onClick={() => setTab('notes')}>{vi ? 'Ghi chú đầy đủ' : 'Full notes'}</button>
        <button type="button" role="tab" aria-selected={tab === 'formulas'} className={tab === 'formulas' ? 'on' : ''} onClick={() => setTab('formulas')}>{ui('formulas')} ({w.formulas.length})</button>
        <button type="button" role="tab" aria-selected={tab === 'cards'} className={tab === 'cards' ? 'on' : ''} onClick={() => setTab('cards')}>{ui('flashcards')} ({w.flashcards.length})</button>
      </div>

      {tab === 'topics' && (
        <ol className="topic-rows">
          {w.spec.topics.map((tp, i) => {
            const ids = w.exercises.filter((e) => e.topic === tp.id).map((e) => e.id)
            const nf = w.formulas.filter((f) => f.topic === tp.id).length
            const tm = mastery(p, ids)
            return (
              <li key={tp.id}>
                <Link to={topicPath(w.key, tp.id)} className="topic-row">
                  <span className="topic-num">{i + 1}</span>
                  <span className="week-main">
                    <span className="week-title">{t(tp.title)}</span>
                    <span className="week-meta">
                      {ids.length} {ui('exercises')}{nf ? ` · ${nf} ${ui('formulas').toLowerCase()}` : ''}
                      {tm.attempted ? ` · ${vi ? 'đã làm' : 'done'} ${tm.attempted}/${ids.length}` : ''}
                    </span>
                  </span>
                  <span className="week-progress">
                    <Bar value={tm.mastery} />
                    <span className="pct">{Math.round(tm.mastery * 100)}%</span>
                  </span>
                  <ChevronRight size={16} className="row-chev" />
                </Link>
              </li>
            )
          })}
        </ol>
      )}

      {tab === 'notes' && (
        <div className="notes-layout">
          <div className="notes" ref={notesRef}>
            <Md doc>{t(w.notes)}</Md>
          </div>
          <aside className="notes-side">
            <p className="toc-title">{ui('topics')}</p>
            <Toc root={notesRef} levels={['H2']} deps={[lang, w.key]} />
          </aside>
        </div>
      )}

      {tab === 'formulas' && <FormulaList key={w.key} formulas={w.formulas.map((f) => ({ ...f, weekKey: w.key }))} />}

      {tab === 'cards' && (
        <div className="card-list">
          {w.flashcards.map((f) => (
            <div key={f.id} className="card mini-card">
              <div className="mini-front"><Md>{t(f.front)}</Md></div>
              <div className="mini-back"><Md>{t(f.back)}</Md></div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
