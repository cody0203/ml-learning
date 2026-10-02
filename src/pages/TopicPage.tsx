import { useMemo, useRef } from 'react'
import { ArrowLeft, ArrowRight, Target } from 'lucide-react'
import { Link, useParams } from 'react-router-dom'
import { weekByKey } from '../content'
import { splitNotes } from '../content/notes'
import { Md } from '../components/Md'
import { Bar } from '../components/Bar'
import { Toc } from '../components/Toc'
import { FormulaCard } from '../components/FormulaList'
import { ExerciseList } from '../components/ExerciseList'
import { useLang } from '../lib/i18n'
import { mastery, useProgress } from '../lib/progress'
import { encodeConfig } from '../lib/session'
import { topicPath } from './WeekPage'

/** Promote headings by one level so a topic's `###` subsections read as the page's sections. */
const promote = (md: string) => {
  let fence = false
  return md
    .split('\n')
    .map((l) => {
      if (/^\s*(```|~~~)/.test(l)) fence = !fence
      return !fence && /^#{3,6}\s/.test(l) ? l.slice(1) : l
    })
    .join('\n')
}

export default function TopicPage() {
  const { course, week, topic } = useParams()
  const w = weekByKey.get(`${course}/${week}`)
  const { t, ui, lang } = useLang()
  const p = useProgress()
  const notesRef = useRef<HTMLDivElement>(null)
  const idx = w ? w.spec.topics.findIndex((x) => x.id === topic) : -1
  const body = useMemo(() => {
    if (!w || idx < 0) return ''
    const s = splitNotes(t(w.notes)).sections[idx]
    return s ? promote(s.body) : ''
  }, [w, idx, t])
  if (!w || idx < 0) return <p>404</p>

  const tp = w.spec.topics[idx]
  const prev = w.spec.topics[idx - 1]
  const next = w.spec.topics[idx + 1]
  const exs = w.exercises.filter((e) => e.topic === tp.id)
  const formulas = w.formulas.filter((f) => f.topic === tp.id).map((f) => ({ ...f, weekKey: w.key }))
  const m = mastery(p, exs.map((e) => e.id))
  const vi = lang === 'vi'
  const practice = (mode: 'new' | 'weak') =>
    `/practice?${encodeConfig({ weeks: [w.key], topic: tp.id, types: [], mode, n: Math.min(10, exs.length) })}&go=1`

  return (
    <div className="page">
      <p className="crumbs">
        <Link to={`/week/${w.key}`}>{t(w.spec.title)}</Link>
        <span aria-hidden> / </span>
        {vi ? 'Chủ đề' : 'Topic'} {idx + 1}/{w.spec.topics.length}
      </p>
      <header className="page-head">
        <h1>{t(tp.title)}</h1>
        {exs.length > 0 && (
          <div className="row gap wrap">
            <Link className="btn primary" to={practice(m.attempted < exs.length ? 'new' : 'weak')}>
              <Target size={15} /> {ui('practiceTopic')}
            </Link>
          </div>
        )}
      </header>
      <div className="row gap center mastery-line">
        <Bar value={m.mastery} />
        <span className="small nowrap muted">{ui('mastery')} {Math.round(m.mastery * 100)}% · {m.attempted}/{exs.length} {ui('exercises')}</span>
      </div>

      <nav className="section-jump" aria-label={vi ? 'Trong trang' : 'On this page'}>
        <a href="#notes" onClick={(e) => { e.preventDefault(); document.getElementById('topic-notes')?.scrollIntoView({ behavior: 'smooth' }) }}>{ui('notes')}</a>
        {formulas.length > 0 && <a href="#formulas" onClick={(e) => { e.preventDefault(); document.getElementById('topic-formulas')?.scrollIntoView({ behavior: 'smooth' }) }}>{ui('formulas')} <span className="muted">{formulas.length}</span></a>}
        {exs.length > 0 && <a href="#practice" onClick={(e) => { e.preventDefault(); document.getElementById('topic-practice')?.scrollIntoView({ behavior: 'smooth' }) }}>{ui('practice')} <span className="muted">{exs.length}</span></a>}
      </nav>

      <div className="notes-layout">
        <div className="topic-main">
          <section id="topic-notes" className="notes" ref={notesRef}>
            <Md doc key={`${lang}-${tp.id}`}>{body}</Md>
          </section>

          {formulas.length > 0 && (
            <section id="topic-formulas" className="topic-section">
              <h2>{ui('formulas')}</h2>
              <div className="formula-grid">
                {formulas.map((f) => <FormulaCard key={f.id} f={f} showWeek={false} />)}
              </div>
            </section>
          )}

          {exs.length > 0 && (
            <section id="topic-practice" className="topic-section">
              <div className="section-head">
                <h2>{ui('practice')}</h2>
                <span className="spacer" />
                {m.attempted > 0 && <Link className="btn small" to={practice('weak')}>{ui('modeWeak')}</Link>}
                <Link className="btn small primary" to={practice('new')}>{ui('modeNew')}</Link>
              </div>
              <div className="card topic-ex">
                <ExerciseList exercises={exs} />
              </div>
            </section>
          )}

          <nav className="pager">
            {prev ? (
              <Link to={topicPath(w.key, prev.id)} className="pager-link">
                <span className="muted small"><ArrowLeft size={13} /> {vi ? 'Trước' : 'Previous'}</span>
                <span>{t(prev.title)}</span>
              </Link>
            ) : <span />}
            {next ? (
              <Link to={topicPath(w.key, next.id)} className="pager-link next">
                <span className="muted small">{vi ? 'Tiếp theo' : 'Next'} <ArrowRight size={13} /></span>
                <span>{t(next.title)}</span>
              </Link>
            ) : (
              <Link to={`/week/${w.key}`} className="pager-link next">
                <span className="muted small">{vi ? 'Xong tuần' : 'Week done'} <ArrowRight size={13} /></span>
                <span>{t(w.spec.title)}</span>
              </Link>
            )}
          </nav>
        </div>
        <aside className="notes-side">
          <p className="toc-title">{vi ? 'Trong chủ đề này' : 'In this topic'}</p>
          <Toc root={notesRef} levels={['H2', 'H3']} deps={[lang, tp.id]} />
        </aside>
      </div>
    </div>
  )
}
