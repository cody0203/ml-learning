import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { exerciseById, topicTitle, weekByKey, type FormulaItem } from '../content'
import { Md } from './Md'
import { useLang } from '../lib/i18n'
import { mastery, useProgress } from '../lib/progress'
import { encodeConfig } from '../lib/session'
import { TYPE_LABEL } from '../exercises/ExerciseCard'

const fold = (s: string) =>
  s.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/đ/g, 'd').replace(/\\[a-z]+/g, (m) => ` ${m.slice(1)} `)

function haystack(f: FormulaItem) {
  return fold([f.name.vi, f.name.en, f.tex, f.where?.vi, f.where?.en, f.numpy, f.topic].filter(Boolean).join(' '))
}

export const practiceFormulaUrl = (ids: string[]) =>
  `/practice?${encodeConfig({ weeks: [], types: [], mode: 'mixed', n: ids.length, ids })}&go=1`

export function FormulaCard({ f, showWeek }: { f: FormulaItem; showWeek: boolean }) {
  const { t, ui, lang } = useLang()
  const p = useProgress()
  const [open, setOpen] = useState(false)
  const ids = f.exercises.filter((id) => exerciseById.has(id))
  const m = mastery(p, ids)
  const w = weekByKey.get(f.weekKey)
  return (
    <article className="card formula" id={f.id}>
      <header className="formula-head">
        <h4>{t(f.name)}</h4>
        {showWeek && w && <Link className="muted small" to={`/week/${w.key}`}>{t(w.spec.title)}</Link>}
      </header>
      <div className="formula-tex">
        <Md>{`$$\n${f.tex}\n$$`}</Md>
      </div>
      {f.where && <div className="formula-where small"><Md>{t(f.where)}</Md></div>}
      {f.numpy && <code className="formula-numpy">{f.numpy}</code>}
      <footer className="formula-foot">
        <span className={`dot ${m.attempted ? (m.mastery >= 0.8 ? 'ok' : 'bad') : ''}`} />
        <button type="button" className="ghost small" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
          {ids.length} {ui('exercises')} {open ? '▴' : '▾'}
        </button>
        <span className="muted small">{m.attempted ? `${Math.round(m.mastery * 100)}% ${ui('mastery').toLowerCase()}` : ''}</span>
        <span className="spacer" />
        <Link className="btn small primary" to={practiceFormulaUrl(ids)}>{ui('practiceFormula')} →</Link>
      </footer>
      {open && (
        <ul className="ex-list">
          {ids.map((id) => {
            const e = exerciseById.get(id)!
            const a = p.attempts[id]
            return (
              <li key={id}>
                <Link to={`/ex/${id}`} className="ex-link">
                  <span className={`dot ${a ? (a.lastCorrect ? 'ok' : 'bad') : ''}`} />
                  <span className="badge small">{TYPE_LABEL[e.type][lang === 'vi' ? 0 : 1]}</span>
                  <span className="ex-title">{t(e.prompt).replace(/[$#*`>]/g, '').slice(0, 90)}</span>
                </Link>
              </li>
            )
          })}
        </ul>
      )}
    </article>
  )
}

/** Searchable formula sheet, grouped by week and topic. */
export function FormulaList({ formulas, showWeek = false }: { formulas: FormulaItem[]; showWeek?: boolean }) {
  const { t, ui } = useLang()
  const [q, setQ] = useState('')
  const index = useMemo(() => formulas.map((f) => ({ f, h: haystack(f) })), [formulas])
  const terms = fold(q).split(/\s+/).filter(Boolean)
  const shown = terms.length ? index.filter(({ h }) => terms.every((x) => h.includes(x))).map(({ f }) => f) : formulas

  const groups: { key: string; title: string; items: FormulaItem[] }[] = []
  for (const f of shown) {
    const key = `${f.weekKey}/${f.topic}`
    let g = groups.find((x) => x.key === key)
    if (!g) {
      const wt = showWeek ? `${t(weekByKey.get(f.weekKey)?.spec.title)} › ` : ''
      g = { key, title: wt + t(topicTitle(f.weekKey, f.topic)), items: [] }
      groups.push(g)
    }
    g.items.push(f)
  }

  return (
    <div className="formula-sheet">
      <div className="formula-search">
        <input
          type="search"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder={ui('searchFormula')}
          aria-label={ui('searchFormula')}
          autoFocus
        />
        <span className="muted small">{shown.length}/{formulas.length}</span>
      </div>
      {!shown.length && <div className="card empty">{ui('noFormula')}</div>}
      {groups.map((g) => (
        <section key={g.key} className="formula-group">
          <h3>{g.title}</h3>
          <div className="formula-grid">
            {g.items.map((f) => (
              <FormulaCard key={f.id} f={f} showWeek={false} />
            ))}
          </div>
        </section>
      ))}
    </div>
  )
}
