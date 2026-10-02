import { Link, useNavigate } from 'react-router-dom'
import { RotateCcw } from 'lucide-react'
import { exerciseById, topicTitle } from '../content'
import { useLang } from '../lib/i18n'
import { progress, useProgress } from '../lib/progress'
import { encodeConfig } from '../lib/session'
import { TYPE_LABEL } from '../exercises/ExerciseCard'

export default function Mistakes() {
  const { t, ui, lang } = useLang()
  const p = useProgress()
  const nav = useNavigate()
  const items = Object.entries(p.mistakes)
    .filter(([id]) => exerciseById.has(id))
    .sort((a, b) => b[1].at - a[1].at)

  return (
    <div className="page narrow">
      <div className="page-head">
        <h1>{ui('mistakes')}</h1>
        {items.length > 0 && (
          <button type="button" className="primary" onClick={() => nav(`/practice?${encodeConfig({ weeks: [], types: [], mode: 'mixed', n: items.length, ids: items.map(([id]) => id) })}&go=1`)}>
            <RotateCcw size={14} /> {ui('practiceMistakes')}
          </button>
        )}
      </div>
      {!items.length && <div className="card empty">{ui('noMistakes')}</div>}
      <ul className="list">
        {items.map(([id, m]) => {
          const e = exerciseById.get(id)!
          const a = p.attempts[id]
          return (
            <li key={id} className="list-row">
              <span className={`dot ${a?.lastCorrect ? 'ok' : 'bad'}`} />
              <div className="grow">
                <Link to={`/ex/${id}`}>{t(e.prompt).replace(/[$#*`>]/g, '').slice(0, 110)}</Link>
                <div className="muted small">
                  {TYPE_LABEL[e.type][lang === 'vi' ? 0 : 1]} · {t(topicTitle(e.weekKey, e.topic))} · ✗×{m.count} · {new Date(m.at).toLocaleDateString()}
                </div>
              </div>
              <button type="button" className="ghost small" onClick={() => progress.clearMistake(id)}>{ui('clear')}</button>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
