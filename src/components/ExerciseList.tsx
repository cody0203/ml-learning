import { Link } from 'react-router-dom'
import type { ExerciseSpec } from '../content/schema'
import { useLang } from '../lib/i18n'
import { useProgress } from '../lib/progress'
import { TYPE_LABEL } from '../exercises/ExerciseCard'

export function ExerciseList({ exercises }: { exercises: ExerciseSpec[] }) {
  const { t, lang } = useLang()
  const p = useProgress()
  return (
    <ul className="ex-list">
      {exercises.map((e) => {
        const a = p.attempts[e.id]
        return (
          <li key={e.id}>
            <Link to={`/ex/${e.id}`} className="ex-link">
              <span className={`dot ${a ? (a.lastCorrect ? 'ok' : 'bad') : ''}`} title={a ? (a.lastCorrect ? '✓' : '✗') : ''} />
              <span className="badge small">{TYPE_LABEL[e.type][lang === 'vi' ? 0 : 1]}</span>
              <span className="ex-title">{t(e.prompt).replace(/[$#*`>]/g, '').slice(0, 110)}</span>
              <span className="diff" aria-label={`difficulty ${e.difficulty}`}>{'●'.repeat(e.difficulty)}</span>
            </Link>
          </li>
        )
      })}
    </ul>
  )
}
