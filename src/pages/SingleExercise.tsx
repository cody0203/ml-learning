import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { allExercises, exerciseById, topicTitle, weekByKey } from '../content'
import { ExerciseCard } from '../exercises/ExerciseCard'
import { useLang } from '../lib/i18n'
import { progress } from '../lib/progress'
import { topicPath } from './WeekPage'

export default function SingleExercise() {
  const { id } = useParams()
  const { t, ui } = useLang()
  const nav = useNavigate()
  const [seed] = useState(() => Math.floor(Math.random() * 1e6))
  const ex = exerciseById.get(id ?? '')
  if (!ex) return <p>404</p>
  // Step through the exercises of the same topic, in the order shown on the topic page.
  const siblings = allExercises.filter((e) => e.weekKey === ex.weekKey && e.topic === ex.topic)
  const idx = siblings.findIndex((e) => e.id === ex.id)
  const w = weekByKey.get(ex.weekKey)!
  return (
    <div className="page narrow">
      <p className="crumbs">
        <Link to={`/week/${w.key}`}>{t(w.spec.title)}</Link>
        <span aria-hidden> / </span>
        <Link to={topicPath(w.key, ex.topic)}>{t(topicTitle(ex.weekKey, ex.topic))}</Link>
        <span aria-hidden> / </span>
        {idx + 1}/{siblings.length}
      </p>
      <ExerciseCard
        ex={ex}
        seed={seed}
        onGraded={(ok) => progress.recordAttempt(ex.id, ok)}
        footer={
          <>
            {idx > 0 && <button type="button" className="ghost" onClick={() => nav(`/ex/${siblings[idx - 1].id}`)}>←</button>}
            {idx < siblings.length - 1 && (
              <button type="button" className="ghost" onClick={() => nav(`/ex/${siblings[idx + 1].id}`)}>{ui('next')} →</button>
            )}
          </>
        }
      />
    </div>
  )
}
