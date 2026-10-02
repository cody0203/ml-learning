import { useEffect, useMemo, useRef, useState } from 'react'
import { allFlashcards, catalog, exerciseById } from '../content'
import { Md } from '../components/Md'
import { ExerciseCard } from '../exercises/ExerciseCard'
import { useLang } from '../lib/i18n'
import { progress, useProgress } from '../lib/progress'
import { shuffle } from '../lib/random'
import { isDue, type Grade } from '../lib/srs'
import type { FlashcardKindT } from '../content/schema'

export const KIND_LABEL: Record<FlashcardKindT, [string, string]> = {
  concept: ['Lý thuyết', 'Concept'],
  formula: ['Công thức', 'Formula'],
  compute: ['Tính nhanh', 'Quick calc'],
}

const NEW_PER_SESSION = 20

export default function Flashcards() {
  const { t, ui, lang } = useLang()
  const p = useProgress()
  const [week, setWeek] = useState<string>('')
  const [kind, setKind] = useState<'' | FlashcardKindT>('')
  const [extra, setExtra] = useState(0)

  // Queue is computed once per filter change so grading doesn't reshuffle mid-session.
  const queue = useMemo(() => {
    const st = progress.get()
    const inWeek = (wk: string) => !week || wk === week
    const cardOk = (id: string) => {
      const f = allFlashcards.find((x) => x.id === id)
      return !!f && inWeek(f.weekKey) && (!kind || f.kind === kind)
    }
    const due = Object.entries(st.cards)
      .filter(([k, s]) => {
        if (!isDue(s)) return false
        const id = k.slice(2)
        if (k.startsWith('f:')) return cardOk(id)
        const wk = exerciseById.get(id)?.weekKey
        return !kind && wk !== undefined && inWeek(wk)
      })
      .sort((a, b) => a[1].due - b[1].due)
      .map(([k]) => k)
    // Shuffle new cards so concept and formula cards are interleaved.
    const fresh = shuffle(allFlashcards.filter((f) => cardOk(f.id) && !st.cards[`f:${f.id}`]).map((f) => `f:${f.id}`))
    let q = [...due, ...fresh.slice(0, NEW_PER_SESSION)]
    if (!q.length && extra) q = shuffle(allFlashcards.filter((f) => cardOk(f.id)).map((f) => `f:${f.id}`)).slice(0, 20)
    return q
  }, [week, kind, extra])

  const [pos, setPos] = useState(0)
  const [flipped, setFlipped] = useState(false)
  const key = queue[pos]

  const grade = (g: Grade) => {
    progress.reviewCard(key, g)
    setFlipped(false)
    setPos((x) => x + 1)
  }

  const header = (
    <div className="page-head">
      <h1>{ui('flashcards')}</h1>
      <select value={week} onChange={(e) => { setWeek(e.target.value); setPos(0) }}>
        <option value="">{ui('allWeeks')}</option>
        {catalog.courses.map((c) => (
          <optgroup key={c.spec.id} label={t(c.spec.title)}>
            {c.weeks.map((w) => (
              <option key={w.key} value={w.key}>{t(w.spec.title)}</option>
            ))}
          </optgroup>
        ))}
      </select>
      <div className="chips fc-kinds">
        {(['', 'concept', 'formula', 'compute'] as const).map((k) => (
          <button type="button" key={k} className={`chip ${kind === k ? 'on' : ''}`} onClick={() => { setKind(k); setPos(0); setFlipped(false) }}>
            {k ? KIND_LABEL[k][lang === 'vi' ? 0 : 1] : ui('allTypes')}
          </button>
        ))}
      </div>
    </div>
  )

  if (!key)
    return (
      <div className="page narrow">
        {header}
        <div className="card empty">
          <p>{ui('noDue')}</p>
          <button type="button" onClick={() => { setExtra((x) => x + 1); setPos(0) }}>{ui('studyAhead')}</button>
        </div>
      </div>
    )

  const counter = <p className="muted small">{pos + 1} / {queue.length}</p>

  if (key.startsWith('e:')) {
    const ex = exerciseById.get(key.slice(2))!
    return (
      <div className="page narrow">
        {header}
        {counter}
        <ExerciseCard
          ex={ex}
          seed={pos * 7919 + 13}
          context={<span className="badge warn">{ui('mistakes')}</span>}
          onGraded={(ok) => progress.recordAttempt(ex.id, ok)}
          footer={<button type="button" className="primary" onClick={() => setPos(pos + 1)}>{ui('next')} →</button>}
        />
      </div>
    )
  }

  const card = allFlashcards.find((f) => f.id === key.slice(2))!
  const s = p.cards[key]
  return (
    <div className="page narrow">
      {header}
      {counter}
      <div className={`card flashcard ${flipped ? 'flipped' : ''}`} onClick={() => setFlipped(true)}>
        <div className="fc-badges">
          <span className={`badge kind-${card.kind}`}>{KIND_LABEL[card.kind][lang === 'vi' ? 0 : 1]}</span>
          {!s && <span className="badge new">new</span>}
        </div>
        <div className="fc-front"><Md>{t(card.front)}</Md></div>
        {flipped ? (
          <div className="fc-back"><Md>{t(card.back)}</Md></div>
        ) : (
          <button type="button" className="primary" onClick={() => setFlipped(true)}>{ui('flip')} (space)</button>
        )}
      </div>
      {flipped && (
        <div className="grade-row">
          <button type="button" className="g-again" onClick={() => grade('again')}>{ui('again')}</button>
          <button type="button" className="g-hard" onClick={() => grade('hard')}>{ui('hard')}</button>
          <button type="button" className="g-good" onClick={() => grade('good')}>{ui('good')}</button>
          <button type="button" className="g-easy" onClick={() => grade('easy')}>{ui('easy')}</button>
        </div>
      )}
      <KeyHandler onKey={(k) => {
        if (!flipped && (k === ' ' || k === 'Enter')) setFlipped(true)
        else if (flipped) {
          const map: Record<string, Grade> = { '1': 'again', '2': 'hard', '3': 'good', '4': 'easy' }
          if (map[k]) grade(map[k])
        }
      }} />
    </div>
  )
}

function KeyHandler({ onKey }: { onKey: (k: string) => void }) {
  useKey(onKey)
  return null
}

function useKey(fn: (k: string) => void) {
  const ref = useRef(fn)
  useEffect(() => {
    ref.current = fn
  })
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT' || (e.target as HTMLElement)?.tagName === 'TEXTAREA') return
      if (e.key === ' ') e.preventDefault()
      ref.current(e.key)
    }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [])
}
