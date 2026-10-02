import { useSyncExternalStore } from 'react'
import { newSrs, review, type Grade, type SrsState } from './srs'

export interface Attempt {
  n: number
  correct: number
  lastCorrect: boolean
  lastAt: number
}

export interface Progress {
  version: 1
  attempts: Record<string, Attempt>
  /** SRS state for flashcards (`f:<id>`) and missed exercises (`e:<id>`). */
  cards: Record<string, SrsState>
  mistakes: Record<string, { at: number; count: number }>
  /** yyyy-mm-dd → number of answered items. */
  activity: Record<string, number>
}

const KEY = 'ml-learning-progress'
const empty = (): Progress => ({ version: 1, attempts: {}, cards: {}, mistakes: {}, activity: {} })

function load(): Progress {
  try {
    const raw = localStorage.getItem(KEY)
    if (raw) return { ...empty(), ...JSON.parse(raw) }
  } catch {
    /* corrupted storage: start fresh */
  }
  return empty()
}

let state = load()
const listeners = new Set<() => void>()

function commit(next: Progress) {
  state = next
  localStorage.setItem(KEY, JSON.stringify(state))
  listeners.forEach((l) => l())
}

export const today = (d = new Date()) =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`

function bump(p: Progress): Progress['activity'] {
  const k = today()
  return { ...p.activity, [k]: (p.activity[k] ?? 0) + 1 }
}

export const progress = {
  get: () => state,
  subscribe(l: () => void) {
    listeners.add(l)
    return () => listeners.delete(l)
  },
  recordAttempt(id: string, correct: boolean) {
    const a = state.attempts[id] ?? { n: 0, correct: 0, lastCorrect: false, lastAt: 0 }
    const mistakes = { ...state.mistakes }
    const cards = { ...state.cards }
    if (!correct) {
      mistakes[id] = { at: Date.now(), count: (mistakes[id]?.count ?? 0) + 1 }
      // A missed exercise joins the review queue, due again soon.
      cards[`e:${id}`] = review(cards[`e:${id}`] ?? newSrs(), 'again')
    } else if (cards[`e:${id}`]) {
      cards[`e:${id}`] = review(cards[`e:${id}`], 'good')
    }
    commit({
      ...state,
      attempts: { ...state.attempts, [id]: { n: a.n + 1, correct: a.correct + (correct ? 1 : 0), lastCorrect: correct, lastAt: Date.now() } },
      mistakes,
      cards,
      activity: bump(state),
    })
  },
  reviewCard(cardKey: string, grade: Grade) {
    commit({
      ...state,
      cards: { ...state.cards, [cardKey]: review(state.cards[cardKey] ?? newSrs(), grade) },
      activity: bump(state),
    })
  },
  clearMistake(id: string) {
    const mistakes = { ...state.mistakes }
    delete mistakes[id]
    commit({ ...state, mistakes })
  },
  export: () => JSON.stringify(state, null, 1),
  import(json: string) {
    const data = JSON.parse(json)
    if (data?.version !== 1 || typeof data.attempts !== 'object') throw new Error('Invalid progress file')
    commit({ ...empty(), ...data })
  },
  reset: () => commit(empty()),
}

export const useProgress = () => useSyncExternalStore(progress.subscribe, progress.get)

export function streak(activity: Record<string, number>): number {
  let n = 0
  const d = new Date()
  if (!activity[today(d)]) d.setDate(d.getDate() - 1)
  while (activity[today(d)]) {
    n++
    d.setDate(d.getDate() - 1)
  }
  return n
}

/** Mastery of a set of exercise ids in [0,1]: last-attempt correctness, unseen counts as 0. */
export function mastery(p: Progress, ids: string[]): { mastery: number; attempted: number; accuracy: number } {
  if (!ids.length) return { mastery: 0, attempted: 0, accuracy: 0 }
  let good = 0
  let attempted = 0
  let n = 0
  let c = 0
  for (const id of ids) {
    const a = p.attempts[id]
    if (!a) continue
    attempted++
    n += a.n
    c += a.correct
    if (a.lastCorrect) good++
  }
  return { mastery: good / ids.length, attempted, accuracy: n ? c / n : 0 }
}
