/** SM-2 spaced repetition. */
export interface SrsState {
  ef: number
  interval: number // days
  reps: number
  due: number // epoch ms
  lapses: number
}

export type Grade = 'again' | 'hard' | 'good' | 'easy'
const Q: Record<Grade, number> = { again: 1, hard: 3, good: 4, easy: 5 }
const DAY = 86_400_000

export const newSrs = (now = Date.now()): SrsState => ({ ef: 2.5, interval: 0, reps: 0, due: now, lapses: 0 })

export function review(s: SrsState, grade: Grade, now = Date.now()): SrsState {
  const q = Q[grade]
  const ef = Math.max(1.3, s.ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)))
  if (q < 3) return { ef, interval: 0, reps: 0, due: now + 10 * 60_000, lapses: s.lapses + 1 }
  const reps = s.reps + 1
  let interval = reps === 1 ? 1 : reps === 2 ? 6 : Math.round(s.interval * ef)
  if (grade === 'hard') interval = Math.max(1, Math.round(interval * 0.6))
  if (grade === 'easy') interval = Math.round(interval * 1.3) + 1
  return { ef, interval, reps, due: now + interval * DAY, lapses: s.lapses }
}

export const isDue = (s: SrsState, now = Date.now()) => s.due <= now
