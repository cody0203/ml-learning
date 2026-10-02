import { allExercises } from '../content'
import type { ExerciseType } from '../content/schema'
import type { Progress } from './progress'
import { shuffle } from './random'

export type Mode = 'mixed' | 'weak' | 'new' | 'exam'

export interface SessionConfig {
  weeks: string[] // week keys; empty = all
  topic?: string
  types: ExerciseType[] // empty = all
  mode: Mode
  n: number
  ids?: string[]
  difficulty?: number[]
}

export interface QueueItem {
  id: string
  seed: number
}

export function pool(cfg: SessionConfig) {
  if (cfg.ids?.length) return allExercises.filter((e) => cfg.ids!.includes(e.id))
  return allExercises.filter(
    (e) =>
      (!cfg.weeks.length || cfg.weeks.includes(e.weekKey)) &&
      (!cfg.topic || e.topic === cfg.topic) &&
      (!cfg.types.length || cfg.types.includes(e.type)) &&
      (!cfg.difficulty?.length || cfg.difficulty.includes(e.difficulty)),
  )
}

function weightedSample<T>(items: T[], weight: (t: T) => number, n: number): T[] {
  const left = items.map((it) => ({ it, w: Math.max(weight(it), 0.01) }))
  const out: T[] = []
  while (out.length < n && left.length) {
    const total = left.reduce((s, x) => s + x.w, 0)
    let r = Math.random() * total
    let k = 0
    while (k < left.length - 1 && (r -= left[k].w) > 0) k++
    out.push(left[k].it)
    left.splice(k, 1)
  }
  return out
}

export function buildQueue(cfg: SessionConfig, p: Progress): QueueItem[] {
  const items = pool(cfg)
  const n = cfg.ids?.length ? items.length : Math.min(cfg.n, items.length)
  let picked = items
  if (cfg.mode === 'new') {
    const unseen = shuffle(items.filter((e) => !p.attempts[e.id]))
    const seen = shuffle(items.filter((e) => p.attempts[e.id]))
    picked = [...unseen, ...seen].slice(0, n)
  } else if (cfg.mode === 'weak') {
    const topicAcc = new Map<string, { c: number; n: number }>()
    for (const e of allExercises) {
      const a = p.attempts[e.id]
      if (!a) continue
      const t = topicAcc.get(e.topic) ?? { c: 0, n: 0 }
      topicAcc.set(e.topic, { c: t.c + a.correct, n: t.n + a.n })
    }
    picked = weightedSample(
      items,
      (e) => {
        const a = p.attempts[e.id]
        const t = topicAcc.get(e.topic)
        const topicWeak = t && t.n ? 1 - t.c / t.n : 0.5
        if (!a) return 1.5 + topicWeak * 2
        return (a.lastCorrect ? 0.3 : 4) + (1 - a.correct / a.n) * 2 + topicWeak * 2
      },
      n,
    )
  } else {
    picked = shuffle(items).slice(0, n)
  }
  return picked.map((e) => ({ id: e.id, seed: Math.floor(Math.random() * 2 ** 31) }))
}

export function encodeConfig(c: SessionConfig): string {
  const q = new URLSearchParams()
  if (c.weeks.length) q.set('weeks', c.weeks.join(','))
  if (c.topic) q.set('topic', c.topic)
  if (c.types.length) q.set('types', c.types.join(','))
  if (c.difficulty?.length) q.set('diff', c.difficulty.join(','))
  if (c.ids?.length) q.set('ids', c.ids.join(','))
  q.set('mode', c.mode)
  q.set('n', String(c.n))
  return q.toString()
}

export function decodeConfig(q: URLSearchParams): SessionConfig {
  const list = (k: string) => (q.get(k) ? q.get(k)!.split(',').filter(Boolean) : [])
  return {
    weeks: list('weeks'),
    topic: q.get('topic') ?? undefined,
    types: list('types') as ExerciseType[],
    difficulty: list('diff').map(Number),
    ids: list('ids'),
    mode: (q.get('mode') as Mode) ?? 'mixed',
    n: Number(q.get('n') ?? 10),
  }
}
