import { buildCatalog, type RawFiles } from './catalog'

const modules = import.meta.glob('/content/**/*.{yaml,yml,md}', { query: '?raw', import: 'default', eager: true }) as Record<
  string,
  string
>

const files: RawFiles = {}
for (const [p, text] of Object.entries(modules)) files[p.replace(/^\/content\//, '')] = text

export const catalog = buildCatalog(files)
if (catalog.errors.length) console.warn('Content errors:\n' + catalog.errors.join('\n'))

export const allWeeks = catalog.courses.flatMap((c) => c.weeks)
export const allExercises = allWeeks.flatMap((w) => w.exercises.map((e) => ({ ...e, weekKey: w.key })))
export const allFlashcards = allWeeks.flatMap((w) => w.flashcards.map((f) => ({ ...f, weekKey: w.key })))
export const exerciseById = new Map(allExercises.map((e) => [e.id, e]))
export const flashcardById = new Map(allFlashcards.map((f) => [f.id, f]))
export const weekByKey = new Map(allWeeks.map((w) => [w.key, w]))
export const allFormulas = allWeeks.flatMap((w) => w.formulas.map((f) => ({ ...f, weekKey: w.key })))
export type FormulaItem = (typeof allFormulas)[number]
/** Exercise id → formulas that list it. */
export const formulasByExercise = new Map<string, FormulaItem[]>()
for (const f of allFormulas) for (const id of f.exercises) formulasByExercise.set(id, [...(formulasByExercise.get(id) ?? []), f])

export function topicTitle(weekKey: string, topicId: string) {
  return weekByKey.get(weekKey)?.spec.topics.find((t) => t.id === topicId)?.title
}
