import { parse } from 'yaml'
import type { ZodType } from 'zod'
import {
  Course, ExerciseFile, FlashcardFile, FormulaFile, Week,
  type CourseSpec, type ExerciseSpec, type FlashcardSpec, type FormulaSpec, type Localized, type WeekSpec,
} from './schema'
import { checkAll } from '../lib/checkers'
import { generators, instantiate } from '../lib/generators'
import { mulberry32 } from '../lib/random'
import { evaluate } from '../lib/expr'
import { splitNotes } from './notes'

export interface WeekData {
  spec: WeekSpec
  courseId: string
  key: string
  notes: Localized
  exercises: ExerciseSpec[]
  flashcards: FlashcardSpec[]
  formulas: FormulaSpec[]
}
export interface CourseData {
  spec: CourseSpec
  weeks: WeekData[]
}
export interface Catalog {
  courses: CourseData[]
  errors: string[]
}

/** Files keyed by path relative to the content root, e.g. `c1/w1/week.yaml`. */
export type RawFiles = Record<string, string>

function parseWith<T>(schema: ZodType<T>, path: string, text: string, errors: string[]): T | null {
  let data: unknown
  try {
    data = parse(text)
  } catch (e) {
    errors.push(`${path}: YAML error: ${(e as Error).message}`)
    return null
  }
  const r = schema.safeParse(data)
  if (!r.success) {
    for (const issue of r.error.issues.slice(0, 10)) errors.push(`${path}: ${issue.path.join('.')}: ${issue.message}`)
    return null
  }
  return r.data
}

export function buildCatalog(files: RawFiles): Catalog {
  const errors: string[] = []
  const courses = new Map<string, CourseData>()
  for (const [path, text] of Object.entries(files)) {
    const m = /^([^/]+)\/course\.yaml$/.exec(path)
    if (!m) continue
    const spec = parseWith(Course, path, text, errors)
    if (spec) courses.set(m[1], { spec, weeks: [] })
  }
  for (const [path, text] of Object.entries(files)) {
    const m = /^([^/]+)\/([^/]+)\/week\.yaml$/.exec(path)
    if (!m) continue
    const [, c, w] = m
    const course = courses.get(c)
    if (!course) {
      errors.push(`${path}: missing ${c}/course.yaml`)
      continue
    }
    const spec = parseWith(Week, path, text, errors)
    if (!spec) continue
    const prefix = `${c}/${w}/`
    const week: WeekData = {
      spec, courseId: course.spec.id, key: `${course.spec.id}/${spec.id}`,
      notes: { vi: files[`${prefix}notes.vi.md`] ?? '', en: files[`${prefix}notes.en.md`] ?? '' },
      exercises: [], flashcards: [], formulas: [],
    }
    for (const [p, t] of Object.entries(files)) {
      if (!p.startsWith(prefix)) continue
      if (/^exercises\/[^/]+\.ya?ml$/.test(p.slice(prefix.length))) {
        const f = parseWith(ExerciseFile, p, t, errors)
        if (f) week.exercises.push(...f.exercises)
      } else if (p === `${prefix}flashcards.yaml`) {
        const f = parseWith(FlashcardFile, p, t, errors)
        if (f) week.flashcards.push(...f.flashcards)
      } else if (p === `${prefix}formulas.yaml`) {
        const f = parseWith(FormulaFile, p, t, errors)
        if (f) week.formulas.push(...f.formulas)
      }
    }
    course.weeks.push(week)
  }
  const list = [...courses.values()].sort((a, b) => a.spec.order - b.spec.order)
  for (const c of list) c.weeks.sort((a, b) => a.spec.order - b.spec.order)
  const catalog = { courses: list, errors }
  errors.push(...semanticErrors(catalog))
  return catalog
}

export function semanticErrors(cat: Catalog): string[] {
  const errs: string[] = []
  const ids = new Set<string>()
  const dup = (id: string, where: string) => {
    if (ids.has(id)) errs.push(`${where}: duplicate id "${id}"`)
    ids.add(id)
  }
  for (const c of cat.courses)
    for (const w of c.weeks) {
      const topics = new Set(w.spec.topics.map((t) => t.id))
      const where = w.key
      if (!w.notes.vi || !w.notes.en) errs.push(`${where}: missing notes.vi.md or notes.en.md`)
      // The site splits notes into per-topic pages at `## ` headings, in week.yaml topic order.
      for (const lang of ['vi', 'en'] as const) {
        const n = splitNotes(w.notes[lang]).sections.length
        if (w.notes[lang] && n !== w.spec.topics.length)
          errs.push(`${where}: notes.${lang}.md has ${n} "## " sections but week.yaml has ${w.spec.topics.length} topics (need one per topic, same order)`)
      }
      // Reject padded/templated content: the same substantive line or explanation must not be copy-pasted around.
      for (const lang of ['vi', 'en'] as const) {
        const counts = new Map<string, number>()
        for (const line of w.notes[lang].replace(/```[\s\S]*?```/g, '').replace(/\$\$[\s\S]*?\$\$/g, '').split('\n')) {
          const l = line.trim()
          // Long lines must be unique; shorter filler bullets may not appear 3+ times.
          if (l.length > 50 || (l.length > 20 && !l.startsWith('#'))) counts.set(l, (counts.get(l) ?? 0) + 1)
        }
        const rep = [...counts].filter(([l, n]) => (l.length > 50 ? n > 1 : n > 2))
        if (rep.length)
          errs.push(`${where}: notes.${lang}.md repeats ${rep.length} line(s) (templated filler?), e.g. ×${rep[0][1]} "${rep[0][0].slice(0, 70)}"`)
      }
      const seenPrompt = new Map<string, string>()
      const seenExpl = new Map<string, number>()
      for (const e of w.exercises) {
        if (e.type === 'parametric') continue
        const p = e.prompt.en.trim()
        if (seenPrompt.has(p)) errs.push(`${where}: ${e.id} has the same prompt as ${seenPrompt.get(p)}`)
        seenPrompt.set(p, e.id)
        const x = e.explanation.en.trim()
        seenExpl.set(x, (seenExpl.get(x) ?? 0) + 1)
      }
      for (const [x, n] of seenExpl) if (n > 1) errs.push(`${where}: ${n} exercises share the same explanation "${x.slice(0, 60)}…"`)
      // Templated prompts ("Which option best describes '<topic>'?") and recycled distractors.
      const skeleton = (s: string) =>
        s.toLowerCase().replace(/\$\$[\s\S]*?\$\$|\$[^$]*\$/g, 'M').replace(/'[^']*'|"[^"]*"|“[^”]*”|`[^`]*`/g, 'Q').replace(/\d+(\.\d+)?/g, 'N').replace(/\s+/g, ' ').trim()
      const skel = new Map<string, string[]>()
      const optUse = new Map<string, Set<string>>()
      for (const e of w.exercises) {
        if (e.type === 'parametric' || e.type === 'code') continue
        const k = skeleton(e.prompt.en)
        if (k.length >= 25) skel.set(k, [...(skel.get(k) ?? []), e.id])
        const opts = e.type === 'mcq' ? e.options.map((o) => o.en) : e.type === 'multi-step' ? e.parts.flatMap((p) => (p.type === 'mcq' ? p.options.map((o) => o.en) : [])) : e.type === 'true-false' ? (e.reasons ?? []).map((r) => r.en) : []
        for (const o of opts) if (o.trim().length > 25) optUse.set(o.trim(), (optUse.get(o.trim()) ?? new Set()).add(e.id))
      }
      for (const [k, ids] of skel) if (ids.length >= 3) errs.push(`${where}: ${ids.length} exercises use the same prompt template "${k.slice(0, 60)}" (${ids.slice(0, 3).join(', ')}…)`)
      for (const [o, ids] of optUse) if (ids.size >= 3) errs.push(`${where}: option "${o.slice(0, 60)}" is recycled in ${ids.size} exercises (${[...ids].slice(0, 3).join(', ')}…)`)
      const fronts = new Set<string>()
      for (const f of w.flashcards) {
        const key = f.front.en.trim().toLowerCase()
        if (fronts.has(key)) errs.push(`${where}: flashcard ${f.id} repeats the question "${f.front.en}"`)
        fronts.add(key)
      }
      if (w.flashcards.length >= 10) {
        const n = (k: string) => w.flashcards.filter((f) => f.kind === k).length / w.flashcards.length
        if (n('formula') + n('compute') < 0.4) errs.push(`${where}: formula+compute flashcards must be ≥ 40% (now ${Math.round((n('formula') + n('compute')) * 100)}%)`)
        if (n('concept') < 0.3) errs.push(`${where}: concept flashcards must be ≥ 30% (now ${Math.round(n('concept') * 100)}%)`)
      }
      for (const f of w.flashcards) {
        dup(f.id, where)
        if (!topics.has(f.topic)) errs.push(`${where}: flashcard ${f.id} has unknown topic ${f.topic}`)
      }
      for (const e of w.exercises) {
        const at = `${where}/${e.id}`
        const figs = [e.figure, ...('options' in e && Array.isArray(e.options) ? e.options.map((o) => (typeof o === 'object' ? o.figure : undefined)) : [])]
        for (const fig of figs) {
          if (!fig) continue
          for (const f of [...fig.functions, ...fig.areas])
            try {
              evaluate(f.f, { x: (fig.xRange[0] + fig.xRange[1]) / 2 + 0.123 })
            } catch (err) {
              errs.push(`${at}: figure function "${f.f}" invalid: ${(err as Error).message}`)
            }
        }
        dup(e.id, where)
        if (!topics.has(e.topic)) errs.push(`${at}: unknown topic ${e.topic}`)
        const inRange = (i: number, n: number, what: string) => {
          if (i < 0 || i >= n) errs.push(`${at}: ${what} index ${i} out of range (0..${n - 1})`)
        }
        switch (e.type) {
          case 'mcq':
            e.answer.forEach((a) => inRange(a, e.options.length, 'answer'))
            break
          case 'true-false':
            if (e.reasons && e.reasonAnswer === undefined) errs.push(`${at}: reasons given without reasonAnswer`)
            if (e.reasons && e.reasonAnswer !== undefined) inRange(e.reasonAnswer, e.reasons.length, 'reasonAnswer')
            break
          case 'find-error':
            inRange(e.answer, e.steps.length, 'answer')
            break
          case 'predict-output':
            inRange(e.answer, e.options.length, 'answer')
            break
          case 'matrix':
            if (new Set(e.answer.map((r) => r.length)).size !== 1) errs.push(`${at}: answer rows have different lengths`)
            break
          case 'construct': {
            if (e.example.length !== e.rows || e.example.some((r) => r.length !== e.cols))
              errs.push(`${at}: example must be ${e.rows}x${e.cols}`)
            else
              checkAll(e.constraints, e.example).forEach((r, i) => {
                if (!r.ok) errs.push(`${at}: example fails constraint #${i} (${e.constraints[i].kind}) ${r.detail ?? ''}`)
              })
            break
          }
          case 'multi-step':
            e.parts.forEach((p, i) => {
              if (p.type === 'mcq') p.answer.forEach((a) => inRange(a, p.options.length, `parts[${i}].answer`))
            })
            break
          case 'parametric':
            if (!generators[e.generator]) errs.push(`${at}: unknown generator "${e.generator}"`)
            else
              for (let s = 1; s <= 25; s++) {
                try {
                  instantiate(e, mulberry32(s))
                } catch (err) {
                  errs.push(`${at}: generator failed on seed ${s}: ${(err as Error).message}`)
                  break
                }
              }
            break
        }
      }
    }
  // Formulas: links must resolve; every formula needs practice; no duplicates.
  const allEx = new Set(cat.courses.flatMap((c) => c.weeks.flatMap((w) => w.exercises.map((e) => e.id))))
  for (const c of cat.courses)
    for (const w of c.weeks) {
      const topics = new Set(w.spec.topics.map((t) => t.id))
      const texSeen = new Map<string, string>()
      for (const f of w.formulas) {
        const at = `${w.key}/formulas/${f.id}`
        dup(f.id, w.key)
        if (!topics.has(f.topic)) errs.push(`${at}: unknown topic ${f.topic}`)
        for (const id of f.exercises) if (!allEx.has(id)) errs.push(`${at}: links to unknown exercise "${id}"`)
        if (new Set(f.exercises).size !== f.exercises.length) errs.push(`${at}: duplicate exercise links`)
        const k = f.tex.replace(/\s+/g, '')
        if (texSeen.has(k)) errs.push(`${at}: same formula as ${texSeen.get(k)}`)
        texSeen.set(k, f.id)
      }
      if (w.exercises.length && !w.formulas.length) errs.push(`${w.key}: missing formulas.yaml`)
    }
  return errs
}
