import { z } from 'zod'

export const L = z.object({ vi: z.string().min(1), en: z.string().min(1) })
export type Localized = z.infer<typeof L>

const num = z.number()
const vec = z.array(num)
const mat = z.array(z.array(num))

export const Figure = z.object({
  xRange: z.tuple([num, num]).default([-6, 6]),
  yRange: z.tuple([num, num]).default([-6, 6]),
  lines: z
    .array(z.object({ a: num, b: num, c: num, label: z.string().optional(), color: z.string().optional() }))
    .default([]),
  vectors: z
    .array(
      z.object({
        to: z.tuple([num, num]),
        from: z.tuple([num, num]).default([0, 0]),
        label: z.string().optional(),
        color: z.string().optional(),
      }),
    )
    .default([]),
  points: z
    .array(z.object({ at: z.tuple([num, num]), label: z.string().optional(), color: z.string().optional() }))
    .default([]),
  /** Graphs y = f(x); `f` uses the answer-expression syntax with variable x, e.g. 'x^2 - 2*x', 'exp(-x^2/2)'. */
  functions: z
    .array(
      z.object({
        f: z.string(),
        label: z.string().optional(),
        color: z.string().optional(),
        domain: z.tuple([num, num]).optional(),
        dashed: z.boolean().optional(),
      }),
    )
    .default([]),
  /** Shaded area between y = f(x) and y = 0 on [from, to] (e.g. a probability under a density). */
  areas: z.array(z.object({ f: z.string(), from: num, to: num, color: z.string().optional() })).default([]),
  /** Bar chart / histogram / PMF bars centred at x. */
  bars: z
    .array(z.object({ x: num, h: num, width: num.optional(), label: z.string().optional(), color: z.string().optional() }))
    .default([]),
  xLabel: z.string().optional(),
  yLabel: z.string().optional(),
})
export type FigureSpec = z.infer<typeof Figure>

export const Constraint = z.discriminatedUnion('kind', [
  z.object({ kind: z.literal('det'), value: num }),
  z.object({ kind: z.literal('rank'), value: z.number().int() }),
  z.object({ kind: z.literal('singular') }),
  z.object({ kind: z.literal('nonsingular') }),
  z.object({ kind: z.literal('rowEchelon') }),
  z.object({ kind: z.literal('reducedRowEchelon') }),
  z.object({ kind: z.literal('noZeroEntries') }),
  z.object({ kind: z.literal('integerEntries') }),
  z.object({ kind: z.literal('nonzero') }),
  z.object({ kind: z.literal('notDiagonal') }),
  z.object({ kind: z.literal('entryRange'), min: num, max: num }),
  z.object({ kind: z.literal('dot'), with: vec, value: num }),
  z.object({ kind: z.literal('norm'), p: z.union([z.literal(1), z.literal(2), z.literal('inf')]), value: num }),
  z.object({ kind: z.literal('solves'), A: mat, b: vec }),
  z.object({ kind: z.literal('mapsTo'), x: vec, b: vec }),
  z.object({ kind: z.literal('notParallelTo'), of: vec }),
  z.object({ kind: z.literal('parallelTo'), of: vec }),
  z.object({ kind: z.literal('noSolutionWith'), b: vec }),
  z.object({ kind: z.literal('infiniteSolutionsWith'), b: vec }),
  z.object({ kind: z.literal('uniqueSolutionWith'), b: vec }),
  // Statistics on a user-entered data vector (1×n or n×1).
  z.object({ kind: z.literal('mean'), value: num }),
  z.object({ kind: z.literal('median'), value: num }),
  z.object({ kind: z.literal('variance'), value: num, sample: z.boolean().default(false) }),
  z.object({ kind: z.literal('std'), value: num, sample: z.boolean().default(false) }),
  z.object({ kind: z.literal('sum'), value: num }),
  z.object({ kind: z.literal('range'), value: num }),
  z.object({ kind: z.literal('meanGreaterThanMedian') }),
  z.object({ kind: z.literal('meanLessThanMedian') }),
  z.object({ kind: z.literal('nonnegative') }),
  z.object({ kind: z.literal('distinctValues'), min: z.number().int() }),
  /** Entries are a valid probability distribution: all ≥ 0 and sum to 1. */
  z.object({ kind: z.literal('probabilityVector') }),
  /** For a PMF over values `values` (user enters probabilities): expected value equals `value`. */
  z.object({ kind: z.literal('expectation'), values: vec, value: num }),
])
export type ConstraintSpec = z.infer<typeof Constraint>

const Base = {
  id: z.string().regex(/^[a-z0-9-]+$/),
  topic: z.string(),
  difficulty: z.union([z.literal(1), z.literal(2), z.literal(3)]).default(1),
  tags: z.array(z.string()).default([]),
  prompt: L,
  explanation: L,
  hints: z.array(L).default([]),
  figure: Figure.optional(),
}

const Option = L.extend({ figure: Figure.optional() })
export type OptionSpec = z.infer<typeof Option>

const Mcq = z.object({
  ...Base,
  type: z.literal('mcq'),
  options: z.array(Option).min(2),
  answer: z.array(z.number().int()).min(1),
  shuffle: z.boolean().default(true),
})
const TrueFalse = z.object({
  ...Base,
  type: z.literal('true-false'),
  answer: z.boolean(),
  reasons: z.array(L).optional(),
  reasonAnswer: z.number().int().optional(),
})
const NumAnswer = z.object({ label: z.string().optional(), value: num })
const Numeric = z.object({ ...Base, type: z.literal('numeric'), answers: z.array(NumAnswer).min(1), tolerance: num.default(1e-3) })
const Matrix = z.object({ ...Base, type: z.literal('matrix'), answer: mat, tolerance: num.default(1e-3) })
const Construct = z.object({
  ...Base,
  type: z.literal('construct'),
  rows: z.number().int().min(1),
  cols: z.number().int().min(1),
  constraints: z.array(Constraint).min(1),
  /** One valid answer, used by the validator and shown as a sample solution. */
  example: mat,
})
const FindError = z.object({ ...Base, type: z.literal('find-error'), steps: z.array(L).min(3), answer: z.number().int() })
/** Steps are written in the correct order; the UI shuffles them. */
const Parsons = z.object({ ...Base, type: z.literal('parsons'), steps: z.array(L).min(3) })
const Matching = z.object({ ...Base, type: z.literal('matching'), pairs: z.array(z.object({ left: L, right: L })).min(3) })
const Predict = z.object({
  ...Base,
  type: z.literal('predict-output'),
  code: z.string(),
  options: z.array(z.string()).min(2),
  answer: z.number().int(),
})
const Code = z.object({ ...Base, type: z.literal('code'), starter: z.string(), tests: z.string(), solution: z.string() })
const Parametric = z.object({
  ...Base,
  type: z.literal('parametric'),
  generator: z.string(),
  params: z.record(z.string(), z.unknown()).default({}),
})

const partBase = { prompt: L, explanation: L.optional() }
export const Part = z.discriminatedUnion('type', [
  z.object({ ...partBase, type: z.literal('mcq'), options: z.array(Option).min(2), answer: z.array(z.number().int()).min(1), shuffle: z.boolean().default(true) }),
  z.object({ ...partBase, type: z.literal('true-false'), answer: z.boolean() }),
  z.object({ ...partBase, type: z.literal('numeric'), answers: z.array(NumAnswer).min(1), tolerance: num.default(1e-3) }),
  z.object({ ...partBase, type: z.literal('matrix'), answer: mat, tolerance: num.default(1e-3) }),
])
const MultiStep = z.object({ ...Base, type: z.literal('multi-step'), parts: z.array(Part).min(2) })

export const Exercise = z.discriminatedUnion('type', [
  Mcq, TrueFalse, Numeric, Matrix, Construct, FindError, Parsons, Matching, Predict, Code, Parametric, MultiStep,
])
export type ExerciseSpec = z.infer<typeof Exercise>
export type PartSpec = z.infer<typeof Part>
export type ExerciseType = ExerciseSpec['type']
export type Of<T extends ExerciseType> = Extract<ExerciseSpec, { type: T }>

export const ExerciseFile = z.object({ exercises: z.array(Exercise) })

export const FlashcardKind = z.enum(['concept', 'formula', 'compute'])
export type FlashcardKindT = z.infer<typeof FlashcardKind>
export const Flashcard = z.object({
  id: z.string().regex(/^[a-z0-9-]+$/),
  topic: z.string(),
  /** concept = theory/why; formula = recall a formula/definition in math; compute = small mental calculation. */
  kind: FlashcardKind,
  front: L,
  back: L,
})
export const FlashcardFile = z.object({ flashcards: z.array(Flashcard) })
export type FlashcardSpec = z.infer<typeof Flashcard>

export const Formula = z.object({
  id: z.string().regex(/^[a-z0-9-]+$/),
  topic: z.string(),
  name: L,
  /** Display-mode LaTeX without surrounding $$. */
  tex: z.string().min(1),
  /** What the symbols mean / when to use it (markdown, may contain $...$). */
  where: L.optional(),
  /** Equivalent NumPy/SciPy call, if any. */
  numpy: z.string().optional(),
  /** Exercise ids (any week) that practise this formula. */
  exercises: z.array(z.string()).min(1),
})
export const FormulaFile = z.object({ formulas: z.array(Formula) })
export type FormulaSpec = z.infer<typeof Formula>

export const Week = z.object({
  id: z.string(),
  order: z.number(),
  title: L,
  source: z.string().optional(),
  topics: z.array(z.object({ id: z.string(), title: L })).min(1),
})
export type WeekSpec = z.infer<typeof Week>

export const Course = z.object({ id: z.string(), order: z.number(), title: L, description: L.optional() })
export type CourseSpec = z.infer<typeof Course>
