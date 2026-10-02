---
name: ingest-syllabus
description: Turn a new course slide PDF (in syllabus/) into study notes, flashcards and a large, diverse exercise set for the ML review website. Use when the user adds a PDF to syllabus/ or asks to "add week N", "ingest slides", "tạo bài tập từ PDF", or wants more exercises for an existing week.
---

# Ingest syllabus PDF → site content

The site (Vite + React, repo root) reads everything from `content/`. You only write YAML/Markdown there — no app code changes are needed.

## 1. Extract the PDF

```powershell
$env:PYTHONIOENCODING="utf-8"; python scripts/extract_pdf.py syllabus/c1-linear-algebra/W4.pdf > $env:TEMP\c1w4.txt
```
PDFs live in `syllabus/<course-id>/W<n>.pdf` (a week may be split: `W3-part-1.pdf`, `W3-part-2.pdf` → one week). If the user drops PDFs elsewhere (e.g. directly in `content/`), classify them by course/week from the title slides and move them there first. Courses: `c1-linear-algebra`, `c2-calculus`, `c3-probability-statistics`.

(`pip install pypdf` if missing.) The script drops animation-build duplicates. Read the whole text and list the **sections** (section title slides) — they become topics. Note every quiz in the slides: they show what the course expects.

## 2. Folder layout

```
content/<course-id>/course.yaml          # exists for c1-linear-algebra
content/<course-id>/<week-dir>/
  week.yaml          id, order, source, title{vi,en}, topics[{id,title{vi,en}}]
  notes.vi.md        Vietnamese notes
  notes.en.md        English notes (same structure)
  flashcards.yaml    flashcards: [...]
  formulas.yaml      formulas: [...]   (quick-reference formula sheet, see §4b)
  exercises/<topic-id>.yaml   exercises: [...]   (one file per topic, plus optional mixed.yaml)
```
IDs: lowercase-kebab, globally unique, prefixed `c<course>w<week>-` (e.g. `c1w4-eig-mcq-03`).
New course? Create `content/c2-calculus/course.yaml` with `id, order, title{vi,en}`.

## 3. Notes (notes.vi.md / notes.en.md)

- One `## ` heading per topic (same order as week.yaml topics), `### ` for sub-ideas. The site splits the notes at `## ` into one page per topic (notes + that topic's formulas + its exercises), and the validator requires #`## ` == #topics.
- **Topic granularity:** aim for 6–12 topics per week, each a coherent lesson with ≥ 5 exercises. Don't create one topic per slide section; group related sections (they become `### ` subsections).
- Intuition first (the slides' ML motivation), then definitions, formulas (KaTeX `$...$`, `$$...$$`), worked examples, common mistakes (`> ⚠️ ...`), and a short "Tóm tắt / Summary" list.
- Vietnamese keeps English technical terms in parentheses the first time: "định thức (determinant)".
- Go beyond the slides where it helps understanding (geometric view, NumPy one-liners), keeping course notation.

## 4. Flashcards

30–45 per week, each with `kind`:
- `concept` (≥ 30 %): meaning, "why", "when does X happen", geometric interpretation, connection to ML. E.g. *"Why does det = 0 mean the rows are dependent?"*
- `formula` (the core of the deck): recall the exact formula/definition written in LaTeX. E.g. front *"$\det\begin{bmatrix}a&b\\c&d\end{bmatrix} = ?$"*, back *"$ad-bc$"*. Also cover the NumPy function that computes it (`np.linalg.det`).
- `compute`: a 5-second mental calculation with small numbers that exercises one formula. E.g. *"$\|(3,-4)\|_2 = ?$"* → *"$5$"*.

`formula` + `compute` must be ≥ 40 %. Every question must be specific and unique. Never use generic fronts like "What is the key idea of this topic?" (the validator rejects repeated questions). Front: one short question. Back: the answer first, then at most 1–2 lines of reasoning. Fields: `id, topic, kind, front{vi,en}, back{vi,en}`.

## 4b. Formula sheet (formulas.yaml) — write AFTER the exercises

The week's "Công thức / Formulas" tab and the global formula search read this file. Every formula a learner might want to look up goes here, each linked to the exercises that practise it.

```yaml
formulas:
  - id: c1w1-f-det-2x2              # globally unique, prefix c<course>w<week>-f-
    topic: determinant-2x2          # a topic id from week.yaml
    name: { vi: 'Định thức ma trận 2×2', en: 'Determinant of a 2×2 matrix' }
    tex: '\det\begin{bmatrix}a&b\\c&d\end{bmatrix} = ad - bc'   # display LaTeX, no $$
    where:                          # optional: symbols + when to use / key condition (markdown, $...$ ok)
      vi: '$\det \neq 0$ ⇔ hệ có nghiệm duy nhất (non-singular).'
      en: '$\det \neq 0$ ⇔ the system has a unique solution (non-singular).'
    numpy: 'np.linalg.det(A)'        # optional
    exercises: [c1w1-det2-num-01, c1w1-det2-code-06]   # ≥ 1 exercise id (any week) that uses this formula
```

- Cover every formula/definition/rule of the week that has a symbolic form (typically 12–30 per week): definitions, rules, special cases, key identities, update rules (gradient descent, Newton), test statistics, etc. Don't duplicate the same formula.
- `exercises`: link the exercises whose solution actually uses that formula (read prompts + explanations; include parametric drills). Aim for ≥ 2 links per formula; if a formula has no practice yet, add an exercise for it. Links may point to other weeks' exercises when relevant.
- `name` short and searchable; `where` explains symbols and when it applies (1–2 lines, no essays).
- After answering an exercise, the site shows the formulas that link to it, so good links matter in both directions.
## 5. Exercises — DIVERSITY IS THE GOAL

**Hand-write everything.** Never generate notes, flashcards or exercises with scripts, loops or fill-in templates.

**Encoding trap (Windows):** non-ASCII text (Vietnamese!) written through PowerShell — command strings, here-strings, `Set-Content`, piped `python -`/`node` scripts — gets mangled (`Với` → `V?i` or `Voi`). Write and modify content files only with the `create`/`edit` tools. The validator rejects `?`-corrupted letters and Vietnamese strings without diacritics.

The validator also rejects templated output: a notes line (> 50 chars) appearing twice, two exercises with the same prompt, or two exercises sharing an explanation. Fewer genuinely different exercises beat many clones.

Target **≥ 60 exercises per week**, spread over **≥ 9 types**, each topic covered by ≥ 4 types. Do NOT pad with number-swapped clones: `parametric` (random numbers) must be ≤ 15 % of a week. Vary the *thinking*, not the numbers.

| type | tests | fields / notes |
|---|---|---|
| `mcq` | concepts, interpretation, "which is true" | `options[{vi,en,figure?}]`, `answer` = list of correct indices (multi-select if >1). Distractors = real misconceptions. |
| `true-false` | precise statements | `answer: true/false`; optional `reasons[{vi,en}]` + `reasonAnswer` → user must also pick WHY |
| `numeric` | a value | `answers: [{label: "a", value: 2}, ...]`, `tolerance` (relative, default 1e-3); users can type `7/3`, `sqrt(2)` |
| `matrix` | a matrix/vector result | `answer: [[..],[..]]`; column vector = `[[1],[2]]` |
| `construct` | build an example satisfying properties → many correct answers | `rows, cols, constraints[], example` (one valid answer). E.g. "singular 2×2 with no zero entries", "vector orthogonal to (2,1) with L1-norm 3", "3×3 matrix of rank 2". |
| `find-error` | spot the wrong step in a worked solution | `steps[{vi,en}]`, `answer` = index of the first wrong step |
| `parsons` | order steps of a procedure | `steps` in the CORRECT order (UI shuffles) |
| `matching` | concept ↔ formula/picture/example | `pairs: [{left{vi,en}, right{vi,en}}]`, 3–6 pairs |
| `predict-output` | read NumPy code | `code`, `options` (plain strings), `answer` index |
| `code` | implement in Python/NumPy (Pyodide in browser; `scipy` is also available but prefer NumPy-only) | `starter`, `solution`, `tests` = functions `def test_x(): assert cond, "msg"`. Starter must FAIL, solution must PASS. If the task is to implement an algorithm, say that `np.linalg.*` shortcuts are not allowed and test on cases that show it. |
| `multi-step` | word problems / chains | `parts[]`, each `mcq`/`true-false`/`numeric`/`matrix` with its own `prompt` (and optional `explanation`) |
| `parametric` | drill with fresh numbers each time | `generator` ∈ linear algebra: solve-system, determinant, classify-system, rank, rref, dot-product, norm, vector-combination, matrix-vector, matrix-multiply, vector-angle · calculus: derivative-poly, partial-derivative, gradient-step, newton-step · probability/statistics: descriptive-stats, binomial-prob, expected-value, bayes, z-score, confidence-interval; optional `params: {size: 3}` (solve-system, determinant, rank, rref, dot-product). Prompt/explanation fields are required but replaced by the generator. |

Construct constraint kinds (see `src/content/schema.ts`): `det{value}`, `rank{value}`, `singular`, `nonsingular`, `rowEchelon`, `reducedRowEchelon`, `noZeroEntries`, `integerEntries`, `nonzero`, `notDiagonal`, `entryRange{min,max}`, `dot{with,value}` (user vector), `norm{p:1|2|inf,value}`, `solves{A,b}` (user vector x solves Ax=b), `mapsTo{x,b}` (user matrix A has Ax=b), `parallelTo{of}`, `notParallelTo{of}`, `noSolutionWith{b}`, `infiniteSolutionsWith{b}`, `uniqueSolutionWith{b}` (user matrix = coefficient matrix).

Statistics constraints (user enters a data row, `rows: 1, cols: n`): `mean{value}`, `median{value}`, `variance{value, sample}`, `std{value, sample}` (`sample: true` divides by n−1), `sum{value}`, `range{value}`, `meanGreaterThanMedian`, `meanLessThanMedian`, `nonnegative`, `distinctValues{min}`, `probabilityVector` (≥0, sums to 1), `expectation{values, value}` (user enters the PMF for the given values). E.g. *"Build a dataset of 5 values with mean 4 and median 2"*, *"Give a PMF on {0,1,2} with E[X] = 1.5"*.

`figure` (on exercise or mcq option): `{xRange:[-6,6], yRange:[-6,6], lines:[{a,b,c,label,color}], vectors:[{to:[x,y], from:[0,0], label, color}], points:[{at:[x,y],label}]}` — a line is `a·x + b·y = c`. Also available: `functions: [{f: 'x^2 - 2*x', label, color, domain:[a,b], dashed}]` (graph y = f(x); expression syntax supports + - * / ^, sqrt, exp, log/ln (natural), log2, log10, sin, cos, tan, abs, sigmoid, pi, e — write multiplication explicitly `2*x`), `areas: [{f, from, to, color}]` (shaded area under a curve, e.g. a probability), `bars: [{x, h, width, label, color}]` (PMF / histogram), `xLabel`, `yLabel`. Axes keep a 1:1 scale only when the ranges are similar; for densities use e.g. `xRange: [-4, 4], yRange: [0, 0.5]`. Numeric answers also accept `exp(2)`, `log(3)`, `sigmoid(1)`.

Every exercise: `id, topic, type, difficulty (1 easy | 2 medium | 3 hard), tags[], prompt{vi,en}, explanation{vi,en}, hints[{vi,en}]` (hints encouraged for difficulty ≥ 2). Explanations must teach the reasoning, not just state the answer.

Idea generators to escape repetition — for each concept ask:
1. Definition / recognition (mcq, matching)
2. Reverse direction: given the answer, build the question (construct)
3. Geometric view (figure mcq)
4. Edge cases & misconceptions (true-false with reasons, find-error)
5. Procedure (parsons, multi-step)
6. Application in ML / word problem (multi-step)
7. Code: implement it, predict NumPy output
8. Connections to earlier weeks (mixed.yaml)

YAML tips: prefer block scalars (`|`) or single quotes for LaTeX and code (backslashes stay literal). In double quotes you must write `\\`. Quote strings containing `: ` or starting with `[`, `{`, `*`, `&`, `!`, `%`, `@`, `` ` ``.

## 6. Validate (must pass)

```powershell
npm run validate
```
Use `npm run validate -- --only <course-id>/w<n>` while working on one week (runs only that week's code and shows only its errors); run the full command at the end. Checks schema, unique ids, topics, answer indices, construct `example` against its constraints, generators, and runs every code exercise (solution passes, starter fails) in Pyodide. Then `npm run build:lite` (regenerates JupyterLite notebooks from code exercises) and `npm run build`. Report the per-type counts it prints.
