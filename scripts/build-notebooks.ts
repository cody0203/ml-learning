/* Generate JupyterLite notebooks from content/: one per week (all code exercises) + one per code exercise + a playground. */
import { mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { join, relative, sep } from 'node:path'
import { buildCatalog, type RawFiles } from '../src/content/catalog'

const root = join(import.meta.dirname, '..')
const contentDir = join(root, 'content')
const outDir = join(root, 'jupyterlite', 'content')

const files: RawFiles = {}
const walk = (d: string) => {
  for (const f of readdirSync(d)) {
    const p = join(d, f)
    if (statSync(p).isDirectory()) walk(p)
    else if (/\.(ya?ml|md)$/.test(f)) files[relative(contentDir, p).split(sep).join('/')] = readFileSync(p, 'utf8')
  }
}
walk(contentDir)
const cat = buildCatalog(files)
if (cat.errors.length) {
  console.error(cat.errors.join('\n'))
  process.exit(1)
}

type Cell = { cell_type: 'markdown' | 'code'; source: string; metadata?: Record<string, unknown> }
const lines = (s: string) => s.replace(/\s+$/, '').split('\n').map((l, i, a) => (i < a.length - 1 ? l + '\n' : l))
const nb = (cells: Cell[]) => ({
  nbformat: 4,
  nbformat_minor: 5,
  metadata: {
    kernelspec: { name: 'python', display_name: 'Python (Pyodide)', language: 'python' },
    language_info: { name: 'python' },
  },
  cells: cells.map((c, i) => ({
    id: `c${i}`,
    cell_type: c.cell_type,
    metadata: c.metadata ?? {},
    source: lines(c.source),
    ...(c.cell_type === 'code' ? { outputs: [], execution_count: null } : {}),
  })),
})

const RUNNER = `def run_tests(ns=None):
    """Run every test_* function defined so far and print a report."""
    ns = ns or globals()
    tests = [(n, f) for n, f in list(ns.items()) if n.startswith("test_") and callable(f)]
    passed = 0
    for n, f in tests:
        try:
            f()
            passed += 1
            print("✅", n)
        except AssertionError as e:
            print("❌", n, "-", e)
        except Exception as e:
            print("💥", n, "-", type(e).__name__, e)
    print(f"\\n{passed}/{len(tests)} tests passed")
    for n, _ in tests:
        ns.pop(n, None)`

const exerciseCells = (e: Extract<(typeof cat.courses)[number]['weeks'][number]['exercises'][number], { type: 'code' }>, heading: string): Cell[] => [
  { cell_type: 'markdown', source: `${heading} ${e.id}\n\n**EN.** ${e.prompt.en}\n\n**VI.** ${e.prompt.vi}` },
  { cell_type: 'code', source: e.starter },
  { cell_type: 'code', source: `# Tests — run after your solution cell\n${e.tests.replace(/\s+$/, '')}\n\nrun_tests()` },
]

rmSync(outDir, { recursive: true, force: true })
mkdirSync(join(outDir, 'exercises'), { recursive: true })

const setup: Cell = { cell_type: 'code', source: `import numpy as np\n\n${RUNNER}` }
let count = 0
for (const c of cat.courses)
  for (const w of c.weeks) {
    const code = w.exercises.filter((e) => e.type === 'code')
    const weekCells: Cell[] = [
      { cell_type: 'markdown', source: `# ${w.spec.title.en}\n# ${w.spec.title.vi}\n\nRun the setup cell first, then solve each exercise and run its test cell.\nChạy cell setup trước, sau đó làm từng bài và chạy cell test của bài đó.` },
      setup,
    ]
    for (const e of code) {
      if (e.type !== 'code') continue
      weekCells.push(...exerciseCells(e, '##'))
      writeFileSync(
        join(outDir, 'exercises', `${e.id}.ipynb`),
        JSON.stringify(nb([setup, ...exerciseCells(e, '#')]), null, 1),
      )
      count++
    }
    writeFileSync(join(outDir, `${w.courseId}-${w.spec.id}.ipynb`), JSON.stringify(nb(weekCells), null, 1))
  }

writeFileSync(
  join(outDir, 'playground.ipynb'),
  JSON.stringify(
    nb([
      { cell_type: 'markdown', source: '# Playground\n\nFree-form NumPy experiments. Edits are saved in your browser (IndexedDB).' },
      { cell_type: 'code', source: 'import numpy as np\n\nA = np.array([[1, 1], [1, 2]])\nb = np.array([10, 12])\nprint("det:", np.linalg.det(A))\nprint("rank:", np.linalg.matrix_rank(A))\nprint("x:", np.linalg.solve(A, b))' },
    ]),
    null,
    1,
  ),
)
console.log(`Generated ${cat.courses.reduce((s, c) => s + c.weeks.length, 0)} week notebooks + ${count} exercise notebooks in jupyterlite/content`)
