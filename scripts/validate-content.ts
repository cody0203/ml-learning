/* Validate all content under content/.
 * Usage: npm run validate [-- --skip-code] [-- --only c2-calculus/w1]  (--only limits code runs and reported errors to matching weeks) */
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative, sep } from 'node:path'
import katex from 'katex'
import { unified } from 'unified'
import remarkParse from 'remark-parse'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import { visit } from 'unist-util-visit'

const mdParser = unified().use(remarkParse).use(remarkGfm).use(remarkMath)
import { buildCatalog, type RawFiles } from '../src/content/catalog'
import { buildHarness } from '../src/lib/pyHarness'

const root = join(import.meta.dirname, '..', 'content')
const files: RawFiles = {}
const walk = (d: string) => {
  for (const f of readdirSync(d)) {
    const p = join(d, f)
    if (statSync(p).isDirectory()) walk(p)
    else if (/\.(ya?ml|md)$/.test(f)) files[relative(root, p).split(sep).join('/')] = readFileSync(p, 'utf8')
  }
}
walk(root)

const onlyIdx = process.argv.indexOf('--only')
const only = onlyIdx > 0 ? process.argv[onlyIdx + 1] : undefined
const inScope = (s: string) => !only || s.includes(only)

const cat = buildCatalog(files)
const errors = [...cat.errors]

// Every $...$ / $$...$$ must compile with KaTeX; also catch control characters from broken escapes (\b, \f, \a...).
const checkTex = (raw: string, where: string) => {
  const s = raw.replace(/```[\s\S]*?```/g, '').replace(/`[^`\n]*`/g, '')
  // eslint-disable-next-line no-control-regex
  if (/[\x00-\x08\x0b\x0c\x0e-\x1f]/.test(s)) errors.push(`${where}: contains control characters (broken backslash escape?)`)
  const re = /\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$/g
  let m
  while ((m = re.exec(s))) {
    const tex = m[1] ?? m[2]
    try {
      katex.renderToString(tex, { throwOnError: true, displayMode: !!m[1] })
    } catch (e) {
      errors.push(`${where}: ${(e as Error).message.slice(0, 100)} in: ${tex.slice(0, 60)}`)
    }
  }
}
const CODE_KEYS = new Set(['code', 'starter', 'solution', 'tests', 'options'])
// Text mangled by a wrong encoding: "Nh?n di?n" (lost Vietnamese letters) or U+FFFD.
const MOJIBAKE = /[A-Za-zÀ-ỹ]\?[A-Za-zÀ-ỹ]|\d\?\d|\uFFFD/
// Vietnamese text whose diacritics were stripped (long string, no non-ASCII letters at all).
const looksDeaccented = (s: string) => {
  const plain = s.replace(/\$\$[\s\S]*?\$\$|\$[^$]*\$|`[^`]*`|```[\s\S]*?```/g, '')
  return plain.replace(/[^A-Za-z]/g, '').length > 60 && !/[À-ỹĐđ]/.test(plain)
}
const checkText = (s: string, where: string, vi: boolean) => {
  if (MOJIBAKE.test(s.replace(/```[\s\S]*?```|`[^`]*`/g, ''))) errors.push(`${where}: corrupted characters (encoding?) "${s.slice(0, 50)}"`)
  if (vi && looksDeaccented(s)) errors.push(`${where}: Vietnamese text without diacritics "${s.slice(0, 50)}"`)
}
const walkText = (o: unknown, where: string, vi = false): void => {
  if (typeof o === 'string') checkText(o, where, vi)
  else if (Array.isArray(o)) o.forEach((v, i) => walkText(v, `${where}[${i}]`, vi))
  else if (o && typeof o === 'object')
    for (const [k, v] of Object.entries(o)) if (!['code', 'starter', 'solution', 'tests', 'tex', 'numpy'].includes(k)) walkText(v, `${where}.${k}`, k === 'vi' || vi)
}
const walkTex = (o: unknown, where: string, codeOpts = false): void => {
  if (typeof o === 'string') checkTex(o, where)
  else if (Array.isArray(o)) o.forEach((v, i) => walkTex(v, `${where}[${i}]`, codeOpts))
  else if (o && typeof o === 'object')
    for (const [k, v] of Object.entries(o)) {
      if (CODE_KEYS.has(k) && (k !== 'options' || codeOpts)) continue
      walkTex(v, `${where}.${k}`, k === 'exercises' ? false : codeOpts)
    }
}
for (const c of cat.courses)
  for (const w of c.weeks) {
    for (const lang of ['vi', 'en'] as const) {
      const where = `${w.key}/notes.${lang}.md`
      checkTex(w.notes[lang], where)
      // Parse exactly like the site (remark-math) so mismatched $/$$ delimiters are caught too.
      const tree = mdParser.parse(w.notes[lang])
      visit(tree, (node: { type: string; value?: string }) => {
        if ((node.type === 'math' || node.type === 'inlineMath') && node.value !== undefined)
          try {
            katex.renderToString(node.value, { throwOnError: true, displayMode: node.type === 'math' })
          } catch (e) {
            errors.push(`${where}: ${(e as Error).message.slice(0, 100)} in: ${node.value.slice(0, 60)}`)
          }
      })
      // Lines starting with the tail of a LaTeX command whose backslash escape was swallowed (\right → CR + "ight").
      w.notes[lang].split('\n').forEach((l, i) => {
        if (/^(ight|ext\{|heta|imes|abla|rac\{|egin\{|eq\b|ilde|ar\{)/.test(l)) errors.push(`${where}:${i + 1}: broken LaTeX escape "${l.slice(0, 40)}"`)
      })
    }
    for (const e of w.exercises) walkTex(e, `${w.key}/${e.id}`, e.type === 'predict-output')
    for (const f of w.flashcards) walkTex(f, `${w.key}/${f.id}`)
    for (const e of w.exercises) walkText(e, `${w.key}/${e.id}`)
    for (const f of w.flashcards) walkText(f, `${w.key}/${f.id}`)
    for (const f of w.formulas) walkText(f, `${w.key}/formulas/${f.id}`)
    walkText(w.spec, `${w.key}/week.yaml`)
    for (const lang of ['vi', 'en'] as const)
      w.notes[lang].split('\n').forEach((l, i) => {
        if (MOJIBAKE.test(l.replace(/`[^`]*`/g, ''))) errors.push(`${w.key}/notes.${lang}.md:${i + 1}: corrupted characters "${l.slice(0, 50)}"`)
      })
    if (w.notes.vi && !/[À-ỹĐđ]/.test(w.notes.vi)) errors.push(`${w.key}/notes.vi.md: no Vietnamese diacritics at all`)
    for (const f of w.formulas) {
      try {
        katex.renderToString(f.tex, { throwOnError: true, displayMode: true })
      } catch (e) {
        errors.push(`${w.key}/formulas/${f.id}: ${(e as Error).message.slice(0, 100)}`)
      }
      walkTex({ name: f.name, where: f.where }, `${w.key}/formulas/${f.id}`)
    }
  }

let stats = ''
for (const c of cat.courses)
  for (const w of c.weeks) {
    const byType: Record<string, number> = {}
    for (const e of w.exercises) byType[e.type] = (byType[e.type] ?? 0) + 1
    const kinds: Record<string, number> = {}
    for (const f of w.flashcards) kinds[f.kind] = (kinds[f.kind] ?? 0) + 1
    stats += `\n${w.key}: ${w.exercises.length} exercises, ${w.formulas.length} formulas, ${w.flashcards.length} flashcards (${Object.entries(kinds)
      .map(([k, v]) => `${k}=${v}`)
      .join(' ')})\n  ${Object.entries(byType)
      .map(([k, v]) => `${k}=${v}`)
      .join(' ')}`
  }

if (!process.argv.includes('--skip-code')) {
  const code = cat.courses.flatMap((c) => c.weeks.filter((w) => inScope(w.key)).flatMap((w) => w.exercises.filter((e) => e.type === 'code')))
  if (code.length) {
    const { loadPyodide } = await import('pyodide')
    const py = await loadPyodide()
    await py.loadPackage('numpy', { messageCallback: () => {} })
    for (const e of code) {
      if (e.type !== 'code') continue
      await py.loadPackagesFromImports(e.solution + '\n' + e.tests + '\n' + e.starter, { messageCallback: () => {} })
      const src = buildHarness(e.solution, e.tests)
      try {
        const out = JSON.parse(await py.runPythonAsync(src))
        if (!out.ok) errors.push(`code ${e.id}: reference solution fails: ${out.error}`)
      } catch (err) {
        errors.push(`code ${e.id}: ${(err as Error).message}`)
      }
      try {
        const out = JSON.parse(await py.runPythonAsync(buildHarness(e.starter, e.tests)))
        if (out.ok) errors.push(`code ${e.id}: starter code already passes the tests`)
      } catch {
        /* starter failing is expected */
      }
    }
  }
}

console.log(stats.trim())
const shown = errors.filter((e) => inScope(e) || e.startsWith('code '))
if (only && shown.length < errors.length) console.log(`\n(${errors.length - shown.length} error(s) outside --only ${only} hidden)`)
if (shown.length) {
  console.error(`\n${shown.length} error(s):\n` + shown.map((e) => ' - ' + e).join('\n'))
  process.exit(1)
}
console.log('\nContent OK')
