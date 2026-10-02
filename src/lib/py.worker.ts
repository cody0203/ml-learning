/// <reference lib="webworker" />
import { buildHarness } from './pyHarness'

const PYODIDE_URL = 'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/'

interface PyodideLike {
  runPythonAsync(code: string): Promise<unknown>
  loadPackage(names: string[]): Promise<void>
  loadPackagesFromImports(code: string): Promise<void>
}

let ready: Promise<PyodideLike> | null = null

function init(): Promise<PyodideLike> {
  ready ??= (async () => {
    const mod = await import(/* @vite-ignore */ `${PYODIDE_URL}pyodide.mjs`)
    const py: PyodideLike = await mod.loadPyodide({ indexURL: PYODIDE_URL })
    await py.loadPackage(['numpy'])
    return py
  })()
  return ready
}

const plainHarness = (code: string) => `
import json, sys, io, traceback
_buf = io.StringIO(); _old = sys.stdout; sys.stdout = _buf; sys.stderr = _buf
_err = None
try:
    exec(compile(${JSON.stringify(code)}, "main.py", "exec"), {"__name__": "__main__"})
except Exception:
    _err = traceback.format_exc(limit=3)
finally:
    sys.stdout = _old; sys.stderr = sys.__stderr__
json.dumps({"ok": _err is None, "error": _err, "stdout": _buf.getvalue()[-20000:]})
`

self.onmessage = async (ev: MessageEvent<{ id: number; kind: 'init' | 'run' | 'test'; code?: string; tests?: string }>) => {
  const { id, kind, code = '', tests = '' } = ev.data
  try {
    const py = await init()
    if (kind === 'init') return self.postMessage({ id, result: { ok: true } })
    await py.loadPackagesFromImports(code)
    const src = kind === 'test' ? buildHarness(code, tests) : plainHarness(code)
    const out = (await py.runPythonAsync(src)) as string
    self.postMessage({ id, result: JSON.parse(out) })
  } catch (e) {
    self.postMessage({ id, result: { ok: false, error: String((e as Error).message ?? e), stdout: '' } })
  }
}
