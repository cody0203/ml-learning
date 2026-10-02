import { useEffect, useState } from 'react'

export interface PyResult {
  ok: boolean
  error?: string | null
  stdout: string
  passed?: number
  total?: number
  failures?: string[]
}

const TIMEOUT_MS = 20_000

let worker: Worker | null = null
let seq = 0
const pending = new Map<number, { resolve: (r: PyResult) => void; timer: number }>()
let status: 'idle' | 'loading' | 'ready' = 'idle'
const statusListeners = new Set<(s: typeof status) => void>()

function setStatus(s: typeof status) {
  status = s
  statusListeners.forEach((l) => l(s))
}

function getWorker() {
  if (worker) return worker
  worker = new Worker(new URL('./py.worker.ts', import.meta.url), { type: 'module' })
  worker.onmessage = (ev: MessageEvent<{ id: number; result: PyResult }>) => {
    const p = pending.get(ev.data.id)
    if (!p) return
    clearTimeout(p.timer)
    pending.delete(ev.data.id)
    if (status !== 'ready') setStatus('ready')
    p.resolve(ev.data.result)
  }
  return worker
}

function call(kind: 'init' | 'run' | 'test', code = '', tests = ''): Promise<PyResult> {
  if (status === 'idle') setStatus('loading')
  const w = getWorker()
  const id = ++seq
  return new Promise((resolve) => {
    const timeout = kind === 'init' ? 120_000 : status === 'ready' ? TIMEOUT_MS : 120_000
    const timer = window.setTimeout(() => {
      // Likely an infinite loop: kill the worker; the next call starts a fresh one.
      pending.delete(id)
      w.terminate()
      worker = null
      setStatus('idle')
      pending.forEach((p) => p.resolve({ ok: false, error: 'Interrupted', stdout: '' }))
      pending.clear()
      resolve({ ok: false, error: `Timed out after ${timeout / 1000}s (infinite loop?)`, stdout: '' })
    }, timeout)
    pending.set(id, { resolve, timer })
    w.postMessage({ id, kind, code, tests })
  })
}

export const py = {
  preload: () => call('init'),
  run: (code: string) => call('run', code),
  test: (code: string, tests: string) => call('test', code, tests),
}

export function usePyStatus() {
  const [s, setS] = useState(status)
  useEffect(() => {
    statusListeners.add(setS)
    return () => {
      statusListeners.delete(setS)
    }
  }, [])
  return s
}
