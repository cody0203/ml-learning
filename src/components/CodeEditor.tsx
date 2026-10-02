import type { KeyboardEvent } from 'react'

/** Minimal code editor: textarea with Tab indentation and Ctrl+Enter to run. */
export function CodeEditor({ value, onChange, onRun, rows = 12 }: { value: string; onChange: (s: string) => void; onRun?: () => void; rows?: number }) {
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    const ta = e.currentTarget
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      onRun?.()
      return
    }
    if (e.key === 'Tab') {
      e.preventDefault()
      const { selectionStart: s, selectionEnd: end } = ta
      const next = value.slice(0, s) + '    ' + value.slice(end)
      onChange(next)
      requestAnimationFrame(() => ta.setSelectionRange(s + 4, s + 4))
    }
    if (e.key === 'Enter') {
      const s = ta.selectionStart
      const line = value.slice(value.lastIndexOf('\n', s - 1) + 1, s)
      let indent = /^\s*/.exec(line)![0]
      if (line.trimEnd().endsWith(':')) indent += '    '
      if (indent) {
        e.preventDefault()
        const next = value.slice(0, s) + '\n' + indent + value.slice(ta.selectionEnd)
        onChange(next)
        requestAnimationFrame(() => ta.setSelectionRange(s + 1 + indent.length, s + 1 + indent.length))
      }
    }
  }
  return (
    <textarea
      className="code-editor"
      spellCheck={false}
      value={value}
      rows={Math.max(rows, value.split('\n').length + 1)}
      onChange={(e) => onChange(e.target.value)}
      onKeyDown={onKey}
    />
  )
}
