import { tryEvaluate } from '../lib/expr'

export type Grid = string[][]

export const emptyGrid = (r: number, c: number): Grid => Array.from({ length: r }, () => Array(c).fill(''))

export function gridValues(g: Grid): number[][] | null {
  const out: number[][] = []
  for (const row of g) {
    const r: number[] = []
    for (const cell of row) {
      const v = tryEvaluate(cell)
      if (v === null) return null
      r.push(v)
    }
    out.push(r)
  }
  return out
}

interface Props {
  value: Grid
  onChange: (g: Grid) => void
  disabled?: boolean
  resizable?: boolean
  cellState?: (i: number, j: number) => 'ok' | 'bad' | undefined
}

export function MatrixInput({ value, onChange, disabled, resizable, cellState }: Props) {
  const rows = value.length
  const cols = value[0]?.length ?? 0
  const set = (i: number, j: number, s: string) => onChange(value.map((r, a) => (a === i ? r.map((c, b) => (b === j ? s : c)) : r)))
  const resize = (r: number, c: number) => {
    if (r < 1 || c < 1 || r > 6 || c > 6) return
    onChange(Array.from({ length: r }, (_, i) => Array.from({ length: c }, (_, j) => value[i]?.[j] ?? '')))
  }
  return (
    <div className="matrix-input-wrap">
      <div className="matrix-input" style={{ gridTemplateColumns: `repeat(${cols}, minmax(3.2rem, 4.5rem))` }}>
        {value.map((row, i) =>
          row.map((cell, j) => {
            const st = cellState?.(i, j)
            const invalid = cell.trim() !== '' && tryEvaluate(cell) === null
            return (
              <input
                key={`${i}-${j}`}
                className={`cell ${st ?? ''} ${invalid ? 'invalid' : ''}`}
                value={cell}
                disabled={disabled}
                inputMode="decimal"
                aria-label={`row ${i + 1} col ${j + 1}`}
                onChange={(e) => set(i, j, e.target.value)}
                onKeyDown={(e) => {
                  const move = (di: number, dj: number) => {
                    const el = e.currentTarget.parentElement?.querySelector<HTMLInputElement>(`[data-cell="${i + di}-${j + dj}"]`)
                    el?.focus()
                  }
                  if (e.key === 'ArrowDown') move(1, 0)
                  if (e.key === 'ArrowUp') move(-1, 0)
                }}
                data-cell={`${i}-${j}`}
              />
            )
          }),
        )}
      </div>
      {resizable && !disabled && (
        <div className="resize">
          <span>{rows}×{cols}</span>
          <button type="button" onClick={() => resize(rows + 1, cols)}>+row</button>
          <button type="button" onClick={() => resize(rows - 1, cols)}>−row</button>
          <button type="button" onClick={() => resize(rows, cols + 1)}>+col</button>
          <button type="button" onClick={() => resize(rows, cols - 1)}>−col</button>
        </div>
      )}
    </div>
  )
}
