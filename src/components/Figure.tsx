import { useId } from 'react'
import type { FigureSpec } from '../content/schema'
import { evaluate } from '../lib/expr'

const W = 260
const PALETTE = ['#2563eb', '#dc2626', '#16a34a', '#9333ea', '#ea580c', '#0891b2']

export function compile(src: string): ((x: number) => number) | null {
  try {
    evaluate(src, { x: 0.5 })
  } catch {
    return null
  }
  return (x) => {
    try {
      return evaluate(src, { x })
    } catch {
      return NaN
    }
  }
}

export function Figure({ spec, size = W }: { spec: FigureSpec; size?: number }) {
  const uid = useId().replace(/:/g, '')
  const [x0, x1] = spec.xRange ?? [-6, 6]
  const [y0, y1] = spec.yRange ?? [-6, 6]
  const ratio = (y1 - y0) / (x1 - x0)
  // Keep 1:1 scale for geometry (vectors, lines); use a fixed aspect for plots with very different axis ranges.
  const h = ratio > 0.4 && ratio < 1.6 ? size * ratio : size * 0.65
  const sx = (x: number) => ((x - x0) / (x1 - x0)) * size
  const sy = (y: number) => h - ((y - y0) / (y1 - y0)) * h
  const niceStep = (span: number) => {
    const raw = span / 10
    const p = 10 ** Math.floor(Math.log10(raw))
    const m = raw / p
    return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p
  }
  const ticks = (a: number, b: number) => {
    const step = niceStep(b - a)
    const out: number[] = []
    for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10))
    return out
  }
  const labelEvery = (a: number, b: number) => {
    const ts = ticks(a, b)
    const k = Math.ceil(ts.length / 6)
    return ts.filter((_, i) => i % k === 0 && Math.abs(ts[i]) > 1e-12)
  }
  const ax = Math.min(Math.max(0, y0), y1) // y of the x-axis
  const ay = Math.min(Math.max(0, x0), x1) // x of the y-axis

  const lineSeg = (a: number, b: number, c: number): [number, number, number, number] | null => {
    if (Math.abs(b) > 1e-12) return [x0, (c - a * x0) / b, x1, (c - a * x1) / b]
    if (Math.abs(a) > 1e-12) return [c / a, y0, c / a, y1]
    return null
  }

  return (
    <svg className="figure" width={size} height={h} viewBox={`0 0 ${size} ${h}`} role="img">
      <defs>
        <clipPath id={`${uid}clip`}>
          <rect x={0} y={0} width={size} height={h} />
        </clipPath>
        {PALETTE.map((c, i) => (
          <marker key={c} id={`${uid}a${i}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill={c} />
          </marker>
        ))}
      </defs>
      {ticks(x0, x1).map((v) => (
        <line key={`gx${v}`} x1={sx(v)} x2={sx(v)} y1={0} y2={h} className="grid" />
      ))}
      {ticks(y0, y1).map((v) => (
        <line key={`gy${v}`} y1={sy(v)} y2={sy(v)} x1={0} x2={size} className="grid" />
      ))}
      {x0 <= 0 && x1 >= 0 && <line x1={sx(0)} x2={sx(0)} y1={0} y2={h} className="axis" />}
      {y0 <= 0 && y1 >= 0 && <line y1={sy(0)} y2={sy(0)} x1={0} x2={size} className="axis" />}
      {labelEvery(x0, x1).map((v) => (
        <text key={`tx${v}`} x={sx(v)} y={Math.min(sy(ax) + 11, h - 2)} textAnchor="middle" className="tick">{+v.toPrecision(4)}</text>
      ))}
      {labelEvery(y0, y1).map((v) => (
        <text key={`ty${v}`} x={Math.max(sx(ay) - 3, 22)} y={sy(v) + 3} textAnchor="end" className="tick">{+v.toPrecision(4)}</text>
      ))}
      {spec.xLabel && <text x={size - 4} y={Math.max(sy(ax) - 4, 10)} textAnchor="end" className="flabel">{spec.xLabel}</text>}
      {spec.yLabel && <text x={Math.min(sx(ay) + 4, size - 40)} y={11} className="flabel">{spec.yLabel}</text>}
      {spec.areas.map((a, i) => {
        const f = compile(a.f)
        const lo = Math.max(a.from, x0)
        const hi = Math.min(a.to, x1)
        if (!f || hi <= lo) return null
        const pts: string[] = [`${sx(lo)},${sy(0)}`]
        for (let k = 0; k <= 80; k++) {
          const x = lo + ((hi - lo) * k) / 80
          const y = f(x)
          if (Number.isFinite(y)) pts.push(`${sx(x)},${sy(Math.max(Math.min(y, y1), y0))}`)
        }
        pts.push(`${sx(hi)},${sy(0)}`)
        return <polygon key={`a${i}`} points={pts.join(' ')} fill={a.color ?? PALETTE[i % PALETTE.length]} fillOpacity={0.25} stroke="none" />
      })}
      {spec.bars.map((b, i) => {
        const w = b.width ?? 0.8
        const color = b.color ?? PALETTE[0]
        const top = sy(Math.max(b.h, 0))
        const bottom = sy(Math.min(b.h, 0))
        return (
          <g key={`b${i}`}>
            <rect x={sx(b.x - w / 2)} y={top} width={sx(b.x + w / 2) - sx(b.x - w / 2)} height={Math.max(bottom - top, 0.5)} fill={color} fillOpacity={0.75} />
            {b.label && <text x={sx(b.x)} y={top - 4} textAnchor="middle" className="flabel">{b.label}</text>}
          </g>
        )
      })}
      {spec.functions.map((fn, i) => {
        const f = compile(fn.f)
        if (!f) return null
        const [d0, d1] = fn.domain ?? [x0, x1]
        const lo = Math.max(d0, x0)
        const hi = Math.min(d1, x1)
        const color = fn.color ?? PALETTE[i % PALETTE.length]
        // Split the path where the function leaves the window or is undefined.
        let d = ''
        let pen = false
        let last: [number, number] | null = null
        const N = 240
        for (let k = 0; k <= N; k++) {
          const x = lo + ((hi - lo) * k) / N
          const y = f(x)
          const inside = Number.isFinite(y) && y >= y0 - (y1 - y0) && y <= y1 + (y1 - y0)
          if (inside) {
            d += `${pen ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(y).toFixed(1)} `
            pen = true
            if (y >= y0 && y <= y1) last = [x, y]
          } else pen = false
        }
        return (
          <g key={`f${i}`}>
            <path d={d} fill="none" stroke={color} strokeWidth={2} strokeDasharray={fn.dashed ? '5 4' : undefined} clipPath={`url(#${uid}clip)`} />
            {fn.label && last && <text x={Math.min(sx(last[0]) + 4, size - 50)} y={Math.max(sy(last[1]) - 6, 12)} fill={color} className="flabel">{fn.label}</text>}
          </g>
        )
      })}
      {spec.lines.map((l, i) => {
        const seg = lineSeg(l.a, l.b, l.c)
        if (!seg) return null
        const color = l.color ?? PALETTE[i % PALETTE.length]
        const lx = Math.min(Math.max(sx(seg[2]), 4), size - 40)
        const ly = Math.min(Math.max(sy(seg[3]), 12), h - 4)
        return (
          <g key={`l${i}`}>
            <line x1={sx(seg[0])} y1={sy(seg[1])} x2={sx(seg[2])} y2={sy(seg[3])} stroke={color} strokeWidth={2} />
            {l.label && <text x={lx} y={ly} fill={color} className="flabel">{l.label}</text>}
          </g>
        )
      })}
      {spec.vectors.map((v, i) => {
        const ci = PALETTE.indexOf(v.color ?? '') >= 0 ? PALETTE.indexOf(v.color!) : i % PALETTE.length
        const color = PALETTE[ci]
        const [fx, fy] = v.from ?? [0, 0]
        return (
          <g key={`v${i}`}>
            <line x1={sx(fx)} y1={sy(fy)} x2={sx(v.to[0])} y2={sy(v.to[1])} stroke={color} strokeWidth={2.5} markerEnd={`url(#${uid}a${ci})`} />
            {v.label && <text x={sx(v.to[0]) + 4} y={sy(v.to[1]) - 4} fill={color} className="flabel">{v.label}</text>}
          </g>
        )
      })}
      {spec.points.map((p, i) => (
        <g key={`p${i}`}>
          <circle cx={sx(p.at[0])} cy={sy(p.at[1])} r={4} fill={p.color ?? '#111827'} />
          {p.label && <text x={sx(p.at[0]) + 6} y={sy(p.at[1]) - 6} className="flabel">{p.label}</text>}
        </g>
      ))}
    </svg>
  )
}
