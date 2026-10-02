import { useEffect, useState } from 'react'

/** Table of contents built from rendered headings inside `root`. */
export function Toc({ root, levels = ['H2', 'H3'], deps }: { root: React.RefObject<HTMLElement | null>; levels?: string[]; deps: unknown[] }) {
  const [items, setItems] = useState<{ id: string; text: string; level: number }[]>([])
  useEffect(() => {
    const hs = root.current?.querySelectorAll(levels.join(',')) ?? []
    setItems(
      [...hs].map((h) => {
        const c = h.cloneNode(true) as HTMLElement
        c.querySelectorAll('.katex-mathml').forEach((n) => n.remove())
        return { id: h.id, text: c.textContent ?? '', level: h.tagName === levels[0] ? 2 : 3 }
      }),
    )
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  if (!items.length) return null
  return (
    <nav className="toc">
      {items.map((it) => (
        <a
          key={it.id}
          href={`#${it.id}`}
          className={`toc-l${it.level}`}
          onClick={(e) => {
            e.preventDefault()
            document.getElementById(it.id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
          }}
        >
          {it.text}
        </a>
      ))}
    </nav>
  )
}
