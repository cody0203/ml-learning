/** Split week notes into an optional preamble and one section per `## ` heading (code fences respected). */
export function splitNotes(md: string): { preamble: string; sections: { title: string; body: string }[] } {
  const lines = md.split(/\r?\n/)
  const sections: { title: string; body: string[] }[] = []
  const pre: string[] = []
  let fence = false
  for (const line of lines) {
    if (/^\s*(```|~~~)/.test(line)) fence = !fence
    const m = !fence && /^##\s+(.+?)\s*#*\s*$/.exec(line)
    if (m) sections.push({ title: m[1], body: [] })
    else (sections.length ? sections[sections.length - 1].body : pre).push(line)
  }
  return { preamble: pre.join('\n').trim(), sections: sections.map((s) => ({ title: s.title, body: s.body.join('\n').trim() })) }
}
