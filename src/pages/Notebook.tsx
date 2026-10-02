import { useMemo } from 'react'
import { ExternalLink } from 'lucide-react'
import { useLocation, useNavigate } from 'react-router-dom'
import { allWeeks } from '../content'
import { useLang } from '../lib/i18n'
import { liteUrl, notebookRoute } from '../lib/notebook'

export default function Notebook() {
  const { t, ui } = useLang()
  const loc = useLocation()
  const nav = useNavigate()
  const path = new URLSearchParams(loc.search).get('path') ?? 'playground.ipynb'
  const src = useMemo(() => liteUrl(path), [path])
  const options = [
    { path: 'playground.ipynb', label: 'Playground' },
    ...allWeeks.map((w) => ({ path: `${w.courseId}-${w.spec.id}.ipynb`, label: t(w.spec.title) })),
  ]
  return (
    <div className="notebook-page">
      <div className="notebook-bar">
        <h1>JupyterLite</h1>
        <select value={options.some((o) => o.path === path) ? path : ''} onChange={(e) => nav(notebookRoute(e.target.value))}>
          {!options.some((o) => o.path === path) && <option value="">{path}</option>}
          {options.map((o) => (
            <option key={o.path} value={o.path}>{o.label}</option>
          ))}
        </select>
        <span className="muted small grow">{ui('notebookHint')}</span>
        <a className="btn small" href={src} target="_blank" rel="noreferrer"><ExternalLink size={14} /> {ui('openNewTab')}</a>
      </div>
      <iframe key={src} className="notebook-frame" src={src} title="JupyterLite" allow="clipboard-read; clipboard-write" />
    </div>
  )
}
