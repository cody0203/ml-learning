import { useRef, useState } from 'react'
import { Download, Upload } from 'lucide-react'
import { useLang } from '../lib/i18n'
import { progress, today } from '../lib/progress'

export default function Settings() {
  const { ui, lang, setLang } = useLang()
  const file = useRef<HTMLInputElement>(null)
  const [msg, setMsg] = useState('')

  const doExport = () => {
    const blob = new Blob([progress.export()], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `ml-learning-progress-${today()}.json`
    a.click()
    URL.revokeObjectURL(a.href)
  }
  const doImport = async (f: File) => {
    try {
      progress.import(await f.text())
      setMsg('✓')
    } catch (e) {
      setMsg(`✗ ${(e as Error).message}`)
    }
  }

  return (
    <div className="page narrow">
      <h1>{ui('settings')}</h1>
      <div className="card form">
        <fieldset>
          <legend>{ui('language')}</legend>
          <div className="chips">
            <button type="button" className={`chip ${lang === 'vi' ? 'on' : ''}`} onClick={() => setLang('vi')}>Tiếng Việt</button>
            <button type="button" className={`chip ${lang === 'en' ? 'on' : ''}`} onClick={() => setLang('en')}>English</button>
          </div>
        </fieldset>
        <fieldset>
          <legend>Backup</legend>
          <p className="muted small">{ui('backup')}</p>
          <div className="row gap wrap">
            <button type="button" onClick={doExport}><Download size={14} /> {ui('export')}</button>
            <button type="button" onClick={() => file.current?.click()}><Upload size={14} /> {ui('import')}</button>
            <input ref={file} type="file" accept="application/json" hidden onChange={(e) => e.target.files?.[0] && doImport(e.target.files[0])} />
            <span>{msg}</span>
          </div>
        </fieldset>
        <fieldset>
          <legend>{lang === 'vi' ? 'Vùng nguy hiểm' : 'Danger zone'}</legend>
          <button type="button" className="danger" onClick={() => confirm(ui('confirmReset')) && progress.reset()}>{ui('resetAll')}</button>
        </fieldset>
      </div>
    </div>
  )
}
