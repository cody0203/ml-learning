import { useState } from 'react'
import { allFormulas, catalog } from '../content'
import { FormulaList } from '../components/FormulaList'
import { useLang } from '../lib/i18n'

export default function Formulas() {
  const { t, ui } = useLang()
  const [course, setCourse] = useState('')
  const list = course ? allFormulas.filter((f) => f.weekKey.startsWith(`${course}/`)) : allFormulas
  return (
    <div className="page">
      <div className="page-head">
        <h1>{ui('formulas')}</h1>
        <div className="chips">
          <button type="button" className={`chip ${!course ? 'on' : ''}`} onClick={() => setCourse('')}>{ui('allTypes')}</button>
          {catalog.courses.map((c) => (
            <button type="button" key={c.spec.id} className={`chip ${course === c.spec.id ? 'on' : ''}`} onClick={() => setCourse(c.spec.id)}>
              {t(c.spec.title)}
            </button>
          ))}
        </div>
      </div>
      <FormulaList key={course} formulas={list} showWeek />
    </div>
  )
}
