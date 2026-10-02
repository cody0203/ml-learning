import { useEffect, useState } from 'react'
import { NavLink, Route, Routes, useLocation } from 'react-router-dom'
import {
  ChevronRight, CircleAlert, Layers, LayoutDashboard, Menu, Moon, Settings as SettingsIcon, Sigma, SquareTerminal, Sun, Target,
} from 'lucide-react'
import { catalog } from './content'
import { useLang } from './lib/i18n'
import { useProgress } from './lib/progress'
import { isDue } from './lib/srs'
import Home from './pages/Home'
import WeekPage from './pages/WeekPage'
import TopicPage from './pages/TopicPage'
import Practice from './pages/Practice'
import Flashcards from './pages/Flashcards'
import Mistakes from './pages/Mistakes'
import Notebook from './pages/Notebook'
import Formulas from './pages/Formulas'
import Settings from './pages/Settings'
import SingleExercise from './pages/SingleExercise'

export default function App() {
  const { t, ui, lang, setLang } = useLang()
  const p = useProgress()
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') ?? 'light')
  const [menu, setMenu] = useState(false)
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('theme', theme)
  }, [theme])
  const due = Object.values(p.cards).filter((c) => isDue(c)).length
  const mistakes = Object.keys(p.mistakes).length
  const loc = useLocation()
  const [closed, setClosed] = useState<string[]>(() => JSON.parse(localStorage.getItem('nav-closed') ?? '[]'))
  const toggleCourse = (id: string) => {
    const next = closed.includes(id) ? closed.filter((x) => x !== id) : [...closed, id]
    setClosed(next)
    localStorage.setItem('nav-closed', JSON.stringify(next))
  }
  // "Tuần 3: Vector…" → number + title, so long week names stay scannable in the sidebar.
  const splitWeek = (title: string) => {
    const m = /^\D*(\d+)\s*[:.–-]\s*(.*)$/.exec(title)
    return m ? { n: m[1], title: m[2] } : { n: '', title }
  }

  return (
    <div className={`layout ${menu ? 'menu-open' : ''}`}>
      <div className="scrim" onClick={() => setMenu(false)} aria-hidden />
      <aside className="sidebar" onClick={(e) => (e.target as HTMLElement).closest('a') && setMenu(false)}>
        <NavLink to="/" className="brand">
          <span className="brand-mark" aria-hidden>∑</span>
          <span>ML Review</span>
        </NavLink>
        <nav className="nav">
          <NavLink to="/" end><LayoutDashboard size={16} />{ui('home')}</NavLink>
          <NavLink to="/practice"><Target size={16} />{ui('practice')}</NavLink>
          <NavLink to="/formulas"><Sigma size={16} />{ui('formulas')}</NavLink>
          <NavLink to="/flashcards">
            <Layers size={16} />{ui('flashcards')} {due > 0 && <span className="pill">{due}</span>}
          </NavLink>
          <NavLink to="/mistakes">
            <CircleAlert size={16} />{ui('mistakes')} {mistakes > 0 && <span className="pill warn">{mistakes}</span>}
          </NavLink>
          <NavLink to="/notebook"><SquareTerminal size={16} />{ui('notebook')}</NavLink>
        </nav>
        {catalog.courses.map((c) => {
          const active = loc.pathname.startsWith(`/week/${c.spec.id}/`)
          const open = active || !closed.includes(c.spec.id)
          return (
            <div key={c.spec.id} className="nav-course">
              <button type="button" className="nav-heading" aria-expanded={open} onClick={() => toggleCourse(c.spec.id)}>
                <ChevronRight size={14} className="chev" />
                <span>{t(c.spec.title).replace(/^.*?:\s*/, '')}</span>
              </button>
              {open && (
                <nav className="nav nav-weeks">
                  {c.weeks.map((w) => {
                    const s = splitWeek(t(w.spec.title))
                    return (
                      <NavLink key={w.key} to={`/week/${w.key}`} title={t(w.spec.title)}>
                        <span className="wk-num">{s.n}</span>
                        <span className="wk-title">{s.title}</span>
                      </NavLink>
                    )
                  })}
                </nav>
              )}
            </div>
          )
        })}
        <div className="sidebar-foot nav">
          <NavLink to="/settings"><SettingsIcon size={16} />{ui('settings')}</NavLink>
        </div>
      </aside>
      <div className="main">
        <header className="topbar">
          <button type="button" className="icon-btn menu-btn" onClick={() => setMenu((m) => !m)} aria-label="menu">
            <Menu size={18} />
          </button>
          <span className="spacer" />
          <div className="seg" role="group" aria-label={ui('language')}>
            <button type="button" className={lang === 'vi' ? 'on' : ''} onClick={() => setLang('vi')}>VI</button>
            <button type="button" className={lang === 'en' ? 'on' : ''} onClick={() => setLang('en')}>EN</button>
          </div>
          <button type="button" className="icon-btn" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} aria-label={ui('theme')}>
            {theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
          </button>
        </header>
        <main className="content">
          {catalog.errors.length > 0 && (
            <details className="card content-errors">
              <summary>{ui('contentErrors')} ({catalog.errors.length})</summary>
              <pre>{catalog.errors.join('\n')}</pre>
            </details>
          )}
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/week/:course/:week" element={<WeekPage />} />
            <Route path="/week/:course/:week/t/:topic" element={<TopicPage />} />
            <Route path="/practice" element={<Practice />} />
            <Route path="/flashcards" element={<Flashcards />} />
            <Route path="/mistakes" element={<Mistakes />} />
            <Route path="/notebook" element={<Notebook />} />
            <Route path="/formulas" element={<Formulas />} />
            <Route path="/playground" element={<Notebook />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/ex/:id" element={<SingleExercise />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}
