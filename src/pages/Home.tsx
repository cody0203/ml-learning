import { Link } from 'react-router-dom'
import { allExercises, allWeeks, catalog } from '../content'
import { useLang } from '../lib/i18n'
import { mastery, streak, today, useProgress } from '../lib/progress'
import { isDue } from '../lib/srs'
import { encodeConfig } from '../lib/session'
import { Bar } from '../components/Bar'
import { ChevronRight, CircleAlert, Layers, Zap } from 'lucide-react'

function Heatmap({ activity }: { activity: Record<string, number> }) {
  const now = new Date()
  const start = new Date(now)
  start.setDate(now.getDate() - 49 - ((now.getDay() + 6) % 7)) // Monday, 7 weeks ago
  const days: { k: string; n: number }[] = []
  for (const x = new Date(start); x <= now; x.setDate(x.getDate() + 1)) days.push({ k: today(x), n: activity[today(x)] ?? 0 })
  const lvl = (n: number) => (n === 0 ? 0 : n < 5 ? 1 : n < 15 ? 2 : n < 30 ? 3 : 4)
  return (
    <div className="heatmap">
      {days.map((x) => (
        <span key={x.k} className={`hm l${lvl(x.n)}`} title={`${x.k}: ${x.n}`} />
      ))}
    </div>
  )
}

export default function Home() {
  const { t, ui, lang } = useLang()
  const p = useProgress()
  const all = mastery(p, allExercises.map((e) => e.id))
  const due = Object.values(p.cards).filter((c) => isDue(c)).length
  const newCards = allWeeks.reduce((s, w) => s + w.flashcards.filter((f) => !p.cards[`f:${f.id}`]).length, 0)
  const mistakes = Object.keys(p.mistakes).length
  const days = streak(p.activity)

  const topicStats = allWeeks.flatMap((w) =>
    w.spec.topics.map((tp) => {
      const ids = w.exercises.filter((e) => e.topic === tp.id).map((e) => e.id)
      return { w, tp, ids, ...mastery(p, ids) }
    }),
  )
  const weak = topicStats.filter((s) => s.attempted > 0 && s.accuracy < 0.75).sort((a, b) => a.accuracy - b.accuracy).slice(0, 5)
  const vi = lang === 'vi'

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>{ui('home')}</h1>
          <p className="lede">
            {vi
              ? <>Đã làm <b>{all.attempted}</b>/{allExercises.length} bài · độ chính xác <b>{Math.round(all.accuracy * 100)}%</b> · chuỗi <b>{days}</b> ngày</>
              : <><b>{all.attempted}</b>/{allExercises.length} exercises attempted · <b>{Math.round(all.accuracy * 100)}%</b> accuracy · <b>{days}</b>-day streak</>}
          </p>
        </div>
      </header>

      <div className="home-grid">
        <div className="home-main">
          {catalog.courses.map((c) => (
            <section key={c.spec.id} className="course-block">
              <h2>{t(c.spec.title)}</h2>
              <ul className="week-rows">
                {c.weeks.map((w) => {
                  const m = mastery(p, w.exercises.map((e) => e.id))
                  const title = t(w.spec.title)
                  const [num, rest] = title.includes(':') ? [title.slice(0, title.indexOf(':')), title.slice(title.indexOf(':') + 1).trim()] : ['', title]
                  return (
                    <li key={w.key}>
                      <Link to={`/week/${w.key}`} className="week-row">
                        <span className="week-num">{num}</span>
                        <span className="week-main">
                          <span className="week-title">{rest}</span>
                          <span className="week-meta">
                            {w.exercises.length} {ui('exercises')} · {w.formulas.length} {ui('formulas').toLowerCase()} · {w.flashcards.length} {ui('cards')}
                          </span>
                        </span>
                        <span className="week-progress">
                          <Bar value={m.mastery} />
                          <span className="pct">{Math.round(m.mastery * 100)}%</span>
                        </span>
                        <ChevronRight size={16} className="row-chev" />
                      </Link>
                    </li>
                  )
                })}
              </ul>
            </section>
          ))}
        </div>

        <aside className="home-side">
          <section className="panel">
            <h3>{vi ? 'Hôm nay' : 'Today'}</h3>
            <Link className="action-row" to="/flashcards">
              <Layers size={18} />
              <span className="grow">
                <b>{ui('continueReview')}</b>
                <span className="muted small">{due} {ui('due')} · {Math.min(newCards, 20)} {vi ? 'thẻ mới' : 'new'}</span>
              </span>
              <ChevronRight size={16} className="row-chev" />
            </Link>
            <Link className="action-row" to={`/practice?${encodeConfig({ weeks: [], types: [], mode: 'weak', n: 10 })}&go=1`}>
              <Zap size={18} />
              <span className="grow">
                <b>{ui('quickPractice')}</b>
                <span className="muted small">{vi ? 'Ưu tiên phần bạn còn yếu' : 'Weighted towards weak spots'}</span>
              </span>
              <ChevronRight size={16} className="row-chev" />
            </Link>
            {mistakes > 0 && (
              <Link className="action-row" to="/mistakes">
                <CircleAlert size={18} />
                <span className="grow">
                  <b>{ui('mistakes')}</b>
                  <span className="muted small">{mistakes} {vi ? 'câu cần xem lại' : 'to revisit'}</span>
                </span>
                <ChevronRight size={16} className="row-chev" />
              </Link>
            )}
          </section>

          {weak.length > 0 && (
            <section className="panel">
              <h3>{ui('weakTopics')}</h3>
              <ul className="weak-list">
                {weak.map((s) => (
                  <li key={`${s.w.key}/${s.tp.id}`}>
                    <Link to={`/practice?${encodeConfig({ weeks: [s.w.key], topic: s.tp.id, types: [], mode: 'weak', n: 10 })}&go=1`}>
                      <span className="grow">{t(s.tp.title)}</span>
                      <span className="pct bad-text">{Math.round(s.accuracy * 100)}%</span>
                    </Link>
                  </li>
                ))}
              </ul>
            </section>
          )}

          <section className="panel">
            <h3>{ui('recentActivity')}</h3>
            <Heatmap activity={p.activity} />
          </section>
        </aside>
      </div>
    </div>
  )
}
