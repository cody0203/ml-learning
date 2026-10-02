import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import type { Localized } from '../content/schema'

export type Lang = 'vi' | 'en'

const UI = {
  home: ['Tổng quan', 'Dashboard'],
  notes: ['Ghi chú', 'Notes'],
  practice: ['Luyện tập', 'Practice'],
  flashcards: ['Flashcards', 'Flashcards'],
  mistakes: ['Sổ lỗi sai', 'Mistake log'],
  playground: ['Python', 'Python'],
  settings: ['Cài đặt', 'Settings'],
  check: ['Kiểm tra', 'Check'],
  next: ['Tiếp', 'Next'],
  finish: ['Kết thúc', 'Finish'],
  retry: ['Làm lại', 'Retry'],
  correct: ['Chính xác!', 'Correct!'],
  incorrect: ['Chưa đúng', 'Not quite'],
  explanation: ['Giải thích', 'Explanation'],
  hint: ['Gợi ý', 'Hint'],
  showHint: ['Xem gợi ý', 'Show hint'],
  showAnswer: ['Xem đáp án', 'Show answer'],
  run: ['Chạy', 'Run'],
  runTests: ['Chạy test', 'Run tests'],
  reset: ['Đặt lại', 'Reset'],
  solution: ['Lời giải mẫu', 'Reference solution'],
  loadingPython: ['Đang tải Python (lần đầu hơi lâu)…', 'Loading Python (first time is slow)…'],
  exercises: ['bài tập', 'exercises'],
  cards: ['thẻ', 'cards'],
  due: ['đến hạn ôn', 'due'],
  mastery: ['Thành thạo', 'Mastery'],
  attempted: ['Đã làm', 'Attempted'],
  accuracy: ['Độ chính xác', 'Accuracy'],
  streak: ['Chuỗi ngày', 'Day streak'],
  topics: ['Chủ đề', 'Topics'],
  start: ['Bắt đầu', 'Start'],
  mode: ['Chế độ', 'Mode'],
  modeMixed: ['Trộn ngẫu nhiên', 'Mixed'],
  modeWeak: ['Ưu tiên phần yếu', 'Weak spots'],
  modeNew: ['Chưa làm', 'Unseen first'],
  modeExam: ['Thi thử (tính giờ)', 'Exam (timed)'],
  count: ['Số câu', 'Questions'],
  types: ['Dạng bài', 'Types'],
  allTypes: ['Tất cả', 'All'],
  scope: ['Phạm vi', 'Scope'],
  allWeeks: ['Tất cả các tuần', 'All weeks'],
  allTopics: ['Tất cả chủ đề', 'All topics'],
  difficulty: ['Độ khó', 'Difficulty'],
  score: ['Điểm', 'Score'],
  timeLeft: ['Còn lại', 'Time left'],
  review: ['Xem lại', 'Review'],
  noExercises: ['Không có bài tập phù hợp.', 'No matching exercises.'],
  again: ['Quên', 'Again'],
  hard: ['Khó', 'Hard'],
  good: ['Nhớ', 'Good'],
  easy: ['Dễ', 'Easy'],
  flip: ['Lật thẻ', 'Flip'],
  noDue: ['Không còn thẻ đến hạn 🎉', 'No cards due 🎉'],
  studyAhead: ['Ôn thêm thẻ ngẫu nhiên', 'Study random cards anyway'],
  noMistakes: ['Chưa có lỗi sai nào.', 'No mistakes yet.'],
  practiceMistakes: ['Luyện lại các câu sai', 'Practice these again'],
  clear: ['Xoá', 'Clear'],
  export: ['Xuất tiến độ (JSON)', 'Export progress (JSON)'],
  import: ['Nhập tiến độ', 'Import progress'],
  resetAll: ['Xoá toàn bộ tiến độ', 'Reset all progress'],
  confirmReset: ['Chắc chắn xoá toàn bộ tiến độ?', 'Really delete all progress?'],
  language: ['Ngôn ngữ', 'Language'],
  theme: ['Giao diện', 'Theme'],
  contentErrors: ['Lỗi nội dung', 'Content errors'],
  yourAnswer: ['Câu trả lời của bạn', 'Your answer'],
  sample: ['Ví dụ một đáp án đúng', 'One valid answer'],
  rows: ['hàng', 'rows'],
  cols: ['cột', 'cols'],
  exprHint: ['Có thể nhập 7/3, sqrt(2), pi…', 'You can type 7/3, sqrt(2), pi…'],
  true: ['Đúng', 'True'],
  false: ['Sai', 'False'],
  why: ['Vì sao?', 'Why?'],
  selectWrongStep: ['Chọn bước SAI đầu tiên', 'Select the first WRONG step'],
  dragOrder: ['Sắp xếp các bước theo đúng thứ tự (dùng ↑ ↓)', 'Put the steps in order (use ↑ ↓)'],
  matchHint: ['Chọn một mục bên trái rồi một mục bên phải', 'Pick a left item, then a right item'],
  multiSelect: ['(chọn tất cả đáp án đúng)', '(select all that apply)'],
  newDrill: ['Số mới', 'New numbers'],
  practiceTopic: ['Luyện chủ đề này', 'Practice this topic'],
  practiceWeek: ['Luyện cả tuần', 'Practice this week'],
  readNotes: ['Đọc ghi chú', 'Read notes'],
  result: ['Kết quả', 'Result'],
  output: ['Kết quả chạy', 'Output'],
  tests: ['Test', 'Tests'],
  passed: ['đạt', 'passed'],
  skip: ['Bỏ qua', 'Skip'],
  weekProgress: ['Tiến độ theo tuần', 'Progress by week'],
  recentActivity: ['Hoạt động 8 tuần qua', 'Last 8 weeks'],
  continueReview: ['Ôn flashcards', 'Review flashcards'],
  quickPractice: ['Luyện nhanh 10 câu', 'Quick 10-question practice'],
  weakTopics: ['Chủ đề cần luyện thêm', 'Topics to work on'],
  backup: ['Tiến độ lưu trong trình duyệt (localStorage). Hãy xuất file định kỳ để sao lưu.', 'Progress is stored in this browser (localStorage). Export regularly as a backup.'],
  notebook: ['Notebook', 'Notebook'],
  formulas: ['Công thức', 'Formulas'],
  searchFormula: ['Tìm công thức: tên, ký hiệu, numpy… (vd: det, đạo hàm, sigmoid)', 'Search formulas: name, symbol, numpy… (e.g. det, chain rule, sigmoid)'],
  noFormula: ['Không tìm thấy công thức phù hợp.', 'No matching formula.'],
  practiceFormula: ['Luyện công thức này', 'Practise this formula'],
  relatedFormulas: ['Công thức liên quan', 'Related formulas'],
  openNotebook: ['Mở trong JupyterLite', 'Open in JupyterLite'],
  openNewTab: ['Mở tab mới', 'Open in new tab'],
  notebookHint: [
    'Chạy Python + NumPy ngay trong trình duyệt. Bài sửa được lưu trong trình duyệt; tải .ipynb về (File → Download) để sao lưu.',
    'Python + NumPy in your browser. Edits are saved in this browser; use File → Download to keep a .ipynb copy.',
  ],
} as const satisfies Record<string, readonly [string, string]>

export type UIKey = keyof typeof UI

interface Ctx {
  lang: Lang
  setLang: (l: Lang) => void
  t: (x: Localized | undefined) => string
  ui: (k: UIKey) => string
}

const LangCtx = createContext<Ctx | null>(null)

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>(() => (localStorage.getItem('lang') as Lang) || 'vi')
  useEffect(() => {
    localStorage.setItem('lang', lang)
    document.documentElement.lang = lang
  }, [lang])
  const value: Ctx = {
    lang,
    setLang,
    t: (x) => (x ? x[lang] || x.en || x.vi : ''),
    ui: (k) => UI[k][lang === 'vi' ? 0 : 1],
  }
  return <LangCtx.Provider value={value}>{children}</LangCtx.Provider>
}

// eslint-disable-next-line react-refresh/only-export-components
export function useLang() {
  const c = useContext(LangCtx)
  if (!c) throw new Error('useLang outside provider')
  return c
}
