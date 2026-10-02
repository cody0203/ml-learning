# ML Review — Ôn tập Machine Learning

Website tĩnh để lưu kiến thức và luyện bài tập cho chuyên ngành *Mathematics for Machine Learning and Data Science* (DeepLearning.AI): Đại số tuyến tính, Giải tích, Xác suất & Thống kê. Song ngữ Việt/Anh, chạy hoàn toàn trên trình duyệt; tiến độ lưu ở `localStorage`.

## Chạy

```powershell
npm install
npm run dev        # http://localhost:5173
npm run build      # xuất ra dist/ (mở được trên GitHub Pages)
npm test           # unit test (linalg, checker, generator, SRS)
npm run validate   # kiểm tra toàn bộ nội dung trong content/
```

## Tính năng

- **Ghi chú** theo tuần, có KaTeX và mục lục.
- **Công thức**: mỗi tuần có tab *Công thức* (và trang *Công thức* tìm kiếm toàn bộ khoá) — tìm theo tên, ký hiệu hay hàm NumPy; mỗi công thức liên kết tới các bài tập dùng nó và có nút *Luyện công thức này*. Sau khi trả lời một bài, site hiện các công thức liên quan.
- **12 dạng bài tập**: trắc nghiệm, đúng/sai kèm lý do, điền số (nhập được `7/3`, `sqrt(2)`), ma trận, *construct* (tự tạo ví dụ, chấm bằng tính chất như det = 0, rank = 2… nên có vô số đáp án đúng), tìm bước sai, sắp xếp bước, ghép cặp, đoán output NumPy, lập trình Python (Pyodide + NumPy chạy ngay trong trình duyệt), bài nhiều bước, và bài luyện số ngẫu nhiên.
- **Chế độ luyện**: ưu tiên phần yếu, bài chưa làm, trộn ngẫu nhiên, thi thử có tính giờ.
- **Flashcards** lặp lại ngắt quãng (SM-2), chia 3 loại: *Lý thuyết*, *Công thức*, *Tính nhanh* (lọc được theo loại). Câu làm sai tự vào hàng đợi ôn tập.
- **Sổ lỗi sai**, dashboard tiến độ, heatmap hoạt động, xuất/nhập file tiến độ JSON.
- **JupyterLite** đầy đủ nhúng trong site (trang *Notebook*): mỗi tuần có 1 notebook gồm các bài code, mỗi bài code có nút *Mở trong JupyterLite*. Bài sửa trong notebook lưu trong trình duyệt; dùng File → Download để giữ file .ipynb. Chấm điểm và lưu tiến độ vẫn làm trong trang bài tập.

## Thêm nội dung khi học tới phần mới

1. Bỏ file PDF slide vào `syllabus/<khoá>/W<n>.pdf` (ví dụ `syllabus/c2-calculus/W4.pdf`; tuần bị tách thì `W3-part-1.pdf`, `W3-part-2.pdf`). Nếu để lẫn ở chỗ khác, Copilot sẽ tự phân loại.
2. Nhờ Copilot: *"ingest syllabus/c2-calculus/W4.pdf"*. Skill `.github/skills/ingest-syllabus/SKILL.md` hướng dẫn Copilot viết ghi chú, flashcards và ≥ 60 bài tập đa dạng vào `content/`.
3. `npm run validate` phải báo `Content OK`. Validator kiểm tra schema, chống nội dung sinh theo khuôn (dòng ghi chú lặp, đề bài/lời giải trùng, phương án sai tái sử dụng), id trùng, chỉ số đáp án, ví dụ *construct* thoả ràng buộc, mọi công thức KaTeX, và chạy lời giải của từng bài code (lời giải phải qua test, code khởi đầu phải trượt).

Cấu trúc nội dung:

```
content/<course>/course.yaml
content/<course>/<week>/week.yaml
content/<course>/<week>/notes.vi.md, notes.en.md
content/<course>/<week>/flashcards.yaml
content/<course>/<week>/formulas.yaml
content/<course>/<week>/exercises/*.yaml
```

Schema nằm ở `src/content/schema.ts`.

## Deploy (GitHub Pages)

Workflow `.github/workflows/deploy.yml`:

- **Push lên `main`** (hoặc bấm *Run workflow*): cài Node 22 + Python 3.11 → unit test → typecheck → `npm run validate` (gồm chạy lời giải mọi bài code trong Pyodide) → build JupyterLite → build site → deploy lên Pages.
- **Pull request vào `main`**: chạy toàn bộ kiểm tra và build, nhưng **không** deploy.

Thiết lập một lần:

1. Tạo repo trên GitHub, push code lên nhánh `main`.
2. Repo → **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. Push (hoặc *Actions → Deploy to GitHub Pages → Run workflow*). URL hiện ở job *Deploy*, dạng `https://<user>.github.io/<repo>/`.

Site dùng đường dẫn tương đối (`base: './'`) và HashRouter, nên chạy được ở bất kỳ tên repo nào mà không cần cấu hình thêm.

> Slide PDF trong `syllabus/` bị `.gitignore` bỏ qua (có bản quyền, ~130 MB) — chúng chỉ cần ở máy bạn để Copilot tạo nội dung.
