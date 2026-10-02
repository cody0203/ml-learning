"""Extract de-duplicated slide text from a PDF. Usage: python scripts/extract_pdf.py file.pdf"""
import re
import sys

from pypdf import PdfReader

pages = [re.sub(r"\s+", " ", p.extract_text() or "").strip() for p in PdfReader(sys.argv[1]).pages]
for i, text in enumerate(pages):
    nxt = pages[i + 1] if i + 1 < len(pages) else ""
    # Animation builds repeat the slide with more text; keep only the last build.
    words = text.split()
    if words and len(nxt) >= len(text) and set(words) <= set(nxt.split()):
        continue
    print(f"--- slide {i + 1} ---\n{text}\n")
