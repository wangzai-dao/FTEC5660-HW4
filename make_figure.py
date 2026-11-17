"""Draws docs/pipeline.png, the figure of the pipeline used in README.md (Task 5).

    python3 make_figure.py          (needs Pillow; the bundled runtime already has it)
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG = (255, 255, 255)
INK = (24, 32, 44)
MUTED = (92, 104, 120)
ACCENT = (28, 92, 168)
INDEX_FILL = (233, 241, 251)
QUERY_FILL = (232, 247, 239)
ABSTAIN_FILL = (252, 238, 233)
BORDER = (176, 192, 210)
LEFT, WIDTH = 250, 800
TITLE_SIZE, DETAIL_SIZE = 26, 18

FONTS = ["/System/Library/Fonts/Supplemental/Arial.ttf",
         "/System/Library/Fonts/Helvetica.ttc",
         "/Library/Fonts/Arial.ttf"]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in FONTS:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


INDEXING = [
    ("documents/corpus/*.html",
     ["one HTML annual report, earnings-call transcript or broker report per file"]),
    ("parse_document()  —  BeautifulSoup + lxml",
     ["h1-h3 / p / li in document order, section path kept", "nav, script and style dropped"]),
    ("table_to_text()  —  Task 1.1",
     ["caption, then one line per row:",
      '"Revenue | 2025: 2,710.4 | 2024: 2,629.1"']),
    ("chunk_document()  —  Task 1.2",
     ["narrative chunks by section; table chunks hold whole rows",
      "<= 1500 characters, metadata: source, company, fiscal_year, doc_type",
      "+ section, kind, caption"]),
    ("build_index()",
     ["BM25 (rank_bm25, k1=1.2, b=0.35) over stemmed tokens",
      "chunks grouped by company, document type and years covered"]),
]

RETRIEVAL = [
    ("question", ['e.g. "Which company held more cash at 31 December 2024: A or B?"']),
    ("parse_query()  —  Task 2.2",
     ["deepseek-v4-flash + rules  ->  companies, fiscal_year, doc_type",
      "names are copied as printed in the question; brokers and auditors dropped"]),
    ("retrieve()  —  Tasks 2.1 / 2.3 / 3",
     ["filter on company, document type and the years a document covers",
      "BM25 rank; a company missing from a comparison is swapped in",
      "abstain: unknown company / uncovered year / unseen metric"]),
    ("<= 5 chunks  —  generate() (fixed)",
     ["deepseek-v4-flash, temperature 0", "one number, one name, or NOT_FOUND"]),
]


def main() -> None:
    padding, gap, band_h = 34, 46, 30
    heights = []
    for title, lines in INDEXING + RETRIEVAL:
        heights.append(16 + TITLE_SIZE + 14 + len(lines) * (DETAIL_SIZE + 7) + 12)
    total = (padding * 2 + TITLE_SIZE + 26 + band_h * 2 + sum(heights)
             + gap * (len(heights) - 1) + gap)
    img = Image.new("RGB", (LEFT + WIDTH + 200, int(total)), BG)
    draw = ImageDraw.Draw(img)

    draw.text((LEFT, padding), "Retrieval-augmented assistant over the report library",
              font=font(29, True), fill=INK)
    y = padding + TITLE_SIZE + 30

    def band(text: str) -> None:
        nonlocal y
        draw.text((LEFT, y), text, font=font(19, True), fill=ACCENT)
        y += band_h

    def box(title: str, lines: list[str], height: int, fill) -> None:
        nonlocal y
        draw.rounded_rectangle([LEFT, y, LEFT + WIDTH, y + height], radius=14,
                               fill=fill, outline=BORDER, width=2)
        draw.text((LEFT + 22, y + 14), title, font=font(TITLE_SIZE, True), fill=INK)
        dy = y + 16 + TITLE_SIZE + 14
        for line in lines:
            draw.text((LEFT + 22, dy), line, font=font(DETAIL_SIZE), fill=MUTED)
            dy += DETAIL_SIZE + 7
        y += height

    def arrow(label: str = "") -> None:
        nonlocal y
        draw.line([(LEFT + WIDTH // 2, y + 6), (LEFT + WIDTH // 2, y + gap - 12)],
                  fill=ACCENT, width=3)
        draw.polygon([(LEFT + WIDTH // 2 - 8, y + gap - 12), (LEFT + WIDTH // 2 + 8, y + gap - 12),
                      (LEFT + WIDTH // 2, y + gap)], fill=ACCENT)
        if label:
            draw.text((LEFT + WIDTH // 2 + 16, y + gap // 2 - 12), label, font=font(17), fill=MUTED)
        y += gap

    band("indexing: once per document")
    for i, (title, lines) in enumerate(INDEXING):
        box(title, lines, heights[i], INDEX_FILL)
        if i < len(INDEXING) - 1:
            arrow()

    y += gap - 20
    band("retrieval: per question")
    offset = len(INDEXING)
    for i, (title, lines) in enumerate(RETRIEVAL):
        fill = ABSTAIN_FILL if title.startswith("retrieve") else QUERY_FILL
        box(title, lines, heights[offset + i], fill)
        if i < len(RETRIEVAL) - 1:
            arrow()

    notes = [
        (110, "no embedding model:\nBM25 + metadata\nfiltering only"),
        (int(total * 0.52), "abstention\nreturns [] so that\nthe generator answers\nNOT_FOUND"),
    ]
    for top, text in notes:
        draw.multiline_text((30, top), text, font=font(18), fill=MUTED, spacing=6)

    out = Path("docs")
    out.mkdir(exist_ok=True)
    img.save(out / "pipeline.png")
    print(f"wrote {out / 'pipeline.png'} ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
