"""Builds REFLECTION.docx from REFLECTION.md, for conversion to REFLECTION.pdf.

    python3 make_reflection_pdf.py .build/REFLECTION.docx
    <libreoffice-kit> convert --input .build/REFLECTION.docx --output REFLECTION.pdf

The Markdown file stays the source of truth: the text is parsed from it, not duplicated here.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

INK = RGBColor(0x1A, 0x22, 0x2E)
MUTED = RGBColor(0x5C, 0x68, 0x78)
ACCENT = RGBColor(0x1C, 0x5C, 0xA8)
RULE = "C8D2DE"
BODY_FONT = "Georgia"


def inline(text: str) -> str:
    """Strip the Markdown emphasis markers; the docx carries the emphasis in its styles."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)
    return text.replace("`", "").strip()


def parse(markdown: str) -> dict:
    lines = markdown.splitlines()
    title = next(line[2:].strip() for line in lines if line.startswith("# "))
    meta, quote, sections = [], [], []
    current = None
    buffer: list[str] = []

    def flush() -> None:
        if current is not None and current["body"]:
            current["body"] = inline(" ".join(current["body"]))

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            continue
        if stripped.startswith("## "):
            flush()
            current = {"title": inline(stripped[3:]), "body": []}
            sections.append(current)
            continue
        if current is None:
            if stripped.startswith(">"):
                quote.append(stripped.lstrip("> "))
            elif stripped:
                meta.append(inline(stripped))
            continue
        if not stripped:
            continue
        current["body"].append(inline(stripped))
    flush()
    quote = [inline(" ".join(quote))] if quote else []
    return {"title": inline(title), "meta": meta, "quote": quote, "sections": sections}


def para_border(paragraph, colour: str, side: str = "bottom", size: int = 6) -> None:
    pPr = paragraph._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    edge = OxmlElement(f"w:{side}")
    edge.set(qn("w:val"), "single")
    edge.set(qn("w:sz"), str(size))
    edge.set(qn("w:space"), "6")
    edge.set(qn("w:color"), colour)
    borders.append(edge)
    pPr.append(borders)


def add_page_number(paragraph) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instr)
    run._r.append(end)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".build/REFLECTION.docx")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = parse(Path("REFLECTION.md").read_text())

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2.4)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.6)
    section.right_margin = Cm(2.6)

    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = INK
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    normal.paragraph_format.line_spacing = 1.22
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)

    title = doc.add_paragraph()
    title.paragraph_format.space_after = Pt(2)
    run = title.add_run("Reflection")
    run.font.name = BODY_FONT
    run.font.size = Pt(24)
    run.font.bold = True
    run.font.color.rgb = ACCENT

    subtitle = doc.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(14)
    run = subtitle.add_run("FTEC5660 Homework 4 · Agentic AI for Business and FinTech")
    run.font.name = BODY_FONT
    run.font.size = Pt(9.5)
    run.font.color.rgb = MUTED
    para_border(subtitle, RULE, "bottom", 8)

    if data["quote"]:
        quote = doc.add_paragraph()
        quote.paragraph_format.left_indent = Cm(0.7)
        quote.paragraph_format.right_indent = Cm(1.4)
        quote.paragraph_format.space_before = Pt(4)
        quote.paragraph_format.space_after = Pt(14)
        run = quote.add_run(" ".join(data["quote"]))
        run.font.name = BODY_FONT
        run.font.size = Pt(9.5)
        run.font.italic = True
        run.font.color.rgb = MUTED
        para_border(quote, "1C5CA8", "left", 12)

    for index, section_data in enumerate(data["sections"], 1):
        heading = doc.add_paragraph()
        heading.paragraph_format.space_before = Pt(13 if index > 1 else 4)
        heading.paragraph_format.space_after = Pt(3)
        heading.paragraph_format.keep_with_next = True
        run = heading.add_run(f"{index}.  {section_data['title']}")
        run.font.name = BODY_FONT
        run.font.size = Pt(11.5)
        run.font.bold = True
        run.font.color.rgb = ACCENT

        body = doc.add_paragraph()
        body.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        run = body.add_run(section_data["body"])
        run.font.name = BODY_FONT
        run.font.size = Pt(10.5)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = footer.add_run("FTEC5660 Homework 4 · Reflection · ")
    run.font.name = BODY_FONT
    run.font.size = Pt(8)
    run.font.color.rgb = MUTED
    add_page_number(footer)
    for run in footer.runs:
        run.font.name = BODY_FONT
        run.font.size = Pt(8)
        run.font.color.rgb = MUTED

    doc.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
