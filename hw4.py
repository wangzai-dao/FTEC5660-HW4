"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================

import math
from collections import Counter, defaultdict

from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi

# One flag per component of the pipeline (Section "Homework 4 solution" of README.md).  The
# ablation study switches them on one at a time; the values below are the submitted pipeline.
CONFIG = {
    "table_text": True,       # V1: keep the row/column structure of every <table>
    "structural": True,       # V2: section-aware chunks, a table chunk holds whole rows
    "context_header": True,   # V3: company / document / section header on every chunk
    "rich_tokens": True,      # V4: stemming, plural folding, fiscal years, synonyms
    "metadata_filter": True,  # V5: filter candidates on company, document type and year
    "per_company": True,      # V6: one retrieval per company when a question compares two
    "abstain": True,          # V7: return [] when the corpus cannot answer
    "llm_parse": True,        # parse_query asks the model, with a rule-based fallback
    "llm_check": False,       # optional second answerability check (see README)
    "table_chunk_chars": 700,
    "text_chunk_chars": 950,
}

_DOC_LABELS = {"annual_report": "annual report",
               "earnings_call": "earnings call",
               "research_report": "research report"}
_YEAR_CELL_RE = re.compile(r"(?:FY\s?)?(?:19|20)\d{2}[AEP]?", re.I)
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"

_WS_RE = re.compile(r"\s+")
_LEGAL = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}
_TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
_BROKER_TAILS = {"securities", "capital", "research", "equity", "partners", "markets",
                 "investments", "advisors", "advisers", "broker", "brokerage", "trust", "bank"}

# What chunk_document learns from the corpus itself (no answer or question is hard-coded).
_COMPANY_REGISTRY: dict[str, str] = {}
_BROKER_KEYS: set[str] = set()
_AUDITOR_KEYS: set[str] = set()
_PARSE_CACHE: dict[str, dict] = {}
_INDEX: dict = {}


# ----------------------------------------------------------------------------- text utilities

def _clean(text: Any) -> str:
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", str(text or ""))).strip()


def _words(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", _clean(text).lower())


def _ckey(text: Any) -> str:
    """Company key: lower case words without the legal suffix (the same idea as company_key below)."""
    return " ".join(w for w in _words(text) if w not in _LEGAL and w not in _TITLES)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _stem(word: str) -> str:
    """A deliberately small stemmer: plurals, -ing, -ed, and a silent final e."""
    w = word
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 4 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


_SYNONYM_GROUPS = (
    {"employee", "employees", "headcount", "staff", "workforce", "personnel"},
    {"revenue", "revenues", "sales", "turnover", "topline"},
    {"borrowing", "borrowings", "debt", "loan", "loans", "leverage"},
    {"audit", "auditor", "auditors", "auditing", "audited"},
    {"headquarters", "headquartered", "hq", "domiciled", "city", "cities", "located", "based"},
    {"inventory", "inventories", "stock", "stocks"},
    {"repurchase", "repurchases", "buyback", "buybacks"},
    {"guidance", "guided", "guide", "guides", "outlook", "expected", "forecast", "forecasts",
     "estimate", "estimates", "estimation", "consensus"},
    {"cash", "liquidity"},
    {"dividend", "dividends", "payout"},
    {"capex", "capital", "expenditure", "expenditures"},
    {"employees", "people", "headcount"},
)
_SYNONYMS: dict[str, set[str]] = {}
for _group in _SYNONYM_GROUPS:
    _stems = {_stem(w) for w in _group}
    for _w in _group:
        _SYNONYMS.setdefault(_stem(_w), set()).update(_stems)

# Question frame words that are not part of the fact the question asks for.
_METRIC_STOP = set()


def _add_stop(*words: str) -> None:
    for w in words:
        _METRIC_STOP.add(_stem(w))
        _METRIC_STOP.add(w)


_add_stop(*"""a an the this that these those of for in on at by with to from into over under about
what which who whom whose how much many more most less least higher lower larger smaller greater
did does do done is was were are be been being has have had having will would can could should
company companies firm firms corporation business its it their they them he she we you i our your
hold held holds holding report reported reports reporting generate generated generates generating
earn earned earns earning announce announced announces announcing set sets setting give gives given
go goes going value amount amounts figure figures number numbers level total all both each per
usd million millions billion billions percent percentage thousand thousands unit units
end ending beginning start starting close closing day days date year years fiscal fy
quarter quarterly call review reviews results result statement statements second third fourth first
approximately roughly nearly around about estimated estimate estimates expect expects expected
forecast forecasts projected projection bidding bidding
during since until between compared comparison versus vs than also just only still even
ask asked say said says according note notes table tables section sections
stock stocks price prices current currently latest recent recently new news
largest larger large biggest highest lowest smallest various major main key""".split())

_TOKEN_RE = re.compile(r"fy\s?(\d{4})|[a-z][a-z0-9]*|[0-9][0-9,]*(?:\.[0-9]+)?")


def _tokens(text: Any) -> list[str]:
    """Lower-case word and fiscal-year tokens; other numbers are dropped (they are not searchable)."""
    out = []
    for m in _TOKEN_RE.finditer(_clean(text).lower()):
        chunk = m.group(0)
        if m.group(1):
            out.append("fy" + m.group(1))
            out.append(m.group(1))
        elif chunk[0].isdigit():
            number = chunk.replace(",", "")
            if len(number) == 4 and number[:2] in ("19", "20"):
                out.append(number)
        else:
            out.append(_stem(chunk))
    return out


def _bm25_tokens(text: Any) -> list[str]:
    if not CONFIG["rich_tokens"]:
        return _clean(text).lower().split()
    return _tokens(text)


def _expand(tokens: list[str]) -> list[str]:
    out = []
    for t in tokens:
        out.append(t)
        out.extend(sorted(_SYNONYMS.get(t, ())))
    return out


def _split_text(text: str, size: int) -> list[str]:
    """Split a long text at sentence boundaries (or brutally) into pieces of at most `size`."""
    if size < 50:
        size = 50
    if len(text) <= size:
        return [text]
    pieces, current = [], ""
    for part in re.split(r"(?<=[.;:!?])\s+", text):
        while len(part) > size:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(part[:size])
            part = part[size:]
        if current and len(current) + len(part) + 1 > size:
            pieces.append(current)
            current = ""
        current = (current + " " + part).strip() if current else part
    if current:
        pieces.append(current)
    return [p for p in pieces if p.strip()]


# ----------------------------------------------------------------------------- Task 1.1: tables

def _span_attr(cell, name: str) -> int:
    try:
        return max(1, int(str(cell.get(name, 1)).strip() or 1))
    except Exception:
        return 1


def _table_matrix(table) -> list[list[str]]:
    """The table as a rectangular grid of cell texts, colspan/rowspan expanded."""
    grid, covered = [], {}
    for r, tr in enumerate(table.find_all("tr")):
        row, c = [], 0
        for cell in tr.find_all(["td", "th"], recursive=False):
            while (r, c) in covered:
                row.append(covered[(r, c)])
                c += 1
            text = _clean(cell.get_text(" ", strip=True))
            row.append(text)
            cols, rows = _span_attr(cell, "colspan"), _span_attr(cell, "rowspan")
            for i in range(rows):
                for j in range(cols):
                    covered[(r + i, c + j)] = text
            c += cols
        while (r, c) in covered:
            row.append(covered[(r, c)])
            c += 1
        grid.append(row)
    if not grid:
        return []
    width = max(len(row) for row in grid)
    return [row + [""] * (width - len(row)) for row in grid]


def _header_rows(table, grid: list[list[str]]) -> int:
    thead = table.find("thead")
    if thead is not None:
        n = len(thead.find_all("tr"))
        if 0 < n < len(grid):
            return n
    first = table.find("tr")
    if first is not None:
        cells = first.find_all(["td", "th"], recursive=False)
        if cells and all(cell.name == "th" for cell in cells):
            return 1
        # a header row written without <th>: a row of year labels ("", "2025", "2024")
        if len(grid) > 1:
            year_like = sum(1 for value in grid[0]
                            if not _clean(value) or _YEAR_CELL_RE.fullmatch(_clean(value)))
            if year_like >= max(2, len(grid[0]) - 1):
                return 1
    return 0


def _column_headers(header_rows: list[list[str]], width: int) -> list[str]:
    headers = []
    for c in range(width):
        parts = []
        for row in header_rows:
            value = row[c] if c < len(row) else ""
            if value and value not in parts:
                parts.append(value)
        headers.append(" / ".join(parts))
    return headers


def _table_to_text(table) -> str:
    caption_el = table.find("caption")
    caption = _clean(caption_el.get_text(" ", strip=True)) if caption_el is not None else ""
    grid = _table_matrix(table)
    if not grid:
        return caption
    n_head = _header_rows(table, grid)
    headers = _column_headers(grid[:n_head], len(grid[0]))
    lines = []
    if caption:
        lines.append(caption)
    header_line = " | ".join(h for h in headers if h)
    if header_line:
        lines.append(header_line)
    for row in grid[n_head:]:
        cells = [_clean(c) for c in row]
        label = cells[0] if cells else ""
        pairs = []
        for c in range(1, len(cells)):
            value = cells[c]
            if not value:
                continue
            head = headers[c] if c < len(headers) else ""
            pairs.append(f"{head}: {value}" if head else value)
        if label and pairs:
            lines.append(f"{label} | " + " | ".join(pairs))
        elif label:
            lines.append(label)
        elif pairs:
            lines.append(" | ".join(pairs))
    return "\n".join(line for line in lines if line)


def table_to_text(table_html: str) -> str:
    """Task 1.1: turn one HTML <table> into text that keeps every value with its row and its column.

    Put each value on a line (at most 400 characters) together with its row label and its column
    header, e.g. "Revenue | 2025: 2,710.4 | 2024: 2,629.1", one table row per line, keep the text of
    every cell unchanged, and include the caption (it holds the unit) in the output. Do not call the
    LLM here.
    """
    if not _clean(table_html):
        return ""
    try:
        soup = BeautifulSoup(str(table_html), "lxml")
    except Exception:
        soup = BeautifulSoup(str(table_html), "html.parser")
    table = soup.find("table") or soup
    return _table_to_text(table)


# ----------------------------------------------------------------------------- document metadata

def _detect_doc_type(text: str) -> str:
    low = _clean(text).lower()
    if re.search(r"(?:earnings|conference)\s+call", low):
        return "earnings_call"
    if re.search(r"results\s+review", low):
        return "research_report"
    if re.search(r"annual\s+report", low):
        return "annual_report"
    return "annual_report"


def _company_name(text: str) -> str:
    text = _clean(text)
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
    text = re.sub(r"\s*[|—–]\s*(?:Annual\s+Report|FY\s?\d{4}).*$", "", text, flags=re.I).strip()
    return text


def _document_year(front: str, doc_type: str) -> int:
    if doc_type == "annual_report":
        match = re.search(r"annual\s+report\s+(\d{4})", front, re.I)
        if not match:
            match = re.search(r"fiscal\s+year\s+ended\s+\d{1,2}\s+\w+\s+(\d{4})", front, re.I)
    elif doc_type == "earnings_call":
        match = re.search(r"(?:full|fiscal)\s+year\s+(\d{4})", front, re.I)
    else:
        match = re.search(r"fy\s?(\d{4})", front, re.I)
    return int(match.group(1)) if match else 2025


def _document_meta(file_name: str, html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    body = soup.body or soup
    title = _clean(soup.title.get_text(" ", strip=True)) if soup.title is not None else ""
    h1_el = body.find("h1")
    h1 = _clean(h1_el.get_text(" ", strip=True)) if h1_el is not None else ""
    body_text = _clean(body.get_text(" ", strip=True))
    front = " ".join(x for x in (title, h1) if x)
    doc_type = _detect_doc_type(front + " " + body_text[:400])
    company = _company_name(h1 or title) or "Unknown"
    fiscal_year = _document_year(front + " " + body_text[:400], doc_type)

    key = _ckey(company)
    if key:
        _COMPANY_REGISTRY.setdefault(key, company)
    if doc_type == "research_report":
        broker = re.search(r"([A-Z][A-Za-z&.\-]*(?:\s+[A-Z][A-Za-z&.\-]*){0,3})\s*\|\s*Equity\s+Research",
                           body_text)
        if broker:
            _BROKER_KEYS.add(_ckey(broker.group(1)))
        broker = re.search(r"(?:prepared\s+by|published\s+by)\s+([A-Z][A-Za-z&.\-]*(?:\s+[A-Z][A-Za-z&.\-]*){0,3})",
                           body_text)
        if broker:
            _BROKER_KEYS.add(_ckey(broker.group(1)))
    auditor = re.search(r"([A-Z][A-Za-z&.\-]*(?:\s+[A-Z][A-Za-z&.\-]*){0,3}\s+"
                        r"(?:LLP|LLC|Plc|plc|Ltd\.?|Inc\.?))\s+(?:audited|is\s+the\s+independent\s+auditor)",
                        body_text)
    if auditor:
        _AUDITOR_KEYS.add(_ckey(auditor.group(1)))
    return {"company": company, "fiscal_year": fiscal_year, "doc_type": doc_type}


def _base_metadata(file_name: str, meta: dict) -> dict:
    return {"source": file_name, "company": meta["company"],
            "fiscal_year": meta["fiscal_year"], "doc_type": meta["doc_type"]}


def _header_line(meta: dict, section: str) -> str:
    label = _DOC_LABELS.get(meta["doc_type"], str(meta["doc_type"]).replace("_", " "))
    parts = [meta["company"], f"{label} {meta.get('fiscal_year')}"]
    if section:
        parts.append(section)
    return " | ".join(p for p in parts if p)[:150]


# ----------------------------------------------------------------------------- Task 1.2: chunks

def _make_chunk(chunks: list[Document], header: str, body: str, base: dict, extra: dict) -> None:
    body = body.strip()
    if not body:
        return
    text = f"{header}\n{body}" if header else body
    if len(text) > MAX_CHUNK_CHARS:
        text = text[:MAX_CHUNK_CHARS].rstrip()
    metadata = dict(base)
    metadata.update(extra)
    chunks.append(Document(page_content=text, metadata=metadata))


def _table_chunks(chunks: list[Document], table, section: str, base: dict, meta: dict) -> None:
    if CONFIG["table_text"]:
        text = _table_to_text(table)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        caption, rows = (lines[0], lines[1:]) if lines else ("", [])
    else:
        caption, rows = "", [_clean(table.get_text(" ", strip=True))]
    if not rows:
        return
    header = _header_line(meta, section) if CONFIG["context_header"] else ""
    prefix = "\n".join(p for p in (header, caption) if p)
    budget = MAX_CHUNK_CHARS - (len(prefix) + 1 if prefix else 0)
    target = max(200, min(CONFIG["table_chunk_chars"], budget))
    group, size = [], 0

    def emit() -> None:
        nonlocal group, size
        if not group:
            return
        _make_chunk(chunks, prefix, "\n".join(group), base,
                    {"section": section, "kind": "table", "caption": caption[:200]})
        group, size = [], 0

    for line in rows:
        for piece in _split_text(line, target):
            if group and size + len(piece) + 1 > target:
                emit()
            group.append(piece)
            size += len(piece) + 1
    emit()


def _structural_chunks(file_name: str, html: str, meta: dict) -> list[Document]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["script", "style", "nav"]):
        tag.decompose()
    body = soup.body or soup
    items: list[tuple[str, str, Any]] = []
    state = {"h2": "", "h3": ""}

    def section() -> str:
        return " / ".join(x for x in (state["h2"], state["h3"]) if x)

    def walk(node) -> None:
        for child in node.children:
            name = getattr(child, "name", None)
            if not name or name in ("script", "style", "nav", "h1"):
                continue
            if name in ("h2", "h3", "h4"):
                text = _clean(child.get_text(" ", strip=True))
                if name == "h2":
                    state["h2"], state["h3"] = text, ""
                elif name == "h3":
                    state["h3"] = text
                else:
                    state["h3"] = f"{state['h3']} / {text}".strip(" /")
                continue
            if name == "table":
                items.append(("table", section(), child))
                continue
            if name in ("p", "li", "blockquote", "dd", "dt"):
                if child.find("table") is not None:
                    walk(child)
                    continue
                text = _clean(child.get_text(" ", strip=True))
                if text:
                    items.append(("text", section(), text))
                continue
            walk(child)

    walk(body)

    base = _base_metadata(file_name, meta)
    chunks: list[Document] = []
    buffer: list[str] = []
    box = {"section": "", "size": 0}

    def flush() -> None:
        if not buffer:
            return
        header = _header_line(meta, box["section"]) if CONFIG["context_header"] else ""
        budget = MAX_CHUNK_CHARS - (len(header) + 1 if header else 0)
        for piece in _split_text("\n".join(buffer), budget):
            _make_chunk(chunks, header, piece, base, {"section": box["section"], "kind": "text"})
        buffer.clear()
        box["size"] = 0

    for kind, sec, payload in items:
        if kind == "text":
            if buffer and (sec != box["section"]
                           or box["size"] + len(payload) + 1 > CONFIG["text_chunk_chars"]):
                flush()
            if not buffer:
                box["section"] = sec
            buffer.append(payload)
            box["size"] += len(payload) + 1
        else:
            flush()
            _table_chunks(chunks, payload, sec, base, meta)
    flush()
    return chunks


def _naive_chunks(file_name: str, html: str, meta: dict) -> list[Document]:
    """The baseline of Task 5 (B0/V1): one flat text, RecursiveCharacterTextSplitter."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["script", "style", "nav"]):
        tag.decompose()
    body = soup.body or soup
    if CONFIG["table_text"]:
        for table in body.find_all("table"):
            block = soup.new_tag("p")
            block.string = "\n" + _table_to_text(table) + "\n"
            table.replace_with(block)
    text = body.get_text("\n", strip=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    base = _base_metadata(file_name, meta)
    chunks = []
    for piece in splitter.split_text(text):
        piece = _clean(piece)
        if not piece:
            continue
        header = ""
        if CONFIG["context_header"]:
            header = _header_line(meta, "")
        _make_chunk(chunks, header, piece, base, {"section": "", "kind": "text"})
    return chunks


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Task 1.2: split one document into chunks of at most MAX_CHUNK_CHARS characters.

    Every chunk needs the metadata "source" (= file_name), "company", "fiscal_year" (an int) and
    "doc_type" (one of DOC_TYPES). Do not call the LLM here.
    """
    meta = _document_meta(file_name, html)
    if not CONFIG["structural"]:
        return _naive_chunks(file_name, html, meta)
    return _structural_chunks(file_name, html, meta)


# ----------------------------------------------------------------------------- Task 2.2: parsing

_POSSESSIVE_RE = re.compile(r"['’]s?$")


def _bare(word: str) -> str:
    return _POSSESSIVE_RE.sub("", word)


def _ends_name(word: str) -> bool:
    """True for a token that closes a name: a possessive such as "Software's" or "Robotics'"."""
    return bool(_POSSESSIVE_RE.search(word))


def _printed_span(name: str, question: str) -> str | None:
    """The company name as printed in the question (keeps its exact form), or None."""
    words = [w for w in _words(name) if w not in _LEGAL] or _words(name)
    if not words:
        return None
    tokens = [(m.group(0), m.start(), m.end())
              for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9&.\-’']*", question)]
    lowered = [_bare(t[0].lower()) for t in tokens]
    connectors = {"of", "and", "&", "de", "van", "von"}
    for i in range(len(tokens)):
        j, k = i, 0
        while k < len(words) and j < len(tokens) and lowered[j] == words[k]:
            j += 1
            k += 1
        if k < len(words):
            continue
        last = j - 1
        if _ends_name(tokens[last][0]):             # "Software's" ends the name
            span = _clean(question[tokens[i][1]:tokens[last][2]].rstrip(".,;:?"))
            span = _clean(_POSSESSIVE_RE.sub("", span))
            if span:
                return span
            continue
        while last + 1 < len(tokens) and last - i < 5:
            nxt = tokens[last + 1][0]
            low_next = lowered[last + 1]
            if _ends_name(nxt):                      # "Software's" ends the name
                last += 1
                break
            if nxt[:1].isupper() or nxt[:1].isdigit() or low_next in connectors:
                last += 1
                continue
            break
        span = _clean(question[tokens[i][1]:tokens[last][2]].rstrip(".,;:?"))
        span = _clean(_POSSESSIVE_RE.sub("", span))
        if span:
            return span
    return None


def _is_excluded_company(key: str) -> bool:
    if not key:
        return True
    if key in _BROKER_KEYS or key in _AUDITOR_KEYS:
        return True
    return key.split()[-1] in _BROKER_TAILS


def _resolve_companies(names: list[str], question: str) -> list[str]:
    out, keys = [], set()
    for raw in names or []:
        if not isinstance(raw, str):
            continue
        name = re.sub(r"^(?:the|its|their|a|an)\s+", "", _clean(raw), flags=re.I)
        name = re.sub(r"['’]s$", "", name).strip()
        if not name:
            continue
        printed = _printed_span(name, question)
        if printed is None:
            continue                                  # the model made the name up
        key = _ckey(printed)
        if key in keys or _is_excluded_company(key):
            continue
        keys.add(key)
        out.append(printed)
    return out


def _library_companies(question: str) -> list[str]:
    padded = " " + " ".join(_words(question)) + " "
    found = []
    for key, canonical in sorted(_COMPANY_REGISTRY.items(), key=lambda kv: -len(kv[0])):
        if f" {key} " in padded:
            found.append((key, canonical))
    return [(k, _printed_span(c, question) or c) for k, c in found]


def _union_companies(companies: list[str], question: str) -> list[str]:
    out, keys = [], set()
    for name in companies:
        key = _ckey(name)
        if key and key not in keys:
            keys.add(key)
            out.append(name)
    for key, printed in _library_companies(question):
        if key not in keys:
            keys.add(key)
            out.append(printed)
    return out


def _rule_companies(question: str) -> list[str]:
    names = [printed for _, printed in _library_companies(question)]
    if names:
        return names
    patterns = (r"([A-Z][A-Za-z&.\-]*(?:\s+[A-Z][A-Za-z&.\-]*){0,2})['’]s?(?=[\s,:;?]|$)",
                r"([A-Z][A-Za-z&.\-]*(?:\s+[A-Z][A-Za-z&.\-]*){0,2})\s+or\s+"
                r"([A-Z][A-Za-z&.\-]*(?:\s+[A-Z][A-Za-z&.\-]*){0,2})",
                r"(?:did|does|for|of|give|gives|from|to|with)\s+"
                r"([A-Z][A-Za-z&.\-]*(?:\s+[A-Z][A-Za-z&.\-]*){0,2})")
    for pattern in patterns:
        for match in re.finditer(pattern, question):
            names.extend(g for g in match.groups() if g)
    out, keys = [], set()
    for name in names:
        key = _ckey(name)
        if key and key not in keys and not _is_excluded_company(key):
            keys.add(key)
            out.append(name)
    return out


def _rule_year(question: str) -> int | None:
    match = re.search(r"\bFY\s?(\d{4})\b", question, re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(?:\d{1,2})\s+(?:" + _MONTHS + r")\s+(\d{4})\b", question, re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(?:19|20)\d{2}\b", question)
    if match:
        return int(match.group(0))
    return None


def _rule_doc_type(question: str) -> str:
    low = question.lower()
    if re.search(r"(?:earnings|conference)\s+call|\bq[1-4]\b|fourth[- ]quarter", low):
        return "earnings_call"
    if re.search(r"results\s+review|target\s+price|\brating\b|broker|forecast|equity\s+research"
                 r"|12[- ]month|analyst|estimate", low):
        return "research_report"
    return "annual_report"


_COMPANY_PROMPT = """You extract the companies that a question asks about.

Companies in the library: {library}

Reply with JSON only, on one line: {{"companies": ["..."]}}

Rules:
- Copy each company name exactly as printed in the question, without the possessive 's.
- List every company the question asks about, in the order in which they appear.
- Never list a bank, broker, research house or audit firm: those end in Securities, Capital, Research, Equity, Partners, Markets, Investments or Advisors.
- A company that is not in the library is still listed.
- Use [] when the question asks about no company.

Examples:
Question: Which company had the higher net income in FY2024: Zephyr Aerospace or Orion Water?
JSON: {{"companies": ["Zephyr Aerospace", "Orion Water"]}}
Question: What investment rating does Lighthouse Securities give Zephyr Aerospace in its FY2025 results review?
JSON: {{"companies": ["Zephyr Aerospace"]}}
Question: How much cash did Halvard Robotic Systems hold at 31 December 2025?
JSON: {{"companies": ["Halvard Robotic Systems"]}}
Question: Who was Orion Water's Chief Financial Officer at the end of FY2025?
JSON: {{"companies": ["Orion Water"]}}

Question: {question}
JSON:"""


def _llm_companies(question: str) -> list[str] | None:
    library = ", ".join(sorted(set(_COMPANY_REGISTRY.values()))) or "(none)"
    answer = make_llm(max_tokens=120).invoke(
        _COMPANY_PROMPT.format(library=library, question=question))
    text = str(getattr(answer, "content", answer))
    match = re.search(r"\{.*?\}", text, re.S)
    if not match:
        return None
    data = json.loads(match.group(0))
    names = data.get("companies") if isinstance(data, dict) else None
    if not isinstance(names, list):
        return None
    return [n for n in names if isinstance(n, str)]


def parse_query(question: str) -> dict:
    """Task 2.2: {"companies": [names], "fiscal_year": int or None, "doc_type": one of DOC_TYPES}.

    The rules for each field are under Task 2.2 in the handout. You may use make_llm() here.
    """
    question = str(question)
    cached = _PARSE_CACHE.get(question)
    if cached is not None:
        return dict(cached)
    parsed = {"companies": _rule_companies(question),
              "fiscal_year": _rule_year(question),
              "doc_type": _rule_doc_type(question)}
    if CONFIG["llm_parse"]:
        try:
            names = _llm_companies(question)
            resolved = _resolve_companies(names or [], question)
            if resolved:
                parsed["companies"] = resolved
        except Exception:
            pass
    parsed["companies"] = _union_companies(parsed["companies"], question)
    _PARSE_CACHE[question] = dict(parsed)
    return parsed


# ----------------------------------------------------------------------------- Tasks 2 and 3

def _doc_covers_year(doc_type: str, doc_year: int | None, year: int) -> bool:
    if doc_year is None or year is None:
        return True
    if doc_type == "annual_report":
        return year in (doc_year, doc_year - 1)
    if doc_type == "earnings_call":
        return year in (doc_year, doc_year + 1)
    return year in (doc_year - 1, doc_year, doc_year + 1, doc_year + 2)


def build_index(chunks: list[Document]) -> Any:
    """Task 2: index the chunks of the whole corpus (the output of chunk_document), once per run.

    Return anything retrieve() needs: a BM25 or TF-IDF index, the chunks grouped by metadata, ...
    """
    global _INDEX
    chunks = [c for c in (chunks or []) if getattr(c, "page_content", "")]
    tokens = [_bm25_tokens(c.page_content) for c in chunks]
    tokens = [t if t else ["empty"] for t in tokens]
    document_frequency: Counter = Counter()
    for row in tokens:
        document_frequency.update(set(row))
    # a tokenizer-independent vocabulary, used only by the abstention check
    vocabulary: Counter = Counter()
    for chunk in chunks:
        vocabulary.update(set(_tokens(chunk.page_content)))
    bm25 = BM25Okapi(tokens, k1=1.2, b=0.35) if tokens else None
    by_company: dict[str, list[int]] = defaultdict(list)
    for i, chunk in enumerate(chunks):
        by_company[_ckey((chunk.metadata or {}).get("company"))].append(i)
    _INDEX = {"chunks": chunks, "tokens": tokens, "df": document_frequency, "vocabulary": vocabulary,
              "bm25": bm25, "by_company": by_company, "company_keys": set(by_company)}
    return _INDEX


def _query_tokens(question: str, companies: list[str]) -> list[str]:
    if not CONFIG["rich_tokens"]:
        return _clean(question).lower().split()
    text = question
    if CONFIG["metadata_filter"]:
        # the metadata filter already pins the company, so its name is only noise in the query
        for name in companies:
            text = re.sub(re.escape(name), " ", text, flags=re.I)
    text = re.sub(r"['’]s\b", " ", text)
    tokens = [t for t in _tokens(text) if t not in _METRIC_STOP]
    return _expand(tokens) if tokens else []


def _metric_terms(question: str, companies: list[str]) -> list[str]:
    """The words of the question that name the fact it asks for (company, year and frame removed)."""
    text = question
    for name in companies:
        text = re.sub(r"\b" + re.escape(name) + r"(?:['’]s)?\b", " ", text, flags=re.I)
    # a broker or audit firm named in the question is not the fact that is asked for
    text = re.sub(r"\b(?:[A-Z][A-Za-z&.\-]*\s+){0,3}"
                  r"(?:Securities|Capital|Research|Equity|Partners|Markets|Investments|Advisors"
                  r"|Advisers|Brokerage|LLP|LLC)\b", " ", text)
    text = re.sub(r"['’]s\b", " ", text)
    text = re.sub(r",?\s*in\s+(?:USD|US\s?dollars?|percent|percentage)\b[^?]*$", " ", text, flags=re.I)
    text = re.sub(r"\b(?:fourth|third|second|first)[- ]quarter\b", " ", text, flags=re.I)
    text = re.sub(r"\bQ[1-4]\b", " ", text, flags=re.I)
    text = re.sub(r"\bFY\s?\d{4}\b", " ", text, flags=re.I)
    text = re.sub(r"\b\d{1,2}\s+(?:" + _MONTHS + r")\s+\d{4}\b", " ", text, flags=re.I)
    text = re.sub(r"\b(?:19|20)\d{2}\b", " ", text)
    text = re.sub(r"\b12[- ]month\b", " ", text, flags=re.I)
    terms = []
    for token in _tokens(text):
        if token.isdigit() or len(token) < 3 or token in _METRIC_STOP:
            continue
        if token not in terms:
            terms.append(token)
    return terms


def _metric_coverage(terms: list[str], text: str) -> float:
    if not terms:
        return 1.0
    present = set(_tokens(text))
    hits = sum(1 for t in terms if t in present or (set(_SYNONYMS.get(t, ())) & present))
    return hits / len(terms)


_CHECK_PROMPT = """You check whether retrieved passages contain the evidence needed to answer a question.

Question: {question}

Passages:
{context}

Does the evidence above state the exact fact the question asks for?
- The passages must be about the company the question names.
- They must state the year the question names (a prior-year column of a later report is fine).
- They must state the specific item asked for, with any qualifier: revenue "from its largest single customer", "lobbying expenses", "carbon offset purchases" must be stated as such.
- A passage that only states a similar figure for another item does not count.

Answer with one word, YES or NO."""


def _llm_answerable(question: str, chunks: list[Document]) -> bool:
    context = format_context(chunks)[:6000]
    answer = make_llm(max_tokens=4).invoke(_CHECK_PROMPT.format(question=question, context=context))
    return "yes" in str(getattr(answer, "content", answer)).strip().lower()


def _is_answerable(index: dict, question: str, picked: list[int], companies: list[str]) -> bool:
    terms = _metric_terms(question, companies)
    df = index.get("vocabulary") or index.get("df") or {}
    unseen = []
    for term in terms:
        variants = {term} | set(_SYNONYMS.get(term, ()))
        if not any(df.get(v, 0) > 0 for v in variants):
            unseen.append(term)
    if unseen:
        return False                    # the corpus never mentions the item that is asked for
    if CONFIG["llm_check"]:
        chunks = [index["chunks"][i] for i in picked]
        coverage = max((_metric_coverage(terms, c.page_content) for c in chunks), default=0.0)
        if coverage < 0.95 and not _llm_answerable(question, chunks):
            return False
    return True


def retrieve(index: Any, question: str) -> list[Document]:
    """Tasks 2 and 3: at most MAX_RETRIEVED of the chunks given to build_index, best first.

    Return [] when the corpus cannot answer the question: the generator then answers NOT_FOUND.
    """
    if not index or not index.get("chunks"):
        return []
    chunks = index["chunks"]
    question = str(question)
    parsed = parse_query(question)
    companies = [c for c in (parsed.get("companies") or []) if _ckey(c)]
    keys: list[str] = []
    for name in companies:
        key = _ckey(name)
        if key not in keys:
            keys.append(key)
    year = _as_int(parsed.get("fiscal_year"))
    doc_type = parsed.get("doc_type")
    if doc_type not in _DOC_LABELS:
        doc_type = "annual_report"

    if CONFIG["metadata_filter"]:
        known = [k for k in keys if k in index["company_keys"]]
        if CONFIG["abstain"] and len(known) < len(keys):
            return []                   # the question is about a company that is not in the library
        keys = known

    candidates = []
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        if CONFIG["metadata_filter"]:
            if keys and _ckey(meta.get("company")) not in keys:
                continue
            if meta.get("doc_type") != doc_type:
                continue
            if year is not None and not _doc_covers_year(doc_type, _as_int(meta.get("fiscal_year")), year):
                continue
        candidates.append(i)
    if not candidates:
        return []

    query_tokens = _query_tokens(question, companies)
    scores = (index["bm25"].get_scores(query_tokens)
              if (query_tokens and index.get("bm25") is not None) else [0.0] * len(chunks))

    def score(i: int) -> float:
        value = float(scores[i])
        meta = chunks[i].metadata or {}
        if year is not None and _as_int(meta.get("fiscal_year")) == year:
            value *= 1.15                # prefer the report of the year the question names
        return value

    ranked = sorted(candidates, key=lambda i: (-score(i), i))
    picked = ranked[:MAX_RETRIEVED]
    if CONFIG["per_company"] and len(keys) > 1:
        # A comparison needs a chunk of every company it names.  Start from the BM25 top-5 and,
        # if one of the companies is missing, swap the weakest chunk of the best represented
        # company for that company's best chunk.
        for key in keys:
            if any(_ckey(chunks[i].metadata.get("company")) == key for i in picked):
                continue
            best = next((i for i in ranked if _ckey(chunks[i].metadata.get("company")) == key), None)
            if best is None or not picked:
                break
            target = None
            for position in sorted(range(len(picked)), key=lambda pos: (score(picked[pos]), pos)):
                other = _ckey(chunks[picked[position]].metadata.get("company"))
                if sum(1 for i in picked if _ckey(chunks[i].metadata.get("company")) == other) > 1:
                    target = position
                    break
            if target is None:
                target = len(picked) - 1
            picked[target] = best
        picked.sort(key=lambda i: (-score(i), i))

    picked = picked[:MAX_RETRIEVED]
    if not picked:
        return []
    if CONFIG["abstain"] and not _is_answerable(index, question, picked, companies):
        return []
    return [chunks[i] for i in picked]


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
