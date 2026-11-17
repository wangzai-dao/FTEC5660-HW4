"""Diagnostics: why each question is kept or abstained, and what is retrieved."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import hw4

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "public_test")
corpus = hw4.load_corpus(folder / "corpus")
questions = hw4.load_json(folder / "questions.json") or {}
truth = hw4.load_json(folder / "ground_truth.json") or {}
docs_meta = hw4.load_json(folder / "docs.json") or {}

chunks_by_file = {f: hw4.chunk_document(f, h) for f, h in corpus.items()}
all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
              if isinstance(c, hw4.Document) and len(c.page_content) <= hw4.MAX_CHUNK_CHARS]
index = hw4.build_index(all_chunks)

for qid in sorted(questions):
    question = questions[qid]
    parsed = hw4.parse_query(question)
    companies = parsed.get("companies") or []
    terms = hw4._metric_terms(question, companies)
    df = index["df"]
    seen = []
    for t in terms:
        variants = {t} | set(hw4._SYNONYMS.get(t, ()))
        if not any(df.get(v, 0) > 0 for v in variants):
            seen.append(t)
    unseen = seen
    got = hw4.retrieve(index, question)
    gold = truth.get(qid, {})
    hit = hw4.retrieval_hit(got, gold) if gold.get("evidence") else None
    flag = "OK " if (got and (hit is not False)) or (not got and not gold.get("evidence")) else "BAD"
    print(f"{flag} {qid} n={len(got)} hit={hit} unseen={unseen} terms={terms[:8]}")
    print(f"     {question}")
    print(f"     parsed={parsed}")
    for c in got:
        print(f"       - {c.metadata.get('source')} | {c.metadata.get('kind')} | "
              f"{c.metadata.get('section', '')[:45]} | {c.page_content[60:150]!r}")
