"""Task 5: the ablation study of FTEC5660 Homework 4.

Every variant flips one component of the pipeline relative to the row above it.  The variants are
the CONFIG flags of hw4.py, so the table below is exactly what the submitted code does.

    python3 ablate.py --data-folder public_test [--no-generate] [--out ablation_results.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import hw4

# name, what it adds, the flags it changes (cumulative, in this order)
VARIANTS = [
    ("B0", "naive RAG (flat text, recursive split, BM25 top-5, no parsing, never [])",
     {"table_text": False, "structural": False, "context_header": False, "rich_tokens": False,
      "metadata_filter": False, "per_company": False, "abstain": False}),
    ("V1", "+ table_to_text: every table becomes one line per row", {"table_text": True}),
    ("V2", "+ structural chunks: section-aware, table rows grouped", {"structural": True}),
    ("V3", "+ contextual chunk header (company | document | section)", {"context_header": True}),
    ("V4", "+ metadata filtering from parse_query (company, doc type, year)", {"metadata_filter": True}),
    ("V5", "+ rich BM25 tokens (stems, plurals, fiscal years, synonyms)", {"rich_tokens": True}),
    ("V6", "+ per-company retrieval for comparison questions", {"per_company": True}),
    ("V7", "+ abstention: unknown company, uncovered year, unseen metric", {"abstain": True}),
]

BASE = dict(table_text=False, structural=False, context_header=False, rich_tokens=False,
            metadata_filter=False, per_company=False, abstain=False, llm_parse=True,
            llm_check=False, table_chunk_chars=700, text_chunk_chars=950)


def measure(folder: Path, generate: bool) -> dict:
    corpus = hw4.load_corpus(folder / "corpus")
    questions = hw4.load_json(folder / "questions.json") or {}
    truth = hw4.load_json(folder / "ground_truth.json") or {}
    docs_meta = hw4.load_json(folder / "docs.json") or {}
    tables = hw4.load_json(folder / "tables.json") or []

    table_scores = [hw4.table_score(hw4.table_to_text(t["html"]), t) for t in tables]
    table_points = hw4.POINTS["1.1"] * sum(table_scores) / len(table_scores) if tables else 0.0

    chunks_by_file = {}
    for file_name, html in corpus.items():
        chunks_by_file[file_name] = list(hw4.chunk_document(file_name, html) or [])
    problems = {f: hw4.document_problems(chunks_by_file.get(f), None, m, f) for f, m in docs_meta.items()}
    passed = sum(not p for p in problems.values()) / max(len(problems), 1)
    chunk_points = hw4.POINTS["1.2"] * passed

    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, hw4.Document) and len(c.page_content) <= hw4.MAX_CHUNK_CHARS]
    index = hw4.build_index(all_chunks)

    parsed = {qid: hw4.parse_query(q) for qid, q in questions.items()}
    retrieved, valid = {}, {}
    for qid, question in questions.items():
        result = hw4.retrieve(index, question)
        retrieved[qid] = list(result or [])[:hw4.MAX_RETRIEVED]
        valid[qid] = isinstance(result, list) and len(result) <= hw4.MAX_RETRIEVED

    graded = [q for q in questions if q in truth]
    parse_mean = sum(hw4.parse_score(parsed[q], truth[q]) for q in graded) / max(len(graded), 1)
    answerable = [q for q in graded if truth[q]["evidence"]]
    unanswerable = [q for q in graded if not truth[q]["evidence"]]
    hits = {q: hw4.retrieval_hit(retrieved[q], truth[q]) for q in answerable}
    recall = sum(hits.values()) / max(len(answerable), 1)
    non_empty = sum(1 for q in answerable if valid[q] and retrieved[q]) / max(len(answerable), 1)
    kept = sum(1 for q in answerable if retrieved[q]) / max(len(answerable), 1)
    rejected = sum(1 for q in unanswerable if not retrieved[q]) / max(len(unanswerable), 1)

    # diagnostics that separate ranking quality from the saturated recall@5
    reciprocal = []
    for q in answerable:
        rank = None
        for position, chunk in enumerate(retrieved[q], 1):
            if any(any(chunk.metadata.get("source") == f and hw4.contains(chunk.page_content, printed)
                       for f, printed in group) for group in truth[q]["evidence"]):
                rank = position
                break
        reciprocal.append(1.0 / rank if rank else 0.0)
    comparisons = [q for q in answerable if truth[q]["category"] == "comparison"]
    both = sum(1 for q in comparisons
               if len({hw4.company_key(c.metadata.get("company")) for c in retrieved[q]}) >= 2)
    row = {"table": round(table_points, 2), "chunks": round(chunk_points, 2),
           "mrr": round(sum(reciprocal) / max(len(reciprocal), 1), 3),
           "two_company": round(both / max(len(comparisons), 1), 3),
           "recall_comparison": round(sum(hits[q] for q in comparisons) / max(len(comparisons), 1), 3),
           "valid": round(non_empty, 3), "parse": round(parse_mean, 3),
           "recall": round(recall, 3), "kept": round(kept, 3), "rejected": round(rejected, 3),
           "n_chunks": len(all_chunks)}
    row["p21"] = round(hw4.POINTS["2.1"] * non_empty, 2)
    row["p22"] = round(hw4.POINTS["2.2"] * parse_mean, 2)
    row["p23"] = round(hw4.POINTS["2.3"] * hw4.clip((recall - hw4.RECALL_ZERO)
                                                    / (hw4.RECALL_FULL - hw4.RECALL_ZERO)), 2)
    row["p3"] = round(hw4.POINTS["3"] * hw4.clip(kept + rejected - 1), 2)

    if generate:
        answers, correct, by_category = {}, {}, {}
        for qid in questions:
            try:
                answers[qid] = hw4.generate(questions[qid], retrieved[qid])
            except Exception:
                answers[qid] = ""
        for q in graded:
            correct[q] = hw4.answer_correct(answers[q], truth[q])
            by_category.setdefault(truth[q]["category"], []).append(correct[q])
        accuracy = sum(correct.values()) / max(len(graded), 1)
        row["accuracy"] = round(accuracy, 3)
        row["p4"] = round(hw4.POINTS["4"] * accuracy, 2)
        row["by_category"] = {k: round(sum(v) / len(v), 3) for k, v in sorted(by_category.items())}
    total = sum(row[k] for k in ("table", "chunks", "p21", "p22", "p23", "p3"))
    row["total"] = round(total + row.get("p4", 0.0), 2)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", default="public_test")
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--out", default="ablation_results.csv")
    args = parser.parse_args()
    folder = Path(args.data_folder)
    generate = not args.no_generate

    rows = []
    for position, (name, description, delta) in enumerate(VARIANTS):
        hw4.CONFIG.update(BASE)
        for _, _, earlier in VARIANTS[:position + 1]:   # cumulative: B0 + V1 + ... + Vn
            hw4.CONFIG.update(earlier)
        start = time.perf_counter()
        row = measure(folder, generate)
        row["variant"], row["description"] = name, description
        row["seconds"] = round(time.perf_counter() - start, 1)
        rows.append(row)
        print(f"{name:>3} {description[:60]:<60} "
              + "  ".join(f"{k}={row.get(k)}" for k in ("recall", "mrr", "two_company", "p23",
                                                        "p3", "p4", "total"))
              + f"  ({row['seconds']}s)")

    fields = ["variant", "description", "n_chunks", "table", "chunks", "valid", "parse", "recall",
              "mrr", "two_company", "recall_comparison",
              "kept", "rejected", "p21", "p22", "p23", "p3", "accuracy", "p4", "total", "seconds",
              "by_category"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row = dict(row)
            if "by_category" in row:
                row["by_category"] = json.dumps(row["by_category"])
            writer.writerow(row)
    print(f"\nwrote {args.out}")
    print(f"usage {hw4.USAGE.summary()}")
    print(f"generator {hw4.GENERATOR_USAGE.summary()}")


if __name__ == "__main__":
    main()
