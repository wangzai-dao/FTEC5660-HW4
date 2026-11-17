"""Fast local evaluation of Tasks 1.1-3 on the public set (no generator calls unless --generate).

    python3 eval.py --data-folder public_test [--generate] [--verbose]
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import hw4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", default="public_test")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a hw4.CONFIG entry, e.g. --set table_text=False")
    args = parser.parse_args()

    for item in args.set:
        key, _, value = item.partition("=")
        value = value.strip()
        if value.lower() in ("true", "false"):
            hw4.CONFIG[key] = value.lower() == "true"
        else:
            hw4.CONFIG[key] = json.loads(value) if value[:1].isdigit() else value

    folder = Path(args.data_folder)
    corpus = hw4.load_corpus(folder / "corpus")
    questions = hw4.load_json(folder / "questions.json") or {}
    truth = hw4.load_json(folder / "ground_truth.json") or {}
    docs_meta = hw4.load_json(folder / "docs.json") or {}
    tables = hw4.load_json(folder / "tables.json") or []
    start = time.perf_counter()

    print(f"{len(corpus)} documents, {len(questions)} questions")

    # ---- Task 1.1
    scores = []
    if tables:
        for table in tables:
            try:
                scores.append(hw4.table_score(hw4.table_to_text(table["html"]), table))
            except Exception:
                traceback.print_exc()
                scores.append(0.0)
        mean = sum(scores) / len(scores)
        print(f"Task 1.1 table cells kept: {mean:.1%}  ({hw4.POINTS['1.1'] * mean:.2f}/8)")
        if args.verbose:
            for table, score in zip(tables, scores):
                print(f"   {score:5.0%}  {table['caption'][:70]}")

    # ---- Task 1.2
    chunks_by_file, errors = {}, {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(hw4.chunk_document(file_name, html) or [])
        except Exception:
            traceback.print_exc()
            chunks_by_file[file_name] = []
            errors[file_name] = "error"
    problems = {f: hw4.document_problems(chunks_by_file.get(f), errors.get(f), m, f)
                for f, m in docs_meta.items()}
    passed = sum(not p for p in problems.values()) / max(len(problems), 1)
    print(f"Task 1.2 documents passing: {passed:.1%}  ({hw4.POINTS['1.2'] * passed:.2f}/7)")
    if args.verbose:
        total = sum(len(v) for v in chunks_by_file.values())
        print(f"   {total} chunks, mean {total / max(len(chunks_by_file), 1):.1f} per document")
        for f, p in problems.items():
            if p:
                print(f"   {f}: {'; '.join(p)}")

    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, hw4.Document) and len(c.page_content) <= hw4.MAX_CHUNK_CHARS]
    print(f"{len(all_chunks)} chunks to index")

    # ---- Task 2.2
    parsed = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = hw4.parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    index = hw4.build_index(all_chunks)
    retrieved, valid = {}, {}
    for qid, question in questions.items():
        try:
            result = hw4.retrieve(index, question)
            retrieved[qid] = list(result or [])[:hw4.MAX_RETRIEVED]
            valid[qid] = isinstance(result, list) and len(result) <= hw4.MAX_RETRIEVED
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False

    graded = [q for q in questions if q in truth]
    parse_scores = {q: hw4.parse_score(parsed[q], truth[q]) for q in graded}
    parse_mean = sum(parse_scores.values()) / max(len(graded), 1)
    print(f"Task 2.2 parse fields: {parse_mean:.1%}  ({hw4.POINTS['2.2'] * parse_mean:.2f}/10)")

    answerable = [q for q in graded if truth[q]["evidence"]]
    unanswerable = [q for q in graded if not truth[q]["evidence"]]
    hits = {q: hw4.retrieval_hit(retrieved[q], truth[q]) for q in answerable}
    recall = sum(hits.values()) / max(len(answerable), 1)
    non_empty = sum(1 for q in answerable if valid[q] and retrieved[q]) / max(len(answerable), 1)
    kept = sum(1 for q in answerable if retrieved[q]) / max(len(answerable), 1)
    rejected = sum(1 for q in unanswerable if not retrieved[q]) / max(len(unanswerable), 1)
    recall_points = hw4.POINTS["2.3"] * hw4.clip((recall - hw4.RECALL_ZERO)
                                                 / (hw4.RECALL_FULL - hw4.RECALL_ZERO))
    abstain_points = hw4.POINTS["3"] * hw4.clip(kept + rejected - 1)
    print(f"Task 2.1 valid non-empty: {non_empty:.1%}  ({hw4.POINTS['2.1'] * non_empty:.2f}/5)")
    print(f"Task 2.3 recall@5: {recall:.1%}  ({recall_points:.2f}/15)")
    print(f"Task 3 kept {kept:.1%} | rejected {rejected:.1%}  ({abstain_points:.2f}/10)")

    accuracy = None
    if args.generate:
        answers = {}
        for qid in questions:
            try:
                answers[qid] = hw4.generate(questions[qid], retrieved[qid])
            except Exception:
                traceback.print_exc()
                answers[qid] = ""
        correct = {q: hw4.answer_correct(answers[q], truth[q]) for q in graded}
        accuracy = sum(correct.values()) / max(len(graded), 1)
        print(f"Task 4 accuracy: {accuracy:.1%}  ({hw4.POINTS['4'] * accuracy:.2f}/20)")

    if args.verbose:
        print("\n--- parse mistakes")
        for q in graded:
            if parse_scores[q] < 1:
                print(f"  {q}: expected {truth[q]['query']} got {parsed[q]}")
                print(f"      {questions[q]}")
        print("\n--- recall misses")
        for q in answerable:
            if not hits[q]:
                print(f"  {q}: {questions[q]}")
                print(f"      evidence {truth[q]['evidence']}")
                for c in retrieved[q]:
                    print(f"      [{c.metadata.get('source')} {c.metadata.get('section', '')[:40]}]"
                          f" {c.page_content[:160]!r}")
        print("\n--- abstention mistakes")
        for q in unanswerable:
            if retrieved[q]:
                print(f"  {q} (should be []): {questions[q]}")
                print(f"      got {[c.metadata.get('source') for c in retrieved[q]]}")
        for q in answerable:
            if not retrieved[q]:
                print(f"  {q} (should not be []): {questions[q]} parsed={parsed[q]}")
        if accuracy is not None:
            print("\n--- wrong answers")
            for q in graded:
                if not correct[q]:
                    print(f"  {q} [{truth[q]['type']}] expected {truth[q]['answer']!r} "
                          f"got {answers[q]!r} | {questions[q]}")
                    print(f"      {[ (c.metadata.get('source'), c.metadata.get('section','')[:30]) for c in retrieved[q]]}")

    points = {"1.1": hw4.POINTS["1.1"] * (sum(scores) / len(scores) if scores else 0),
              "1.2": hw4.POINTS["1.2"] * passed,
              "2.1": hw4.POINTS["2.1"] * non_empty,
              "2.2": hw4.POINTS["2.2"] * parse_mean,
              "2.3": recall_points,
              "3": abstain_points}
    if accuracy is not None:
        points["4"] = hw4.POINTS["4"] * accuracy
    print("\n" + "  ".join(f"{k} {v:.2f}" for k, v in points.items())
          + f"  | total {sum(points.values()):.2f}/{sum(hw4.POINTS[k] for k in points)}")
    print(f"time {time.perf_counter() - start:.1f}s | usage {hw4.USAGE.summary()}")


if __name__ == "__main__":
    main()
