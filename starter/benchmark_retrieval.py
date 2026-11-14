"""Benchmark Day 7 retrieval methods without exposing gold to retrievers."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from baseline import hit, pack_context, validate
from retrieval import build_retrievers


METHODS = (
    ("tfidf", "tfidf", False),
    ("dense", "dense", False),
    ("dense_filter", "dense", True),
    ("hybrid_rrf", "hybrid", True),
    ("hybrid_dense2", "hybrid_dense2", True),
    ("hybrid_lexical2", "hybrid_lexical2", True),
    ("hybrid_interleave", "hybrid_interleave", True),
)


def evaluate_method(
    method_name: str,
    retriever: Any,
    questions: list[dict[str, Any]],
    *,
    use_filters: bool,
    top_k: int,
    budget: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces = []
    for question in questions:
        filters = question.get("filter") if use_filters else None
        started = time.perf_counter()
        results = retriever.search(
            question["question"], filters=filters, top_k=top_k
        )
        context, context_ids = pack_context(results, budget=budget)
        elapsed_ms = (time.perf_counter() - started) * 1000
        marked = [{**row, "hit": hit(row, question["gold"])} for row in results]
        traces.append(
            {
                "query_id": question["id"],
                "split": question["split"],
                "question": question["question"],
                "method": method_name,
                "filter": filters,
                "top_k": top_k,
                "budget_chars": budget,
                "results": marked,
                "context": context,
                "context_ids": context_ids,
                "context_chars": len(context),
                "elapsed_ms": elapsed_ms,
                "hit1": bool(marked and marked[0]["hit"]),
                "hit3": any(row["hit"] for row in marked[:3]),
                "context_hit": any(
                    row["hit"] and row["id"] in context_ids for row in marked
                ),
            }
        )
    count = len(traces)
    summary = {
        "method": method_name,
        "split": questions[0]["split"] if questions else None,
        "n": count,
        "hit1": sum(trace["hit1"] for trace in traces),
        "hit3": sum(trace["hit3"] for trace in traces),
        "context_hit": sum(trace["context_hit"] for trace in traces),
        "hit1_rate": sum(trace["hit1"] for trace in traces) / count if count else 0,
        "hit3_rate": sum(trace["hit3"] for trace in traces) / count if count else 0,
        "context_evidence_rate": (
            sum(trace["context_hit"] for trace in traces) / count if count else 0
        ),
        "median_ms": statistics.median(
            trace["elapsed_ms"] for trace in traces
        ) if traces else 0,
        "avg_context_chars": statistics.mean(
            trace["context_chars"] for trace in traces
        ) if traces else 0,
        "top_k": top_k,
        "budget_chars": budget,
        "uses_metadata_filter": use_filters,
        "service_cost_usd": 0,
    }
    return summary, traces


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="So sánh TF-IDF, dense Day 6 và hybrid RRF"
    )
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=1600)
    parser.add_argument("--dense-strategy", default="B")
    parser.add_argument(
        "--methods",
        help="Danh sách method cách nhau bằng dấu phẩy; mặc định chạy tất cả",
    )
    args = parser.parse_args()
    if args.top_k < 3:
        parser.error("--top-k phải ít nhất 3 để tính Evidence Hit@3")
    if args.budget < 1:
        parser.error("--budget phải lớn hơn 0")

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    all_questions = json.loads(args.questions.read_text(encoding="utf-8"))
    # Unanswerable cases belong to agent abstention evaluation, not retrieval
    # Hit@k. Only inspect gold for answerable questions in the selected split.
    questions = [
        question
        for question in all_questions
        if question.get("split") == args.split
        and question.get("answerable", True)
    ]
    validate(chunks, questions)
    retrievers = build_retrievers(
        args.chunks, dense_strategy=args.dense_strategy
    )

    selected_methods = METHODS
    if args.methods:
        requested = {name.strip() for name in args.methods.split(",") if name.strip()}
        known = {name for name, _, _ in METHODS}
        unknown = requested - known
        if unknown:
            parser.error(f"Method không hỗ trợ: {sorted(unknown)}")
        selected_methods = tuple(row for row in METHODS if row[0] in requested)

    summaries = []
    all_traces = []
    for method_name, retriever_name, use_filters in selected_methods:
        summary, traces = evaluate_method(
            method_name,
            retrievers[retriever_name],
            questions,
            use_filters=use_filters,
            top_k=args.top_k,
            budget=args.budget,
        )
        summaries.append(summary)
        all_traces.extend(traces)

    output = {
        "split": args.split,
        "answerable_questions": len(questions),
        "selection_note": (
            "Chỉ dùng dev để thử/chọn cấu hình; không dùng test để tinh chỉnh."
            if args.split == "dev"
            else "Test chỉ chạy sau khi đã chốt cấu hình trên dev."
        ),
        "methods": summaries,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "traces.json").write_text(
        json.dumps(all_traces, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
