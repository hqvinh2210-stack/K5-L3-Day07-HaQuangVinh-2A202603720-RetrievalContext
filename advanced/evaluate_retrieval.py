"""Evaluate advanced retrieval with multi-region gold evidence.

Only DEV may be used to select a configuration.  TEST is run once after the
pipeline is locked.  Unanswerable cases are reserved for agent evaluation.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))

from baseline import pack_context  # noqa: E402
from retrieval import build_retrievers  # noqa: E402


METHODS = (
    "tfidf",
    "dense",
    "hybrid",
    "hybrid_dense2",
    "hybrid_lexical2",
    "hybrid_interleave",
)


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("‐", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", value).strip().casefold()


def row_matches_gold(row: dict[str, Any], gold: dict[str, Any]) -> bool:
    return (
        row.get("source_id") == gold["source_id"]
        and int(row.get("page", -1)) == int(gold["page"])
        and norm(gold["evidence"]) in norm(row.get("text", ""))
    )


def coverage(
    rows: list[dict[str, Any]], gold_items: list[dict[str, Any]], *, require_all: bool
) -> tuple[bool, list[int]]:
    matched = [
        index
        for index, gold in enumerate(gold_items)
        if any(row_matches_gold(row, gold) for row in rows)
    ]
    enough = len(matched) == len(gold_items) if require_all else bool(matched)
    return enough, matched


def evaluate(
    name: str,
    retriever: Any,
    questions: list[dict[str, Any]],
    *,
    top_k: int,
    budget: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    for question in questions:
        started = time.perf_counter()
        rows = retriever.search(question["question"], filters=None, top_k=top_k)
        context, context_ids = pack_context(rows, budget=budget)
        require_all = bool(question.get("requires_all_gold"))
        hit1, gold_at_1 = coverage(rows[:1], question["gold"], require_all=require_all)
        hit3, gold_at_3 = coverage(rows[:3], question["gold"], require_all=require_all)
        context_rows = [row for row in rows if row["id"] in context_ids]
        context_hit, gold_in_context = coverage(
            context_rows, question["gold"], require_all=require_all
        )
        marked_rows = []
        for rank, row in enumerate(rows, start=1):
            marked_rows.append({
                **row,
                "rank": rank,
                "matched_gold": [
                    index
                    for index, gold in enumerate(question["gold"])
                    if row_matches_gold(row, gold)
                ],
            })
        traces.append({
            "query_id": question["id"],
            "group": question["group"],
            "question": question["question"],
            "method": name,
            "requires_all_gold": require_all,
            "gold_regions": len(question["gold"]),
            "top_k": top_k,
            "budget_chars": budget,
            "results": marked_rows,
            "context_ids": context_ids,
            "context_chars": len(context),
            "hit1": hit1,
            "hit3": hit3,
            "context_hit": context_hit,
            "matched_gold_at_1": gold_at_1,
            "matched_gold_at_3": gold_at_3,
            "matched_gold_in_context": gold_in_context,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        })

    by_group: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "hit3": 0, "context_hit": 0})
    for trace in traces:
        group = by_group[trace["group"]]
        group["n"] += 1
        group["hit3"] += int(trace["hit3"])
        group["context_hit"] += int(trace["context_hit"])
    n = len(traces)
    summary = {
        "method": name,
        "n": n,
        "hit1": sum(int(t["hit1"]) for t in traces),
        "hit3": sum(int(t["hit3"]) for t in traces),
        "context_hit": sum(int(t["context_hit"]) for t in traces),
        "hit1_rate": sum(int(t["hit1"]) for t in traces) / n,
        "hit3_rate": sum(int(t["hit3"]) for t in traces) / n,
        "context_evidence_rate": sum(int(t["context_hit"]) for t in traces) / n,
        "median_ms": statistics.median(t["elapsed_ms"] for t in traces),
        "avg_context_chars": statistics.mean(t["context_chars"] for t in traces),
        "by_group": dict(sorted(by_group.items())),
        "top_k": top_k,
        "budget_chars": budget,
        "service_cost_usd": 0,
    }
    return summary, traces


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=ROOT / "advanced" / "data" / "chunks.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "advanced" / "benchmark" / "questions.json")
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dense-strategy", default="ADVANCED_DSFB")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--budget", type=int, default=8000)
    args = parser.parse_args()
    if args.top_k < 3 or args.budget < 1:
        parser.error("top-k phải >=3 và budget phải >0")

    payload = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = [
        q for q in payload["questions"]
        if q["split"] == args.split and q["answerable"]
    ]
    requested = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = set(requested) - set(METHODS)
    if unknown:
        parser.error(f"method không hỗ trợ: {sorted(unknown)}")
    retrievers = build_retrievers(args.chunks, dense_strategy=args.dense_strategy)

    summaries = []
    traces = []
    for method in requested:
        summary, method_traces = evaluate(
            method, retrievers[method], questions, top_k=args.top_k, budget=args.budget
        )
        summaries.append(summary)
        traces.extend(method_traces)

    output = {
        "split": args.split,
        "answerable_questions": len(questions),
        "multi_hop_rule": "all gold regions must be present",
        "hit_rule": "gold source, physical PDF page, and evidence substring must match",
        "selection_note": (
            "DEV is used for configuration selection."
            if args.split == "dev"
            else "TEST was evaluated only after the pipeline was locked."
        ),
        "methods": summaries,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
