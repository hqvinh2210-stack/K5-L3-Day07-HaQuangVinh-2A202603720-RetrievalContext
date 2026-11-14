"""Create copyright-safe result artifacts from local full traces.

The submitted trace keeps IDs, pages, regions, ranks, answers and timings but
removes book text and cumulative context strings.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "advanced" / "results"
OUTPUT = ROOT / "advanced" / "submission"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(name: str, value: Any) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def usage_totals(summary: dict[str, Any]) -> dict[str, int]:
    usage = summary.get("usage", [])
    return {
        "model_calls": int(summary.get("model_calls", 0)),
        "input_tokens": sum(int(row.get("input_tokens", 0)) for row in usage),
        "output_tokens": sum(int(row.get("output_tokens", 0)) for row in usage),
        "thinking_tokens": sum(
            int((row.get("output_tokens_details") or {}).get("thinking_tokens", 0))
            for row in usage
        ),
    }


def result_ref(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    return {
        "id": row["id"],
        "source_id": row.get("source_id"),
        "page": row.get("page"),
        "content_type": metadata.get("content_type"),
        "label": metadata.get("label"),
        "bbox": metadata.get("bbox"),
        "score": row.get("score"),
        "strategy": row.get("strategy"),
        "component_ranks": row.get("component_ranks"),
    }


def sanitize_trace(trace: dict[str, Any]) -> dict[str, Any]:
    searches = []
    for event in trace.get("events", []):
        if event.get("event") != "search":
            continue
        searches.append({
            "query": event.get("query"),
            "results": [result_ref(row) for row in event.get("results", [])],
            "context_ids": event.get("context_ids", []),
            "context_chars": len(event.get("context", "")),
        })
    return {
        "query_id": trace["query_id"],
        "split": trace["split"],
        "group": trace.get("group"),
        "answerable": trace.get("answerable"),
        "question": trace.get("question"),
        "status": trace.get("status"),
        "answer": trace.get("answer"),
        "citations": trace.get("citations", []),
        "searches": searches,
        "elapsed_ms": trace.get("elapsed_ms"),
        "error_type": trace.get("error_type"),
        "error": trace.get("error"),
    }


def main() -> None:
    test_dir = RESULTS / "agent_test_final"
    test_summary = load(test_dir / "summary.json")
    dev_summary = load(RESULTS / "agent_dev" / "summary.json")
    test_usage = usage_totals(test_summary)
    dev_usage = usage_totals(dev_summary)
    pricing = {
        "source": "Anthropic Claude Haiku 5.5 announcement, checked 2026-11-14",
        "source_url": "https://www.anthropic.com/claude-haiku-5-5",
        "prompt_tier": "each prompt below 100k tokens",
        "input_usd_per_million": 0.10,
        "output_usd_per_million": 0.50,
    }
    test_cost = (
        test_usage["input_tokens"] * pricing["input_usd_per_million"]
        + test_usage["output_tokens"] * pricing["output_usd_per_million"]
    ) / 1_000_000
    dev_cost = (
        dev_usage["input_tokens"] * pricing["input_usd_per_million"]
        + dev_usage["output_tokens"] * pricing["output_usd_per_million"]
    ) / 1_000_000

    dump("test_trace_sanitized.json", [
        sanitize_trace(trace) for trace in load(test_dir / "traces.json")
    ])
    dump("test_reviews.json", load(test_dir / "reviews.json"))
    dump("test_metrics.json", load(test_dir / "metrics.json"))
    dump(
        "retrieval_test_summary.json",
        load(RESULTS / "retrieval_test_final" / "summary.json"),
    )
    dump(
        "run_costs.json",
        {
            "pricing": pricing,
            "test": {**test_usage, "estimated_usd": round(test_cost, 8)},
            "dev_main_run": {**dev_usage, "estimated_usd": round(dev_cost, 8)},
            "billing_note": "Estimate only; actual billed amount must be checked in the Anthropic Console.",
        },
    )
    manifest = load(ROOT / "advanced" / "data" / "manifest.json")
    dump(
        "index_manifest_public.json",
        {key: value for key, value in manifest.items() if key != "page_stats"},
    )


if __name__ == "__main__":
    main()
