"""Run the real Day 7 agent and save reproducible per-question traces."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from agentic import run_agent
from model_controller import AnthropicController, OpenAIController
from retrieval import build_retrievers


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Chạy agent thật trên một split")
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument(
        "--method",
        choices=(
            "tfidf",
            "dense",
            "hybrid",
            "hybrid_dense2",
            "hybrid_lexical2",
            "hybrid_interleave",
        ),
        default="hybrid_interleave",
    )
    parser.add_argument("--model", help="Mặc định đọc OPENAI_MODEL")
    parser.add_argument(
        "--provider", choices=("anthropic", "openai"), default="anthropic"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-searches", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=1600)
    parser.add_argument("--dense-strategy", default="B")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--question-id", help="Chạy một câu để smoke test")
    args = parser.parse_args()

    payload = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = payload["questions"] if isinstance(payload, dict) else payload
    selected = [question for question in questions if question.get("split") == args.split]
    if args.question_id:
        selected = [question for question in selected if question.get("id") == args.question_id]
    if not selected:
        parser.error("Không có câu hỏi phù hợp")

    retriever = build_retrievers(
        args.chunks, dense_strategy=args.dense_strategy
    )[args.method]
    controller_class = AnthropicController if args.provider == "anthropic" else OpenAIController
    if args.provider == "anthropic":
        controller = controller_class(
            model=args.model, max_tokens=args.max_output_tokens
        )
    else:
        controller = controller_class(model=args.model)
    traces = []
    for question in selected:
        started = time.perf_counter()
        runtime_question = question["question"]
        if question.get("group") == "multi_hop":
            runtime_question += (
                "\nYêu cầu retrieval: đây là câu multi-hop. Chỉ trả lời sau khi "
                "đã tìm và trích dẫn bằng chứng cho từng ý từ ít nhất hai trang "
                "hoặc hai vùng nguồn độc lập."
            )
        try:
            result = run_agent(
                runtime_question,
                controller,
                retriever,
                filters=question.get("filter"),
                max_searches=args.max_searches,
                top_k=args.top_k,
                budget=args.budget,
            )
            trace = {
                "query_id": question["id"],
                "split": question["split"],
                "answerable": question.get("answerable", True),
                "group": question.get("group"),
                "question": question["question"],
                "method": args.method,
                "provider": args.provider,
                "model": controller.model,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                **result,
            }
        except Exception as exc:  # Preserve provider/retriever failures explicitly.
            trace = {
                "query_id": question["id"],
                "split": question["split"],
                "answerable": question.get("answerable", True),
                "group": question.get("group"),
                "question": question["question"],
                "method": args.method,
                "provider": args.provider,
                "model": controller.model,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "status": "error",
                "answer": "",
                "citations": [],
                "events": [],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        traces.append(trace)
        print(f"{trace['query_id']}: {trace['status']}")

    output = {
        "split": args.split,
        "method": args.method,
        "dense_strategy": args.dense_strategy,
        "max_searches": args.max_searches,
        "top_k": args.top_k,
        "budget_chars": args.budget,
        "max_output_tokens": args.max_output_tokens,
        "provider": args.provider,
        "model": controller.model,
        "questions": len(traces),
        "answered": sum(trace["status"] == "answered" for trace in traces),
        "abstained": sum(trace["status"] == "abstain" for trace in traces),
        "errors": sum(trace["status"] == "error" for trace in traces),
        "model_calls": controller.calls,
        "usage": controller.usage,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
