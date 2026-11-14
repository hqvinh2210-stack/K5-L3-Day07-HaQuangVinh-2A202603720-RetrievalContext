"""Aggregate mandatory human pass/fail labels for the advanced benchmark."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--reviews", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payload = load(args.benchmark)
    cases = [q for q in payload["questions"] if q["split"] == args.split]
    traces = load(args.traces)
    reviews = load(args.reviews)
    trace_by_id = {row["query_id"]: row for row in traces}
    review_by_id = {row["query_id"]: row for row in reviews}
    if len(trace_by_id) != len(traces) or len(review_by_id) != len(reviews):
        raise ValueError("trace/review có ID trùng")

    rows = []
    by_group: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
    elapsed = []
    errors = 0
    for case in cases:
        cid = case["id"]
        if cid not in trace_by_id or cid not in review_by_id:
            raise ValueError(f"{cid}: thiếu trace hoặc nhãn review")
        trace = trace_by_id[cid]
        review = review_by_id[cid]
        passed = review.get("passed")
        if type(passed) is not bool:
            raise ValueError(f"{cid}: passed phải là boolean")
        if not isinstance(review.get("reason"), str) or not review["reason"].strip():
            raise ValueError(f"{cid}: thiếu lý do review")
        if case["answerable"]:
            checks = review.get("evidence_checks")
            if not isinstance(checks, list) or len(checks) != len(case["gold"]):
                raise ValueError(f"{cid}: cần một evidence_check cho mỗi vùng gold")
            if any(type(value) is not bool for value in checks):
                raise ValueError(f"{cid}: evidence_checks phải là boolean")
            if passed and (trace.get("status") != "answered" or not all(checks)):
                raise ValueError(f"{cid}: nhãn pass không khớp answer/evidence")
        elif passed and trace.get("status") != "abstain":
            raise ValueError(f"{cid}: câu thiếu evidence chỉ pass khi agent abstain")
        if trace.get("status") == "error":
            errors += 1
            if passed:
                raise ValueError(f"{cid}: provider/system error không thể pass")
        elapsed.append(float(trace.get("elapsed_ms", 0)))
        by_group[case["group"]]["total"] += 1
        by_group[case["group"]]["passed"] += int(passed)
        rows.append({
            "query_id": cid,
            "group": case["group"],
            "answerable": case["answerable"],
            "status": trace.get("status"),
            "passed": passed,
            "reason": review["reason"],
        })

    passed_count = sum(int(row["passed"]) for row in rows)
    total = len(rows)
    report = {
        "split": args.split,
        "passed": passed_count,
        "total": total,
        "pass_rate": passed_count / total if total else 0,
        "threshold_met": passed_count / total >= 0.9 if total else False,
        "system_or_provider_errors": errors,
        "elapsed_ms": {
            "total": sum(elapsed),
            "median": statistics.median(elapsed) if elapsed else 0,
        },
        "by_group": dict(sorted(by_group.items())),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
