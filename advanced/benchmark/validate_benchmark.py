from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = ROOT / "advanced" / "benchmark" / "questions.json"
DEFAULT_CHUNKS = ROOT / "advanced" / "data" / "chunks.json"


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("‐", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", value).strip().casefold()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    questions = benchmark["questions"]
    errors: list[str] = []

    ids = [q["id"] for q in questions]
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate ids: {duplicates}")

    expected = {
        ("dev", "table"): 3,
        ("dev", "image"): 3,
        ("dev", "chart"): 3,
        ("dev", "multi_hop"): 3,
        ("dev", "text"): 3,
        ("dev", "unanswerable"): 3,
        ("test", "table"): 6,
        ("test", "image"): 5,
        ("test", "chart"): 5,
        ("test", "multi_hop"): 4,
        ("test", "text"): 4,
        ("test", "unanswerable"): 6,
    }
    actual = Counter((q["split"], q["group"]) for q in questions)
    if actual != Counter(expected):
        errors.append(f"distribution mismatch: {dict(actual)}")

    by_page: dict[tuple[str, int], list[str]] = {}
    for chunk in chunks:
        key = (chunk["source_id"], int(chunk["page"]))
        by_page.setdefault(key, []).append(norm(chunk["text"]))

    for q in questions:
        if q["answerable"]:
            if not q.get("gold") or not q.get("expected_answer"):
                errors.append(f"{q['id']}: answerable item lacks gold/answer")
                continue
            if q["group"] == "multi_hop" and len(q["gold"]) < 2:
                errors.append(f"{q['id']}: multi-hop item has fewer than two gold regions")
            for index, gold in enumerate(q["gold"], start=1):
                texts = by_page.get((gold["source_id"], int(gold["page"])), [])
                needle = norm(gold["evidence"])
                if not any(needle in text for text in texts):
                    errors.append(
                        f"{q['id']} gold#{index}: evidence not found on page {gold['page']}: "
                        f"{gold['evidence']!r}"
                    )
        elif q.get("expected_behavior") != "search_and_abstain" or not q.get("absence_reason"):
            errors.append(f"{q['id']}: unanswerable item lacks abstention metadata")

    print(json.dumps({
        "questions": len(questions),
        "dev": sum(q["split"] == "dev" for q in questions),
        "test": sum(q["split"] == "test" for q in questions),
        "distribution": {f"{split}/{group}": count for (split, group), count in sorted(actual.items())},
        "errors": errors,
    }, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
