"""Connect Day 7 to the PostgreSQL/pgvector index produced in Day 6.

The adapter intentionally reuses the exact Day 6 embedding model, vector
dimension, table, and strategy.  It can also export the selected strategy to
the Day 7 JSON contract so lexical baselines do not need a live database.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# Day 6 pins and caches this model locally; retrieval must remain reproducible
# without contacting the model hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DIMENSIONS = 384
DEFAULT_STRATEGY = "B"
ALLOWED_FILTERS = {"section", "year", "document_type", "language"}

REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS_ROOT = REPO_ROOT.parent
DEFAULT_DAY6_ROOT = DOWNLOADS_ROOT / "lab06"
DEFAULT_DAY6_ENV = DEFAULT_DAY6_ROOT / "starter" / ".env"
DEFAULT_DAY6_QUESTIONS = DEFAULT_DAY6_ROOT / "benchmark" / "questions.json"


def load_database_url(env_file: Path | None = None) -> str:
    """Load the Day 6 connection without copying its secret into this repo."""
    from dotenv import load_dotenv

    # Allow a local, ignored .env to point at a non-default Day 6 location.
    load_dotenv(REPO_ROOT / ".env", override=False)
    selected_env = env_file or Path(
        os.getenv("DAY6_ENV_FILE", str(DEFAULT_DAY6_ENV))
    )
    if selected_env.exists():
        load_dotenv(selected_env, override=False)

    database_url = os.getenv("DAY6_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "Thiếu DAY6_DATABASE_URL/DATABASE_URL và không đọc được cấu hình "
            f"từ {selected_env}"
        )
    return database_url


class Day6DenseRetriever:
    """Dense retriever compatible with ``agentic.run_agent``.

    ``strategy='B'`` is the Day 6 configuration selected on the dev split.
    Strategy C contains the same chunks/vectors and differs only by whether a
    trusted metadata filter is supplied at query time.
    """

    def __init__(
        self,
        database_url: str | None = None,
        *,
        strategy: str = DEFAULT_STRATEGY,
        model_name: str = DEFAULT_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS,
        device: str | None = None,
    ) -> None:
        if not strategy.strip() or dimensions <= 0:
            raise ValueError("strategy và dimensions phải hợp lệ")
        self.database_url = database_url or load_database_url()
        self.strategy = strategy
        self.model_name = model_name
        self.dimensions = dimensions
        self.device = device
        self._model = None
        self._query_cache: dict[str, list[float]] = {}

    def _embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise ValueError("query không được rỗng")
        if query in self._query_cache:
            return list(self._query_cache[query])
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            actual_dimensions = self._model.get_embedding_dimension()
            if actual_dimensions != self.dimensions:
                raise RuntimeError(
                    "Embedding model không khớp Day 6: "
                    f"{actual_dimensions} != {self.dimensions}"
                )
        vector = self._model.encode_query(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        result = vector.tolist()
        self._query_cache[query] = result
        return list(result)

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Embed a query and search the persisted Day 6 pgvector rows."""
        import psycopg

        if top_k < 1:
            raise ValueError("top_k phải lớn hơn 0")
        filters = filters or {}
        unknown_filters = set(filters) - ALLOWED_FILTERS
        if unknown_filters:
            raise ValueError(f"Filter không hỗ trợ: {sorted(unknown_filters)}")

        query_vector = self._embed_query(query)
        if len(query_vector) != self.dimensions:
            raise RuntimeError("Query embedding không đúng 384 chiều của Day 6")
        vector_literal = "[" + ",".join(str(value) for value in query_vector) + "]"

        clauses = ["strategy = %s"]
        where_params: list[Any] = [self.strategy]
        for key, value in filters.items():
            clauses.append("metadata ->> %s = %s")
            where_params.extend([key, str(value)])

        statement = f"""
            SELECT chunk_id, document_id, source, page, text_content,
                   metadata, 1 - (embedding <=> %s::vector) AS score
            FROM document_chunks
            WHERE {' AND '.join(clauses)}
            ORDER BY embedding <=> %s::vector, chunk_id
            LIMIT %s
        """
        params = [vector_literal, *where_params, vector_literal, top_k]
        with psycopg.connect(self.database_url) as connection:
            rows = connection.execute(statement, params).fetchall()

        return [
            {
                "id": row[0],
                "source_id": row[1],
                "source": row[2],
                "page": row[3],
                "text": row[4],
                "metadata": row[5],
                "score": float(row[6]),
                "strategy": f"day6-dense-{self.strategy}",
            }
            for row in rows
        ]


def check_connection(
    database_url: str | None = None,
    *,
    strategy: str = DEFAULT_STRATEGY,
) -> dict[str, Any]:
    """Verify the Day 6 table, strategy, row count, and vector dimension."""
    import psycopg

    url = database_url or load_database_url()
    statement = """
        SELECT COUNT(*), MIN(vector_dims(embedding)), MAX(vector_dims(embedding))
        FROM document_chunks
        WHERE strategy = %s
    """
    with psycopg.connect(url) as connection:
        row = connection.execute(statement, (strategy,)).fetchone()
    count = int(row[0]) if row else 0
    min_dimensions = int(row[1]) if row and row[1] is not None else None
    max_dimensions = int(row[2]) if row and row[2] is not None else None
    if count == 0:
        raise RuntimeError(f"Day 6 không có chunk cho strategy {strategy!r}")
    if min_dimensions != DEFAULT_DIMENSIONS or max_dimensions != DEFAULT_DIMENSIONS:
        raise RuntimeError("Vector trong Day 6 không đồng nhất ở 384 chiều")
    return {
        "status": "ok",
        "strategy": strategy,
        "chunks": count,
        "dimensions": min_dimensions,
        "embedding_model": DEFAULT_MODEL,
    }


def export_day6_data(
    output_dir: Path,
    *,
    database_url: str | None = None,
    strategy: str = DEFAULT_STRATEGY,
    questions_path: Path = DEFAULT_DAY6_QUESTIONS,
) -> dict[str, Any]:
    """Export pgvector chunks and Day 6 questions to the Day 7 JSON schema."""
    import psycopg

    url = database_url or load_database_url()
    statement = """
        SELECT chunk_id, document_id, source, page, text_content, metadata
        FROM document_chunks
        WHERE strategy = %s
        ORDER BY page, chunk_id
    """
    with psycopg.connect(url) as connection:
        rows = connection.execute(statement, (strategy,)).fetchall()
    if not rows:
        raise RuntimeError(f"Day 6 không có chunk cho strategy {strategy!r}")

    chunks = [
        {
            "id": row[0],
            "source_id": row[1],
            "source": row[2],
            "page": row[3],
            "text": row[4],
            "metadata": row[5],
        }
        for row in rows
    ]
    raw_questions = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = []
    for raw in raw_questions:
        converted = {
            "id": raw["id"],
            "split": raw["split"],
            "question": raw["question"],
            "expected_answer": raw["expected_answer"],
            "gold": [
                {
                    "source_id": raw["gold_source_id"],
                    "page": raw["gold_page"],
                    "evidence": raw["evidence_quote"],
                }
            ],
        }
        if raw.get("filter"):
            converted["filter"] = raw["filter"]
        if raw.get("question_type"):
            converted["question_type"] = raw["question_type"]
        questions.append(converted)

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in (
        ("day6_chunks.json", chunks),
        ("day6_questions.json", questions),
    ):
        (output_dir / filename).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return {
        "status": "ok",
        "strategy": strategy,
        "chunks": len(chunks),
        "questions": len(questions),
        "output": str(output_dir.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kết nối và xuất dữ liệu pgvector Day 6 cho Day 7"
    )
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--export", type=Path, metavar="OUTPUT_DIR")
    parser.add_argument("--questions", type=Path, default=DEFAULT_DAY6_QUESTIONS)
    args = parser.parse_args()
    if not args.check and args.export is None:
        parser.error("chọn --check, --export hoặc cả hai")

    database_url = load_database_url(args.env_file)
    if args.check:
        print(json.dumps(check_connection(database_url, strategy=args.strategy), ensure_ascii=False, indent=2))
    if args.export is not None:
        print(
            json.dumps(
                export_day6_data(
                    args.export,
                    database_url=database_url,
                    strategy=args.strategy,
                    questions_path=args.questions,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
