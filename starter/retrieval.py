"""Retrieval methods used by the Day 7 experiments.

All retrievers implement the interface expected by ``agentic.run_agent``.
The module contains three genuinely different methods: lexical TF-IDF, dense
pgvector retrieval from Day 6, and rank-based hybrid fusion.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from baseline import TfidfIndex
from day6_connection import Day6DenseRetriever


class Retriever(Protocol):
    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        ...


def metadata_matches(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Apply trusted user/benchmark metadata filters consistently."""
    if not filters:
        return True
    metadata = chunk.get("metadata") or {}
    return all(str(metadata.get(key)) == str(value) for key, value in filters.items())


class TfidfRetriever:
    """Lexical retrieval over the exported Day 6 chunks."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            raise ValueError("chunks không được rỗng")
        self.chunks = chunks
        self.index = TfidfIndex(chunks)

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k phải lớn hơn 0")
        # Rank first, then apply the same metadata semantics as pgvector. Ask
        # for the full lexical ranking so filtering cannot hide later matches.
        ranked = self.index.search(query, top_k=len(self.chunks))
        return [
            {**chunk, "strategy": "tfidf"}
            for chunk in ranked
            if metadata_matches(chunk, filters)
        ][:top_k]


class HybridRetriever:
    """Fuse dense and lexical rankings with Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        dense: Retriever,
        lexical: Retriever,
        *,
        candidate_k: int = 10,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
    ) -> None:
        if candidate_k < 1 or rrf_k < 1 or dense_weight <= 0 or lexical_weight <= 0:
            raise ValueError("candidate_k, rrf_k và trọng số phải lớn hơn 0")
        self.dense = dense
        self.lexical = lexical
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k phải lớn hơn 0")
        candidate_k = max(top_k, self.candidate_k)
        rankings = [
            (self.dense.search(query, filters=filters, top_k=candidate_k), self.dense_weight),
            (self.lexical.search(query, filters=filters, top_k=candidate_k), self.lexical_weight),
        ]
        fused: dict[str, dict[str, Any]] = {}
        for ranking, weight in rankings:
            for rank, chunk in enumerate(ranking, start=1):
                chunk_id = chunk["id"]
                if chunk_id not in fused:
                    fused[chunk_id] = {
                        **chunk,
                        "score": 0.0,
                        "strategy": "hybrid-rrf",
                        "component_ranks": {},
                    }
                component = chunk.get("strategy", "unknown")
                fused[chunk_id]["component_ranks"][component] = rank
                fused[chunk_id]["score"] += weight / (self.rrf_k + rank)
        return sorted(
            fused.values(),
            key=lambda chunk: (-chunk["score"], chunk["id"]),
        )[:top_k]


class InterleaveHybridRetriever:
    """Round-robin dense and lexical ranks while removing duplicate chunks."""

    def __init__(self, dense: Retriever, lexical: Retriever, *, candidate_k: int = 10) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k phải lớn hơn 0")
        self.dense = dense
        self.lexical = lexical
        self.candidate_k = candidate_k

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k phải lớn hơn 0")
        candidate_k = max(top_k, self.candidate_k)
        dense_rows = self.dense.search(query, filters=filters, top_k=candidate_k)
        lexical_rows = self.lexical.search(query, filters=filters, top_k=candidate_k)
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rank in range(candidate_k):
            for source, rows in (("dense", dense_rows), ("tfidf", lexical_rows)):
                if rank >= len(rows) or rows[rank]["id"] in seen:
                    continue
                row = rows[rank]
                seen.add(row["id"])
                output.append(
                    {
                        **row,
                        "score": 1.0 / (rank + 1),
                        "strategy": "hybrid-interleave",
                        "selected_from": source,
                        "component_rank": rank + 1,
                    }
                )
                if len(output) == top_k:
                    return output
        return output


def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("File chunks phải là JSON array không rỗng")
    return chunks


def build_retrievers(
    chunks_path: Path,
    *,
    dense_strategy: str = "B",
) -> dict[str, Retriever]:
    """Build retrievers once so the model and TF-IDF index are reused."""
    chunks = load_chunks(chunks_path)
    lexical = TfidfRetriever(chunks)
    dense = Day6DenseRetriever(strategy=dense_strategy)
    hybrid = HybridRetriever(dense, lexical)
    hybrid_dense2 = HybridRetriever(dense, lexical, dense_weight=2.0)
    hybrid_lexical2 = HybridRetriever(dense, lexical, lexical_weight=2.0)
    hybrid_interleave = InterleaveHybridRetriever(dense, lexical)
    return {
        "tfidf": lexical,
        "dense": dense,
        "hybrid": hybrid,
        "hybrid_dense2": hybrid_dense2,
        "hybrid_lexical2": hybrid_lexical2,
        "hybrid_interleave": hybrid_interleave,
    }
