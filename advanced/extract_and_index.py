"""Extract and index the 130-page advanced-challenge PDF.

The generated chunks are local-only because the source is a copyrighted book.
Benchmark evidence contains only short excerpts needed for verification.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

# The course machine already caches the pinned model.  Prevent Hugging Face
# metadata probes from adding long network retries in an offline/sandbox run.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pdfplumber
import psycopg
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = REPO_ROOT / "Data Science for Business_1.pdf"
DEFAULT_OUTPUT = REPO_ROOT / "advanced" / "data"
DEFAULT_ANNOTATIONS = REPO_ROOT / "advanced" / "config" / "visual_annotations.json"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIMENSIONS = 384
STRATEGY = "ADVANCED_DSFB"
DOCUMENT_ID = "dsfb_2013_excerpt"


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    # Some mathematical glyphs are exposed by pypdf as lone UTF-16 surrogates.
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    text = text.replace("\u00ad", "").replace("�", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, size: int = 1400, overlap: int = 200) -> list[str]:
    if size < 200 or overlap < 0 or overlap >= size:
        raise ValueError("size/overlap không hợp lệ")
    flat = re.sub(r"\s+", " ", text).strip()
    if not flat:
        return []
    parts = []
    start = 0
    while start < len(flat):
        end = min(len(flat), start + size)
        if end < len(flat):
            boundary = flat.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        parts.append(flat[start:end].strip())
        if end >= len(flat):
            break
        start = max(start + 1, end - overlap)
    return parts


def table_to_text(rows: list[list[str | None]]) -> str:
    normalized = []
    for row in rows:
        cells = [clean_text(cell or "") for cell in row]
        if any(cells):
            normalized.append(" | ".join(cells))
    return "\n".join(normalized)


def load_visual_annotations(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("visual_annotations.json phải là array")
    seen = set()
    for item in data:
        for key in ("id", "page", "label", "visual_type", "description", "bbox"):
            if key not in item:
                raise ValueError(f"visual annotation thiếu {key}")
        if item["id"] in seen:
            raise ValueError(f"visual ID trùng: {item['id']}")
        seen.add(item["id"])
    return data


def extract(pdf_path: Path, annotations_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reader = PdfReader(pdf_path)
    annotations = load_visual_annotations(annotations_path)
    chunks: list[dict[str, Any]] = []
    page_stats = []
    table_count = 0
    image_count = 0

    with pdfplumber.open(pdf_path) as plumber_pdf:
        for page_number, (pdf_page, layout_page) in enumerate(
            zip(reader.pages, plumber_pdf.pages, strict=True), start=1
        ):
            text = clean_text(pdf_page.extract_text() or "")
            text_parts = split_text(text)
            for index, part in enumerate(text_parts):
                chunks.append(
                    {
                        "id": f"{DOCUMENT_ID}:p{page_number:03d}:text:{index:03d}",
                        "source_id": DOCUMENT_ID,
                        "source": pdf_path.name,
                        "page": page_number,
                        "text": part,
                        "metadata": {
                            "content_type": "text",
                            "language": "en",
                            "year": 2013,
                            "page": page_number,
                        },
                    }
                )

            page_tables = layout_page.find_tables()
            for table_index, table in enumerate(page_tables):
                table_text = table_to_text(table.extract())
                if not table_text:
                    continue
                table_count += 1
                bbox = [round(value, 2) for value in table.bbox]
                chunks.append(
                    {
                        "id": f"{DOCUMENT_ID}:p{page_number:03d}:table:{table_index:02d}",
                        "source_id": DOCUMENT_ID,
                        "source": pdf_path.name,
                        "page": page_number,
                        "text": f"Table on PDF page {page_number}.\n{table_text}",
                        "metadata": {
                            "content_type": "table",
                            "language": "en",
                            "year": 2013,
                            "page": page_number,
                            "bbox": bbox,
                        },
                    }
                )

            image_bboxes = [
                [round(image[key], 2) for key in ("x0", "top", "x1", "bottom")]
                for image in layout_page.images
            ]
            image_count += len(image_bboxes)
            page_stats.append(
                {
                    "page": page_number,
                    "text_chars": len(text),
                    "text_chunks": len(text_parts),
                    "tables": len(page_tables),
                    "images": len(image_bboxes),
                    "image_bboxes": image_bboxes,
                }
            )

    for annotation in annotations:
        chunks.append(
            {
                "id": annotation["id"],
                "source_id": DOCUMENT_ID,
                "source": pdf_path.name,
                "page": annotation["page"],
                "text": (
                    f"{annotation['label']} on PDF page {annotation['page']}. "
                    f"{annotation['description']}"
                ),
                "metadata": {
                    "content_type": annotation["visual_type"],
                    "language": "en",
                    "year": 2013,
                    "page": annotation["page"],
                    "label": annotation["label"],
                    "bbox": annotation["bbox"],
                    "annotation_source": "manual_visual_verification",
                },
            }
        )

    ids = [chunk["id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Chunk ID bị trùng")
    manifest = {
        "document_id": DOCUMENT_ID,
        "source": pdf_path.name,
        "title": "Data Science for Business",
        "authors": ["Foster Provost", "Tom Fawcett"],
        "copyright_year": 2013,
        "physical_pages": len(reader.pages),
        "text_chars": sum(row["text_chars"] for row in page_stats),
        "chunks": len(chunks),
        "text_chunks": sum(c["metadata"]["content_type"] == "text" for c in chunks),
        "table_chunks": sum(c["metadata"]["content_type"] == "table" for c in chunks),
        "visual_chunks": sum(c["metadata"]["content_type"] in {"image", "diagram", "chart"} for c in chunks),
        "detected_tables": table_count,
        "detected_image_objects": image_count,
        "pages_below_50_text_chars": [row["page"] for row in page_stats if row["text_chars"] < 50],
        "page_stats": page_stats,
        "rights_note": (
            "Copyrighted book; source PDF and extracted full-text chunks are for local "
            "coursework only and should not be redistributed. Submit source metadata, "
            "code, benchmark short excerpts, and reproduction instructions instead."
        ),
        "embedding_model": MODEL_NAME,
        "embedding_dimensions": DIMENSIONS,
        "vector_strategy": STRATEGY,
    }
    return chunks, manifest


def load_database_url() -> str:
    load_dotenv(REPO_ROOT / ".env", override=False)
    env_file = Path(os.getenv("DAY6_ENV_FILE", str(REPO_ROOT.parent / "lab06" / "starter" / ".env")))
    if not env_file.is_absolute():
        env_file = (REPO_ROOT / env_file).resolve()
    if env_file.exists():
        load_dotenv(env_file, override=False)
    database_url = os.getenv("DAY6_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Thiếu DAY6_DATABASE_URL/DATABASE_URL")
    return database_url


def index_chunks(chunks: list[dict[str, Any]], database_url: str) -> dict[str, Any]:
    model = SentenceTransformer(MODEL_NAME)
    if model.get_embedding_dimension() != DIMENSIONS:
        raise RuntimeError("Embedding dimension không khớp")
    started = time.perf_counter()
    vectors = model.encode_document(
        [chunk["text"] for chunk in chunks],
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()
    rows = [
        (
            STRATEGY,
            chunk["id"],
            chunk["source_id"],
            chunk["source"],
            chunk["page"],
            chunk["text"],
            json.dumps(chunk["metadata"], ensure_ascii=False),
            "[" + ",".join(str(value) for value in vector) + "]",
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    statement = """
        INSERT INTO document_chunks (
            strategy, chunk_id, document_id, source, page,
            text_content, metadata, embedding
        ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
    """
    with psycopg.connect(database_url) as connection:
        with connection.transaction():
            connection.execute("DELETE FROM document_chunks WHERE strategy = %s", (STRATEGY,))
            with connection.cursor() as cursor:
                cursor.executemany(statement, rows)
        count = connection.execute(
            "SELECT COUNT(*) FROM document_chunks WHERE strategy = %s", (STRATEGY,)
        ).fetchone()[0]
    if count != len(chunks):
        raise RuntimeError(f"Row count sai: {count} != {len(chunks)}")
    return {
        "strategy": STRATEGY,
        "rows": count,
        "seconds": time.perf_counter() - started,
        "model": MODEL_NAME,
        "dimensions": DIMENSIONS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract/index advanced PDF")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    chunks, manifest = extract(args.pdf, args.annotations)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not args.extract_only:
        manifest["index"] = index_chunks(chunks, load_database_url())
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "page_stats"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
