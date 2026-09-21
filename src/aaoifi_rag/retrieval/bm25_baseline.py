"""Fixed-token and clause-level BM25 baselines.

The module provides Config A's no-clause-awareness fixed-token chunks and a
separate clause-level index. It deliberately excludes dense retrieval,
reranking, SAC, generation, and evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable
import pickle
from pathlib import Path
import re
from typing import Any

from rank_bm25 import BM25Okapi
import tiktoken


FIXED_TOKENIZER_NAME = "o200k_base"
FIXED_CHUNK_SIZE = 256
FIXED_CHUNK_OVERLAP = 32
BM25_TOKEN_RE = re.compile(r"[\w]+(?:[’'][\w]+)?|\d+(?:/\d+)*", re.UNICODE)


def bm25_tokenize(text: str) -> list[str]:
    """A deterministic Unicode word tokenizer for BM25 term matching."""
    return [token.casefold() for token in BM25_TOKEN_RE.findall(text)]


def build_fixed_token_chunks(pages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create 256-token chunks with a 32-token overlap, ignoring clauses."""
    encoding = tiktoken.get_encoding(FIXED_TOKENIZER_NAME)
    chunks: list[dict[str, Any]] = []
    pages_by_standard: dict[str, list[dict[str, Any]]] = {}
    for page in pages:
        pages_by_standard.setdefault(page["standard_id"], []).append(page)

    step = FIXED_CHUNK_SIZE - FIXED_CHUNK_OVERLAP
    for standard_id, standard_pages in pages_by_standard.items():
        token_stream: list[int] = []
        page_spans: list[tuple[int, int, int]] = []
        for page in sorted(standard_pages, key=lambda item: item["source_page"]):
            start = len(token_stream)
            token_stream.extend(encoding.encode(page["text"], disallowed_special=()))
            end = len(token_stream)
            page_spans.append((page["source_page"], start, end))
        for offset in range(0, len(token_stream), step):
            token_ids = token_stream[offset : offset + FIXED_CHUNK_SIZE]
            if not token_ids:
                continue
            source_pages = [
                page_number
                for page_number, page_start, page_end in page_spans
                if page_start < offset + len(token_ids) and page_end > offset
            ]
            text = encoding.decode(token_ids)
            chunks.append(
                {
                    "chunk_id": f"fixed:{standard_id}:{offset:06d}",
                    "standard_id": standard_id,
                    "source_pages": source_pages,
                    "token_start": offset,
                    "token_end": offset + len(token_ids),
                    "token_count": len(token_ids),
                    "text": text,
                    "bm25_text": f"{standard_id} pages {' '.join(map(str, source_pages))} {text}",
                }
            )
    return chunks


def build_clause_chunks(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use each extracted clause/sub-clause record as exactly one BM25 chunk."""
    chunks: list[dict[str, Any]] = []
    for record in records:
        suffix = f":{record['sub_clause_id']}" if record["sub_clause_id"] else ""
        occurrence_index = record.get("occurrence_index", 0)
        chunks.append(
            {
                "chunk_id": (
                    f"clause:{record['standard_id']}:{record['clause_id']}"
                    f"{suffix}:occurrence:{occurrence_index}"
                ),
                "standard_id": record["standard_id"],
                "section_id": record["section_id"],
                "clause_id": record["clause_id"],
                "occurrence_index": occurrence_index,
                "sub_clause_id": record["sub_clause_id"],
                "source_page": record["source_page"],
                "text": record["text"],
                "bm25_text": (
                    f"{record['standard_id']} section {record['section_id']} "
                    f"clause {record['clause_id']} {record['sub_clause_id'] or ''} {record['text']}"
                ).strip(),
            }
        )
    return chunks


def persist_bm25_index(chunks: list[dict[str, Any]], output_path: Path, index_name: str) -> None:
    """Persist a local, private BM25 index and alignment metadata."""
    tokenized_corpus = [bm25_tokenize(chunk["bm25_text"]) for chunk in chunks]
    index = BM25Okapi(tokenized_corpus)
    payload = {
        "index_name": index_name,
        "bm25_tokenizer": "unicode_word_casefold_v1",
        "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
        "index": index,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        pickle.dump(payload, stream)


def search_bm25(chunks: list[dict[str, Any]], index_path: Path, query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Load a locally generated index and return scored chunks in rank order."""
    with index_path.open("rb") as stream:
        payload = pickle.load(stream)
    scores = payload["index"].get_scores(bm25_tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda index: float(scores[index]), reverse=True)[:top_k]
    return [{**chunks[index], "score": float(scores[index])} for index in ranked]
