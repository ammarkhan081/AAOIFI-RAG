"""Colab handoff for BGE-M3 retrieval and BGE-reranker-v2-m3.

Run this .py file cell-by-cell in Google Colab. It contains no AAOIFI text.
Upload the private Phase-2 ``clause_chunks.jsonl`` output before Cell 2.

The input, Chroma directory, archive, and results may contain licensed AAOIFI
text. Keep them private; do not commit or publish them without confirmed
redistribution rights.
"""

# %% [Cell 1] Install dependencies in a fresh Colab runtime
import os
from pathlib import Path
import subprocess
import sys
import time

# This keeps downloaded model files in the temporary Colab filesystem so the
# script can report their actual footprint in the final output metadata.
os.environ.setdefault("HF_HOME", "/content/huggingface_cache")

_install_started = time.perf_counter()
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "FlagEmbedding==1.4.0",
        "chromadb==1.5.9",
    ]
)
INSTALL_SECONDS = time.perf_counter() - _install_started
print(f"Dependency installation completed in {INSTALL_SECONDS:.1f}s")


# %% [Cell 2] Upload location and experiment configuration
# Upload the private Phase-2 clause_chunks.jsonl through Colab's Files panel as
# /content/clause_chunks.jsonl. Alternatively, uncomment these two lines:
# from google.colab import files
# files.upload()

INPUT_CHUNKS_PATH = Path("/content/clause_chunks.jsonl")
OUTPUT_DIR = Path("/content/aaoifi_dense_output")
CHROMA_DIR = OUTPUT_DIR / "chroma_bge_m3"
RESULTS_PATH = OUTPUT_DIR / "dense_sanity_results.json"
HF_HOME = Path(os.environ["HF_HOME"])

COLLECTION_NAME = "aaoifi_clause_chunks_bge_m3"
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
EMBEDDING_BATCH_SIZE = 8
EMBEDDING_MAX_LENGTH = 8192
RERANK_MAX_LENGTH = 1024
DENSE_CANDIDATE_K = 10

# Deliberate retrieval choice: BGE's English search instruction applies only
# to queries. Documents remain the exact private Phase-2 clause text.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

SANITY_QUERIES = [
    {
        "id": "ss8_murabahah_automatic_conclusion",
        "query": "What does SS8 say about the automatic conclusion of a Murabahah contract?",
    },
    {
        "id": "ss9_ijarah_return_without_consent",
        "query": "What does SS9 say about rental when an Ijarah asset is returned without the lessor's consent?",
    },
    {
        "id": "ss17_certificates_of_leased_assets",
        "query": "What does SS17 say about certificates of ownership of leased assets?",
    },
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if not INPUT_CHUNKS_PATH.is_file():
    raise FileNotFoundError(
        f"Missing {INPUT_CHUNKS_PATH}. Upload the private Phase-2 "
        "clause_chunks.jsonl before continuing."
    )


# %% [Cell 3] Imports and helper functions
import importlib.metadata
import json
import shutil
from typing import Any

import chromadb
from FlagEmbedding import BGEM3FlagModel
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    required = {
        "chunk_id",
        "standard_id",
        "section_id",
        "clause_id",
        "occurrence_index",
        "sub_clause_id",
        "source_page",
        "text",
    }
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = required.difference(record)
            if missing:
                raise ValueError(f"JSONL line {line_number} is missing: {sorted(missing)}")
            records.append(record)
    if not records:
        raise ValueError("The uploaded JSONL contains no records.")
    return records


def normalized_dense_vectors(model: BGEM3FlagModel, texts: list[str]) -> np.ndarray:
    encoded = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        max_length=EMBEDDING_MAX_LENGTH,
    )
    vectors = np.asarray(encoded["dense_vecs"], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("BGE-M3 returned a zero-norm embedding.")
    return vectors / norms


def chroma_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "standard_id": str(chunk["standard_id"]),
        "section_id": str(chunk["section_id"]),
        "clause_id": str(chunk["clause_id"]),
        "occurrence_index": int(chunk["occurrence_index"]),
        "source_page": int(chunk["source_page"]),
    }
    if chunk["sub_clause_id"] is not None:
        metadata["sub_clause_id"] = str(chunk["sub_clause_id"])
    return metadata


def directory_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def print_ranked_results(label: str, results: list[dict[str, Any]]) -> None:
    print(f"\n{label}")
    for rank, result in enumerate(results, start=1):
        meta = result["metadata"]
        if "reranker_score" in result:
            score = f"reranker_score={result['reranker_score']:.6f}"
        else:
            score = f"chroma_distance={result['chroma_distance']:.6f}"
        print(
            f"{rank}. {meta['standard_id']} §{meta['clause_id']} "
            f"(occurrence {meta['occurrence_index']}, page {meta['source_page']}; {score})"
        )
        print(result["text"])


class DirectBGEReranker:
    """Use BAAI's direct AutoTokenizer/sequence-classification pattern."""

    def __init__(self, model_id: str, device: str) -> None:
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def score_pairs(
        self,
        pairs: list[tuple[str, str]],
        *,
        batch_size: int,
        max_length: int,
    ) -> list[float]:
        scores: list[float] = []
        for offset in range(0, len(pairs), batch_size):
            batch = pairs[offset : offset + batch_size]
            queries = [pair[0] for pair in batch]
            passages = [pair[1] for pair in batch]
            tokenized = self.tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokenized = {name: tensor.to(self.device) for name, tensor in tokenized.items()}
            with torch.no_grad():
                logits = self.model(**tokenized, return_dict=True).logits.view(-1)
            scores.extend(float(value) for value in logits.float().cpu().tolist())
        return scores


# %% [Cell 4] Load and validate the private Phase-2 clause chunks
chunks = read_jsonl(INPUT_CHUNKS_PATH)
chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
if len(chunk_ids) != len(set(chunk_ids)):
    raise ValueError("clause_chunks.jsonl contains duplicate chunk_id values.")
print(f"Loaded {len(chunks)} clause-level chunks.")
if len(chunks) != 362:
    print("WARNING: this is not the verified 362-record Phase-2 corpus.")


# %% [Cell 5] Load BGE-M3 and create dense embeddings
gpu_available = torch.cuda.is_available()
device_description = torch.cuda.get_device_name(0) if gpu_available else "CPU"
use_fp16 = gpu_available
print(f"Device: {device_description}; use_fp16={use_fp16}")

_model_started = time.perf_counter()
embedding_model = BGEM3FlagModel(EMBEDDING_MODEL_ID, use_fp16=use_fp16)
EMBEDDING_MODEL_LOAD_SECONDS = time.perf_counter() - _model_started

_embedding_started = time.perf_counter()
document_vectors = normalized_dense_vectors(embedding_model, [chunk["text"] for chunk in chunks])
EMBEDDING_SECONDS = time.perf_counter() - _embedding_started
if document_vectors.shape[0] != len(chunks):
    raise RuntimeError("Embedding count does not match the uploaded chunk count.")
print(
    f"Generated {document_vectors.shape[0]} embeddings of dimension "
    f"{document_vectors.shape[1]} in {EMBEDDING_SECONDS:.1f}s"
)


# %% [Cell 6] Create and verify a persistent local Chroma vector store
# This removes only this explicit Colab output directory, so a rerun cannot mix
# a previous vector store with the currently uploaded source JSONL.
if CHROMA_DIR.exists():
    shutil.rmtree(CHROMA_DIR)

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)
collection.add(
    ids=chunk_ids,
    embeddings=document_vectors.tolist(),
    documents=[str(chunk["text"]) for chunk in chunks],
    metadatas=[chroma_metadata(chunk) for chunk in chunks],
)
if collection.count() != len(chunks):
    raise RuntimeError(f"Chroma stored {collection.count()} records; expected {len(chunks)}.")
print(f"Persisted and verified {collection.count()} vectors at {CHROMA_DIR}")


# %% [Cell 7] Dense-only retrieval and dense-plus-reranker sanity checks
_reranker_started = time.perf_counter()
reranker = DirectBGEReranker(RERANKER_MODEL_ID, "cuda" if gpu_available else "cpu")
RERANKER_MODEL_LOAD_SECONDS = time.perf_counter() - _reranker_started


def dense_retrieve(query: str, top_k: int = DENSE_CANDIDATE_K) -> list[dict[str, Any]]:
    query_vector = normalized_dense_vectors(embedding_model, [QUERY_INSTRUCTION + query])[0]
    response = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "text": document,
            "metadata": metadata,
            # This is raw cosine distance from Chroma, not a calibrated UQ score.
            "chroma_distance": float(distance),
            "derived_cosine_similarity": float(1.0 - distance),
        }
        for document, metadata, distance in zip(
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
            strict=True,
        )
    ]


def rerank(query: str, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    pairs = [(query, candidate["text"]) for candidate in candidates]
    scores = reranker.score_pairs(
        pairs,
        batch_size=min(16, len(pairs)),
        max_length=RERANK_MAX_LENGTH,
    )
    if len(scores) != len(candidates):
        raise RuntimeError("Reranker returned a score count different from the candidate count.")
    scored = [
        {**candidate, "reranker_score": float(score)}
        for candidate, score in zip(candidates, scores, strict=True)
    ]
    scored.sort(key=lambda item: item["reranker_score"], reverse=True)
    top_1 = scored[0]["reranker_score"] if scored else None
    top_2 = scored[1]["reranker_score"] if len(scored) > 1 else None
    return scored, {
        "top_1_score": top_1,
        "top_2_score": top_2,
        "top_1_top_2_margin": (top_1 - top_2) if top_1 is not None and top_2 is not None else None,
    }


_sanity_started = time.perf_counter()
sanity_results: list[dict[str, Any]] = []
print("\nTHROWAWAY SANITY CHECKS ONLY — not pilot evaluation data or performance claims.")
for query_spec in SANITY_QUERIES:
    dense_candidates = dense_retrieve(query_spec["query"])
    reranked_candidates, signal = rerank(query_spec["query"], dense_candidates)
    dense_top_3 = dense_candidates[:3]
    reranked_top_3 = reranked_candidates[:3]
    print(f"\n=== {query_spec['id']} ===\nQuery: {query_spec['query']}")
    print_ranked_results("Dense-only top 3", dense_top_3)
    print_ranked_results("Dense + BGE-reranker top 3", reranked_top_3)
    print(
        "Reranker reliability signal: "
        f"top_1_score={signal['top_1_score']:.6f}; "
        f"top_1_top_2_margin={signal['top_1_top_2_margin']:.6f}"
    )
    sanity_results.append(
        {
            **query_spec,
            "dense_only_top_3": dense_top_3,
            "dense_candidate_count_before_rerank": len(dense_candidates),
            "dense_plus_reranker_top_3": reranked_top_3,
            "reranker_reliability_signal": signal,
        }
    )
SANITY_SECONDS = time.perf_counter() - _sanity_started

# Make the SS17 case explicit: earlier clause-level BM25 placed the exact
# target at rank 5 in this sanity check. This script does not infer success.
ss17_result = next(item for item in sanity_results if item["id"] == "ss17_certificates_of_leased_assets")
print("\n=== SS17 certificates comparison target ===")
print("Compare manually with prior clause-level BM25 observation: exact clause at rank 5.")
print_ranked_results("Dense-only SS17 top 3", ss17_result["dense_only_top_3"])
print_ranked_results("Dense + reranker SS17 top 3", ss17_result["dense_plus_reranker_top_3"])


# %% [Cell 8] Save private artifacts, measure actual costs, and download them
chroma_zip_path = Path(
    shutil.make_archive(
        base_name=str(OUTPUT_DIR / "chroma_bge_m3"),
        format="zip",
        root_dir=str(CHROMA_DIR),
    )
)

run_metadata = {
    "input_chunks_path": str(INPUT_CHUNKS_PATH),
    "input_chunk_count": len(chunks),
    "collection_name": COLLECTION_NAME,
    "embedding_model_id": EMBEDDING_MODEL_ID,
    "reranker_model_id": RERANKER_MODEL_ID,
    "query_instruction": QUERY_INSTRUCTION,
    "device": device_description,
    "use_fp16": use_fp16,
    "package_versions": {
        "FlagEmbedding": importlib.metadata.version("FlagEmbedding"),
        "chromadb": importlib.metadata.version("chromadb"),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
    },
    "timing_seconds": {
        "dependency_install": INSTALL_SECONDS,
        "embedding_model_load": EMBEDDING_MODEL_LOAD_SECONDS,
        "embedding_generation": EMBEDDING_SECONDS,
        "reranker_model_load": RERANKER_MODEL_LOAD_SECONDS,
        "sanity_queries_and_reranking": SANITY_SECONDS,
    },
    "disk_bytes": {
        "huggingface_cache": directory_size_bytes(HF_HOME),
        "chroma_persist_directory": directory_size_bytes(CHROMA_DIR),
        "chroma_zip": chroma_zip_path.stat().st_size,
    },
    "embedding_shape": list(document_vectors.shape),
    "chroma_record_count": collection.count(),
    "notes": [
        "Sanity queries are throwaway inspection checks, not pilot evaluation data.",
        "Chroma distances and reranker scores are raw signals, not calibrated confidence or UQ values.",
        "Artifacts may contain licensed AAOIFI text and must remain private.",
    ],
}

with RESULTS_PATH.open("w", encoding="utf-8") as stream:
    json.dump(json_safe({"run_metadata": run_metadata, "sanity_results": sanity_results}), stream, ensure_ascii=False, indent=2)
    stream.write("\n")

print("\nSaved private artifacts:")
print(f"- Results: {RESULTS_PATH} ({RESULTS_PATH.stat().st_size} bytes)")
print(f"- Chroma ZIP: {chroma_zip_path} ({chroma_zip_path.stat().st_size} bytes)")
print(json.dumps(run_metadata, indent=2))

try:
    from google.colab import files

    files.download(str(RESULTS_PATH))
    files.download(str(chroma_zip_path))
except ModuleNotFoundError:
    print("Not running in Colab; download the two saved artifacts manually.")
