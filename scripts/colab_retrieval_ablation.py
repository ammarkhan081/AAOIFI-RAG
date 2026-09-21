"""Colab retrieval ablation: isolated retrieval component testing.

THIS IS A 7-ITEM RETRIEVAL-ONLY ABULATION, NOT A PILOT EVALUATION.
It does not score generation correctness or statistical significance.
Clause Recall@5 is a retrieval metric only — not a Shari'ah judgment.

Run this .py file cell-by-cell in Google Colab (GPU runtime). Do not run it
in the local sandbox.

BEFORE RUNNING:
  - Runtime > Change runtime type > GPU (T4).
  - Colab Secrets: HUGGINGFACE_HUB_TOKEN, notebook access ON.
  - Hugging Face: accept terms on https://huggingface.co/BAAI/bge-m3
  - Hugging Face: accept terms on https://huggingface.co/BAAI/bge-reranker-v2-m3
  - Locally: python scripts/pack_colab_e2e_handoff.py
    (re-pack so the zip contains the current 7-item hard_set.jsonl)
  - Upload colab_e2e_handoff.zip to /content. Licensed AAOIFI text; Colab
    session only; do not commit or share.

RETRIEVAL CONFIGS TESTED:
  - Config A: fixed-token BM25 only, no reranker, top-5 by BM25 score.
  - Config B: Config A's candidates, reranked by BGE-reranker-v2-m3, top-5 by rerank score.
  - Config B' (PARTIAL): clause-level BM25 only, no reranker, top-5 by BM25 score.
  - Config B'+rerank: clause-level BM25, reranked, top-5 by rerank score.
  - Config Hybrid: BM25(clause-level) ∪ dense (BGE-M3+Chroma), reranked, top-5.

Note: Config B' is labeled as "clause-level chunking only, SAC not yet implemented"
because the Sentence-Aware Chunking (SAC) component is not built yet.
"""

# %% [Cell 1] Environment setup
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/content/huggingface_cache")

_install_start = time.perf_counter()
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "rank-bm25==0.2.2",
        "tiktoken==0.14.0",
        "FlagEmbedding==1.4.0",
        "chromadb==1.5.9",
        "huggingface_hub>=0.24.0",
        "git+https://github.com/huggingface/transformers.git",
    ]
)
INSTALL_SECONDS = time.perf_counter() - _install_start
print(f"Dependencies installed in {INSTALL_SECONDS:.1f}s")

import importlib.metadata
import torch

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"VRAM total (GB) : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}")
else:
    raise RuntimeError("This retrieval ablation requires a Colab GPU runtime.")

print("Package versions (installed):")
for pkg in (
    "rank-bm25",
    "tiktoken",
    "FlagEmbedding",
    "chromadb",
    "huggingface-hub",
    "transformers",
):
    print(f"  {pkg}: {importlib.metadata.version(pkg)}")


# %% [Cell 2] Private artifacts — upload colab_e2e_handoff.zip
import zipfile

CONTENT = Path("/content")
HANDOFF_ZIP = CONTENT / "colab_e2e_handoff.zip"
CLAUSE_CHUNKS_PATH = CONTENT / "clause_chunks.jsonl"
HARD_SET_PATH = CONTENT / "hard_set.jsonl"
CHROMA_ZIP_PATH = CONTENT / "chroma_bge_m3.zip"
CHROMA_DIR = CONTENT / "chroma_bge_m3"
SRC_ROOT = CONTENT / "src"
RESULTS_JSON_PATH = CONTENT / "retrieval_ablation_results.json"


def _handoff_ready() -> list[str]:
    missing: list[str] = []
    if not CLAUSE_CHUNKS_PATH.is_file():
        missing.append(str(CLAUSE_CHUNKS_PATH))
    if not HARD_SET_PATH.is_file():
        missing.append(str(HARD_SET_PATH))
    if not (CHROMA_DIR / "chroma.sqlite3").is_file():
        missing.append(str(CHROMA_DIR / "chroma.sqlite3"))
    if not (SRC_ROOT / "aaoifi_rag" / "retrieval" / "bm25_baseline.py").is_file():
        missing.append(str(SRC_ROOT / "aaoifi_rag" / "retrieval" / "bm25_baseline.py"))
    if not (SRC_ROOT / "aaoifi_rag" / "retrieval" / "bge_reranker.py").is_file():
        missing.append(str(SRC_ROOT / "aaoifi_rag" / "retrieval" / "bge_reranker.py"))
    return missing


def _zip_prefix(names: list[str]) -> str:
    tops = {name.split("/", 1)[0] for name in names if name and not name.startswith("__")}
    tops = {item for item in tops if item}
    if len(tops) == 1:
        top = next(iter(tops))
        if all(name == top or name.startswith(top + "/") for name in names if name):
            return top + "/"
    return ""


def _extract_zip(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if name and not name.endswith("/")]
        prefix = _zip_prefix(names)
        for name in names:
            rel = name[len(prefix) :] if prefix and name.startswith(prefix) else name
            if not rel or rel.endswith(".pdf"):
                if rel.endswith(".pdf"):
                    print(f"Skipping PDF inside zip (must not be used): {name}")
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as src, target.open("wb") as out:
                out.write(src.read())
    print(f"Extracted {zip_path} -> {dest}")


def _unpack_chroma_zip_if_needed() -> None:
    global CHROMA_DIR
    if (CHROMA_DIR / "chroma.sqlite3").is_file():
        return
    if not CHROMA_ZIP_PATH.is_file():
        return
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(CHROMA_ZIP_PATH) as archive:
        archive.extractall(CHROMA_DIR)
    nested = CHROMA_DIR / "chroma_bge_m3" / "chroma.sqlite3"
    if not (CHROMA_DIR / "chroma.sqlite3").is_file() and nested.is_file():
        CHROMA_DIR = nested.parent


def _try_extract_handoff() -> None:
    if HANDOFF_ZIP.is_file():
        _extract_zip(HANDOFF_ZIP, CONTENT)
    src_zip = CONTENT / "src.zip"
    if src_zip.is_file() and not (SRC_ROOT / "aaoifi_rag").is_dir():
        _extract_zip(src_zip, CONTENT)
        if (CONTENT / "aaoifi_rag" / "retrieval").is_dir() and not (
            SRC_ROOT / "aaoifi_rag"
        ).is_dir():
            SRC_ROOT.mkdir(parents=True, exist_ok=True)
            (CONTENT / "aaoifi_rag").replace(SRC_ROOT / "aaoifi_rag")
    _unpack_chroma_zip_if_needed()


_try_extract_handoff()
_missing = _handoff_ready()
if _missing:
    print("Handoff files are not in /content yet.")
    print("Upload colab_e2e_handoff.zip (built locally by pack_colab_e2e_handoff.py).")
    print("Still missing:\n  " + "\n  ".join(_missing))
    try:
        from google.colab import files as colab_files
    except ModuleNotFoundError as error:
        raise FileNotFoundError(
            "Not in Colab and handoff files are missing. "
            "Place colab_e2e_handoff.zip under /content and re-run Cell 2."
        ) from error
    uploaded = colab_files.upload()
    if not uploaded:
        raise FileNotFoundError(
            "No file was uploaded. Build colab_e2e_handoff.zip locally "
            "(python scripts/pack_colab_e2e_handoff.py) and upload that one zip."
        )
    _try_extract_handoff()
    for uploaded_name in uploaded:
        uploaded_path = CONTENT / uploaded_name
        if uploaded_path.suffix.lower() == ".zip" and uploaded_path.is_file():
            if uploaded_path.name != HANDOFF_ZIP.name:
                _extract_zip(uploaded_path, CONTENT)
            _unpack_chroma_zip_if_needed()
    _missing = _handoff_ready()

if _missing:
    raise FileNotFoundError(
        "Still missing after upload. Expected a zip with clause_chunks.jsonl, "
        "hard_set.jsonl, chroma_bge_m3/chroma.sqlite3, and "
        "src/aaoifi_rag/retrieval/*.py at the zip root (or one wrapper folder).\n  "
        + "\n  ".join(_missing)
    )

sys.path.insert(0, str(SRC_ROOT))
print(f"clause_chunks : {CLAUSE_CHUNKS_PATH}")
print(f"hard_set      : {HARD_SET_PATH}")
print(f"chroma dir    : {CHROMA_DIR}")
print(f"src on path   : {SRC_ROOT}")
print(f"results json  : {RESULTS_JSON_PATH} (Colab-local; do not commit)")


# %% [Cell 3] Hugging Face auth (Colab Secret — token is never printed)
try:
    from google.colab import userdata

    _token = userdata.get("HUGGINGFACE_HUB_TOKEN")
    if not _token:
        raise ValueError(
            "HUGGINGFACE_HUB_TOKEN secret is empty. "
            "Add it in Colab Secrets and re-run."
        )
except ModuleNotFoundError as error:
    raise RuntimeError(
        "google.colab not available — this script must run inside Google Colab."
    ) from error

import huggingface_hub

huggingface_hub.login(token=_token, add_to_git_credential=False)
whoami = huggingface_hub.whoami()
print(f"AUTH SUCCESS: logged in as '{whoami['name']}'")
del _token


# %% [Cell 4] Batch configuration
import json
from typing import Any

from aaoifi_rag.retrieval.bge_reranker import DirectBGEReranker, rerank_records
from aaoifi_rag.retrieval.bm25_baseline import persist_bm25_index, search_bm25

EXPECTED_HARD_SET_SIZE = 7
EXPECTED_CORPUS_SIZE = 362
COLLECTION_NAME = "aaoifi_clause_chunks_bge_m3"
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
EMBEDDING_BATCH_SIZE = 8
EMBEDDING_MAX_LENGTH = 8192
RERANK_MAX_LENGTH = 1024

# Ablation-specific k values
BM25_FIXED_TOKEN_K = 10  # Config A: fixed-token BM25
BM25_CLAUSE_LEVEL_K = 10  # Config B': clause-level BM25
DENSE_CANDIDATE_K = 10  # Config Hybrid: dense candidates
TOP_K_AFTER_RERANK = 5  # All configs: final top-5


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"{path} contains no records.")
    return records


def _norm_sub_clause_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    return value


def gold_clause_key(gold: dict[str, Any]) -> tuple[Any, ...]:
    return (
        gold["standard_id"],
        gold["clause_id"],
        int(gold.get("occurrence_index", 0)),
        _norm_sub_clause_id(gold.get("sub_clause_id")),
    )


def retrieved_clause_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["standard_id"],
        record["clause_id"],
        int(record.get("occurrence_index", 0)),
        _norm_sub_clause_id(record.get("sub_clause_id")),
    )


def clause_label(record: dict[str, Any]) -> str:
    sub = record.get("sub_clause_id")
    sub_bit = f"({sub})" if sub else ""
    return (
        f"{record['standard_id']} §{record['clause_id']}{sub_bit} "
        f"occ={record.get('occurrence_index', 0)} "
        f"page={record.get('source_page', '?')} "
        f"id={record.get('chunk_id', '?')}"
    )


hard_set_items = read_jsonl(HARD_SET_PATH)
hard_set_items.sort(key=lambda row: row["item_id"])
if len(hard_set_items) != EXPECTED_HARD_SET_SIZE:
    print(
        f"WARNING: hard_set.jsonl has {len(hard_set_items)} items; "
        f"this ablation script expected {EXPECTED_HARD_SET_SIZE}."
    )
print(f"HARD-SET ITEMS ({len(hard_set_items)}): {[row['item_id'] for row in hard_set_items]}")
print("verification note: corpus_cross_reference is NOT qualified scholar review.")
for row in hard_set_items:
    if row.get("verification_basis") != "corpus_cross_reference":
        print(
            f"WARNING: {row['item_id']} verification_basis="
            f"{row.get('verification_basis')!r}"
        )
    if row.get("status") == "scholar_validated":
        print(f"WARNING: {row['item_id']} is marked scholar_validated.")


# %% [Cell 5] Load retrieval stack and clause chunks
import gc

import chromadb
import numpy as np
from FlagEmbedding import BGEM3FlagModel

clause_chunks = read_jsonl(CLAUSE_CHUNKS_PATH)
if len(clause_chunks) != EXPECTED_CORPUS_SIZE:
    print(
        f"WARNING: loaded {len(clause_chunks)} clause chunks; "
        f"verified Phase-2 corpus is {EXPECTED_CORPUS_SIZE}."
    )
required_chunk_fields = {
    "chunk_id",
    "standard_id",
    "section_id",
    "clause_id",
    "occurrence_index",
    "sub_clause_id",
    "source_page",
    "text",
    "bm25_text",
}
for line_number, chunk in enumerate(clause_chunks, start=1):
    missing_fields = required_chunk_fields.difference(chunk)
    if missing_fields:
        raise ValueError(
            f"clause_chunks.jsonl line {line_number} missing {sorted(missing_fields)}"
        )

# Build both BM25 indices
BM25_FIXED_TOKEN_PATH = CONTENT / "bm25_fixed_token_colab.pkl"
BM25_CLAUSE_LEVEL_PATH = CONTENT / "bm25_clause_level_colab.pkl"

persist_bm25_index(clause_chunks, BM25_FIXED_TOKEN_PATH, "bm25_fixed_token")
print(f"Rebuilt fixed-token BM25 index at {BM25_FIXED_TOKEN_PATH}")

persist_bm25_index(clause_chunks, BM25_CLAUSE_LEVEL_PATH, "bm25_clause_level")
print(f"Rebuilt clause-level BM25 index at {BM25_CLAUSE_LEVEL_PATH}")

# Load Chroma for dense retrieval
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_collection(name=COLLECTION_NAME)
if collection.count() != len(clause_chunks):
    raise RuntimeError(
        f"Chroma count {collection.count()} != clause chunk count {len(clause_chunks)}."
    )
print(f"Opened Chroma collection {COLLECTION_NAME!r} with {collection.count()} vectors.")

# Load BGE-M3 for query encoding
print("Loading BGE-M3 for query encoding only (documents are already in Chroma)...")
_embed_start = time.perf_counter()
embedding_model = BGEM3FlagModel(EMBEDDING_MODEL_ID, use_fp16=True)
EMBED_LOAD_SECONDS = time.perf_counter() - _embed_start
print(f"BGE-M3 loaded in {EMBED_LOAD_SECONDS:.1f}s")

# Load reranker
print(f"Loading reranker {RERANKER_MODEL_ID} via DirectBGEReranker...")
_rerank_start = time.perf_counter()
reranker = DirectBGEReranker(RERANKER_MODEL_ID, device="cuda:0")
RERANK_LOAD_SECONDS = time.perf_counter() - _rerank_start
print(f"Reranker loaded in {RERANK_LOAD_SECONDS:.1f}s")


# %% [Cell 6] Retrieval functions for each config
def dense_retrieve(query: str, top_k: int) -> list[dict[str, Any]]:
    encoded = embedding_model.encode(
        [QUERY_INSTRUCTION + query],
        batch_size=EMBEDDING_BATCH_SIZE,
        max_length=EMBEDDING_MAX_LENGTH,
    )
    vectors = np.asarray(encoded["dense_vecs"], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("BGE-M3 returned a zero-norm query embedding.")
    query_vector = (vectors / norms)[0]
    response = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    ids = response["ids"][0]
    documents = response["documents"][0]
    metadatas = response["metadatas"][0]
    distances = response["distances"][0]
    hits: list[dict[str, Any]] = []
    for chunk_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances, strict=True
    ):
        hits.append(
            {
                "chunk_id": chunk_id,
                "standard_id": metadata["standard_id"],
                "section_id": metadata.get("section_id"),
                "clause_id": metadata["clause_id"],
                "occurrence_index": int(metadata["occurrence_index"]),
                "sub_clause_id": metadata.get("sub_clause_id"),
                "source_page": int(metadata["source_page"]),
                "text": document,
                "chroma_distance": float(distance),
                "derived_cosine_similarity": float(1.0 - distance),
                "retrieval_sources": ["dense"],
            }
        )
    return hits


def merge_by_chunk_id(
    bm25_hits: list[dict[str, Any]],
    dense_hits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for hit in bm25_hits:
        record = dict(hit)
        record["retrieval_sources"] = ["bm25"]
        merged[record["chunk_id"]] = record
    for hit in dense_hits:
        existing = merged.get(hit["chunk_id"])
        if existing is None:
            merged[hit["chunk_id"]] = dict(hit)
            continue
        existing["retrieval_sources"] = sorted(
            set(existing.get("retrieval_sources", []) + hit.get("retrieval_sources", []))
        )
        existing["chroma_distance"] = hit.get("chroma_distance")
        existing["derived_cosine_similarity"] = hit.get("derived_cosine_similarity")
    return list(merged.values())


def compute_clause_recall(
    gold_clause_ids: list[dict[str, Any]],
    context_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute Clause Recall@5 for a single item."""
    rank_by_key: dict[tuple[Any, ...], int] = {}
    for rank, record in enumerate(context_records, start=1):
        key = retrieved_clause_key(record)
        if key not in rank_by_key:
            rank_by_key[key] = rank
    
    gold_hit_details: list[dict[str, Any]] = []
    for gold in gold_clause_ids:
        key = gold_clause_key(gold)
        rank = rank_by_key.get(key)
        gold_hit_details.append(
            {
                "standard_id": gold["standard_id"],
                "clause_id": gold["clause_id"],
                "occurrence_index": int(gold.get("occurrence_index", 0)),
                "sub_clause_id": gold.get("sub_clause_id"),
                "in_top5": rank is not None,
                "rank": rank,
            }
        )
    
    n_gold = len(gold_clause_ids)
    n_gold_in_top5 = sum(1 for hit in gold_hit_details if hit["in_top5"])
    recall = n_gold_in_top5 / n_gold if n_gold > 0 else 0.0
    
    return {
        "n_gold": n_gold,
        "n_gold_in_top5": n_gold_in_top5,
        "recall": recall,
        "gold_hit_details": gold_hit_details,
    }


# %% [Cell 7] Run retrieval ablation for all configs
ablation_results: list[dict[str, Any]] = []

for item in hard_set_items:
    item_id = item["item_id"]
    query_text = item["question_text"]
    gold_clause_ids = item["gold_clause_ids"]
    
    print(f"\n{'='*72}")
    print(f"Processing {item_id}: {query_text[:60]}...")
    print(f"{'='*72}")
    
    item_result = {
        "item_id": item_id,
        "question_text": query_text,
        "n_gold_clauses": len(gold_clause_ids),
        "configs": {},
    }
    
    # Config A: fixed-token BM25 only, no reranker
    print(f"\nConfig A: fixed-token BM25 only")
    bm25_fixed_hits = search_bm25(
        clause_chunks, BM25_FIXED_TOKEN_PATH, query_text, top_k=TOP_K_AFTER_RERANK
    )
    recall_a = compute_clause_recall(gold_clause_ids, bm25_fixed_hits)
    print(f"  Recall@5: {recall_a['recall']:.2f} ({recall_a['n_gold_in_top5']}/{recall_a['n_gold']})")
    for rank, hit in enumerate(bm25_fixed_hits[:TOP_K_AFTER_RERANK], start=1):
        print(f"    {rank}. {clause_label(hit)}")
    
    item_result["configs"]["A_fixed_token_bm25"] = {
        "description": "fixed-token BM25 only, no reranker, top-5 by BM25 score",
        "recall": recall_a["recall"],
        "n_gold_in_top5": recall_a["n_gold_in_top5"],
        "n_gold": recall_a["n_gold"],
        "gold_hit_details": recall_a["gold_hit_details"],
        "top5": [
            {
                "rank": rank + 1,
                "chunk_id": hit["chunk_id"],
                "standard_id": hit["standard_id"],
                "clause_id": hit["clause_id"],
                "occurrence_index": hit["occurrence_index"],
                "sub_clause_id": hit.get("sub_clause_id"),
                "source_page": hit["source_page"],
            }
            for rank, hit in enumerate(bm25_fixed_hits[:TOP_K_AFTER_RERANK])
        ],
    }
    
    # Config B: fixed-token BM25 + reranker
    print(f"\nConfig B: fixed-token BM25 + reranker")
    bm25_fixed_candidates = search_bm25(
        clause_chunks, BM25_FIXED_TOKEN_PATH, query_text, top_k=BM25_FIXED_TOKEN_K
    )
    reranked_b, signal_b = rerank_records(
        reranker,
        query_text,
        bm25_fixed_candidates,
        text_key="text",
        batch_size=16,
        max_length=RERANK_MAX_LENGTH,
    )
    recall_b = compute_clause_recall(gold_clause_ids, reranked_b[:TOP_K_AFTER_RERANK])
    print(f"  Recall@5: {recall_b['recall']:.2f} ({recall_b['n_gold_in_top5']}/{recall_b['n_gold']})")
    for rank, hit in enumerate(reranked_b[:TOP_K_AFTER_RERANK], start=1):
        print(f"    {rank}. {clause_label(hit)}")
    
    item_result["configs"]["B_fixed_token_bm25_rerank"] = {
        "description": "fixed-token BM25 candidates, reranked, top-5 by rerank score",
        "recall": recall_b["recall"],
        "n_gold_in_top5": recall_b["n_gold_in_top5"],
        "n_gold": recall_b["n_gold"],
        "gold_hit_details": recall_b["gold_hit_details"],
        "top5": [
            {
                "rank": rank + 1,
                "chunk_id": hit["chunk_id"],
                "standard_id": hit["standard_id"],
                "clause_id": hit["clause_id"],
                "occurrence_index": hit["occurrence_index"],
                "sub_clause_id": hit.get("sub_clause_id"),
                "source_page": hit["source_page"],
                "reranker_score": hit.get("reranker_score"),
            }
            for rank, hit in enumerate(reranked_b[:TOP_K_AFTER_RERANK])
        ],
    }
    
    # Config B': clause-level BM25 only (PARTIAL - SAC not yet implemented)
    print(f"\nConfig B': clause-level BM25 only (PARTIAL - SAC not yet implemented)")
    bm25_clause_hits = search_bm25(
        clause_chunks, BM25_CLAUSE_LEVEL_PATH, query_text, top_k=TOP_K_AFTER_RERANK
    )
    recall_b_prime = compute_clause_recall(gold_clause_ids, bm25_clause_hits)
    print(f"  Recall@5: {recall_b_prime['recall']:.2f} ({recall_b_prime['n_gold_in_top5']}/{recall_b_prime['n_gold']})")
    for rank, hit in enumerate(bm25_clause_hits[:TOP_K_AFTER_RERANK], start=1):
        print(f"    {rank}. {clause_label(hit)}")
    
    item_result["configs"]["B_prime_clause_level_bm25"] = {
        "description": "clause-level BM25 only, no reranker, top-5 by BM25 score (PARTIAL - SAC not yet implemented)",
        "recall": recall_b_prime["recall"],
        "n_gold_in_top5": recall_b_prime["n_gold_in_top5"],
        "n_gold": recall_b_prime["n_gold"],
        "gold_hit_details": recall_b_prime["gold_hit_details"],
        "top5": [
            {
                "rank": rank + 1,
                "chunk_id": hit["chunk_id"],
                "standard_id": hit["standard_id"],
                "clause_id": hit["clause_id"],
                "occurrence_index": hit["occurrence_index"],
                "sub_clause_id": hit.get("sub_clause_id"),
                "source_page": hit["source_page"],
            }
            for rank, hit in enumerate(bm25_clause_hits[:TOP_K_AFTER_RERANK])
        ],
    }
    
    # Config B'+rerank: clause-level BM25 + reranker
    print(f"\nConfig B'+rerank: clause-level BM25 + reranker")
    bm25_clause_candidates = search_bm25(
        clause_chunks, BM25_CLAUSE_LEVEL_PATH, query_text, top_k=BM25_CLAUSE_LEVEL_K
    )
    reranked_b_prime, signal_b_prime = rerank_records(
        reranker,
        query_text,
        bm25_clause_candidates,
        text_key="text",
        batch_size=16,
        max_length=RERANK_MAX_LENGTH,
    )
    recall_b_prime_rerank = compute_clause_recall(gold_clause_ids, reranked_b_prime[:TOP_K_AFTER_RERANK])
    print(f"  Recall@5: {recall_b_prime_rerank['recall']:.2f} ({recall_b_prime_rerank['n_gold_in_top5']}/{recall_b_prime_rerank['n_gold']})")
    for rank, hit in enumerate(reranked_b_prime[:TOP_K_AFTER_RERANK], start=1):
        print(f"    {rank}. {clause_label(hit)}")
    
    item_result["configs"]["B_prime_clause_level_bm25_rerank"] = {
        "description": "clause-level BM25 candidates, reranked, top-5 by rerank score",
        "recall": recall_b_prime_rerank["recall"],
        "n_gold_in_top5": recall_b_prime_rerank["n_gold_in_top5"],
        "n_gold": recall_b_prime_rerank["n_gold"],
        "gold_hit_details": recall_b_prime_rerank["gold_hit_details"],
        "top5": [
            {
                "rank": rank + 1,
                "chunk_id": hit["chunk_id"],
                "standard_id": hit["standard_id"],
                "clause_id": hit["clause_id"],
                "occurrence_index": hit["occurrence_index"],
                "sub_clause_id": hit.get("sub_clause_id"),
                "source_page": hit["source_page"],
                "reranker_score": hit.get("reranker_score"),
            }
            for rank, hit in enumerate(reranked_b_prime[:TOP_K_AFTER_RERANK])
        ],
    }
    
    # Config Hybrid: BM25(clause-level) ∪ dense + reranker (existing E2E policy)
    print(f"\nConfig Hybrid: BM25(clause-level) ∪ dense + reranker")
    bm25_clause_hybrid = search_bm25(
        clause_chunks, BM25_CLAUSE_LEVEL_PATH, query_text, top_k=BM25_CLAUSE_LEVEL_K
    )
    dense_hybrid = dense_retrieve(query_text, DENSE_CANDIDATE_K)
    hybrid_candidates = merge_by_chunk_id(bm25_clause_hybrid, dense_hybrid)
    reranked_hybrid, signal_hybrid = rerank_records(
        reranker,
        query_text,
        hybrid_candidates,
        text_key="text",
        batch_size=16,
        max_length=RERANK_MAX_LENGTH,
    )
    recall_hybrid = compute_clause_recall(gold_clause_ids, reranked_hybrid[:TOP_K_AFTER_RERANK])
    print(f"  Recall@5: {recall_hybrid['recall']:.2f} ({recall_hybrid['n_gold_in_top5']}/{recall_hybrid['n_gold']})")
    for rank, hit in enumerate(reranked_hybrid[:TOP_K_AFTER_RERANK], start=1):
        print(f"    {rank}. {clause_label(hit)}")
    
    item_result["configs"]["Hybrid_bm25_dense_rerank"] = {
        "description": "BM25(clause-level) ∪ dense (BGE-M3+Chroma), reranked, top-5",
        "recall": recall_hybrid["recall"],
        "n_gold_in_top5": recall_hybrid["n_gold_in_top5"],
        "n_gold": recall_hybrid["n_gold"],
        "gold_hit_details": recall_hybrid["gold_hit_details"],
        "top5": [
            {
                "rank": rank + 1,
                "chunk_id": hit["chunk_id"],
                "standard_id": hit["standard_id"],
                "clause_id": hit["clause_id"],
                "occurrence_index": hit["occurrence_index"],
                "sub_clause_id": hit.get("sub_clause_id"),
                "source_page": hit["source_page"],
                "reranker_score": hit.get("reranker_score"),
                "retrieval_sources": hit.get("retrieval_sources", []),
            }
            for rank, hit in enumerate(reranked_hybrid[:TOP_K_AFTER_RERANK])
        ],
    }
    
    ablation_results.append(item_result)


# %% [Cell 8] Compute aggregate statistics and build output table
import pandas as pd

# Build per-item table
config_names = [
    "A_fixed_token_bm25",
    "B_fixed_token_bm25_rerank",
    "B_prime_clause_level_bm25",
    "B_prime_clause_level_bm25_rerank",
    "Hybrid_bm25_dense_rerank",
]

table_data = []
for item_result in ablation_results:
    row = {"item_id": item_result["item_id"]}
    for config_name in config_names:
        config = item_result["configs"][config_name]
        row[f"{config_name}_recall"] = config["recall"]
        row[f"{config_name}_n_gold_in_top5"] = config["n_gold_in_top5"]
        row[f"{config_name}_n_gold"] = config["n_gold"]
    table_data.append(row)

df = pd.DataFrame(table_data)

# Compute aggregate statistics (mean recall across items)
aggregate_row = {"item_id": "AGGREGATE (mean recall)"}
for config_name in config_names:
    recall_col = f"{config_name}_recall"
    aggregate_row[recall_col] = df[recall_col].mean()
    aggregate_row[f"{config_name}_n_gold_in_top5"] = df[f"{config_name}_n_gold_in_top5"].sum()
    aggregate_row[f"{config_name}_n_gold"] = df[f"{config_name}_n_gold"].sum()

df_aggregate = pd.concat([df, pd.DataFrame([aggregate_row])], ignore_index=True)

# Rename columns for display
display_columns = ["item_id"]
for config_name in config_names:
    display_columns.append(f"{config_name}_recall")

df_display = df_aggregate[display_columns].copy()
df_display.columns = ["Item"] + [name.replace("_", " ").title() for name in config_names]

print("\n" + "="*72)
print("CLAUSE RECALL@5 BY CONFIGURATION (n=7, descriptive only)")
print("="*72)
print(df_display.to_string(index=False))


# %% [Cell 9] Save results to JSON
ablation_report = {
    "label": (
        "7-item retrieval ablation, not a pilot evaluation. "
        "Clause Recall@5 is a retrieval metric only — not a Shari'ah judgment. "
        "n=7, descriptive comparison only. No statistical significance is claimed."
    ),
    "verification_note": (
        "Hard-set items are corpus_cross_reference only, not qualified "
        "Shari'ah scholar review."
    ),
    "n_items": len(ablation_results),
    "configs": {
        "A_fixed_token_bm25": "fixed-token BM25 only, no reranker, top-5 by BM25 score",
        "B_fixed_token_bm25_rerank": "fixed-token BM25 candidates, reranked, top-5 by rerank score",
        "B_prime_clause_level_bm25": "clause-level BM25 only, no reranker, top-5 by BM25 score (PARTIAL - SAC not yet implemented)",
        "B_prime_clause_level_bm25_rerank": "clause-level BM25 candidates, reranked, top-5 by rerank score",
        "Hybrid_bm25_dense_rerank": "BM25(clause-level) ∪ dense (BGE-M3+Chroma), reranked, top-5",
    },
    "aggregates": {
        config_name: {
            "mean_recall": df[f"{config_name}_recall"].mean(),
            "total_gold_in_top5": int(df[f"{config_name}_n_gold_in_top5"].sum()),
            "total_gold": int(df[f"{config_name}_n_gold"].sum()),
        }
        for config_name in config_names
    },
    "items": ablation_results,
    "package_versions": {
        "transformers": importlib.metadata.version("transformers"),
        "torch": torch.__version__,
        "FlagEmbedding": importlib.metadata.version("FlagEmbedding"),
        "chromadb": importlib.metadata.version("chromadb"),
        "rank-bm25": importlib.metadata.version("rank-bm25"),
    },
    "load_times": {
        "bge_m3_seconds": round(EMBED_LOAD_SECONDS, 1),
        "reranker_seconds": round(RERANK_LOAD_SECONDS, 1),
    },
}

RESULTS_JSON_PATH.write_text(
    json.dumps(ablation_report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n" + "="*72)
print("RETRIEVAL ABLATION JSON — n=7 DESCRIPTIVE COMPARISON ONLY")
print("="*72)
print(json.dumps(ablation_report, ensure_ascii=False, indent=2))
print()
print(f"Wrote {RESULTS_JSON_PATH}")
print("Keep this file in the Colab session. It is not a public artifact.")

print("\nAGGREGATE SUMMARY:")
for config_name in config_names:
    agg = ablation_report["aggregates"][config_name]
    print(
        f"  {config_name}: mean_recall={agg['mean_recall']:.3f}, "
        f"gold_in_top5={agg['total_gold_in_top5']}/{agg['total_gold']}"
    )
