"""Colab batch smoke test: retrieve+generate for ALL hard-set items in one run.

THIS IS A 7-ITEM SMOKE TEST, NOT THE n=25-30 PILOT EVALUATION.
It does not score correctness, entailment, or statistical significance.
"abstained" vs "answered" is exact string match to the fixed abstention
line only — not a quality or Shari'ah judgment.

Run this .py file cell-by-cell in Google Colab (GPU runtime). Do not run it
in the local sandbox.

VRAM plan (T4): retrieve for every item while BGE-M3 + reranker are loaded,
store top-5 context, then unload retrieval models, load Jais-2 once, and
generate for every item without reloading the generator.

BEFORE RUNNING:
  - Runtime > Change runtime type > GPU (T4).
  - Colab Secrets: HUGGINGFACE_HUB_TOKEN, notebook access ON.
  - Hugging Face: accept terms on https://huggingface.co/inception42/Jais-2-8B-Chat
  - Locally: python scripts/pack_colab_e2e_handoff.py
    (re-pack so the zip contains the current 7-item hard_set.jsonl)
  - Upload colab_e2e_handoff.zip to /content. Licensed AAOIFI text; Colab
    session only; do not commit or share.

Jais-2 load constraints (same as colab_jais2_load_test.py):
  1. BitsAndBytes NF4 4-bit, compute dtype bfloat16 (not float16)
  2. device_map="cuda:0" (not "auto")
  3. apply_chat_template(tokenize=True, return_dict=True)
  4. Primary model ID inception42/Jais-2-8B-Chat (inceptionai only on 404)
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
        "accelerate>=0.30.0",
        "bitsandbytes>=0.43.0",
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
    raise RuntimeError("This smoke test requires a Colab GPU runtime.")

print("Package versions (installed):")
for pkg in (
    "rank-bm25",
    "tiktoken",
    "FlagEmbedding",
    "chromadb",
    "accelerate",
    "bitsandbytes",
    "huggingface-hub",
    "transformers",
):
    print(f"  {pkg}: {importlib.metadata.version(pkg)}")

print(
    "\nNOTE: FlagEmbedding==1.4.0 plus transformers-from-git plus Jais-2 4-bit "
    "in one Colab process was jointly load-tested on 2026-08-29 (H07 then H02 "
    "on a T4). If a later dependency bump breaks the import, stop and report "
    "the traceback rather than patching blindly."
)


# %% [Cell 2] Private artifacts — upload colab_e2e_handoff.zip
import zipfile

CONTENT = Path("/content")
HANDOFF_ZIP = CONTENT / "colab_e2e_handoff.zip"
CLAUSE_CHUNKS_PATH = CONTENT / "clause_chunks.jsonl"
HARD_SET_PATH = CONTENT / "hard_set.jsonl"
CHROMA_ZIP_PATH = CONTENT / "chroma_bge_m3.zip"
CHROMA_DIR = CONTENT / "chroma_bge_m3"
SRC_ROOT = CONTENT / "src"
RESULTS_JSON_PATH = CONTENT / "e2e_batch_smoke_results.json"


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


# %% [Cell 4] Batch configuration (provisional k values — not evaluated)
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

# OPEN DESIGN DECISION (same as colab_e2e_single_test.py): union of BM25
# clause-level k=10 and dense k=10, then rerank, then top 5. Not the pilot policy.
BM25_CANDIDATE_K = 10
DENSE_CANDIDATE_K = 10
CONTEXT_AFTER_RERANK_K = 5

MODEL_ID = "inception42/Jais-2-8B-Chat"
_MODEL_ID_FALLBACK = "inceptionai/Jais-2-8B-Chat"
MAX_NEW_TOKENS = 512

ABSTENTION_LINE = "I cannot answer from the given context"

SYSTEM_PROMPT = (
    "You are answering a question using only the retrieved AAOIFI clause "
    "excerpts provided in the user message. Do not use any other knowledge. "
    "If the provided excerpts are insufficient to answer, reply exactly: "
    f"{ABSTENTION_LINE}"
)


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


def classify_response(text: str) -> str:
    """Exact match after strip. Not a quality, entailment, or Shari'ah label."""
    if text.strip() == ABSTENTION_LINE:
        return "abstained"
    return "answered"


hard_set_items = read_jsonl(HARD_SET_PATH)
hard_set_items.sort(key=lambda row: row["item_id"])
if len(hard_set_items) != EXPECTED_HARD_SET_SIZE:
    print(
        f"WARNING: hard_set.jsonl has {len(hard_set_items)} items; "
        f"this batch script expected {EXPECTED_HARD_SET_SIZE}."
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
print("gold_answer is held for the leak-guard and optional human print only.")
print("It must not enter any model prompt.")


# %% [Cell 5] Load retrieval stack once; retrieve+rerank every item
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

BM25_INDEX_PATH = CONTENT / "bm25_clause_level_colab.pkl"
persist_bm25_index(clause_chunks, BM25_INDEX_PATH, "bm25_clause_level")
print(f"Rebuilt clause-level BM25 index at {BM25_INDEX_PATH}")

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_collection(name=COLLECTION_NAME)
if collection.count() != len(clause_chunks):
    raise RuntimeError(
        f"Chroma count {collection.count()} != clause chunk count {len(clause_chunks)}."
    )
print(f"Opened Chroma collection {COLLECTION_NAME!r} with {collection.count()} vectors.")

print("Loading BGE-M3 for query encoding only (documents are already in Chroma)...")
_embed_start = time.perf_counter()
embedding_model = BGEM3FlagModel(EMBEDDING_MODEL_ID, use_fp16=True)
EMBED_LOAD_SECONDS = time.perf_counter() - _embed_start
print(f"BGE-M3 loaded in {EMBED_LOAD_SECONDS:.1f}s")

print(f"Loading reranker {RERANKER_MODEL_ID} via DirectBGEReranker (not FlagReranker)...")
_rerank_start = time.perf_counter()
reranker = DirectBGEReranker(RERANKER_MODEL_ID, device="cuda:0")
RERANK_LOAD_SECONDS = time.perf_counter() - _rerank_start
print(f"Reranker loaded in {RERANK_LOAD_SECONDS:.1f}s")


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


def gold_hits_in_top5(
    gold_clause_ids: list[dict[str, Any]],
    context_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rank_by_key: dict[tuple[Any, ...], int] = {}
    for rank, record in enumerate(context_records, start=1):
        key = retrieved_clause_key(record)
        if key not in rank_by_key:
            rank_by_key[key] = rank
    hits: list[dict[str, Any]] = []
    for gold in gold_clause_ids:
        key = gold_clause_key(gold)
        rank = rank_by_key.get(key)
        hits.append(
            {
                "standard_id": gold["standard_id"],
                "clause_id": gold["clause_id"],
                "occurrence_index": int(gold.get("occurrence_index", 0)),
                "sub_clause_id": gold.get("sub_clause_id"),
                "in_top5": rank is not None,
                "rank": rank,
            }
        )
    return hits


prepared_items: list[dict[str, Any]] = []
for item in hard_set_items:
    item_id = item["item_id"]
    query_text = item["question_text"]
    gold_answer = item["gold_answer"]
    gold_clause_ids = item["gold_clause_ids"]

    bm25_hits = search_bm25(
        clause_chunks, BM25_INDEX_PATH, query_text, top_k=BM25_CANDIDATE_K
    )
    dense_hits = dense_retrieve(query_text, DENSE_CANDIDATE_K)
    candidates = merge_by_chunk_id(bm25_hits, dense_hits)
    reranked, rerank_signal = rerank_records(
        reranker,
        query_text,
        candidates,
        text_key="text",
        batch_size=16,
        max_length=RERANK_MAX_LENGTH,
    )
    context_records = reranked[:CONTEXT_AFTER_RERANK_K]
    gold_hits = gold_hits_in_top5(gold_clause_ids, context_records)
    n_gold_in_top5 = sum(1 for hit in gold_hits if hit["in_top5"])
    print(
        f"{item_id}: candidates={len(candidates)} "
        f"gold_in_top5={n_gold_in_top5}/{len(gold_hits)} "
        f"rerank_top1={rerank_signal.top_1_score}"
    )
    for rank, record in enumerate(context_records, start=1):
        print(f"  {rank}. {clause_label(record)}")

    prepared_items.append(
        {
            "item": item,
            "query_text": query_text,
            "gold_answer": gold_answer,
            "context_records": context_records,
            "gold_hits": gold_hits,
            "n_candidates": len(candidates),
            "rerank_top_1_score": rerank_signal.top_1_score,
            "rerank_top_2_score": rerank_signal.top_2_score,
            "rerank_margin": rerank_signal.top_1_top_2_margin,
        }
    )

print(f"Retrieval complete for {len(prepared_items)} items. Unload BGE next, then Jais-2.")


# %% [Cell 6] Unload retrieval models before Jais-2 (T4 VRAM budget)
print("Releasing BGE-M3 and reranker from GPU before loading Jais-2...")
del embedding_model
del reranker
gc.collect()
torch.cuda.empty_cache()
print(f"VRAM allocated after retrieval teardown: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"VRAM reserved  after retrieval teardown: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


# %% [Cell 7] Load Jais-2 once
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def _load_jais2(model_id: str):
    print(f"Loading tokenizer from {model_id!r}...")
    tok = AutoTokenizer.from_pretrained(model_id)
    print(f"Loading 4-bit model from {model_id!r} with device_map='cuda:0'...")
    torch.cuda.reset_peak_memory_stats()
    vram_before = torch.cuda.memory_allocated()
    load_started = time.perf_counter()
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=_bnb_config,
        device_map="cuda:0",
    )
    mdl.eval()
    load_seconds = time.perf_counter() - load_started
    vram_after = torch.cuda.memory_allocated()
    return tok, mdl, load_seconds, vram_after - vram_before, torch.cuda.max_memory_allocated()


try:
    tokenizer, model, JAIS_LOAD_SECONDS, vram_model_bytes, vram_peak_bytes = _load_jais2(
        MODEL_ID
    )
    _active_model_id = MODEL_ID
except RepositoryNotFoundError:
    print(f"WARNING: {MODEL_ID!r} returned 404. Retrying {_MODEL_ID_FALLBACK!r}.")
    tokenizer, model, JAIS_LOAD_SECONDS, vram_model_bytes, vram_peak_bytes = _load_jais2(
        _MODEL_ID_FALLBACK
    )
    _active_model_id = _MODEL_ID_FALLBACK

device_map = getattr(model, "hf_device_map", {}) or {}
cpu_layers = [name for name, device in device_map.items() if str(device) == "cpu"]
disk_layers = [name for name, device in device_map.items() if str(device) == "disk"]
if cpu_layers or disk_layers:
    raise RuntimeError(
        f"Jais-2 has offloaded layers (cpu={len(cpu_layers)}, disk={len(disk_layers)}). "
        "Do not generate. Restart the runtime and confirm 4-bit + device_map='cuda:0'."
    )

_hub_sha = huggingface_hub.HfApi().model_info(_active_model_id).sha
print("JAIS-2 LOAD COMPLETE (once for the whole batch)")
print(f"  model_id     : {_active_model_id}")
print(f"  hub_revision : {_hub_sha}")
print(f"  load time    : {JAIS_LOAD_SECONDS:.1f}s")
print(f"  VRAM used    : {vram_model_bytes / 1e9:.2f} GB")
print(f"  VRAM peak    : {vram_peak_bytes / 1e9:.2f} GB")
print(f"  device map   : {device_map if device_map else 'N/A'}")


# %% [Cell 8] Generate for every item (Jais-2 already loaded; never pass gold_answer)
_input_device = torch.device("cuda:0")
item_results: list[dict[str, Any]] = []

for prepared in prepared_items:
    item = prepared["item"]
    item_id = item["item_id"]
    query_text = prepared["query_text"]
    gold_answer = prepared["gold_answer"]
    context_records = prepared["context_records"]
    gold_hits = prepared["gold_hits"]

    context_blocks = []
    top5_meta = []
    for rank, record in enumerate(context_records, start=1):
        context_blocks.append(f"[{rank}] {clause_label(record)}\n{record['text']}")
        top5_meta.append(
            {
                "rank": rank,
                "chunk_id": record.get("chunk_id"),
                "standard_id": record["standard_id"],
                "clause_id": record["clause_id"],
                "occurrence_index": int(record.get("occurrence_index", 0)),
                "sub_clause_id": record.get("sub_clause_id"),
                "retrieval_sources": record.get("retrieval_sources", []),
                "reranker_score": record.get("reranker_score"),
            }
        )
    retrieved_context_text = "\n\n".join(context_blocks)
    user_prompt = (
        "Retrieved clause excerpts (this is the only source you may use):\n\n"
        f"{retrieved_context_text}\n\n"
        f"Question:\n{query_text}\n\n"
        "Answer only from the excerpts above. If they are insufficient, reply exactly: "
        f"{ABSTENTION_LINE}"
    )
    if gold_answer in user_prompt or gold_answer in SYSTEM_PROMPT:
        raise RuntimeError(
            f"Refusing to generate for {item_id}: gold_answer leaked into the prompt."
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(_input_device) for key, value in inputs.items()}
    inputs.pop("token_type_ids", None)
    first_tokens = inputs["input_ids"][0][:4].tolist()
    if first_tokens[0] == 0 and first_tokens[1] == 0:
        raise RuntimeError(f"{item_id}: Double-BOS detected.")

    torch.cuda.reset_peak_memory_stats()
    gen_start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generation_seconds = time.perf_counter() - gen_start
    vram_gen_peak_bytes = torch.cuda.max_memory_allocated()
    prompt_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][prompt_len:]
    new_tokens = len(generated_ids)
    model_answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    response_class = classify_response(model_answer)
    n_gold = len(gold_hits)
    n_gold_in_top5 = sum(1 for hit in gold_hits if hit["in_top5"])
    any_gold_in_top5 = n_gold_in_top5 > 0
    all_gold_in_top5 = n_gold > 0 and n_gold_in_top5 == n_gold

    result = {
        "item_id": item_id,
        "status": item.get("status"),
        "verification_basis": item.get("verification_basis"),
        "gold_answer_leak_guard_passed": True,
        "any_gold_clause_in_top5": any_gold_in_top5,
        "all_gold_clauses_in_top5": all_gold_in_top5,
        "n_gold_clauses": n_gold,
        "n_gold_clauses_in_top5": n_gold_in_top5,
        "gold_clause_hits": gold_hits,
        "top5": top5_meta,
        "model_response": model_answer,
        "response_class": response_class,
        "response_class_rule": (
            "abstained iff model_response.strip() == "
            "the fixed line 'I cannot answer from the given context'; "
            "else answered. Not a correctness label."
        ),
        "generation_seconds": round(generation_seconds, 2),
        "new_tokens": new_tokens,
        "input_token_count": int(prompt_len),
        "vram_peak_generation_gb": round(vram_gen_peak_bytes / 1e9, 2),
    }
    item_results.append(result)
    print(
        f"{item_id}: {response_class}  gold_in_top5={n_gold_in_top5}/{n_gold}  "
        f"new_tokens={new_tokens}  {generation_seconds:.2f}s"
    )
    print(f"  model_response: {model_answer!r}")


# %% [Cell 9] Consolidated JSON — n=7 smoke test, not the pilot evaluation
n_items = len(item_results)
n_abstained = sum(1 for row in item_results if row["response_class"] == "abstained")
n_answered = sum(1 for row in item_results if row["response_class"] == "answered")
n_any_gold = sum(1 for row in item_results if row["any_gold_clause_in_top5"])
n_all_gold = sum(1 for row in item_results if row["all_gold_clauses_in_top5"])

batch_report = {
    "label": (
        "7-item Colab smoke test, not the n=25-30+ pilot evaluation. "
        "Counts are descriptive only. No statistical significance is claimed."
    ),
    "verification_note": (
        "Hard-set items are corpus_cross_reference only, not qualified "
        "Shari'ah scholar review."
    ),
    "n_items": n_items,
    "aggregates": {
        "n_abstained": n_abstained,
        "n_answered": n_answered,
        "n_any_gold_clause_in_top5": n_any_gold,
        "n_all_gold_clauses_in_top5": n_all_gold,
        "abstained_over_n": f"{n_abstained}/{n_items}",
        "answered_over_n": f"{n_answered}/{n_items}",
        "any_gold_in_top5_over_n": f"{n_any_gold}/{n_items}",
        "all_gold_in_top5_over_n": f"{n_all_gold}/{n_items}",
    },
    "retrieval_policy": (
        f"union(BM25 clause-level k={BM25_CANDIDATE_K}, "
        f"dense k={DENSE_CANDIDATE_K}) -> rerank -> top {CONTEXT_AFTER_RERANK_K}"
    ),
    "generation": {
        "model_id": _active_model_id,
        "hub_revision": _hub_sha,
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "abstention_line": ABSTENTION_LINE,
        "jais2_load_seconds": round(JAIS_LOAD_SECONDS, 1),
        "vram_model_gb": round(vram_model_bytes / 1e9, 2),
    },
    "package_versions": {
        "transformers": importlib.metadata.version("transformers"),
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "torch": torch.__version__,
        "FlagEmbedding": importlib.metadata.version("FlagEmbedding"),
    },
    "items": item_results,
}

RESULTS_JSON_PATH.write_text(
    json.dumps(batch_report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("=" * 72)
print("BATCH SMOKE TEST JSON — NOT PILOT EVALUATION — n=7 DESCRIPTIVE COUNTS ONLY")
print("=" * 72)
print(json.dumps(batch_report, ensure_ascii=False, indent=2))
print()
print(f"Wrote {RESULTS_JSON_PATH}")
print("Keep this file in the Colab session. It is not a public artifact.")
print(
    f"aggregates: abstained {n_abstained}/{n_items}; "
    f"any gold in top-5 {n_any_gold}/{n_items}; "
    f"all golds in top-5 {n_all_gold}/{n_items}"
)
