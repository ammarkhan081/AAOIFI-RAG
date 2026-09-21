"""Colab smoke test: retrieve for one hard-set item, then generate with Jais-2.

THIS IS A SINGLE-ITEM SMOKE TEST, NOT THE PILOT EVALUATION.
The script does not score, grade, or claim that the model answer matches gold.
Current default item is H02 (H07 was the first smoke-test item).
For all hard-set items in one Colab session (Jais-2 loaded once), use
scripts/colab_e2e_batch_test.py instead.

Run this .py file cell-by-cell in Google Colab (GPU runtime). Do not run it in
the local sandbox.

BEFORE RUNNING:
  - Runtime > Change runtime type > GPU (T4 is the confirmed Jais-2 4-bit target).
  - Colab Secrets: HUGGINGFACE_HUB_TOKEN, notebook access ON.
  - Hugging Face: accept terms on https://huggingface.co/inception42/Jais-2-8B-Chat
  - On the local machine, run:  python scripts/pack_colab_e2e_handoff.py
    Then in Cell 2, upload the resulting colab_e2e_handoff.zip (or place it
    at /content/colab_e2e_handoff.zip). That zip contains licensed AAOIFI
    clause text. Keep it in this Colab session only; do not commit or share it.

Jais-2 load constraints (required; same as the verified
scripts/colab_jais2_load_test.py path):
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
    "in one Colab process was jointly load-tested on 2026-08-29 (H07 smoke "
    "test on a T4). That run loaded BGE-M3, encoded, reranked via "
    "DirectBGEReranker, then generated with Jais-2. If a later dependency "
    "bump breaks the import, stop and report the traceback rather than "
    "patching blindly."
)


# %% [Cell 2] Private artifacts — upload colab_e2e_handoff.zip
# Cell 1 only installs Python packages. It does not copy your local corpus.
# A new Colab VM is empty, which is why the previous run failed with
# FileNotFoundError for clause_chunks.jsonl, hard_set.jsonl, Chroma, and src.
#
# Locally:  python scripts/pack_colab_e2e_handoff.py
# Then either:
#   - run this cell and use the upload widget for colab_e2e_handoff.zip, or
#   - put that zip at /content/colab_e2e_handoff.zip (Files panel) and re-run.
# Do not upload AAOIFI PDFs.

import zipfile

CONTENT = Path("/content")
HANDOFF_ZIP = CONTENT / "colab_e2e_handoff.zip"
CLAUSE_CHUNKS_PATH = CONTENT / "clause_chunks.jsonl"
HARD_SET_PATH = CONTENT / "hard_set.jsonl"
CHROMA_ZIP_PATH = CONTENT / "chroma_bge_m3.zip"
CHROMA_DIR = CONTENT / "chroma_bge_m3"
SRC_ROOT = CONTENT / "src"


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
    """If every member shares a single top-level folder, return 'folder/'."""
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
    # If the widget uploaded the zip under a different cwd name, extract it.
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


# %% [Cell 4] Smoke-test configuration (provisional k values — not evaluated)
import json
from typing import Any

from aaoifi_rag.retrieval.bge_reranker import DirectBGEReranker, rerank_records
from aaoifi_rag.retrieval.bm25_baseline import persist_bm25_index, search_bm25

SMOKE_ITEM_ID = "H02"
EXPECTED_CORPUS_SIZE = 362
COLLECTION_NAME = "aaoifi_clause_chunks_bge_m3"
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
EMBEDDING_BATCH_SIZE = 8
EMBEDDING_MAX_LENGTH = 8192
RERANK_MAX_LENGTH = 1024

# OPEN DESIGN DECISION (flagged, not evaluated):
# User asked to load BM25 clause-level + BGE-M3 + reranker, then retrieve
# top-k from "the clause-level index" and rerank. This smoke test takes a
# union of BM25 clause-level hits and Chroma dense hits (both are clause-
# level records), then reranks the union. Alternative: BM25-only then
# rerank, or dense-only then rerank. Confirm before treating this as the
# pilot retrieval policy.
BM25_CANDIDATE_K = 10
DENSE_CANDIDATE_K = 10
CONTEXT_AFTER_RERANK_K = 5

MODEL_ID = "inception42/Jais-2-8B-Chat"
_MODEL_ID_FALLBACK = "inceptionai/Jais-2-8B-Chat"
MAX_NEW_TOKENS = 512  # larger than the load-test cap of 60; not previously measured

SYSTEM_PROMPT = (
    "You are answering a question using only the retrieved AAOIFI clause "
    "excerpts provided in the user message. Do not use any other knowledge. "
    "If the provided excerpts are insufficient to answer, reply exactly: "
    'I cannot answer from the given context'
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"{path} contains no records.")
    return records


def load_smoke_item(path: Path) -> dict[str, Any]:
    items = {row["item_id"]: row for row in read_jsonl(path)}
    if SMOKE_ITEM_ID not in items:
        raise KeyError(f"{path} has no item {SMOKE_ITEM_ID!r}. Found: {sorted(items)}")
    item = items[SMOKE_ITEM_ID]
    if item.get("verification_basis") != "corpus_cross_reference":
        print(
            f"WARNING: {SMOKE_ITEM_ID} verification_basis="
            f"{item.get('verification_basis')!r} (expected corpus_cross_reference)."
        )
    if item.get("status") == "scholar_validated":
        print(f"WARNING: {SMOKE_ITEM_ID} is marked scholar_validated in this file.")
    return item


def clause_label(record: dict[str, Any]) -> str:
    sub = record.get("sub_clause_id")
    sub_bit = f"({sub})" if sub else ""
    return (
        f"{record['standard_id']} §{record['clause_id']}{sub_bit} "
        f"occ={record.get('occurrence_index', 0)} "
        f"page={record.get('source_page', '?')} "
        f"id={record.get('chunk_id', '?')}"
    )


smoke_item = load_smoke_item(HARD_SET_PATH)
QUERY_TEXT = smoke_item["question_text"]
GOLD_ANSWER_FOR_HUMAN_ONLY = smoke_item["gold_answer"]
print(f"SMOKE ITEM          : {SMOKE_ITEM_ID}")
print(f"status              : {smoke_item.get('status')}")
print(f"verification_basis  : {smoke_item.get('verification_basis')}")
print("verification note    : corpus_cross_reference is NOT qualified scholar review.")
print(f"answerability       : {smoke_item.get('answerability')}")
print(f"expected_behavior   : {smoke_item.get('expected_behavior')}")
print(f"question_text       : {QUERY_TEXT}")
print("gold_answer is loaded for the final human-comparison print only.")
print("It is not passed to retrieval scoring as a target, and must not enter the model prompt.")


# %% [Cell 5] Load corpus, rebuild clause-level BM25, open Chroma, retrieve, rerank
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
    missing = required_chunk_fields.difference(chunk)
    if missing:
        raise ValueError(f"clause_chunks.jsonl line {line_number} missing {sorted(missing)}")

# Rebuild BM25 in this process from the uploaded JSONL so pickle protocol /
# rank_bm25 version drift cannot silently misalign scores. Uses persist_bm25_index
# from src/aaoifi_rag/retrieval/bm25_baseline.py (same builder as Phase 2).
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


_bm25_start = time.perf_counter()
bm25_hits = search_bm25(clause_chunks, BM25_INDEX_PATH, QUERY_TEXT, top_k=BM25_CANDIDATE_K)
BM25_SECONDS = time.perf_counter() - _bm25_start

_dense_start = time.perf_counter()
dense_hits = dense_retrieve(QUERY_TEXT, DENSE_CANDIDATE_K)
DENSE_SECONDS = time.perf_counter() - _dense_start

candidates = merge_by_chunk_id(bm25_hits, dense_hits)
print(
    f"Candidates before rerank: {len(candidates)} "
    f"(BM25 k={BM25_CANDIDATE_K} in {BM25_SECONDS:.2f}s; "
    f"dense k={DENSE_CANDIDATE_K} in {DENSE_SECONDS:.2f}s)"
)

_rerank_q_start = time.perf_counter()
reranked, rerank_signal = rerank_records(
    reranker,
    QUERY_TEXT,
    candidates,
    text_key="text",
    batch_size=16,
    max_length=RERANK_MAX_LENGTH,
)
RERANK_SECONDS = time.perf_counter() - _rerank_q_start
context_records = reranked[:CONTEXT_AFTER_RERANK_K]
print(f"Reranked {len(reranked)} candidates in {RERANK_SECONDS:.2f}s")
print(
    "Reranker raw signal (not calibrated UQ): "
    f"top_1={rerank_signal.top_1_score}; "
    f"top_2={rerank_signal.top_2_score}; "
    f"margin={rerank_signal.top_1_top_2_margin}"
)
print("\nRetrieved context that will be given to the generator:")
for rank, record in enumerate(context_records, start=1):
    print(f"{rank}. {clause_label(record)}  reranker_score={record['reranker_score']:.6f}")


# %% [Cell 6] Unload retrieval models before Jais-2 (T4 VRAM budget)
# OPEN DESIGN DECISION: unload BGE-M3 and the reranker after retrieval so
# Jais-2 4-bit (~5.7 GB claimed on T4) has headroom. This smoke test does
# not keep a shared multi-model serving setup. I am not certain of the
# exact remaining VRAM after FlagEmbedding teardown; Cell 7 prints it.
print("Releasing BGE-M3 and reranker from GPU before loading Jais-2...")
del embedding_model
del reranker
gc.collect()
torch.cuda.empty_cache()
print(f"VRAM allocated after retrieval teardown: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"VRAM reserved  after retrieval teardown: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


# %% [Cell 7] Load Jais-2 with the four confirmed T4 constraints
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    # bfloat16 is mandatory: Jais-2 Squared-ReLU overflows float16 to NaN.
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
    tokenizer, model, JAIS_LOAD_SECONDS, vram_model_bytes, vram_peak_bytes = _load_jais2(MODEL_ID)
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
print("JAIS-2 LOAD COMPLETE")
print(f"  model_id     : {_active_model_id}")
print(f"  hub_revision : {_hub_sha}")
print(f"  load time    : {JAIS_LOAD_SECONDS:.1f}s")
print(f"  VRAM used    : {vram_model_bytes / 1e9:.2f} GB")
print(f"  VRAM peak    : {vram_peak_bytes / 1e9:.2f} GB")
print(f"  device map   : {device_map if device_map else 'N/A'}")


# %% [Cell 8] Build the generation prompt from retrieved text only (never gold_answer)
context_blocks = []
for rank, record in enumerate(context_records, start=1):
    context_blocks.append(
        f"[{rank}] {clause_label(record)}\n{record['text']}"
    )
retrieved_context_text = "\n\n".join(context_blocks)

user_prompt = (
    "Retrieved clause excerpts (this is the only source you may use):\n\n"
    f"{retrieved_context_text}\n\n"
    f"Question:\n{QUERY_TEXT}\n\n"
    "Answer only from the excerpts above. If they are insufficient, reply exactly: "
    "I cannot answer from the given context"
)

if GOLD_ANSWER_FOR_HUMAN_ONLY in user_prompt or GOLD_ANSWER_FOR_HUMAN_ONLY in SYSTEM_PROMPT:
    raise RuntimeError("Refusing to generate: gold_answer leaked into the prompt.")

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_prompt},
]
print(f"Prompt characters (user): {len(user_prompt)}")
print("Prompt contains gold_answer: False (guard passed)")


# %% [Cell 9] Generate — smoke test only
_input_device = torch.device("cuda:0")
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
print(f"First 4 input token IDs: {first_tokens} (BOS=0 should appear once, not twice)")
if first_tokens[0] == 0 and first_tokens[1] == 0:
    raise RuntimeError("Double-BOS detected. Inspect tokenizer.add_bos_token.")
print(f"Input token count: {inputs['input_ids'].shape[-1]}")

torch.cuda.reset_peak_memory_stats()
_gen_start = time.perf_counter()
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
GENERATION_SECONDS = time.perf_counter() - _gen_start
vram_gen_peak_bytes = torch.cuda.max_memory_allocated()

prompt_len = inputs["input_ids"].shape[-1]
generated_ids = output_ids[0][prompt_len:]
new_tokens = len(generated_ids)
model_answer = tokenizer.decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
print(f"Generated {new_tokens} tokens in {GENERATION_SECONDS:.2f}s")
print(f"VRAM peak during generation: {vram_gen_peak_bytes / 1e9:.2f} GB")


# %% [Cell 10] Human side-by-side print — no automated match judgment
separator = "=" * 72
print(separator)
print(f"SMOKE TEST OUTPUT — {SMOKE_ITEM_ID} ONLY — NOT PILOT EVALUATION")
print("No automated correctness, entailment, or retrieval-hit claim is made.")
print("Compare the three blocks yourself.")
print(separator)
print()
print("--- 1. RETRIEVED CONTEXT (what the model was given) ---")
for rank, record in enumerate(context_records, start=1):
    sources = ",".join(record.get("retrieval_sources", []))
    print(
        f"\n[{rank}] {clause_label(record)}\n"
        f"    sources={sources}  reranker_score={record['reranker_score']:.6f}"
    )
    if "score" in record:
        print(f"    bm25_score={record['score']:.6f}")
    if record.get("chroma_distance") is not None:
        print(f"    chroma_distance={record['chroma_distance']:.6f}")
    print(record["text"])
print()
print("--- 2. MODEL GENERATED ANSWER (Jais-2; gold_answer was not in the prompt) ---")
print(model_answer)
print()
print(
    f"--- 3. {SMOKE_ITEM_ID} gold_answer "
    "(human comparison only; corpus_cross_reference, NOT scholar-validated) ---"
)
print(GOLD_ANSWER_FOR_HUMAN_ONLY)
print()
print(separator)
print("ITEM METADATA (not a grading result)")
print(f"  item_id             : {smoke_item['item_id']}")
print(f"  status              : {smoke_item.get('status')}")
print(f"  verification_basis  : {smoke_item.get('verification_basis')}")
print(f"  gold_clause_ids     : {json.dumps(smoke_item.get('gold_clause_ids'), ensure_ascii=False)}")
print(f"  model_id            : {_active_model_id}")
print(f"  hub_revision        : {_hub_sha}")
print(f"  retrieval_policy    : union(BM25 clause-level k={BM25_CANDIDATE_K}, "
      f"dense k={DENSE_CANDIDATE_K}) -> rerank -> top {CONTEXT_AFTER_RERANK_K}")
print(f"  generation          : do_sample=False, max_new_tokens={MAX_NEW_TOKENS}")
print(f"  generation_seconds  : {GENERATION_SECONDS:.2f}")
print(f"  new_tokens          : {new_tokens}")
print(separator)
