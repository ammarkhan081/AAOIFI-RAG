"""Colab SAC (Summary-Augmented Chunking) Build and Test — v4

THIS IS A COLAB HANDOFF SCRIPT — DO NOT RUN IN THE LOCAL SANDBOX.

Purpose: Implement SAC per Reuter et al. (arXiv:2510.06999) to reduce Document-Level Retrieval Mismatch.
- Generate per-standard summaries (5 total: SS8, SS9, SS13, SS17, SS26)
- Prepend each summary to clause chunks from that standard
- Build SAC-enriched BM25 index
- Test Clause Recall@5 on 7 hard-set items
- Compare against existing B' (clause-level BM25) and B'+rerank

Design choice: Per-standard summaries. A "standard" is this corpus's natural
document unit, so this matches Reuter et al.'s document-level granularity
(not a coarser simplification of it). The summary prompt below asks for
generic coverage rather than clause-type-targeted extraction, which also
matches their finding that generic summaries outperformed domain-expert-
targeted ones.

v4 CHANGELOG (fixes bugs found across 3 prior GPU runs — see
standard_summaries_generation_log.json this run produces for full detail):
  1. Prompts now use tokenizer.apply_chat_template() instead of hand-rolled
     <|system|>/<|user|> or <|im_start|> tags. Jais-2 ships its own
     chat_template.jinja; the hand-rolled formats did not match it, which is
     the most likely root cause of BOTH failure modes seen previously:
     immediate-EOS (1-token output) on SS13/SS17, and mid-generation token
     corruption on SS8/SS9/SS26 that produced garbled terms ("ribaInah",
     repeated-character loops) despite looking like a normal-length,
     non-retried success.
  2. Added min_new_tokens to generate() as a hard floor against immediate-EOS.
  3. Added validate_summary_text(): catches repeated-character runs and
     unexpected scripts (the concrete corruption patterns actually observed)
     as hard failures that trigger a retry, and flags mid-word case-fusion
     ("ribaInah"-style) as a soft warning for human review, since that
     pattern can't be caught mechanically with full confidence.
  4. Cache validity now checks CONTENT, not just file existence. Run 3's
     retry fix was correct, but a naive "file exists -> skip regeneration"
     check would have permanently frozen Run 2's corrupted summaries the
     next time this script ran. Existence + one clean bill of health from
     validate_summary_text() is required, on ALL 5 standards, before the
     cache is trusted.
  5. Removed the char-based max_chars=6000 pre-truncation. It was redundant
     with the tokenizer's own truncation=True, max_length=8192 AND
     imprecise (chars are a poor proxy for tokens) AND cut clause text more
     aggressively than the model's actual context budget required — passing
     the full clause set through only the tokenizer's truncation should let
     MORE real clause text reach the model for the larger standards.
  6. Dependencies fully pinned (previously transformers was git+main and
     bitsandbytes was an open range — both are documented in this project's
     memory as having caused version drift between runs).
  7. Best-effort determinism flags added (CUBLAS_WORKSPACE_CONFIG,
     use_deterministic_algorithms(warn_only=True)). This is not a
     replacement for the caching/freezing design below — it just narrows
     the residual gap for the one generation run that actually happens.
  8. Markdown artifacts (the model wrapping terms in **bold**/*italic*)
     are stripped from summaries before they're cached or indexed.

WHAT V4 DOES NOT AND CANNOT GUARANTEE:
  - That every one of the 5 summaries is substantively perfect. The QC
    checks catch the specific corruption patterns already observed; they
    are not a semantic fact-checker. Read the 5 cached summaries yourself
    before trusting a Recall@5 number built on top of them — this script
    prints them in full at the end for exactly that reason.
  - Any specific improvement in Recall@5 over B'/B'+rerank. That is an
    empirical outcome of running this, not something fixed in the code.
  - Statistical significance at n=7. That was never claimed by this
    pipeline and still isn't.

BEFORE RUNNING:
  - Runtime > Change runtime type > GPU (T4). Do this on a FRESH runtime —
    CUBLAS_WORKSPACE_CONFIG below only takes effect if set before any CUDA
    context exists in this process.
  - Colab Secrets: HUGGINGFACE_HUB_TOKEN, notebook access ON.
  - Hugging Face: accept terms on https://huggingface.co/inception42/Jais-2-8B-Chat
  - If /content/standard_summaries.json already exists from a PRIOR run,
    delete it before running unless you specifically want this script to
    validate and reuse it (Cell 5 will tell you which standards, if any,
    fail validation and force a clean regeneration of just this file).
  - Locally: python scripts/pack_colab_e2e_handoff.py
    (re-pack so the zip contains the current 7-item hard_set.jsonl)
  - Upload colab_e2e_handoff.zip to /content. Licensed AAOIFI text; Colab
    session only; do not commit or share.
"""

# %% [Cell 1] Environment setup
import os

# Must be set before any CUDA context is created in this process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HOME", "/content/huggingface_cache")

import re
import subprocess
import sys
import time
from pathlib import Path

# Set seeds for reproducibility
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# Best-effort: greedy decoding (used throughout this script) removes the
# dominant source of run-to-run divergence seen previously (temperature
# sampling). This narrows the remaining gap from non-deterministic CUDA
# kernels. warn_only=True so an op lacking a deterministic implementation
# (some 4-bit kernels) logs a warning instead of raising.
torch.use_deterministic_algorithms(True, warn_only=True)

# Fully pinned versions. These are the exact versions this pipeline has been
# validated against — see /areas/aaoifi-rag-prototype.md project notes.
# If a pin fails to resolve on a future Colab image, relax that one pin only.
PINNED_VERSIONS = {
    "rank-bm25": "0.2.2",
    "tiktoken": "0.14.0",
    "FlagEmbedding": "1.4.0",
    "chromadb": "1.5.9",
    "huggingface_hub": "1.28.0",
    "transformers": "5.16.0",
    "bitsandbytes": "0.50.2",
}

_install_start = time.perf_counter()
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--quiet"]
    + [f"{pkg}=={ver}" for pkg, ver in PINNED_VERSIONS.items()]
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
    raise RuntimeError("This SAC build script requires a Colab GPU runtime.")

print("Package versions (installed):")
for pkg in PINNED_VERSIONS:
    installed = importlib.metadata.version(pkg.replace("_", "-"))
    expected = PINNED_VERSIONS[pkg]
    flag = "OK" if installed == expected else f"MISMATCH (expected {expected})"
    print(f"  {pkg}: {installed} [{flag}]")


# %% [Cell 2] Private artifacts — upload colab_e2e_handoff.zip
import zipfile

CONTENT = Path("/content")
HANDOFF_ZIP = CONTENT / "colab_e2e_handoff.zip"
CLAUSE_CHUNKS_PATH = CONTENT / "clause_chunks.jsonl"
HARD_SET_PATH = CONTENT / "hard_set.jsonl"
CHROMA_ZIP_PATH = CONTENT / "chroma_bge_m3.zip"
CHROMA_DIR = CONTENT / "chroma_bge_m3"
SRC_ROOT = CONTENT / "src"
STANDARD_SUMMARIES_PATH = CONTENT / "standard_summaries.json"
GENERATION_LOG_PATH = CONTENT / "standard_summaries_generation_log.json"
SAC_CHUNKS_PATH = CONTENT / "clause_chunks_sac.jsonl"
SAC_INDEX_PATH = CONTENT / "bm25_sac_colab.pkl"
RESULTS_JSON_PATH = CONTENT / "sac_ablation_results.json"


def _handoff_ready() -> list[str]:
    missing: list[str] = []
    if not CLAUSE_CHUNKS_PATH.is_file():
        missing.append(str(CLAUSE_CHUNKS_PATH))
    if not HARD_SET_PATH.is_file():
        missing.append(str(HARD_SET_PATH))
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
    _missing = _handoff_ready()

if _missing:
    raise FileNotFoundError(
        "Still missing after upload. Expected a zip with clause_chunks.jsonl, "
        "hard_set.jsonl, and src/aaoifi_rag/retrieval/*.py at the zip root (or one wrapper folder).\n  "
        + "\n  ".join(_missing)
    )

sys.path.insert(0, str(SRC_ROOT))
print(f"clause_chunks : {CLAUSE_CHUNKS_PATH}")
print(f"hard_set      : {HARD_SET_PATH}")
print(f"src on path   : {SRC_ROOT}")
print(f"sac_chunks    : {SAC_CHUNKS_PATH} (to be created)")
print(f"sac_index     : {SAC_INDEX_PATH} (to be created)")
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


# %% [Cell 4] Load clause chunks and group by standard
import json
from typing import Any


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


clause_chunks = read_jsonl(CLAUSE_CHUNKS_PATH)
print(f"Loaded {len(clause_chunks)} clause chunks")

from collections import defaultdict

clauses_by_standard = defaultdict(list)
for chunk in clause_chunks:
    clauses_by_standard[chunk["standard_id"]].append(chunk)

print(f"Found {len(clauses_by_standard)} unique standards:")
for standard_id, chunks in sorted(clauses_by_standard.items()):
    print(f"  {standard_id}: {len(chunks)} clauses")


# %% [Cell 5] Validate any cached summaries; load Jais-2 only if needed
#
# SAC defines the summary as a property of the document, not a per-run
# stochastic draw (Reuter et al.), so a cache is the right design. But a
# cache is only trustworthy if its CONTENT has been checked, not just its
# existence — Run 3's fix could have been silently defeated by Run 2's
# corrupted cache file still sitting on disk. Every summary in the cache
# must independently pass validate_summary_text() before it is trusted.

_UNEXPECTED_SCRIPT_RE = re.compile(
    r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04FF\u0600-\u06FF]"
)  # CJK / Hiragana-Katakana / Hangul / Cyrillic / Arabic — none expected in this corpus
_REPEATED_CHAR_RE = re.compile(r"(.)\1{3,}")  # e.g. "هههههههه", "aaaaaa"
_MIXED_CASE_FUSION_RE = re.compile(r"[a-z][A-Z]")  # e.g. "ribaInah" — soft warn only


def validate_summary_text(text: str, min_length: int = 20) -> dict[str, Any]:
    """Hard-fail on corruption patterns actually observed in prior runs
    (empty/short output, repeated-character loops, stray scripts). Soft-warn
    on mid-word case fusion, which is a real pattern seen ("ribaInah") but
    not one that can be flagged with full confidence by regex alone --
    those get logged for a human pass, not auto-retried."""
    issues: list[str] = []
    hard_fail = False

    stripped = text.strip()
    if len(stripped) < min_length:
        issues.append(f"too_short:{len(stripped)}_chars")
        hard_fail = True
    if _REPEATED_CHAR_RE.search(text):
        issues.append("repeated_character_run")
        hard_fail = True
    if _UNEXPECTED_SCRIPT_RE.search(text):
        issues.append("unexpected_script_detected")
        hard_fail = True
    if _MIXED_CASE_FUSION_RE.search(text):
        issues.append("possible_fused_term_mixed_case:NEEDS_HUMAN_REVIEW")

    return {"valid": not hard_fail, "issues": issues}


def clean_summary_text(text: str) -> str:
    """Strip markdown emphasis the model sometimes adds and normalize
    whitespace. Applied only after validation passes."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\w)\*(.*?)\*(?!\w)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


NEED_SUMMARY_GENERATION = True
_cache_report: dict[str, dict[str, Any]] = {}

if STANDARD_SUMMARIES_PATH.is_file():
    with STANDARD_SUMMARIES_PATH.open("r", encoding="utf-8") as f:
        _cached_summaries = json.load(f)
    expected_standards = set(clauses_by_standard.keys())
    all_valid = set(_cached_summaries.keys()) == expected_standards
    for standard_id in expected_standards:
        text = _cached_summaries.get(standard_id, "")
        result = validate_summary_text(text)
        _cache_report[standard_id] = result
        if not result["valid"]:
            all_valid = False
    if all_valid:
        NEED_SUMMARY_GENERATION = False
        print(f"Cache at {STANDARD_SUMMARIES_PATH} passed validation for all "
              f"{len(expected_standards)} standards -- skipping Jais-2 load entirely.")
    else:
        print(f"Cache at {STANDARD_SUMMARIES_PATH} exists but did NOT pass validation:")
        for standard_id, result in sorted(_cache_report.items()):
            status = "OK" if result["valid"] else "FAIL"
            print(f"  {standard_id}: {status} {result['issues']}")
        print("Regenerating from scratch. The stale file will be overwritten "
              "once new summaries pass validation.")
else:
    print(f"{STANDARD_SUMMARIES_PATH} not found -- will generate fresh summaries.")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

JAIS2_MODEL_ID = "inception42/Jais-2-8B-Chat"

if NEED_SUMMARY_GENERATION:
    # 4-bit nf4 stays mandatory here, not a candidate for bf16: Jais-2-8B in
    # bf16 is ~16GB for weights alone, the entire budget of a T4 (16GB)
    # before activations/KV-cache/reranker share the card.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading Jais-2 for summary generation...")
    _load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(JAIS2_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        JAIS2_MODEL_ID,
        quantization_config=bnb_config,
        device_map="cuda:0",
    )
    model.eval()  # disable dropout etc. -- inference should not be stochastic here
    JAIS2_LOAD_SECONDS = time.perf_counter() - _load_start
    print(f"Jais-2 loaded in {JAIS2_LOAD_SECONDS:.1f}s")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        raise RuntimeError(
            "This tokenizer has no chat_template. The whole point of this "
            "version is to use the model's own template instead of guessing "
            "one -- if this fires, stop and check the Jais-2 tokenizer files "
            "(a chat_template.jinja should ship with the tokenizer)."
        )
else:
    JAIS2_LOAD_SECONDS = 0.0
    tokenizer = None
    model = None


# %% [Cell 6] Generate per-standard summaries (or load the validated cache)
import hashlib

SYSTEM_PROMPT = "You are a helpful assistant that summarizes Islamic finance standards concisely."
SUMMARY_PROMPT_TEMPLATE = """Summarize the following AAOIFI Shari'ah Standard in 2-3 sentences. Focus on what the standard covers, its main topics, and key provisions. Do not invent information — only summarize what is in the provided text.

Standard: {standard_id}
Number of clauses: {n_clauses}

Clauses:
{clauses_text}

Summary:"""


def build_prompt_text(tokenizer, standard_id: str, chunks: list[dict[str, Any]]) -> str:
    clauses_text = "\n\n".join(f"§{c['clause_id']}: {c['text']}" for c in chunks)
    # Pre-truncate by character count to prevent OOM on large standards (SS17 with 86 clauses).
    # This is a safety net before tokenizer truncation. 4000 chars is conservative
    # enough to avoid OOM on T4 while still providing substantial clause content.
    max_chars = 4000
    if len(clauses_text) > max_chars:
        clauses_text = clauses_text[:max_chars] + "... [truncated]"
    user_content = SUMMARY_PROMPT_TEMPLATE.format(
        standard_id=standard_id, n_clauses=len(chunks), clauses_text=clauses_text
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    # Use the model's OWN chat template rather than a hand-rolled guess.
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_one_summary(
    model, tokenizer, standard_id: str, chunks: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """Try the full clause set, then progressively smaller (but still real,
    corpus-derived) slices if generation fails validation. Never falls back
    to a prompt with no clause content -- if every attempt fails, this
    raises rather than silently accepting an ungrounded summary."""
    attempt_plan = [("full", chunks), ("reduced_30", chunks[:30]), ("reduced_15", chunks[:15])]
    attempt_log: list[dict[str, Any]] = []

    for attempt_name, subset in attempt_plan:
        if not subset:
            continue
        prompt_text = build_prompt_text(tokenizer, standard_id, subset)
        inputs = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=8192
        ).to("cuda:0")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                min_new_tokens=40,  # hard floor against immediate-EOS
                do_sample=False,     # greedy: deterministic given fixed weights/kernels
                pad_token_id=tokenizer.pad_token_id,
            )

        raw_summary = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        input_tokens = inputs.input_ids.shape[1]
        output_tokens = outputs.shape[1] - input_tokens
        check = validate_summary_text(raw_summary)

        attempt_log.append(
            {
                "attempt": attempt_name,
                "n_clauses_used": len(subset),
                "n_clauses_total": len(chunks),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "valid": check["valid"],
                "issues": check["issues"],
            }
        )
        print(
            f"  [{attempt_name}] clauses={len(subset)}/{len(chunks)} "
            f"in_tok={input_tokens} out_tok={output_tokens} valid={check['valid']}"
        )
        if check["issues"]:
            print(f"    issues: {check['issues']}")

        if check["valid"]:
            return clean_summary_text(raw_summary), attempt_log

    raise ValueError(
        f"Failed to generate a valid, corpus-grounded summary for {standard_id} "
        f"after {len(attempt_log)} attempts. Refusing to fall back to an "
        f"ungrounded prompt -- inspect attempt_log below and the tokenizer's "
        f"chat_template before proceeding.\nattempt_log={attempt_log}"
    )


standard_summaries: dict[str, str] = {}
generation_log: dict[str, list[dict[str, Any]]] = {}

if NEED_SUMMARY_GENERATION:
    for standard_id, chunks in sorted(clauses_by_standard.items()):
        print(f"\n{'='*72}\nGenerating summary for {standard_id} ({len(chunks)} clauses)...\n{'='*72}")
        summary, attempt_log = generate_one_summary(model, tokenizer, standard_id, chunks)
        standard_summaries[standard_id] = summary
        generation_log[standard_id] = attempt_log
        print(f"Summary (final, after cleanup):\n  {summary}\n")

    with STANDARD_SUMMARIES_PATH.open("w", encoding="utf-8") as f:
        json.dump(standard_summaries, f, ensure_ascii=False, indent=2)
    STANDARD_SUMMARIES_SHA256 = hashlib.sha256(STANDARD_SUMMARIES_PATH.read_bytes()).hexdigest()
    STANDARD_SUMMARIES_PATH.with_suffix(".sha256.txt").write_text(
        STANDARD_SUMMARIES_SHA256 + "\n", encoding="utf-8"
    )
    GENERATION_LOG_PATH.write_text(
        json.dumps(generation_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {STANDARD_SUMMARIES_PATH}")
    print(f"SHA256: {STANDARD_SUMMARIES_SHA256}")
    print(f"Saved per-standard generation log to {GENERATION_LOG_PATH} -- keep this "
          f"alongside the summaries for audit; it records exactly which attempt "
          f"(full clause set or a reduced slice) produced each cached summary.")
    print("Record this hash in your corpus manifest alongside clause_chunks.jsonl.")
else:
    standard_summaries = _cached_summaries
    STANDARD_SUMMARIES_SHA256 = hashlib.sha256(STANDARD_SUMMARIES_PATH.read_bytes()).hexdigest()
    generation_log = {
        sid: [{"attempt": "cached_revalidated", "valid": True, "issues": _cache_report[sid]["issues"]}]
        for sid in standard_summaries
    }
    print(f"Loaded {len(standard_summaries)} pre-validated cached summaries.")
    print(f"SHA256: {STANDARD_SUMMARIES_SHA256}")

print("\n" + "=" * 72)
print("HUMAN REVIEW CHECKPOINT — read all 5 summaries before trusting any")
print("Recall@5 number built on top of them. Automated checks catch the")
print("corruption patterns seen so far; they are not a fact-checker.")
print("=" * 72)
for standard_id, summary in sorted(standard_summaries.items()):
    print(f"\n{standard_id}:\n  {summary}")
print()


# %% [Cell 7] Create SAC-enriched clause chunks
sac_chunks = []
for chunk in clause_chunks:
    sac_chunk = dict(chunk)
    standard_id = chunk["standard_id"]
    summary = standard_summaries[standard_id]
    sac_chunk["bm25_text"] = f"[Standard summary: {summary}] {chunk['bm25_text']}"
    sac_chunk["standard_summary"] = summary
    sac_chunks.append(sac_chunk)

with SAC_CHUNKS_PATH.open("w", encoding="utf-8") as f:
    for chunk in sac_chunks:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print(f"Saved {len(sac_chunks)} SAC-enriched chunks to {SAC_CHUNKS_PATH}")
print("Example SAC bm25_text for first chunk:")
print(f"  {sac_chunks[0]['bm25_text'][:200]}...")


# %% [Cell 8] Build SAC-enriched BM25 index
from aaoifi_rag.retrieval.bm25_baseline import persist_bm25_index

persist_bm25_index(sac_chunks, SAC_INDEX_PATH, "sac_bm25")
print(f"Built SAC-enriched BM25 index at {SAC_INDEX_PATH}")


# %% [Cell 9] Load reranker and hard set
from aaoifi_rag.retrieval.bge_reranker import DirectBGEReranker, rerank_records

RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
RERANK_MAX_LENGTH = 1024

print(f"Loading reranker {RERANKER_MODEL_ID}...")
_rerank_start = time.perf_counter()
reranker = DirectBGEReranker(RERANKER_MODEL_ID, device="cuda:0")
RERANK_LOAD_SECONDS = time.perf_counter() - _rerank_start
print(f"Reranker loaded in {RERANK_LOAD_SECONDS:.1f}s")

hard_set_items = read_jsonl(HARD_SET_PATH)
hard_set_items.sort(key=lambda row: row["item_id"])
print(f"Loaded {len(hard_set_items)} hard-set items")


# %% [Cell 10] Utility functions for recall computation
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


# %% [Cell 11] Run SAC retrieval test
from aaoifi_rag.retrieval.bm25_baseline import search_bm25

BM25_SAC_K = 10  # Candidates before rerank
TOP_K_AFTER_RERANK = 5  # Final top-5

sac_results: list[dict[str, Any]] = []

for item in hard_set_items:
    item_id = item["item_id"]
    query_text = item["question_text"]
    gold_clause_ids = item["gold_clause_ids"]

    print(f"\n{'='*72}\nProcessing {item_id}: {query_text[:60]}...\n{'='*72}")

    item_result = {
        "item_id": item_id,
        "question_text": query_text,
        "n_gold_clauses": len(gold_clause_ids),
        "configs": {},
    }

    print("\nConfig B'' (SAC BM25 only):")
    sac_bm25_hits = search_bm25(sac_chunks, SAC_INDEX_PATH, query_text, top_k=TOP_K_AFTER_RERANK)
    recall_b_double_prime = compute_clause_recall(gold_clause_ids, sac_bm25_hits)
    print(f"  Recall@5: {recall_b_double_prime['recall']:.2f} "
          f"({recall_b_double_prime['n_gold_in_top5']}/{recall_b_double_prime['n_gold']})")

    item_result["configs"]["B_prime_prime_sac_bm25"] = {
        "description": "SAC-enriched clause-level BM25 only, no reranker, top-5 by BM25 score",
        "recall": recall_b_double_prime["recall"],
        "n_gold_in_top5": recall_b_double_prime["n_gold_in_top5"],
        "n_gold": recall_b_double_prime["n_gold"],
        "gold_hit_details": recall_b_double_prime["gold_hit_details"],
    }

    print("\nConfig B''+rerank (SAC BM25 + reranker):")
    sac_bm25_candidates = search_bm25(sac_chunks, SAC_INDEX_PATH, query_text, top_k=BM25_SAC_K)
    reranked_sac, _ = rerank_records(
        reranker, query_text, sac_bm25_candidates,
        text_key="bm25_text", batch_size=16, max_length=RERANK_MAX_LENGTH,
    )
    recall_sac_rerank = compute_clause_recall(gold_clause_ids, reranked_sac[:TOP_K_AFTER_RERANK])
    print(f"  Recall@5: {recall_sac_rerank['recall']:.2f} "
          f"({recall_sac_rerank['n_gold_in_top5']}/{recall_sac_rerank['n_gold']})")

    item_result["configs"]["B_prime_prime_sac_bm25_rerank"] = {
        "description": "SAC-enriched clause-level BM25 candidates, reranked, top-5 by rerank score",
        "recall": recall_sac_rerank["recall"],
        "n_gold_in_top5": recall_sac_rerank["n_gold_in_top5"],
        "n_gold": recall_sac_rerank["n_gold"],
        "gold_hit_details": recall_sac_rerank["gold_hit_details"],
    }

    sac_results.append(item_result)


# %% [Cell 12] Compute aggregate statistics and compare with existing B' and B'+rerank
import pandas as pd

RETRIEVAL_ABLATION_JSON = CONTENT / "retrieval_ablation_results.json"

if RETRIEVAL_ABLATION_JSON.is_file():
    with RETRIEVAL_ABLATION_JSON.open("r", encoding="utf-8") as f:
        ablation_data = json.load(f)
    existing_aggregates = ablation_data["aggregates"]
    EXISTING_B_PRIME_RECALL = existing_aggregates["B_prime_clause_level_bm25"]["mean_recall"]
    EXISTING_B_PRIME_RERANK_RECALL = existing_aggregates["B_prime_clause_level_bm25_rerank"]["mean_recall"]
    print(f"Loaded baseline from {RETRIEVAL_ABLATION_JSON}:")
    print(f"  B' mean_recall: {EXISTING_B_PRIME_RECALL:.3f}")
    print(f"  B'+rerank mean_recall: {EXISTING_B_PRIME_RERANK_RECALL:.3f}")
else:
    print(f"WARNING: {RETRIEVAL_ABLATION_JSON} not found.")
    print("Using placeholder values — ensure retrieval_ablation_results.json is in /content.")
    EXISTING_B_PRIME_RECALL = None
    EXISTING_B_PRIME_RERANK_RECALL = None

config_names = ["B_prime_prime_sac_bm25", "B_prime_prime_sac_bm25_rerank"]

table_data = []
for item_result in sac_results:
    row = {"item_id": item_result["item_id"]}
    for config_name in config_names:
        config = item_result["configs"][config_name]
        row[f"{config_name}_recall"] = config["recall"]
        row[f"{config_name}_n_gold_in_top5"] = config["n_gold_in_top5"]
        row[f"{config_name}_n_gold"] = config["n_gold"]
    table_data.append(row)

df = pd.DataFrame(table_data)

sac_aggregates = {}
for config_name in config_names:
    sac_aggregates[config_name] = {
        "mean_recall": df[f"{config_name}_recall"].mean(),
        "total_gold_in_top5": int(df[f"{config_name}_n_gold_in_top5"].sum()),
        "total_gold": int(df[f"{config_name}_n_gold"].sum()),
    }

print("\n" + "=" * 72)
print("SAC ABLATION: Clause Recall@5 (n=7, descriptive only)")
print("=" * 72)
if EXISTING_B_PRIME_RECALL is not None:
    print(f"\nExisting B' (clause-level BM25 only): mean_recall={EXISTING_B_PRIME_RECALL:.3f}")
    print(f"Existing B'+rerank: mean_recall={EXISTING_B_PRIME_RERANK_RECALL:.3f}")
else:
    print("\nExisting B' (clause-level BM25 only): [baseline not loaded]")
    print("Existing B'+rerank: [baseline not loaded]")
print(f"\nSAC B'' (SAC BM25 only): mean_recall={sac_aggregates['B_prime_prime_sac_bm25']['mean_recall']:.3f}")
print(f"SAC B''+rerank: mean_recall={sac_aggregates['B_prime_prime_sac_bm25_rerank']['mean_recall']:.3f}")

print("\nPer-item SAC results:")
display_columns = ["item_id"] + [f"{name}_recall" for name in config_names]
df_display = df[display_columns].copy()
df_display.columns = ["Item"] + [name.replace("_", " ").title() for name in config_names]
print(df_display.to_string(index=False))


# %% [Cell 13] Save results to JSON
sac_report = {
    "label": (
        "7-item SAC ablation, not a pilot evaluation. "
        "Clause Recall@5 is a retrieval metric only — not a Shari'ah judgment. "
        "n=7, descriptive comparison only. No statistical significance is claimed."
    ),
    "verification_note": (
        "Hard-set items are corpus_cross_reference only, not qualified "
        "Shari'ah scholar review. Standard summaries passed automated corruption "
        "checks (see summary_qc.issues per standard below) but still require a "
        "human read-through -- automated checks are not a semantic fact-checker."
    ),
    "n_items": len(sac_results),
    "sac_design": {
        "granularity": "per-standard (5 summaries: SS8, SS9, SS13, SS17, SS26) "
                       "-- matches Reuter et al.'s document-level granularity, "
                       "since a standard is this corpus's natural document unit",
        "model": "inception42/Jais-2-8B-Chat",
        "summary_length": "2-3 sentences per standard",
        "enrichment": "summary prepended to each clause's 'bm25_text' field "
                       "(raw summary kept separately in 'standard_summary')",
        "generation_mode": "greedy (do_sample=False), model's own chat_template, "
                            "min_new_tokens=40 floor, retry on reduced clause slice "
                            "if validation fails, never falls back to an ungrounded prompt",
        "standard_summaries_sha256": STANDARD_SUMMARIES_SHA256,
        "standard_summaries_path": str(STANDARD_SUMMARIES_PATH),
        "generation_log_path": str(GENERATION_LOG_PATH),
    },
    "summary_qc": {
        standard_id: {
            "n_clauses_used_in_accepted_attempt": log_entries[-1]["n_clauses_used"]
                if "n_clauses_used" in log_entries[-1] else None,
            "n_clauses_total": len(clauses_by_standard[standard_id]),
            "issues": log_entries[-1]["issues"],
        }
        for standard_id, log_entries in generation_log.items()
    },
    "standard_summaries": standard_summaries,
    "configs": {
        "B_prime_prime_sac_bm25": "SAC-enriched clause-level BM25 only, no reranker, top-5 by BM25 score",
        "B_prime_prime_sac_bm25_rerank": "SAC-enriched clause-level BM25 candidates, reranked, top-5 by rerank score",
    },
    "comparison": {
        "existing_B_prime_recall": EXISTING_B_PRIME_RECALL,
        "existing_B_prime_rerank_recall": EXISTING_B_PRIME_RERANK_RECALL,
        "sac_B_prime_prime_recall": sac_aggregates["B_prime_prime_sac_bm25"]["mean_recall"],
        "sac_B_prime_prime_rerank_recall": sac_aggregates["B_prime_prime_sac_bm25_rerank"]["mean_recall"],
    },
    "aggregates": sac_aggregates,
    "items": sac_results,
    "package_versions": {
        pkg: importlib.metadata.version(pkg.replace("_", "-")) for pkg in PINNED_VERSIONS
    },
    "load_times": {
        "jais2_seconds": round(JAIS2_LOAD_SECONDS, 1),
        "reranker_seconds": round(RERANK_LOAD_SECONDS, 1),
    },
}

RESULTS_JSON_PATH.write_text(json.dumps(sac_report, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n" + "=" * 72)
print("SAC ABLATION JSON — n=7 DESCRIPTIVE COMPARISON ONLY")
print("=" * 72)
print(json.dumps(sac_report, ensure_ascii=False, indent=2))
print()
print(f"Wrote {RESULTS_JSON_PATH}")
print("Keep this file in the Colab session. It is not a public artifact.")

print("\nFINAL COMPARISON:")
if EXISTING_B_PRIME_RECALL is not None:
    print(f"  B' (existing):  {EXISTING_B_PRIME_RECALL:.3f}")
    print(f"  B'+rerank (existing): {EXISTING_B_PRIME_RERANK_RECALL:.3f}")
else:
    print("  B' (existing):  [baseline not loaded]")
    print("  B'+rerank (existing): [baseline not loaded]")
print(f"  B'' (SAC):     {sac_aggregates['B_prime_prime_sac_bm25']['mean_recall']:.3f}")
print(f"  B''+rerank (SAC): {sac_aggregates['B_prime_prime_sac_bm25_rerank']['mean_recall']:.3f}")
print("\nRead the 5 summaries printed above (or in sac_ablation_results.json"
      "['standard_summaries']) before treating this as a reportable SAC result.")
