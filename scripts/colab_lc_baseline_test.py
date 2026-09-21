"""Colab Long-Context (LC) baseline test: stuff entire corpus into context.

This tests whether retrieval adds value over stuffing the whole corpus into
Jais-2's context window. Since the corpus is larger than the context limit,
this script measures the gap and reports it rather than truncating.

THIS IS A 7-ITEM SMOKE TEST, NOT THE n=25-30 PILOT EVALUATION.
It does not score correctness, entailment, or statistical significance.
"abstained" vs "answered" is exact string match to the fixed abstention
line only — not a quality or Shari'ah judgment.

Run this .py file cell-by-cell in Google Colab (GPU runtime). Do not run it
in the local sandbox.

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

LC BASELINE SPECIFIC:
  - Constructs prompt with ENTIRE 362-clause corpus (prefixed by clause IDs)
  - Checks if corpus fits in Jais-2's 8192-token context window
  - If NOT: skips generation, records fits_context=false, required_tokens, context_limit
  - This is itself a valid RQ4 finding (corpus too large for long-context baseline)
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
        "tiktoken==0.14.0",
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
    "tiktoken",
    "accelerate",
    "bitsandbytes",
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
SRC_ROOT = CONTENT / "src"
RESULTS_JSON_PATH = CONTENT / "lc_baseline_results.json"


def _handoff_ready() -> list[str]:
    missing: list[str] = []
    if not CLAUSE_CHUNKS_PATH.is_file():
        missing.append(str(CLAUSE_CHUNKS_PATH))
    if not HARD_SET_PATH.is_file():
        missing.append(str(HARD_SET_PATH))
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
        if (CONTENT / "aaoifi_rag").is_dir() and not (SRC_ROOT / "aaoifi_rag").is_dir():
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
        "Still missing after upload. Expected a zip with clause_chunks.jsonl and "
        "hard_set.jsonl at the zip root (or one wrapper folder).\n  "
        + "\n  ".join(_missing)
    )

sys.path.insert(0, str(SRC_ROOT))
print(f"clause_chunks : {CLAUSE_CHUNKS_PATH}")
print(f"hard_set      : {HARD_SET_PATH}")
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


# %% [Cell 4] Load corpus and hard set; build full corpus text
import json
from typing import Any

EXPECTED_HARD_SET_SIZE = 7
EXPECTED_CORPUS_SIZE = 362

MODEL_ID = "inception42/Jais-2-8B-Chat"
_MODEL_ID_FALLBACK = "inceptionai/Jais-2-8B-Chat"
MAX_NEW_TOKENS = 512
JAIS_CONTEXT_LIMIT = 8192  # From Hugging Face model card and config

ABSTENTION_LINE = "I cannot answer from the given context"

SYSTEM_PROMPT = (
    "You are answering a question using only the AAOIFI clause "
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


def classify_response(text: str) -> str:
    """Exact match after strip. Not a quality, entailment, or Shari'ah label."""
    if text.strip() == ABSTENTION_LINE:
        return "abstained"
    return "answered"


# Load clause chunks
clause_chunks = read_jsonl(CLAUSE_CHUNKS_PATH)
if len(clause_chunks) != EXPECTED_CORPUS_SIZE:
    print(
        f"WARNING: loaded {len(clause_chunks)} clause chunks; "
        f"verified Phase-2 corpus is {EXPECTED_CORPUS_SIZE}."
    )

# Build full corpus text with clause ID prefixes (as specified for citation comparability)
corpus_blocks = []
for chunk in clause_chunks:
    clause_id = chunk["chunk_id"]
    text = chunk["text"]
    corpus_blocks.append(f"[{clause_id}] {text}")

FULL_CORPUS_TEXT = "\n\n".join(corpus_blocks)

# Load hard set
hard_set_items = read_jsonl(HARD_SET_PATH)
hard_set_items.sort(key=lambda row: row["item_id"])
if len(hard_set_items) != EXPECTED_HARD_SET_SIZE:
    print(
        f"WARNING: hard_set.jsonl has {len(hard_set_items)} items; "
        f"this batch script expected {EXPECTED_HARD_SET_SIZE}."
    )

print(f"CORPUS SIZE: {len(clause_chunks)} clause chunks")
print(f"HARD-SET ITEMS: {len(hard_set_items)} ({[row['item_id'] for row in hard_set_items]})")
print("verification note: corpus_cross_reference is NOT qualified scholar review.")
for row in hard_set_items:
    if row.get("verification_basis") != "corpus_cross_reference":
        print(
            f"WARNING: {row['item_id']} verification_basis="
            f"{row.get('verification_basis')!r}"
        )
    if row.get("status") == "scholar_validated":
        print(f"WARNING: {row['item_id']} is marked scholar_validated.")


# %% [Cell 5] Estimate token count using tiktoken (cl100k_base approximation)
import tiktoken

# Use cl100k_base (GPT-4 tokenizer) as a reasonable approximation
# The actual Jais tokenizer would be more accurate but requires loading the model first
encoder = tiktoken.get_encoding("cl100k_base")

CORPUS_TOKENS = encoder.encode(FULL_CORPUS_TEXT)
CORPUS_TOKEN_COUNT = len(CORPUS_TOKENS)

print(f"Full corpus token count (tiktoken cl100k_base): {CORPUS_TOKEN_COUNT}")
print(f"Jais-2 context limit: {JAIS_CONTEXT_LIMIT}")
print(f"Fits in context: {CORPUS_TOKEN_COUNT <= JAIS_CONTEXT_LIMIT}")

if CORPUS_TOKEN_COUNT > JAIS_CONTEXT_LIMIT:
    print(f"\nCORPUS DOES NOT FIT IN CONTEXT")
    print(f"Excess tokens: {CORPUS_TOKEN_COUNT - JAIS_CONTEXT_LIMIT}")
    print(f"This is a valid RQ4 finding: the corpus is too large for long-context baseline.")
    print("Generation will be skipped for all items with fits_context=false.")


# %% [Cell 6] Load Jais-2 (only if corpus fits; otherwise skip)
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if CORPUS_TOKEN_COUNT <= JAIS_CONTEXT_LIMIT:
    print("Corpus fits in context. Loading Jais-2 for generation...")

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
    print("JAIS-2 LOAD COMPLETE")
    print(f"  model_id     : {_active_model_id}")
    print(f"  hub_revision : {_hub_sha}")
    print(f"  load time    : {JAIS_LOAD_SECONDS:.1f}s")
    print(f"  VRAM used    : {vram_model_bytes / 1e9:.2f} GB")
    print(f"  VRAM peak    : {vram_peak_bytes / 1e9:.2f} GB")

    _input_device = torch.device("cuda:0")
else:
    print("Corpus does not fit in context. Skipping Jais-2 load.")
    tokenizer = None
    model = None
    _active_model_id = None
    _hub_sha = None
    JAIS_LOAD_SECONDS = 0
    vram_model_bytes = 0
    vram_peak_bytes = 0
    _input_device = None


# %% [Cell 7] Process each hard-set item
item_results: list[dict[str, Any]] = []

for item in hard_set_items:
    item_id = item["item_id"]
    query_text = item["question_text"]
    gold_answer = item["gold_answer"]
    gold_clause_ids = item["gold_clause_ids"]

    # Check if corpus fits in context
    fits_context = CORPUS_TOKEN_COUNT <= JAIS_CONTEXT_LIMIT

    if not fits_context:
        # Skip generation - corpus too large
        result = {
            "item_id": item_id,
            "status": item.get("status"),
            "verification_basis": item.get("verification_basis"),
            "fits_context": False,
            "required_tokens": CORPUS_TOKEN_COUNT,
            "context_limit": JAIS_CONTEXT_LIMIT,
            "excess_tokens": CORPUS_TOKEN_COUNT - JAIS_CONTEXT_LIMIT,
            "generation_skipped_reason": "corpus_exceeds_context_limit",
            "model_response": None,
            "response_class": None,
            "generation_seconds": None,
            "new_tokens": None,
            "input_token_count": None,
            "n_gold_clauses": len(gold_clause_ids),
            "gold_answer_leak_guard_passed": True,  # No prompt constructed, so no leak
        }
        item_results.append(result)
        print(
            f"{item_id}: SKIPPED (corpus {CORPUS_TOKEN_COUNT} tokens > limit {JAIS_CONTEXT_LIMIT})"
        )
        continue

    # Corpus fits - construct prompt and generate
    user_prompt = (
        "AAOIFI clause excerpts (this is the only source you may use):\n\n"
        f"{FULL_CORPUS_TEXT}\n\n"
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

    result = {
        "item_id": item_id,
        "status": item.get("status"),
        "verification_basis": item.get("verification_basis"),
        "fits_context": True,
        "required_tokens": CORPUS_TOKEN_COUNT,
        "context_limit": JAIS_CONTEXT_LIMIT,
        "gold_answer_leak_guard_passed": True,
        "n_gold_clauses": len(gold_clause_ids),
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
        f"{item_id}: {response_class}  new_tokens={new_tokens}  {generation_seconds:.2f}s"
    )
    print(f"  model_response: {model_answer!r}")


# %% [Cell 8] Consolidated JSON — LC baseline, n=7, descriptive only
n_items = len(item_results)
n_fits_context = sum(1 for row in item_results if row["fits_context"])
n_skipped = sum(1 for row in item_results if not row["fits_context"])

batch_report = {
    "label": (
        "LC baseline, n=7, descriptive only. Tests whether retrieval adds value "
        "over stuffing the entire corpus into Jais-2's context window."
    ),
    "verification_note": (
        "Hard-set items are corpus_cross_reference only, not qualified "
        "Shari'ah scholar review."
    ),
    "corpus": {
        "n_clause_chunks": len(clause_chunks),
        "corpus_token_count_estimate": CORPUS_TOKEN_COUNT,
        "tokenizer_used": "tiktoken cl100k_base (approximation)",
        "jais_context_limit": JAIS_CONTEXT_LIMIT,
        "fits_in_context": CORPUS_TOKEN_COUNT <= JAIS_CONTEXT_LIMIT,
        "excess_tokens": (
            CORPUS_TOKEN_COUNT - JAIS_CONTEXT_LIMIT
            if CORPUS_TOKEN_COUNT > JAIS_CONTEXT_LIMIT
            else 0
        ),
    },
    "n_items": n_items,
    "aggregates": {
        "n_items_fitting_context": n_fits_context,
        "n_items_skipped": n_skipped,
        "skipped_reason": "corpus_exceeds_context_limit" if n_skipped > 0 else None,
    },
    "generation": {
        "model_id": _active_model_id,
        "hub_revision": _hub_sha,
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "abstention_line": ABSTENTION_LINE,
        "jais2_load_seconds": round(JAIS_LOAD_SECONDS, 1) if JAIS_LOAD_SECONDS else None,
        "vram_model_gb": round(vram_model_bytes / 1e9, 2) if vram_model_bytes else None,
    },
    "package_versions": {
        "transformers": importlib.metadata.version("transformers"),
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "torch": torch.__version__,
        "tiktoken": importlib.metadata.version("tiktoken"),
    },
    "items": item_results,
}

RESULTS_JSON_PATH.write_text(
    json.dumps(batch_report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("=" * 72)
print("LC BASELINE RESULTS — n=7 DESCRIPTIVE ONLY")
print("=" * 72)
print(json.dumps(batch_report, ensure_ascii=False, indent=2))
print()
print(f"Wrote {RESULTS_JSON_PATH}")
print("Keep this file in the Colab session. It is not a public artifact.")
print(
    f"corpus tokens: {CORPUS_TOKEN_COUNT}, context limit: {JAIS_CONTEXT_LIMIT}, "
    f"fits: {CORPUS_TOKEN_COUNT <= JAIS_CONTEXT_LIMIT}"
)
