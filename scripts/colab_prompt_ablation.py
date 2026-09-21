"""Colab prompt ablation: test all 4 prompt variants on the same n=7 retrieval.

PURPOSE
-------
This script runs each of the 4 prompt variants (v1 baseline, v2 single
abstention, v3 synthesis guidance, v4 specific trigger) against the SAME
retrieved context from the stored n=7 run. This isolates the effect of the
prompt on Jais-2's answer/abstain behavior from retrieval variability.

The stored retrieval results carry chunk_ids but NOT text (to avoid committing
AAOIFI clause prose). We rejoin chunk_ids against the live clause corpus to
reconstruct the exact context the model saw.

BEFORE RUNNING
--------------
  - Runtime > Change runtime type > GPU (T4)
  - Colab Secrets: HUGGINGFACE_HUB_TOKEN, notebook access ON
  - HuggingFace: accept terms on https://huggingface.co/inception42/Jais-2-8B-Chat
  - Upload colab_e2e_handoff.zip to /content
  - IMPORTANT: Update the zip first: python scripts/pack_colab_e2e_handoff.py

OUTPUT
------
  - /content/prompt_ablation_results.json  (download this)
  - Console: per-variant, per-item answer/abstain summary table
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
        "transformers==4.44.2",
    ]
)
INSTALL_SECONDS = time.perf_counter() - _install_start
print(f"Dependencies installed in {INSTALL_SECONDS:.1f}s")

import torch

print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    raise RuntimeError("GPU required.")


# %% [Cell 2] Unpack handoff
import json
import zipfile

HANDOFF = Path("/content/colab_e2e_handoff.zip")
WORK = Path("/content/aaoifi_rag_workspace")
assert HANDOFF.exists(), "Upload colab_e2e_handoff.zip to /content first."

with zipfile.ZipFile(HANDOFF) as zf:
    zf.extractall(WORK)

# Add src to path
SRC = WORK / "src"
sys.path.insert(0, str(SRC))

# Load data
CLAUSE_CHUNKS_PATH = WORK / "data" / "private" / "extracted" / "clause_chunks.jsonl"
HARD_SET_PATH = WORK / "data" / "private" / "hard_set.jsonl"
STORED_RESULTS_PATH = WORK / "reports" / "e2e_batch_smoke_results.json"

def _load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

clause_chunks = _load_jsonl(CLAUSE_CHUNKS_PATH)
hard_set = _load_jsonl(HARD_SET_PATH)
stored_results = json.loads(STORED_RESULTS_PATH.read_text(encoding="utf-8"))

chunk_by_id = {c.get("chunk_id", ""): c for c in clause_chunks}
print(f"Corpus: {len(clause_chunks)} chunks, Hard set: {len(hard_set)} items")
print(f"Stored results: {len(stored_results.get('items', []))} items")


# %% [Cell 3] Reconstruct stored contexts
stored_items = stored_results["items"]
item_contexts = {}

for stored in stored_items:
    item_id = stored["item_id"]
    top5_ids = [r["chunk_id"] for r in stored["top5"]]
    context = []
    for cid in top5_ids:
        chunk = chunk_by_id.get(cid)
        if chunk is None:
            print(f"WARNING: {item_id} chunk {cid} not found in corpus")
            continue
        # Reconstruct the record format the pipeline expects
        record = dict(chunk)
        # Add reranker score from stored result
        stored_rec = next((r for r in stored["top5"] if r["chunk_id"] == cid), None)
        if stored_rec and "reranker_score" in stored_rec:
            record["reranker_score"] = stored_rec["reranker_score"]
        context.append(record)
    item_contexts[item_id] = context

print(f"Reconstructed contexts for {len(item_contexts)} items")
for item_id, ctx in item_contexts.items():
    print(f"  {item_id}: {len(ctx)} chunks")


# %% [Cell 4] Load Jais-2
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "inception42/Jais-2-8B-Chat"
HUB_REVISION = "da0e1639cd92b508b24120f7f77f5270a8465dc4"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print(f"Loading {MODEL_ID} (revision {HUB_REVISION[:12]}...)...")
_load_start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, revision=HUB_REVISION, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=HUB_REVISION,
    quantization_config=quantization_config,
    device_map="cuda:0",
    trust_remote_code=True,
)

_load_seconds = time.perf_counter() - _load_start
print(f"Model loaded in {_load_seconds:.1f}s")
print(f"VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")


# %% [Cell 5] Import prompt configs and generate
from aaoifi_rag.generation.prompt_config import PromptConfig
from aaoifi_rag.generation.types import GenerationConfig

GENERATION_CONFIG = GenerationConfig(
    model_id=MODEL_ID,
    hub_revision=HUB_REVISION,
    do_sample=False,
    max_new_tokens=512,
    seed=42,
)

PROMPT_VARIANTS = PromptConfig.all_configs()
print(f"Testing {len(PROMPT_VARIANTS)} prompt variants:")
for pc in PROMPT_VARIANTS:
    print(f"  - {pc.version}")


def generate_one(prompt_config, item_id, query_text, context_records):
    """Generate with Jais-2 for a single item using a specific prompt config."""
    prompt = prompt_config.build_prompt(
        item_id=item_id,
        query_text=query_text,
        context_records=context_records,
        gold_answer=None,  # Never leak during ablation
    )
    messages = prompt.as_messages()

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    ).to("cuda:0")

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=GENERATION_CONFIG.max_new_tokens,
            do_sample=GENERATION_CONFIG.do_sample,
        )

    new_tokens = outputs[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return {
        "text": text,
        "input_tokens": input_len,
        "new_tokens": len(new_tokens),
    }


# %% [Cell 6] Run all variants × all items
from datetime import datetime, timezone

ABSTENTION_LINE = "I cannot answer from the given context"

all_results = {
    "experiment": "prompt_ablation_v1",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "model": MODEL_ID,
    "hub_revision": HUB_REVISION,
    "generation_config": GENERATION_CONFIG.as_dict(),
    "n_items": len(hard_set),
    "n_variants": len(PROMPT_VARIANTS),
    "variants": {},
}

for pc in PROMPT_VARIANTS:
    print(f"\n{'='*60}")
    print(f"VARIANT: {pc.version}")
    print(f"{'='*60}")

    variant_results = {
        "prompt_version": pc.version,
        "prompt_config": pc.as_dict(),
        "items": [],
    }

    for item in hard_set:
        item_id = item["item_id"]
        query = item["question_text"]
        context = item_contexts.get(item_id, [])

        if not context:
            print(f"  {item_id}: SKIP (no context)")
            continue

        _gen_start = time.perf_counter()
        result = generate_one(pc, item_id, query, context)
        _gen_seconds = time.perf_counter() - _gen_start

        # Classify response
        text = result["text"]
        abstained = ABSTENTION_LINE.lower() in text.lower()
        empty = len(text.strip()) == 0

        status = "abstained" if abstained else ("empty" if empty else "answered")

        item_result = {
            "item_id": item_id,
            "status": status,
            "response_text": text,
            "input_tokens": result["input_tokens"],
            "new_tokens": result["new_tokens"],
            "generation_seconds": round(_gen_seconds, 3),
        }
        variant_results["items"].append(item_result)

        # Truncate for display
        display_text = text[:80] + "..." if len(text) > 80 else text
        print(f"  {item_id}: {status:>10} | {display_text}")

    # Summary
    statuses = [r["status"] for r in variant_results["items"]]
    n_answered = statuses.count("answered")
    n_abstained = statuses.count("abstained")
    n_empty = statuses.count("empty")
    variant_results["summary"] = {
        "n_answered": n_answered,
        "n_abstained": n_abstained,
        "n_empty": n_empty,
        "answer_rate": round(n_answered / len(statuses), 3) if statuses else 0,
    }
    print(f"\n  SUMMARY: {n_answered} answered, {n_abstained} abstained, {n_empty} empty")

    all_results["variants"][pc.version] = variant_results


# %% [Cell 7] Save results and print comparison
OUTPUT_PATH = Path("/content/prompt_ablation_results.json")
OUTPUT_PATH.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nResults saved to {OUTPUT_PATH}")

# Comparison table
print(f"\n{'='*60}")
print("PROMPT ABLATION COMPARISON TABLE")
print(f"{'='*60}")
print(f"{'Variant':<30} {'Answered':>8} {'Abstained':>9} {'Rate':>6}")
print("-" * 60)
for version, vr in all_results["variants"].items():
    s = vr["summary"]
    print(f"{version:<30} {s['n_answered']:>8} {s['n_abstained']:>9} {s['answer_rate']:>6.1%}")

# Per-item breakdown
print(f"\n{'='*60}")
print("PER-ITEM BREAKDOWN")
print(f"{'='*60}")
item_ids = [item["item_id"] for item in hard_set]
header = f"{'Item':<8}" + "".join(f" {v[:20]:>20}" for v in all_results["variants"])
print(header)
print("-" * len(header))
for item_id in item_ids:
    row = f"{item_id:<8}"
    for vr in all_results["variants"].values():
        item_result = next((r for r in vr["items"] if r["item_id"] == item_id), None)
        status = item_result["status"] if item_result else "N/A"
        row += f" {status:>20}"
    print(row)

print(f"\n✅ Download {OUTPUT_PATH} and place it in reports/prompt_ablation_results.json")
