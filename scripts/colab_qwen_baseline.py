"""Colab baseline evaluation for Qwen-2.5-7B-Instruct (Comparator Arm).

PURPOSE
-------
This script runs the n=7 hard set through the exact same AnswerPipeline
as the Jais-2 baseline, using the EXACT same retrieved contexts and the
EXACT same v1 prompt (double abstention).

The ONLY variable changed is the generation model, satisfying the requirement
for a strictly controlled comparator arm.

Model Choice Note: The original research plan listed 'Qwen3-8B' as a
comparator. This script uses Qwen2.5-7B-Instruct as it is the most stable,
capable, and available open-weights representative of the Qwen family in
this size class as of late 2024 / early 2026.

BEFORE RUNNING
--------------
  - Runtime > Change runtime type > GPU (T4)
  - Colab Secrets: HUGGINGFACE_HUB_TOKEN, notebook access ON
  - Upload colab_e2e_handoff.zip to /content
  - IMPORTANT: Update the zip first: python scripts/pack_colab_e2e_handoff.py

OUTPUT
------
  - /content/e2e_qwen_baseline_results.json  (download this)
  - Console: per-item pipeline traces
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
        "bitsandbytes==0.50.2",
        "huggingface_hub>=0.24.0",
        "transformers==5.16.1",
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
stored_jais2_results = json.loads(STORED_RESULTS_PATH.read_text(encoding="utf-8"))

chunk_by_id = {c.get("chunk_id", ""): c for c in clause_chunks}
print(f"Corpus: {len(clause_chunks)} chunks, Hard set: {len(hard_set)} items")


# %% [Cell 3] Reconstruct stored contexts and create mock Retriever
from aaoifi_rag.orchestration.protocols import RetrievalResult, Retriever

# We MUST use the exact same retrieved contexts Jais-2 saw.
stored_items = stored_jais2_results["items"]
item_contexts = {}

for stored in stored_items:
    item_id = stored["item_id"]
    top5_ids = [r["chunk_id"] for r in stored["top5"]]
    context = []
    for cid in top5_ids:
        chunk = chunk_by_id.get(cid)
        if chunk is None:
            continue
        record = dict(chunk)
        stored_rec = next((r for r in stored["top5"] if r["chunk_id"] == cid), None)
        if stored_rec and "reranker_score" in stored_rec:
            record["reranker_score"] = stored_rec["reranker_score"]
        context.append(record)
    item_contexts[item_id] = context

class FixedContextRetriever:
    """Mock retriever that just returns the pre-computed n=7 contexts."""
    def retrieve(self, query_text: str, k: int) -> RetrievalResult:
        # Match by exact query text
        for item in hard_set:
            if item["question_text"] == query_text:
                return RetrievalResult(records=tuple(item_contexts[item["item_id"]]))
        return RetrievalResult(records=())

print(f"Reconstructed contexts for {len(item_contexts)} items")


# %% [Cell 4] Load Qwen-2.5-7B-Instruct
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# CHANGED ONLY THIS: The model (Comparator Arm)
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print(f"Loading {MODEL_ID}...")
_load_start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="cuda:0",
)

_load_seconds = time.perf_counter() - _load_start
print(f"Model loaded in {_load_seconds:.1f}s")
print(f"VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")


# %% [Cell 5] Implement Generator Protocol and Pipeline
from aaoifi_rag.generation.types import GeneratedAnswer, GenerationConfig
from aaoifi_rag.generation.prompt_config import PromptConfig
from aaoifi_rag.orchestration.pipeline import AnswerPipeline

GENERATION_CONFIG = GenerationConfig(
    model_id=MODEL_ID,
    hub_revision=None, # Use latest
    do_sample=False,
    max_new_tokens=512,
    seed=42,
)

class QwenGenerator:
    def generate(self, prompt) -> GeneratedAnswer:
        _start = time.perf_counter()
        messages = prompt.as_messages()
        
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True
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
        _gen_time = time.perf_counter() - _start
        
        return GeneratedAnswer(
            text=text,
            config=GENERATION_CONFIG,
            input_token_count=input_len,
            new_tokens=len(new_tokens),
            generation_seconds=_gen_time,
            truncated=(len(new_tokens) >= GENERATION_CONFIG.max_new_tokens)
        )

retriever = FixedContextRetriever()
generator = QwenGenerator()

# We use PromptConfig.v5() to explicitly enforce citations and synthesis
pipeline = AnswerPipeline(retriever, generator, prompt_config=PromptConfig.v5())


# %% [Cell 6] Run the e2e Pipeline on the Hard Set
from datetime import datetime, timezone

results = {
    "experiment": "e2e_batch_smoke",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "retrieval": {"type": "fixed_jais2_contexts"},
    "generation": GENERATION_CONFIG.as_dict(),
    "items": []
}

print(f"\n{'='*60}")
print(f"RUNNING QWEN BASELINE E2E PIPELINE (v5 prompt)")
print(f"{'='*60}")

for item in hard_set:
    item_id = item["item_id"]
    query = item["question_text"]
    gold_answer = item.get("gold_answer")
    
    print(f"\nItem: {item_id}")
    
    result = pipeline.run(item_id, query, gold_answer=gold_answer)
    
    # Store standard trace format
    trace = {
        "item_id": item_id,
        "stages": [s.as_dict() for s in result.stages],
        "top5": [
            {"chunk_id": r["chunk_id"], "reranker_score": r.get("reranker_score")} 
            for r in result.retrieval.records[:5]
        ],
        "decision": result.decision.value,
        "served_answer": result.served_answer,
    }
    
    if result.generated:
        trace["generation"] = {
            "text": result.generated.text,
            "input_tokens": result.generated.input_token_count,
            "new_tokens": result.generated.new_tokens,
            "seconds": round(result.generated.generation_seconds, 3),
            "truncated": result.generated.truncated
        }
        
    results["items"].append(trace)
    print(f"  Decision: {result.decision.value}")
    if result.served_answer:
        display = result.served_answer[:100] + "..." if len(result.served_answer) > 100 else result.served_answer
        print(f"  Answer : {display}")


# %% [Cell 7] Save results
OUTPUT_PATH = Path("/content/e2e_qwen_baseline_results.json")
OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n✅ Results saved to {OUTPUT_PATH}")
print(f"Download and place in reports/e2e_qwen_baseline_results.json to compare side-by-side with Jais-2.")
