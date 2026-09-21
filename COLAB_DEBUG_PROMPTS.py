# COLAB_DEBUG_PROMPTS.py
# Run this to compare v1 and v3 prompts

import subprocess, sys, os, json, zipfile, time
from pathlib import Path

CONTENT = Path("/content")
HANDOFF_ZIP = CONTENT / "colab_e2e_handoff.zip"
RESULTS_JSON_PATH = CONTENT / "e2e_batch_debug_results.json"

# ── 1. Dependencies (fixed order) ──
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "pillow", "Pillow"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "pillow==11.0.0"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
    "rank-bm25==0.2.2", "accelerate>=0.30.0",
    "bitsandbytes>=0.43.0", "huggingface_hub>=0.24.0"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
    "git+https://github.com/huggingface/transformers.git"])

print("✓ Dependencies installed")

import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

# ── 2. Extract handoff ──
print("Extracting...")
with zipfile.ZipFile(HANDOFF_ZIP) as archive:
    for name in archive.namelist():
        if name.endswith(".pdf") or "chroma" in name.lower():
            continue
        target = CONTENT / Path(name).name
        with archive.open(name) as src:
            target.write_bytes(src.read())
print("✓ Extracted")

# ── 3. Load data ──
def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

clause_chunks = read_jsonl(CONTENT / "clause_chunks.jsonl")
hard_set = read_jsonl(CONTENT / "hard_set.jsonl")
print(f"Loaded: {len(clause_chunks)} chunks, {len(hard_set)} items")

# ── 4. BM25 retrieval ──
from rank_bm25 import BM25Okapi
tokens = [[word for word in chunk["text"].lower().split()] for chunk in clause_chunks]
bm25 = BM25Okapi(tokens)
print(f"BM25 ready: {len(clause_chunks)} docs")

def retrieve_context(query, k=5):
    scores = bm25.get_scores(query.lower().split())
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{"rank": r+1, "chunk_id": clause_chunks[i]["chunk_id"],
             "standard_id": clause_chunks[i]["standard_id"],
             "clause_id": clause_chunks[i]["clause_id"],
             "text": clause_chunks[i]["text"]} for r, i in enumerate(top)]

# ── 5. Load Jais-2 (modern API) ──
print("\nLoading Jais-2-8B-Chat...")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "inception42/Jais-2-8B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    attn_implementation="eager"
)

print("✓ Model loaded")
print(f"  Device map: {getattr(model, 'hf_device_map', 'auto')}")

# ── 6. Generate function (fixed) ──
def generate(query, context_records, system_prompt, user_suffix=""):
    context_text = "\n\n".join(
        f"[{r['rank']}] {r['standard_id']} §{r['clause_id']}\n{r['text']}"
        for r in context_records
    )
    user_prompt = (
        f"Retrieved excerpts:\n\n{context_text}\n\n"
        f"Question:\n{query}\n\n{user_suffix}"
    ).strip()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

# ── 7. Define prompts ──
ABSTENTION_LINE = "I cannot answer from the given context"

# v1 prompt (baseline)
SYSTEM_PROMPT_V1 = (
    "You are answering a question using only the retrieved AAOIFI clause "
    "excerpts provided. Do not use outside knowledge. "
    f"If the provided excerpts are insufficient to answer, reply exactly: '{ABSTENTION_LINE}'"
)
USER_PROMPT_SUFFIX_V1 = ""  # Already in system prompt

# v3 prompt (current)
SYSTEM_PROMPT_V3 = (
    "You are answering a question using only the retrieved AAOIFI clause "
    "excerpts provided. Do not use outside knowledge. "
    "You MAY combine information from multiple excerpts to answer the question. "
    f"Reply with the exact phrase '{ABSTENTION_LINE}' ONLY if the excerpts "
    "contain NO relevant information."
)
USER_PROMPT_SUFFIX_V3 = (
    "Using the excerpts above, synthesize an answer if the information is present "
    "across one or more excerpts. Reply exactly with the phrase below ONLY if "
    f"there is NO relevant information: {ABSTENTION_LINE}"
)

# Run both prompts
results = []
for prompt_name, system_prompt, user_suffix in [
    ("v1_baseline", SYSTEM_PROMPT_V1, USER_PROMPT_SUFFIX_V1),
    ("v3_synthesis", SYSTEM_PROMPT_V3, USER_PROMPT_SUFFIX_V3)
]:
    print(f"\n{'='*60}")
    print(f"RUNNING {prompt_name}")
    print(f"{'='*60}\n")

    prompt_results = []
    for i, item in enumerate(hard_set, 1):
        print(f"[{i}/7] {item['item_id']}...", end=" ", flush=True)
        context = retrieve_context(item["question_text"], k=5)
        response = generate(item["question_text"], context, system_prompt, user_suffix)
        rc = "abstained" if ABSTENTION_LINE.lower() in response.lower() else "answered"
        print(f"{rc} -> '{response[:100]}...'")

        prompt_results.append({
            "item_id": item["item_id"],
            "model_response": response,
            "response_class": rc,
            "context_length": sum(len(c["text"]) for c in context)
        })

    answered = sum(1 for r in prompt_results if r["response_class"] == "answered")
    abstained = 7 - answered
    print(f"\n{prompt_name} RESULTS: Answered {answered}/7, Abstained {abstained}/7")

    results.append({
        "prompt": prompt_name,
        "system_prompt": system_prompt,
        "user_suffix": user_suffix,
        "n_answered": answered,
        "n_abstained": abstained,
        "coverage_pct": 100 * answered / 7,
        "items": prompt_results
    })

# Save results
output = {"experiments": results}
with open(RESULTS_JSON_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Saved to {RESULTS_JSON_PATH}")
print("Compare the coverage between v1 and v3 to see if v3 helped.")