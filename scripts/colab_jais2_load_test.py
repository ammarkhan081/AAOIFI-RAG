"""Colab load-test for inception42/Jais-2-8B-Chat (Jais-2 8B Chat).

Verified-working configuration as of 2026-08-29, confirmed by a real Colab
run that generated: "Hello, this is an AI assistant, and I am working."

This file is the single source of truth for a fresh Colab session. It loads
directly into the working 4-bit NF4 / bfloat16 / device_map="cuda:0" setup.
It does not attempt device_map="auto" or torch_dtype=float16 at any point.

Run this .py file cell-by-cell in Google Colab. It contains no AAOIFI text.
Purpose: confirm HF authentication, model access, and a single test generation
ONLY. This file stays a load test. Retrieval-plus-generation is already wired
in scripts/colab_e2e_single_test.py; do not copy retrieval into this script.

BEFORE RUNNING:
  - In Colab: Runtime > Change runtime type > GPU (T4 is sufficient for load
    testing; A100 will be faster but not required here).
  - In Colab Secrets (the key icon in the left sidebar): add a secret named
    exactly  HUGGINGFACE_HUB_TOKEN  with your HF token as its value, and
    toggle "Grant notebook access" ON.
  - On HuggingFace.co: visit https://huggingface.co/inception42/Jais-2-8B-Chat
    while logged in and accept the contact-sharing terms if prompted.
    The load test will fail at model download with a 403/gated error if you
    have not done this step.
"""

# %% [Cell A] Environment setup — set HF cache dir and install dependencies
import os
import subprocess
import sys
import time

# Keep downloaded model weights in the Colab filesystem so disk usage is
# reported accurately at the end of the test.
os.environ.setdefault("HF_HOME", "/content/huggingface_cache")

_install_start = time.perf_counter()

# transformers from GitHub main is required for the Jais-2 architecture
# (PyPI transformers is not sufficient). See:
# https://huggingface.co/inception42/Jais-2-8B-Chat
#
# bitsandbytes is required for BitsAndBytesConfig in Cell D. The Colab session
# that first confirmed generation may already have had it installed from an
# earlier diagnostic cell; a *fresh* runtime does not. It is therefore part
# of this install even though it was not listed in the original Cell A notes.
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "git+https://github.com/huggingface/transformers.git",
    "huggingface_hub>=0.24.0",
    "accelerate>=0.30.0",
    "bitsandbytes>=0.43.0",
])

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
    raise RuntimeError(
        "No CUDA device. Change the Colab runtime to GPU (T4) and re-run from Cell A."
    )


# %% [Cell B] Authenticate with Hugging Face using the Colab Secret
# google.colab.userdata is the correct API for Colab Secrets — it reads the
# secret from Colab's secure vault, not from os.environ directly. The secret
# is never written to os.environ or any file by this script.
try:
    from google.colab import userdata
    _token = userdata.get("HUGGINGFACE_HUB_TOKEN")
    if not _token:
        raise ValueError(
            "HUGGINGFACE_HUB_TOKEN secret is empty. "
            "Add it in Colab Secrets (key icon, left sidebar) and re-run."
        )
except ModuleNotFoundError:
    raise RuntimeError(
        "google.colab not available — this script must run inside Google Colab. "
        "If running locally, set HUGGINGFACE_HUB_TOKEN in your environment and "
        "replace this block with: _token = os.environ['HUGGINGFACE_HUB_TOKEN']"
    )

import huggingface_hub

huggingface_hub.login(token=_token, add_to_git_credential=False)
whoami = huggingface_hub.whoami()
print(f"AUTH SUCCESS: logged in as '{whoami['name']}'")

# Immediately discard the reference — we only need the login side-effect.
del _token


# %% [Cell C] Configuration

# Model ID: inception42/Jais-2-8B-Chat is the directly-verified live repo
# (browser-confirmed "Gated model — You have been granted access").
# The model card's embedded Python snippet still shows the old alias
# "inceptionai/Jais-2-8B-Chat" — that is a stale code sample; inception42
# is the correct primary. We attempt inception42 first and only fall back
# to inceptionai if HF returns a 404 (e.g. if the repo is renamed again).
MODEL_ID = "inception42/Jais-2-8B-Chat"
_MODEL_ID_FALLBACK = "inceptionai/Jais-2-8B-Chat"  # stale alias — try only on 404

# Test prompt: generic, contains zero AAOIFI content. This is purely a
# "is the model responsive" check. The retrieval pipeline lives in
# scripts/colab_e2e_single_test.py, not here.
TEST_MESSAGES = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Respond concisely in English.",
    },
    {
        "role": "user",
        "content": "Say hello and confirm you are working. Keep your reply to one sentence.",
    },
]

MAX_NEW_TOKENS = 60  # Small cap — this is a load test, not a quality eval.
HF_HOME = os.environ["HF_HOME"]

print(f"Model ID        : {MODEL_ID}")
print(f"HF cache dir    : {HF_HOME}")
print(f"Max new tokens  : {MAX_NEW_TOKENS}")


# %% [Cell D] Load tokenizer and 4-bit model (working config only)
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# NF4 4-bit + bfloat16 compute + explicit cuda:0.
# float16 is not used: Jais-2 Squared-ReLU (relu(x)^2) overflows float16
# (~65504 max) to NaN, which collapses generation.
# device_map="auto" is not used: on T4 it offloaded layers (including
# lm_head) to CPU and generation took 27+ minutes instead of ~2 seconds.
_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def _load_tokenizer_and_model(model_id: str):
    print(f"Loading tokenizer from {model_id!r}...")
    _tok_start = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id)
    tok_seconds = time.perf_counter() - _tok_start
    print(f"Tokenizer loaded in {tok_seconds:.1f}s  (vocab size: {tok.vocab_size:,})")

    print(f"\nLoading 4-bit model from {model_id!r} onto cuda:0")
    print("(~16 GB download on first run — subsequent runs use the local HF cache)")
    _model_start = time.perf_counter()
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=_bnb_config,
        device_map="cuda:0",
    )
    model_seconds = time.perf_counter() - _model_start
    mdl.eval()
    return tok, mdl, tok_seconds, model_seconds


torch.cuda.reset_peak_memory_stats()
_vram_before_load = torch.cuda.memory_allocated()

try:
    tokenizer, model, TOK_LOAD_SECONDS, MODEL_LOAD_SECONDS = _load_tokenizer_and_model(MODEL_ID)
    _active_model_id = MODEL_ID
except RepositoryNotFoundError:
    print(f"\nWARNING: {MODEL_ID!r} returned 404. Retrying with fallback alias {_MODEL_ID_FALLBACK!r}.")
    print("Report this so the script can be corrected for future runs.")
    tokenizer, model, TOK_LOAD_SECONDS, MODEL_LOAD_SECONDS = _load_tokenizer_and_model(_MODEL_ID_FALLBACK)
    _active_model_id = _MODEL_ID_FALLBACK

_vram_after_load = torch.cuda.memory_allocated()
vram_model_bytes = _vram_after_load - _vram_before_load
vram_peak_bytes = torch.cuda.max_memory_allocated()

print(f"\nMODEL LOAD COMPLETE")
print(f"  Load time          : {MODEL_LOAD_SECONDS:.1f}s")
print(f"  Model class        : {model.__class__.__name__}")
print(f"  Device map summary : {getattr(model, 'hf_device_map', 'N/A')}")
print(f"  VRAM used by model : {vram_model_bytes / 1e9:.2f} GB")
print(f"  VRAM peak so far   : {vram_peak_bytes / 1e9:.2f} GB")

device_map = getattr(model, "hf_device_map", None)
if device_map is None:
    _param_devices = {p.device for p in model.parameters()}
    print(f"  Unique parameter devices: {_param_devices}")
    _offloaded = any(str(d) not in ("cuda:0", "cuda") and not str(d).startswith("cuda") for d in _param_devices)
else:
    _offloaded_layers = {
        k: v for k, v in device_map.items() if str(v) in ("cpu", "disk")
    }
    print(f"  Named placements     : {len(device_map)}")
    print(f"  CPU/disk offloaded   : {len(_offloaded_layers)}")
    _offloaded = bool(_offloaded_layers)

if _offloaded:
    raise RuntimeError(
        "CPU/disk offloading detected after the 4-bit cuda:0 load. "
        "Do not generate. Restart the Colab runtime and re-run from Cell A."
    )
_hub_sha = huggingface_hub.HfApi().model_info(_active_model_id).sha
print(f"  Hub revision SHA    : {_hub_sha}")
print("  Placement check     : all layers on GPU")


# %% [Cell E] Single test generation
# tokenize=True + return_dict=True avoids double-BOS (tokenize=False then
# tokenizer(...) prepends a second BOS and generation collapses).

_input_device = torch.device("cuda:0")
print(f"Placing inputs on : {_input_device}")

inputs = tokenizer.apply_chat_template(
    TEST_MESSAGES,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = {k: v.to(_input_device) for k, v in inputs.items()}
inputs.pop("token_type_ids", None)

_first_tokens = inputs["input_ids"][0][:4].tolist()
print(f"First 4 input token IDs : {_first_tokens}  (ID 0 = BOS, should appear once only)")
if _first_tokens[0] == 0 and _first_tokens[1] == 0:
    raise RuntimeError(
        "Double-BOS detected even with tokenize=True. "
        "The tokenizer may be adding BOS again. Inspect tokenizer.add_bos_token."
    )
print(f"Input token count : {inputs['input_ids'].shape[-1]}")

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

_first_gen = generated_ids[:5].tolist()
print(f"First 5 generated token IDs : {_first_gen}  (should NOT all be 0)")

response_text = tokenizer.decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print("\n" + "=" * 60)
print("TEST GENERATION RESULT")
print("=" * 60)
print(f"Model response   : {response_text!r}")
print(f"New tokens       : {new_tokens}")
print(f"Generation time  : {GENERATION_SECONDS:.2f}s")
print(f"Tokens/sec       : {new_tokens / GENERATION_SECONDS:.1f}")
print(f"VRAM peak (gen)  : {vram_gen_peak_bytes / 1e9:.2f} GB")
print("=" * 60)


# %% [Cell F] Final report
def _dir_bytes(path: str) -> int:
    import pathlib
    p = pathlib.Path(path)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0

import json

report = {
    "auth": {
        "hf_username": whoami["name"],
        "login_succeeded": True,
    },
    "model": {
        "model_id_attempted": MODEL_ID,
        "model_id_loaded": _active_model_id,
        "fallback_used": _active_model_id != MODEL_ID,
        "model_class": model.__class__.__name__,
        "vocab_size": tokenizer.vocab_size,
        "context_length": getattr(model.config, "max_position_embeddings", "unknown"),
        "quantization": "nf4_4bit",
        "bnb_4bit_compute_dtype": "bfloat16",
        "device_map": "cuda:0",
        "hub_revision_sha": _hub_sha,
        "hub_revision_note": (
            "SHA of the Hub default revision at report time via "
            "HfApi.model_info; from_pretrained is not pinned to this SHA."
        ),
    },
    "hardware": {
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
    },
    "timing_seconds": {
        "dependency_install": round(INSTALL_SECONDS, 1),
        "tokenizer_load": round(TOK_LOAD_SECONDS, 1),
        # MODEL_LOAD_SECONDS is measured around the successful 4-bit cuda:0
        # from_pretrained call in Cell D. There is no earlier discarded load.
        "model_load": round(MODEL_LOAD_SECONDS, 1),
        "generation": round(GENERATION_SECONDS, 2),
    },
    "memory": {
        "vram_model_gb": round(vram_model_bytes / 1e9, 2),
        "vram_peak_model_load_gb": round(vram_peak_bytes / 1e9, 2),
        "vram_peak_generation_gb": round(vram_gen_peak_bytes / 1e9, 2),
    },
    "generation": {
        "test_prompt": TEST_MESSAGES,
        "response": response_text,
        "new_tokens": new_tokens,
        "tokens_per_second": round(new_tokens / GENERATION_SECONDS, 1),
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
    },
    "disk": {
        "hf_cache_gb": round(_dir_bytes(HF_HOME) / 1e9, 2),
    },
    "package_versions": {
        "transformers": importlib.metadata.version("transformers"),
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        "accelerate": importlib.metadata.version("accelerate"),
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "torch": torch.__version__,
    },
    "notes": [
        "This is a load test only — no AAOIFI corpus content was used.",
        "Verified-working configuration as of 2026-08-29 via a real Colab run "
        "(response: 'Hello, this is an AI assistant, and I am working.').",
        "Load path is 4-bit NF4, bfloat16 compute, device_map='cuda:0' only. "
        "There is no discarded float16 / device_map='auto' attempt, so "
        "timing_seconds.model_load is the successful 4-bit load.",
        "Primary model ID: inception42/Jais-2-8B-Chat. The model card's "
        "embedded code sample shows the stale alias inceptionai/...; that "
        "alias is retained only as a fallback if the primary returns 404.",
        "Token access requires accepting contact-sharing terms on the HF model page.",
        "Transformers installed from GitHub main branch per model card instructions.",
        "corpus_cross_reference and qualified_scholar_review are distinct "
        "verification levels — this script does not touch either.",
    ],
}

print("\n=== LOAD TEST REPORT ===")
print(json.dumps(report, indent=2, ensure_ascii=False))
