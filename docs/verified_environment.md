# Verified Colab environment (Jais-2 + retrieval co-load)

**Recorded:** 2026-08-29 (this file).  
**Purpose:** reproducibility notes for the confirmed T4 load path. This is not a pilot-evaluation result.

## What this file does and does not claim

- Package versions below were **not re-measured in the 2026-08-29 afternoon local session**. They are copied from the researcher's pasted Colab stdout from the same calendar day (fresh T4 load test, then H07 end-to-end smoke test).
- The Hugging Face revision SHA **was** retrieved in that local session via `huggingface_hub.HfApi().model_info("inception42/Jais-2-8B-Chat")` with **no token**. That call returns the Hub **default-revision HEAD at lookup time**, not a hash printed by the earlier Colab `from_pretrained` (those scripts did not pass `revision=`).
- `corpus_cross_reference` in the hard set is **not** qualified Shari'ah scholar review.

## Hardware (from the 2026-08-29 Colab pastes)

- Runtime: Google Colab GPU, Tesla T4
- VRAM total: 15.6 GB
- CUDA available: True

## Python packages (from the 2026-08-29 Colab pastes)

| Package | Version as printed |
| --- | --- |
| torch | 2.11.0+cu128 |
| transformers | 5.16.0.dev0 (installed from `git+https://github.com/huggingface/transformers.git`, not a pinned PyPI release) |
| bitsandbytes | 0.50.2 |
| huggingface-hub | 1.28.0 (printed on the H07 smoke-test Cell 1) |
| accelerate | 1.14.0 (printed on the H07 smoke-test Cell 1) |
| FlagEmbedding | 1.4.0 |
| chromadb | 1.5.9 |
| rank-bm25 | 0.2.2 |
| tiktoken | 0.14.0 |

Install constraint used by the Colab scripts: transformers from GitHub `main` at install time. `5.16.0.dev0` is a moving git snapshot; it is **not** an immutable transformers commit. Pinning a transformers git SHA is still an open choice.

## Jais-2 model identity

- **Primary model ID:** `inception42/Jais-2-8B-Chat` (gated: `auto`).
- **Fallback ID only on Hub 404:** `inceptionai/Jais-2-8B-Chat` (stale alias; not used unless the primary 404s).
- **Hub default revision SHA (this session lookup, 2026-08-29):** `da0e1639cd92b508b24120f7f77f5270a8465dc4`
- **Hub `lastModified` at that lookup:** 2026-08-18 09:59:48+00:00
- **Load config confirmed in the rewritten load-test paste:** `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.bfloat16` (float16 is documented as causing NaN from Squared-ReLU), `device_map="cuda:0"` (not `"auto"`), `apply_chat_template(tokenize=True, return_dict=True)`.

VRAM figures from the **rewritten load-test** paste on a fresh T4: **5.66 GB** weights, **5.88 GB** peak during the short hello generation. Those numbers were not re-measured locally.

## Coexistence (H07 smoke test, 2026-08-29 Colab paste)

FlagEmbedding 1.4.0, transformers-from-git, and Jais-2 4-bit **did** run in one process on that H07 run (BGE-M3 encode + DirectBGEReranker, then Jais-2 generate). That observation comes from the pasted notebook output, not from a rerun in this session.

## Open design decision (not silently assumed)

The Colab scripts still call `from_pretrained(model_id)` **without** `revision=<sha>`. Recording the HEAD SHA here does **not** pin future downloads. Whether experiments should pin `revision="da0e1639cd92b508b24120f7f77f5270a8465dc4"` is a decision for the researcher.
