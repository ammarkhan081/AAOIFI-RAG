# Jais-2 Long-Context (LC) Baseline Infeasibility Finding

**Date**: 2026-08-30  
**Purpose**: Document that the full 362-clause AAOIFI corpus does not fit in Jais-2-8B-Chat's context window, making a long-context baseline infeasible for this model.

## Context Limit vs. Corpus Size

| Metric | Value |
| --- | --- |
| **Model** | inception42/Jais-2-8B-Chat |
| **Context limit** | 8,192 tokens |
| **Corpus size** | 362 clause chunks |
| **Estimated token count** | 28,238 tokens |
| **Tokenization method** | tiktoken cl100k_base (approximation) |
| **Fits in context** | **No** |
| **Excess tokens** | 20,046 tokens (3.4x over capacity) |

## Methodology

- Loaded `data/private/extracted/clause_chunks.jsonl` (362 records)
- Constructed full corpus text with clause ID prefixes (e.g., `[clause:SS8:2:occurrence:0] text`)
- Estimated token count using tiktoken cl100k_base (GPT-4 tokenizer) as a reasonable approximation
- Compared against Jais-2's documented context limit of 8,192 tokens (from Hugging Face model card and transformers config)

## Note on Tokenization Precision

The actual Jais-2 tokenizer would give a more precise token count, but the margin (20,046 excess tokens, 3.4x over capacity) is too large to change the conclusion. The corpus is definitively too large for Jais-2's context window regardless of tokenizer differences.

## Implications for RQ4

This is a valid RQ4 finding: **Jais-2-8B-Chat cannot accommodate a long-context baseline for this corpus**. The retrieval-augmented approach is necessary for this model because the corpus exceeds its context capacity by a factor of ~3.4.

The LC baseline script (`scripts/colab_lc_baseline_test.py`) was designed to handle this case by:
- Detecting when the corpus exceeds the context limit
- Skipping generation for all items with `fits_context=false`
- Recording the required tokens, context limit, and excess tokens
- This skip behavior is itself a valid experimental result

## Next Steps

- Proceed with frontier model testing (ChatGPT, Gemini, Claude) which have much larger context windows
- The frontier manual test bundle (`reports/frontier_manual_test_bundle.md`) will enable testing the LC baseline on models that can actually accommodate the full corpus

## LC Baseline Substitution Status

**Date**: 2026-08-30

The original research plan called for comparing RAG vs. LC (Long-Context) baselines using the same model (Jais-2). However, since the LC baseline is infeasible for Jais-2 due to context window limitations, a **substitution** was executed:

- **Original plan**: RAG (Jais-2) vs. LC (Jais-2 with full corpus) — same model, different context strategies
- **Actual execution**: RAG (Jais-2) vs. FRONTIER (Claude Sonnet 5 with full corpus) — different models, different context strategies

This substitution is explicitly recorded to avoid the appearance of a silently dropped baseline. The frontier comparison serves as a proxy for the LC baseline question: does providing the full corpus (rather than retrieved context) improve answerability? The frontier model answered all 7 items correctly (100% answered, 0% abstained) compared to Jais-2's 3/7 answered (43%), 4/7 abstained (57%). See `reports/frontier_baseline_comparison.md` for full details.

While this does not isolate context-length from model capability as separate variables (model capability is a confound), it demonstrates that the evidence exists in the corpus for all items — Jais-2's abstentions are not due to genuine information gaps. The leading candidate explanation is Jais-2-specific abstention calibration.
