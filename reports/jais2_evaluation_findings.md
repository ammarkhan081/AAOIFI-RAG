# Jais-2-8B-Chat Evaluation Findings

**Date**: 2026-09-16
**Phase**: Stage 1 (Prototype) — Generation Layer Evaluation

This document formally records the empirical findings observed while testing `inception42/Jais-2-8B-Chat` as the primary generative model for the AAOIFI compliance RAG prototype. These are robust scientific observations about the model's behavior under the constraints of deductive compliance answering.

## 1. Chronic Over-Abstention
Under the baseline prompt (which strictly requires the model to answer *only* from the provided context and instructs it to abstain otherwise), Jais-2 exhibits a severe conservative prior, abstaining on 4 out of 7 (57%) of known-answerable hard set items (H02, H04, H06, H07) despite perfect context retrieval.

## 2. Cross-Lingual Token Leakage
Despite being queried in English and provided with English-only AAOIFI context, Jais-2 exhibits a strong internal prior to surface Arabic script. This has been independently observed three times:
1. **SS8 SAC Summary**: The original retrieval summarization injected Arabic script into its output.
2. **H05 Generation**: The model generated phrases like `"policyholdersigently"`, `"fundفرنسية assets"`, and `"fund الشيرازي assets"`.
3. **v5 Forced-Answer Ablation**: When pushed to answer via prompt `v5`, the model generated the entirety of H06 in Arabic, and prefaced H03 with `"الشرح:"` (Explanation:).

This repetition establishes that cross-lingual token leakage is a systematic architectural behavior of Jais-2 under these constraints, not isolated noise. The `script_integrity` gate is specifically designed to detect and escalate this.

## 3. Catastrophic Degradation under Prompt-Forcing
An ablation study (v1-v5) was conducted to determine if the over-abstention was a tunable prompt artifact. We applied `v5` (Strong Guidance), which explicitly permitted synthesis and strictly commanded English-only output.

**Finding**: When Jais-2's abstention prior is forcefully suppressed by the prompt, the model undergoes catastrophic degradation. 
- On item H02 (a simple "No" answer), the forced model hallucinated a 512-token essay, contradicted its own logic, and triggered the `max_new_tokens` hard limit, cutting off mid-sentence.
- It completely ignored the "English only" rule, directly generating Arabic script.

This proves that the over-abstention is not merely a superficial instruction-tuning artifact that can be engineered away; it is a fundamental characteristic of the model. When pushed past its natural abstention threshold, its reasoning and instruction-following capabilities break down.

## Conclusion & Next Steps
These findings validate the necessity of the proposed **Comparator Arm** (Layer 3). We will activate the planned Qwen comparator (using `Qwen/Qwen2.5-7B-Instruct` as the most stable current representative of the Qwen 7B/8B class) to test the exact same contexts and prompts. This will isolate model-intrinsic behavior from the rest of the RAG pipeline architecture.

All Jais-2 results and traces will be preserved alongside the Qwen comparator results.
