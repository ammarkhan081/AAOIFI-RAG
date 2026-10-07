# Final Deliverable: Trustworthy RAG & Selective Prediction Prototype

## A. Final Prototype Status
**Status:** Complete (Stage 1 Pilot).
The end-to-end RAG architecture with deterministic safety routing is fully implemented, verified, and computationally evaluated on a pilot hard set ($n=7$).

## B. Research Contribution
We implemented a **selective-prediction architecture of 13 deterministic gates (10 enabled by default; 5 were triggered by defects observed in the stored n=7 run)**. Rather than relying on an LLM's internal safety priors, this system mechanically verifies outputs (e.g., citation integrity, absence of self-contradiction, exact RAG reliance) and deterministically routes the generation into one of three buckets: `ANSWER`, `ABSTAIN`, or `ESCALATE` (human review).

## C. Research Gap
**Gap Addressed:** Standard RAG pipelines suffer from unmeasured hallucination risk, especially in high-stakes domains like Islamic Finance (AAOIFI compliance), where models frequently drop citations or suffer from language/instruction collapse.
**Solution:** Our architecture explicitly intercepts unsafe generations before they reach the user, converting dangerous hallucinations into safe escalations.

## D. Architecture
1. **Retrieval Layer**: BM25 + BGE-M3 Dense Retriever (Top-K extraction).
2. **Prompt Orchestration**: Strict structural constraints enforcing `[n]` citation syntax and explicit abstention triggers (`v5` Strong Guidance).
3. **Generation Layer**: HuggingFace local inference (Evaluated on Jais-2-8B-Chat and Qwen-2.5-7B-Instruct).
4. **Reliability Layer**: 13 deterministic regex/heuristic gates (10 enabled by default) analyzing the generation string against the retrieved context.
5. **Routing Layer**: Decision engine mapping gate triggers to `ANSWER`, `ABSTAIN`, or `ESCALATE`.

## E. Experiments
We conducted a controlled 2x2 experimental matrix isolating the generation layer (using `fixed_jais2_contexts` to freeze retrieval variance) across two variables:
- **Models**: Jais-2 vs. Qwen-2.5
- **Prompts**: `v1` (Basic constraints) vs. `v5` (Strict citation/RAG constraints)

## F. Baselines
- **Primary Baseline**: Jais-2-8B-Chat under `v1` prompting. (Resulted in high over-abstention [4/7] and uncited reasoning that required escalation).
- **Comparator**: Qwen-2.5-7B-Instruct under `v1` prompting. (Improved synthesis but failed to format citations, triggering mechanical escalations).

## G. Ablations
**Prompt Ablation (Jais-2 on v5)**: To test robustness, we subjected the bilingual Jais-2 model to the strict English `v5` prompt.
*Result:* Catastrophic failure. The model suffered from cross-lingual leakage, generating 512 tokens and spontaneously injecting irrelevant Arabic script (`الشيرازي`, `فرنسية`) into English compliance answers. This proved that strict orchestration can break the safety priors of bilingual models.

## H. Evaluation
Metrics were computed deterministically against gold labels:
- **Answer Rate**: % of items producing a verified `ANSWER`.
- **Abstention Rate**: % of items where the model safely abstained.
- **Escalation Rate**: % of items intercepted by mechanical gates.

## I. Results
**Target System (Qwen-2.5 on v5):**
- **Answer Rate**: 28.6% (2/7)
- **Abstention Rate**: 57.1% (4/7)
- **Escalation Rate**: 14.3% (1/7) 
*Finding:* No served answer was flagged by the gates: both answered items (H03, H05) passed every enabled gate, which measures gate compliance, not answer correctness. The 4 abstentions were on items that are answerable by construction (see Limitation 3), so they count as over-abstention, not as correct abstention. Crucially, on H01, the model attempted to mix a partial answer with an abstention string—a dangerous self-contradiction that the `no_self_contradiction` gate successfully detected and escalated.

## J. Limitations
1. **Sample Size**: The $n=7$ scale is statistically insignificant. It serves only as a mechanical proof-of-concept for the architecture, not a generalized proof of model capability.
2. **Conservative Yield**: The answer rate is only 28.6% (2/7), indicating the system is highly conservative on this pilot. A false-positive (selective-risk) rate is not measured end-to-end here, because the hard set contains only answerable items (Limitation 3).
3. **Lack of True Negatives**: The current hard set contains only "answerable" items. Escalation precision cannot be fully measured until unanswerable probes are introduced.

## K. Reproducibility
- **Traceability**: Every generation, gate trigger, and routing decision is saved as a JSON artifact in the `reports/` directory.
- **Evaluation**: The automated evaluation script (`scripts/run_qwen_evaluation.py`) programmatically recalculates all metrics from the raw traces against the ground-truth `hard_set.jsonl`.
- **Dependencies**: Fixed environments recorded (e.g., `transformers 5.17.0.dev0`).

## L. Future Extension
This prototype forms the verified foundation for a full research publication. The immediate next steps for detailed research are:
1. Scale the hard set from $n=7$ to $n=150$ (incorporating unanswerable probes) to achieve statistical significance.
2. Formally compute Selective Risk (False Positive Rate) on the scaled dataset.
3. Experiment with advanced reasoning models (e.g., Llama-3, Command-R) to improve the 28.6% Answer Rate without compromising the checks of the gate layer.
