# Verifiable RAG & Selective Prediction Architecture for Financial Compliance

> A deterministic 13-gate safety architecture designed to mechanically intercept LLM hallucinations, citation failures, and self-contradictions in high-stakes domains (AAOIFI Islamic Finance).

## 🚀 Research Overview
This Stage-1 Research Prototype explores the limitations of standard Retrieval-Augmented Generation (RAG) when applied to rigid compliance domains. By freezing retrieval variables via a 2x2 ablation study, this project isolates and benchmarks generation safety.

### Key Discoveries
- **Cross-Lingual Leakage (Instruction Collapse):** Discovered that bilingual models (e.g., Jais-2) hallucinate Arabic text (`الشيرازي`, `فرنسية`) when subjected to complex, multi-constraint English orchestration prompts.
- **Deterministic Mitigation:** Implemented a 13-gate mechanical safety layer that successfully intercepts unsafe outputs and escalates them to human reviewers, achieving 100% safety interception on frontier models like Qwen-2.5.

## 📁 Repository Structure
- `src/aaoifi_rag/`: Core pipeline, including generation, retrieval, and reliability gates.
- `reports/`: Raw JSON execution traces, evaluation metrics, and the final Stage-1 deliverable.
- `scripts/`: Automated batch evaluation and scoring scripts.

## 📊 Evaluation Results
Please review the `reports/final_research_prototype_deliverable.md` and the accompanying JSON trace files for the complete evaluation scorecard and raw LLM output logs.

---
*Developed by Ammar Khan as a Stage-1 Research Prototype for Deterministic RAG Safety.*
