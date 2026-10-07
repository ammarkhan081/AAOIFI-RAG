# Prototype Summary

> **Status note (added 2026-10-07):** this summary was last updated 2026-09-02 (Phase 2) and predates the Stage-1 deliverable. For current Stage-1 status see `reports/final_research_prototype_deliverable.md`; for retrieval numbers see `reports/full_ablation_table.md`.

**Last Updated**: 2026-09-02  
**Phase**: Phase 2 - Retrieval Prototype Complete

---

## Status Table

| Component | Status | Notes |
|-----------|--------|-------|
| Corpus Ingestion | ✅ Complete | 5 standards (SS8, SS9, SS13, SS17, SS26), 362 clauses |
| Hard Set Authoring | ✅ Complete | 7 items (H01-H07) with gold clause annotations |
| Retrieval Baseline (B') | ✅ Complete | Clause-level BM25, Recall@5: 0.571 |
| Reranker Integration | ✅ Complete | BGE-reranker-v2-m3, B'+rerank Recall@5: 0.643 |
| Dense Retrieval | ✅ Complete | BGE-M3 + Chroma, tested in Hybrid config |
| SAC (Summary-Augmented Chunking) | ✅ Complete | Per-standard summaries, B'' Recall@5: 0.750, B''+rerank: 0.833 |
| Retrieval Ablation | ✅ Complete | 7 configs tested (A, B, B', B'+rerank, B'', B''+rerank, Hybrid) |
| Generation Prototype | ✅ Complete | Jais-2 end-to-end smoke tests (n=7) |
| Statistical Abstention (C) | ⚠️ Partial | Signal analysis completed, no correlation found, not wired into routing |
| Disagreement Trigger (D) | ⚠️ Partial | Citation entailment check completed, not wired into routing |
| Local Consistency (LC) | ✅ Complete | Documented as infeasible for Jais-2 (8K context limit) |
| Frontier Baseline | ✅ Complete | Claude Sonnet 5 comparison (7/7 answered vs RAG's 3/7) |

---

## Key Results

### Retrieval Performance (Clause Recall@5, n=7)

| Config | Recall@5 | Notes |
|--------|----------|-------|
| B''+rerank (SAC) | 0.833 | Best performing config |
| B'' (SAC) | 0.750 | SAC improvement confirmed |
| B'+rerank | 0.643 | Phase 2 baseline with reranker |
| B (fixed-token + rerank) | 0.643 | Matches clause-level baseline |
| B' (clause-level) | 0.571 | Phase 2 baseline |
| A (fixed-token) | 0.571 | Phase 1 baseline |
| Hybrid (BM25 ∪ dense) | 0.571 | Weakest due to noise on H04 |

See `reports/full_ablation_table.md` for complete ablation table.

### Generation Performance (n=7 Smoke Test)

- **Answered**: 3/7 (H01, H03, H05)
- **Abstained**: 4/7 (H02, H04, H06, H07)
- **Gold in Top-5**: 7/7 (all items had at least one gold clause retrieved)
- **All Golds in Top-5**: 1/7 (H07 only)

See `reports/smoke_test_batch_n7.md` for details.

---

## Remaining Work

1. **Wire C and D into routing logic**: Statistical abstention and disagreement trigger analyses exist but not integrated into pipeline
2. **Scale evaluation**: Expand hard set to n≈25–30+ for statistical significance
3. **Expert validation**: Add qualified Shari'ah scholar review to hard-set annotations

---

## Artifacts

### Data
- `data/private/extracted/clause_chunks.jsonl` - 362 clause chunks
- `data/private/extracted/hard_set.jsonl` - 7 hard-set items with gold annotations
- `data/private/extracted/standard_summaries.json` - 5 SAC summaries

### Indexes
- `data/private/extracted/bm25_clause_level.pkl` - Clause-level BM25 index
- `data/private/extracted/bm25_sac_colab.pkl` - SAC-enriched BM25 index
- `data/private/extracted/chroma_bge_m3/` - Dense retrieval index

### Results
- `data/private/extracted/retrieval_ablation_results.json` - Full retrieval ablation
- `data/private/extracted/sac_ablation_results.json` - SAC ablation results
- `reports/full_ablation_table.md` - Consolidated retrieval ablation table
- `reports/sac_ablation_n7.md` - SAC detailed results
- `reports/smoke_test_batch_n7.md` - Generation smoke test results
