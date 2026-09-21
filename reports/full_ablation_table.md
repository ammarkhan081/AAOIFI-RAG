# Full Retrieval Ablation Table (n=7)

**Date**: 2026-09-02  
**Source**: `retrieval_ablation_results.json` + `sac_ablation_results.json`  
**Metric**: Clause Recall@5 (mean across 7 hard-set items)

---

## Config Rankings (Best to Worst)

| Rank | Configuration | Mean Recall@5 | Total Gold in Top-5 | Total Gold |
|------|---------------|---------------|---------------------|------------|
| 1 | B''+rerank (SAC BM25 + reranker) | 0.833 | 14 | 17 |
| 2 | B'' (SAC BM25 only) | 0.750 | 12 | 17 |
| 3 | B'+rerank (clause-level BM25 + reranker) | 0.643 | 10 | 17 |
| 4 | B (fixed-token BM25 + reranker) | 0.643 | 10 | 17 |
| 5 | B' (clause-level BM25 only) | 0.571 | 9 | 17 |
| 6 | A (fixed-token BM25 only) | 0.571 | 9 | 17 |
| 7 | Hybrid (BM25 ∪ dense + reranker) | 0.571 | 9 | 17 |

**Takeaway**: SAC + reranking (B''+rerank) is the strongest per-item configuration. Hybrid is the weakest (tied with baseline BM25-only configs), confirming the noise-introduction risk observed on H04.

---

## Config Descriptions

- **A**: Fixed-token BM25 only (baseline from Phase 1)
- **B**: Fixed-token BM25 + reranker
- **B'**: Clause-level BM25 only (Phase 2 baseline)
- **B'+rerank**: Clause-level BM25 + reranker (Phase 2 baseline)
- **B''**: SAC BM25 only (per-standard summaries prepended to clause chunks)
- **B''+rerank**: SAC BM25 + reranker (best performing config)
- **Hybrid**: BM25(clause-level) ∪ dense (BGE-M3+Chroma) + reranker

---

## Plan Config Coverage

**Covered by this table**:
- ✅ A (fixed-token BM25 baseline)
- ✅ B (fixed-token BM25 + reranker)
- ✅ B' (clause-level BM25)
- ✅ B'+rerank (clause-level BM25 + reranker)
- ✅ B'' (SAC BM25)
- ✅ B''+rerank (SAC BM25 + reranker)
- ✅ Hybrid (BM25 ∪ dense + reranker)

**NOT covered (remaining work)**:
- ❌ n=25–30 pilot evaluation - current results are n=7 descriptive only
- ❌ Scholar validation - hard-set items are corpus_cross_reference only, not qualified Shari'ah review

**Partial work (not formal pipeline configs)**:
- ⚠️ C (statistical abstention) - reliability signal analysis completed (reports/reliability_signal_analysis_n7.md), no correlation found, not wired into routing logic
- ⚠️ D (disagreement trigger) - citation entailment check completed (reports/citation_entailment_check_n7.md), not wired into routing logic

**Completed (not retrieval configs)**:
- ✅ LC (local consistency check) - documented as infeasible for Jais-2 (reports/lc_baseline_jais2_infeasible.md)
- ✅ FRONTIER - Claude Sonnet 5 comparison completed (reports/frontier_baseline_comparison.md)
