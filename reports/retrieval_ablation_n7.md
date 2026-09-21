# Retrieval Ablation Results (n=7)

**Date**: 2026-08-30  
**Purpose**: Isolated retrieval component testing to evaluate the contribution of each retrieval strategy (BM25, dense, reranker) to Clause Recall@5.  
**Method**: Tested 5 retrieval configurations across all 7 hard-set items. Retrieval-only, no generation.  
**Note**: This is a descriptive n=7 comparison, not a statistically significant pilot evaluation.

---

## Configuration Definitions

| Config | Description |
|--------|-------------|
| **A** | fixed-token BM25 only, no reranker, top-5 by BM25 score |
| **B** | fixed-token BM25 candidates, reranked by BGE-reranker-v2-m3, top-5 by rerank score |
| **B'** | clause-level BM25 only, no reranker, top-5 by BM25 score (PARTIAL - SAC not yet implemented) |
| **B'+rerank** | clause-level BM25 candidates, reranked, top-5 by rerank score |
| **Hybrid** | BM25(clause-level) ∪ dense (BGE-M3+Chroma), reranked, top-5 |

---

## Per-Item Clause Recall@5

| Item | Config A | Config B | Config B' | Config B'+rerank | Config Hybrid |
|------|----------|----------|-----------|------------------|---------------|
| H01 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| H02 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 |
| H03 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| H04 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 |
| H05 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| H06 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| H07 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **AGGREGATE (mean)** | **0.571** | **0.643** | **0.571** | **0.643** | **0.571** |

---

## Key Finding: Dense Retrieval Noise on H04

**Mechanism**: On H04, the Hybrid config (BM25∪dense+rerank) performed strictly worse than all reranked-BM25-only configs (B, B'+rerank). Dense retrieval added two irrelevant candidates to the candidate pool:
- SS17 §5/1
- SS17 §4/4

The reranker ranked these irrelevant candidates above the genuinely relevant SS17 §5/1/8/7, pushing it out of the top-5. This caused Hybrid's Clause Recall@5 to drop to 0.67 (2/3 gold clauses) while all other configs achieved 1.00 (3/3 gold clauses).

**Interpretation**: Dense retrieval introduced noise that the reranker did not fully correct. This is the only item in the n=7 set where any configuration difference was observed — all other items tied exactly across all configs.

**Fragility Note**: This aggregate finding is driven by a single item (H04) out of 7. The result is fragile and should not be over-generalized. A larger pilot (n≈25–30+) is needed to determine whether dense retrieval's noise-introduction risk holds at scale or is specific to H04's query phrasing.

---

## Secondary Observation: H02 Clause-Level Differences

**Observation**: On H02, all configs achieved identical Clause Recall@5 (0.67, 2/3 gold clauses). However, the specific clauses retrieved differed:
- **Config A** (fixed-token BM25): hit SS9 §8/1 and SS9 §8/1(c)
- **Config Hybrid** (BM25∪dense+rerank): hit SS9 §8/1 and SS9 §8/2

**Interpretation**: Clause Recall@5 alone does not fully capture retrieval quality. Identical recall counts can mask differences in which specific clauses are retrieved. This matters for downstream generation if different clauses provide different evidence quality or completeness.

---

## Aggregate Statistics

| Config | Mean Recall@5 | Total Gold in Top-5 | Total Gold Clauses |
|--------|---------------|---------------------|-------------------|
| A | 0.571 | 9 | 17 |
| B | 0.643 | 10 | 17 |
| B' | 0.571 | 9 | 17 |
| B'+rerank | 0.643 | 10 | 17 |
| Hybrid | 0.571 | 9 | 17 |

---

## Retrieval Policy Recommendation (Not Yet Applied)

**Status**: Documentation only — no code changes in this session.

**Recommendation**: Future work should A/B test whether switching the default retrieval policy from Hybrid (BM25∪dense+rerank) to reranked clause-level BM25 (B'+rerank) improves downstream generation results.

**Rationale**:
1. **Performance**: B'+rerank performed at least as well as Hybrid on every tested item (7/7), and strictly better on H04.
2. **Simplicity**: B'+rerank is simpler and cheaper — no dense encoding (BGE-M3) or Chroma vector database needed.
3. **Noise risk**: The H04 finding suggests dense retrieval can introduce noise that the reranker does not fully correct.

**Caveats**:
- This recommendation is based on n=7 items only.
- The H04 finding is fragile (single-item driver).
- Dense retrieval may still add value on a larger, more diverse query set.
- Any policy change should be validated with a larger pilot (n≈25–30+) before adoption.

---

## Verification Note

Hard-set items are `corpus_cross_reference` only, **not** qualified Shari'ah scholar review. This is a technical retrieval ablation, not a fatwa or compliance ruling.
