# Reliability Signal Analysis: Reranker Top-1 Score vs. Retrieval/Answer Quality (n=7)

**Date**: 2026-08-30  
**Purpose**: Exploratory analysis of whether reranker top-1 score predicts retrieval completeness (all gold clauses in top-5) or model response class (answered vs. abstained).  
**Data Source**: `reports/e2e_batch_smoke_results.json` (Hybrid config: BM25∪dense+rerank)  
**Note**: This is a descriptive n=7 analysis, not a statistically significant finding. No formal correlation coefficients computed.

---

## Data Table (Sorted by Rerank Top-1 Score, Descending)

| Item ID | Rerank Top-1 Score | All Golds in Top-5 | Response Class | N Gold Clauses | N Gold in Top-5 |
|---------|-------------------|-------------------|----------------|----------------|----------------|
| H01     | 10.10             | False             | answered       | 3              | 2              |
| H03     | 7.90              | False             | answered       | 2              | 1              |
| H06     | 7.34              | False             | abstained      | 2              | 1              |
| H04     | 7.28              | False             | abstained      | 2              | 1              |
| H02     | 5.18              | False             | abstained      | 4              | 2              |
| H05     | 4.98              | False             | answered       | 3              | 1              |
| H07     | 3.78              | **True**          | abstained      | 1              | 1              |

---

## Pattern Analysis

### Observation 1: No Clear Correlation Between Score and Retrieval Completeness

- **Highest scores (H01: 10.10, H03: 7.90)**: Both answered, but neither achieved `all_golds_in_top5` (H01: 2/3, H03: 1/2).
- **Middle scores (H06: 7.34, H04: 7.28, H02: 5.18)**: All abstained, and none achieved `all_golds_in_top5`.
- **Lowest score (H07: 3.78)**: Abstained, but **was the only item to achieve `all_golds_in_top5`** (1/1).

**Interpretation**: Higher reranker top-1 scores do **not** correlate with better retrieval completeness. The item with the lowest score (H07) had perfect retrieval completeness, while items with the highest scores (H01, H03) had incomplete retrieval.

### Observation 2: No Clear Correlation Between Score and Response Class

- **Answered items**: H01 (10.10), H03 (7.90), H05 (4.98) — scores span the full range (10.10 to 4.98).
- **Abstained items**: H06 (7.34), H04 (7.28), H02 (5.18), H07 (3.78) — scores also span the full range (7.34 to 3.78).

**Interpretation**: Higher reranker top-1 scores do **not** correlate with the model choosing to answer rather than abstain. The highest-scoring item (H01) answered, but the second-highest (H03) also answered while the third-highest (H06) abstained. The lowest-scoring item (H07) abstained despite having perfect retrieval completeness.

### Observation 3: Retrieval Completeness Does Not Guarantee Answering

- **H07**: Perfect retrieval completeness (`all_golds_in_top5=True`), but the model abstained.
- **H01, H03, H05**: Incomplete retrieval completeness, but the model answered.

**Interpretation**: The model's abstention decision is not driven solely by whether all gold clauses are present in the retrieved context. This aligns with the frontier baseline comparison finding that Jais-2's abstentions are not attributable to insufficient evidence. Two independent analyses now point to the same root cause: (1) the frontier model answered all 7 items correctly with the same or superset of context where Jais-2 abstained, and (2) reranker top-1 scores (a retrieval confidence signal) do not predict Jais-2's abstention decisions. Together, these findings suggest Jais-2's over-abstention is model-specific calibration, not evidence-driven.

---

## Conclusion: Reranker Top-1 Score Is Not a Useful Selective-Prediction Signal

Based on this n=7 exploratory analysis, **reranker top-1 score does not predict**:
- Whether all gold clauses are retrieved
- Whether the model will answer or abstain

The signal shows no meaningful pattern that could support a threshold-based selective-prediction policy. The item with the lowest score had the best retrieval completeness, and items with similar scores had opposite response classes.

---

## Draft Selective-Prediction Policy (Rejected)

**Status**: Not recommended for further testing based on this analysis.

**Rejected hypothesis**: A threshold-based rule such as "if `rerank_top1_score < threshold, escalate regardless of model response" would not be effective because:
1. Low scores (e.g., H07: 3.78) can correspond to perfect retrieval completeness.
2. High scores (e.g., H01: 10.10) can correspond to incomplete retrieval completeness.
3. No clear threshold separates "good" from "bad" outcomes in either dimension.

**Alternative direction**: If selective prediction is still desired for the larger pilot, consider:
- **Reranker margin** (top_1 - top_2 score difference): A large margin might indicate higher confidence in the top-1 retrieval, though this was not analyzed here.
- **Model uncertainty signals**: If the generator can expose internal uncertainty (e.g., log probabilities), these may be more predictive of abstention than retrieval scores.
- **Retrieval completeness directly**: Use `all_golds_in_top5` or `n_gold_in_top5 / n_gold` as the signal, though this requires gold clause IDs (not available at inference time).

---

## Verification Note

Hard-set items are `corpus_cross_reference` only, **not** qualified Shari'ah scholar review. This is a technical signal analysis, not a fatwa or compliance ruling.
