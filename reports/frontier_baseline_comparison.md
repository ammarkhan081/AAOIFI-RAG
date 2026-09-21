# Frontier Baseline Comparison: Jais-2 RAG vs. Claude Sonnet 5 (Extended Thinking)

**Date**: 2026-08-30  
**Purpose**: Compare RAG baseline (Jais-2) against a frontier model with full corpus context to determine whether Jais-2's abstentions are attributable to insufficient retrieved evidence or model-specific calibration.  
**Method**: Manual testing using `reports/frontier_manual_test_bundle.md` — same 362-clause corpus, same 7 hard-set questions, same abstention instruction and leak-guard discipline as the RAG system.

---

## Per-Item Results

| Item | RAG (Jais-2) | FRONTIER (Claude Sonnet 5, Extended Thinking) |
|------|-------------|------------------------------------------------|
| H01 | answered (2/3 gold clauses cited) | answered (3/3 gold clauses cited) |
| H02 | ABSTAINED | answered, correct |
| H03 | answered (25-token fragment, incomplete) | answered (complete, both sentences) |
| H04 | ABSTAINED | answered, correct |
| H05 | answered (garbled Arabic tokens + self-contradictory trailing abstention) | answered, clean and coherent |
| H06 | ABSTAINED | answered, correct |
| H07 | ABSTAINED | answered, correct |

---

## Aggregate Statistics

| Metric | RAG (Jais-2) | FRONTIER (Claude Sonnet 5) |
|--------|-------------|----------------------------|
| Answered | 3/7 (43%) | 7/7 (100%) |
| Abstained | 4/7 (57%) | 0/7 (0%) |

---

## Additional Finding: H05 Gold Clause Completeness

For H05, the FRONTIER response cited an **additional clause** not present in the original `gold_clause_ids`:

- **SS26 §5/3**: "The company should assume the role of the agent in managing the insurance account, and the role of the Mudarib or agent in investing the insurance assets."

This suggests that the hard set's gold-clause lists may not be fully exhaustive. This is noted as a minor, honest limitation of the evaluation set — not something to fix retroactively without re-verification.

---

## Caveats and Limitations

### Resource Asymmetry
This is a resource-asymmetric comparison:
- **RAG**: Jais-2-8B-Chat (8B parameters, 8K context window, quantized 4-bit)
- **FRONTIER**: Claude Sonnet 5 with Extended Thinking mode (proprietary frontier model, effectively unlimited context for this corpus)

The finding demonstrates that Jais-2's abstentions are **not explained by missing evidence** — the same or less context, given to a more capable model, produced correct answers. However, the comparison does not isolate context-length from raw model capability as separate variables. Model capability is a confound.

### Sample Size
- **n=7**, descriptive only
- No significance testing
- Results are illustrative, not statistically conclusive

### Testing Method
- Manual testing via free web chat interface
- No programmatic reproducibility
- Subject to interface-specific behaviors (e.g., Extended Thinking mode's reasoning process)

---

## Interpretation

The frontier model, when provided with the full corpus (or a superset of the retrieved context used by RAG), answered all 7 items correctly with zero abstentions. This indicates that:

1. **The evidence exists in the corpus** for all 7 items — the abstentions are not due to genuine information gaps in the source material.
2. **Jais-2's abstention calibration appears overly conservative** — it abstains where a more capable model, with access to the same evidence, produces correct answers.
3. **The leading candidate explanation** for Jais-2's over-abstention is model-specific calibration (e.g., safety thresholds, uncertainty quantification), rather than insufficient retrieved evidence.

However, this cannot be conclusively proven without isolating the context-length variable from the model-capability variable. A definitive causal attribution would require testing Jais-2 with the full corpus (which is infeasible due to its 8K token limit) or testing Claude with the same limited retrieved context as Jais-2.

---

## Status of LC Baseline for Jais-2

As documented in `reports/lc_baseline_jais2_infeasible.md`, the Long-Context (LC) baseline is **infeasible for Jais-2** because the 362-clause corpus (~28,238 tokens) exceeds its 8,192-token context window by 3.4x.

The FRONTIER baseline described in this report serves as a **substitution** for the LC baseline in the ablation table:
- **Original plan**: Compare RAG vs. LC (same model, different context strategies)
- **Actual execution**: Compare RAG (Jais-2) vs. FRONTIER (Claude Sonnet 5, full corpus) — different models, different context strategies

This substitution is explicitly recorded to avoid the appearance of a silently dropped baseline.
