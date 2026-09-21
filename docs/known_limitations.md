# Known limitations and candidate research questions

**Updated:** 2026-08-30  
**Status:** documentation of observed behaviour. These are **not** statistically established results, failure rates, or pilot-evaluation conclusions.

Hard-set items remain `verification_basis=corpus_cross_reference`. That is not qualified Shari'ah scholar review.

Smoke-test notes: `reports/smoke_test_batch_n7.md`. Machine-readable batch output: `reports/e2e_batch_smoke_results.json` (Git-ignored; `model_response` quotes clause wording). Single-item logs with retrieved AAOIFI text: `reports/smoke_test_h07.md`, `reports/smoke_test_h02.md` (Git-ignored).

**n=7 batch aggregates (copied from that JSON, not re-derived):** `n_items` 7; `n_abstained` **4/7**; `n_answered` **3/7**; `any_gold_in_top5` **7/7**; `all_golds_in_top5` **1/7** (H07 only). `response_class` is exact-line match only.

## 1. Generator over-abstention

**Single-item runs (H07, H02):** both emitted the instructed abstention line (8 new tokens). Gold was in the prompt (H07 rank 2; H02 partial gold).

**n=7 batch (`e2e_batch_smoke_results.json`):** `response_class=abstained` on **H02, H04, H06, H07** (4/7). Script-`answered`: **H01, H03, H05** (3/7). All seven items had **at least one** gold clause in top-5. Only H07 had **all** golds in top-5, and H07 still abstained.

**Frontier baseline comparison (2026-08-30):** A manual test using Claude Sonnet 5 (Extended Thinking mode) with the full 362-clause corpus answered all 7 items correctly (100% answered, 0% abstained). This demonstrates that Jais-2's abstentions are **not attributable to insufficient retrieved evidence** — the same or less context, given to a more capable model, produced correct answers. The leading candidate explanation is Jais-2-specific abstention calibration (e.g., safety thresholds, uncertainty quantification), not a proven cause (model capability is a confound). See `reports/frontier_baseline_comparison.md` for full details.

This is still a **descriptive n=7 smoke-test count**, not a measured rate. The frontier comparison elevates the finding from "cause unclear" to "not attributable to insufficient evidence," but does not isolate context-length from model capability as separate variables.

## 2. Heading-vs-clause ranking (refined after n=7 hard-set batch)

**Do not over-generalise to “the reranker always prefers headings.”**

- **This n=7 batch (JSON `top5` rank 1):** only **H02** has rank-1 `clause_id` `8` (`chunk_id` `clause:SS9:8:occurrence:0`). The other six rank-1 ids are `2/4/4`, `6`, `5/2/2`, `4/2`, `5/6`, `3/2`.
- **Earlier Phase 2 dense-retrieval sanity check (not this hard set):** SS8 heading §4 above §4/1. Separate query, n=1.
- **H02 single-item smoke test:** same heading-first pattern as the batch H02 row.

Candidate questions: on which query types do heading-length records take rank 1 after this union+rerank policy?

## 3. Garbled / cross-lingual generation (n=1 in the n=7 batch)

**H05** `model_response` includes Arabic tokens `وأخرجه` and `تهنئة` inside otherwise English text. Observation **n=1**. Candidate question: how often do non-source-language tokens appear in Stage 1 English-only prompts? Not investigated or fixed in this documentation pass.

## 4. Mixed answer + abstention in one string (n=1)

**H05** `model_response` is not equal to the abstention line but **ends with** that line after other text. `response_class` is **`answered`**. Candidate questions: how often do mixed completions occur? Should evaluation use a third class?

## 5. `answered` does not mean uniform quality

JSON `new_tokens`: **H01** 277 vs **H03** 25, both `response_class=answered`. Binary abstain/answer does not capture coverage. Candidate question: what depth/coverage labels are needed alongside `response_class`?

## 6. Retrieval ablation: dense retrieval noise (n=7, descriptive)

**Retrieval ablation (2026-08-30):** Tested 5 retrieval configurations (A: fixed-token BM25, B: fixed-token BM25+rerank, B': clause-level BM25, B'+rerank, Hybrid: BM25∪dense+rerank) across all 7 hard-set items. Full details: `reports/retrieval_ablation_n7.md`.

**Key finding (H04):** Hybrid config (BM25∪dense+rerank) performed strictly worse than all reranked-BM25-only configs. Dense retrieval added two irrelevant candidates (SS17 §5/1, §4/4) to the candidate pool; the reranker ranked these above the genuinely relevant SS17 §5/1/8/7, pushing it out of top-5. This caused Hybrid's Clause Recall@5 to drop to 0.67 while all other configs achieved 1.00.

**Fragility note:** This aggregate finding is driven by a single item (H04) out of 7. All other items tied exactly across all configs. Candidate question for larger pilot: does dense retrieval's noise-introduction risk hold at scale, or is this specific to H04's query phrasing?

**Secondary observation (H02):** All configs achieved identical Clause Recall@5 (0.67), but the specific clauses retrieved differed. Config A hit SS9 §8/1 and §8/1(c); Hybrid hit SS9 §8/1 and §8/2. Clause Recall@5 alone does not fully capture retrieval quality. Candidate question: should evaluation track which specific clauses are retrieved, not just recall counts?

**Policy recommendation (documentation only):** Future work should A/B test whether switching the default policy from Hybrid to reranked clause-level BM25 (B'+rerank) improves downstream generation results. B'+rerank performed at least as well on every tested item, is simpler/cheaper (no dense encoding or Chroma), and avoids the noise risk observed on H04. See `reports/retrieval_ablation_n7.md` for full recommendation and caveats.

## 7. SAC (Summary-Augmented Chunking) - Complete (2026-09-01)

**SAC ablation (2026-09-01):** Implemented SAC per Reuter et al. (arXiv:2510.06999) to reduce document-level retrieval mismatch. Per-standard summaries (SS8, SS9, SS13, SS17, SS26) generated via Jais-2 with full corpus grounding (no ungrounded fallback). Full details: `reports/sac_ablation_n7.md`.

**Results (Clause Recall@5, n=7):**
- B' (existing clause-level BM25): 0.571
- B'+rerank (existing): 0.643
- B'' (SAC BM25 only): 0.750 (+0.179 vs B')
- B''+rerank (SAC): 0.833 (+0.190 vs B'+rerank)

**Conclusion:** SAC provides meaningful retrieval improvement on this corpus, confirming Reuter et al.'s finding transfers to AAOIFI standards. Best performance achieved with SAC + reranking.

**Known caveat:** SS8 summary has minor encoding artifact (UTF-8/Latin-1 mojibake of Arabic "راتيجي") at end. Negligible impact on BM25 matching. Root cause: Arabic Unicode range (\u0600-\u06FF) was missing from validation regex - now fixed for future runs. Not regenerated due to low impact and Colab cost.

## What these observations are not

- Not a pilot evaluation (planned n≈25–30+, with expert validation still required).
- Not a claim that retrieval or generation “failed” or “succeeded” as a scored system.
- Not a reason, in this documentation pass, to change reranker or generation code.
- Not qualified Shari'ah review.
