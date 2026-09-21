# Process-behaviour classification (n=7 batch), plan error taxonomy

**Source:** `reports/e2e_batch_smoke_results.json` (top-5 ids, `gold_clause_hits`, `model_response`, `response_class`). Gold *wording* for the items lives in `data/private/hard_set.jsonl` (not in the JSON). This note does **not** judge legal correctness or Shari'ah; items remain `corpus_cross_reference` only.

**Not** the n=25–30+ pilot. Descriptive n=7 only.

## Taxonomy labels (as listed in the task)

| Category | How it is used here (process only) |
| --- | --- |
| retrieval miss | A listed gold clause is absent from that item’s top-5. |
| retrieval mismatch | Top-5 ranking/type is off (e.g. section heading at rank 1; a non-gold record above the gold). |
| context misuse | Supportive text was in the prompt (same retrieved records) but generation ignored, truncated, or garbled it. |
| citation mismatch | Cited ids/ranks do not match what was retrieved (not used as primary here). |
| domain ambiguity | Question or source does not support a safe process decision (not used as primary; no scholar labels). |
| abstention failure | Answered (or mixed-answered) when a clean abstain would have been the instructed behaviour for insufficient context — **not** used for the four exact-line abstentions. |
| over-abstention | Exact abstention line despite gold clause(s) in the same top-5 / prompt. |
| prompt-injection / instruction-conflict | Output fights the “reply exactly …” instruction (e.g. other text plus the abstention line). |

The task’s parenthetical that called H02/H04/H06/H07 “abstention failure (should have answered)” is labelled **over-abstention** here, so it is not collapsed with “failed to abstain.”

`response_class` is the script’s exact-line rule, copied from the JSON.

## Table

| item_id | response_class | primary_error_category | secondary_category | reasoning |
| --- | --- | --- | --- | --- |
| H01 | answered | retrieval miss | (none) | JSON: 2/3 golds in top-5 (SS8 §2/4/4 rank 1, §2/4/2 rank 5); gold §2/4/1 not in top-5. `model_response` restates those two retrieved records and answers the syndicated arrangement-fee side of the question. Process: generation used the golds that were present; the commitment-fee gold was a retrieval miss. Not scored as citation mismatch (ranks [1] and [5] match the prompt labels). |
| H02 | abstained | over-abstention | retrieval mismatch; retrieval miss | Exact abstention line. 2/4 golds in top-5 (§8/1 occ 0 rank 3, §8/2 rank 5); §8/1(c) and §8/3 missing. Rank 1 is `clause:SS9:8:occurrence:0` (heading), same pattern as the earlier H02 single-item run. The parent gold §8/1 was still in the prompt, so this is over-abstention as a behaviour pattern, not a full retrieval miss. **Resists a single label:** missing §8/1(c)/§8/3 could support “caution,” but that is not a taxonomy bucket. |
| H03 | answered | context misuse | retrieval miss | JSON: gold SS13 §6 at rank 1; gold §3/2 not in top-5. Rank-1 chunk text (corpus) includes both the “permitted to obtain guarantees” sentence **and** the enforcement-limitation sentence. `model_response` is 25 tokens quoting only the first of those. Process: incomplete use of a record that was retrieved. §3/2 is a retrieval miss. **Not** labelled “correct”: no scholar review; only “thin use of the same top-1 record.” |
| H04 | abstained | over-abstention | retrieval miss | Exact abstention line. Gold SS17 §5/2/2 (the prospectus nominal-value *purchase* undertaking) is rank 1; gold §5/1/8/7 not in top-5. The question was rescoped to that issuer purchase rule. Process: the load-bearing gold was in the prompt at rank 1 and the model still used the exact abstention line. |
| H05 | answered | context misuse | instruction-conflict; retrieval miss; *cross-lingual artifact (not in list)* | Script-`answered`. Gold §4/2 at rank 1; golds §2 and §5/5 not in top-5. `model_response` garbles the rank-1 English clause with Arabic `وأخرجه` / `تهنئة`, then a paragraph that matches retrieved SS26 §3/1 (JSON rank 4), then a trailing abstention line. Mixed instruction-following = prompt-injection/instruction-conflict. Cross-lingual tokens are **outside** the listed taxonomy (generation artifact, not a retrieval miss of Arabic source). |
| H06 | abstained | over-abstention | retrieval miss | Exact abstention line. Gold SS8 §5/6 at rank 1; gold §5/8 not in top-5. §5/6 is the delay-donation / not-for-the-Institution rule the question asks. Process: that gold was in the prompt at rank 1 and the model still abstained. |
| H07 | abstained | over-abstention | retrieval mismatch | Exact abstention line. Sole gold SS9 §8/5 in top-5 at rank 2 (`all_gold_clauses_in_top5=true`). Rank 1 is SS9 §3/2 (substantive, not a heading). Process: gold was in the prompt and the model still used the exact abstention line. Secondary: a non-gold record outranked the gold. |

## Items that resist a clean single category

- **H02:** over-abstention vs “legitimate caution” because only 2/4 golds were in top-5 and rank 1 is a heading. Both process facts are in the JSON; they are listed as primary + secondary rather than forced into one name.
- **H03:** “acceptable minimal answer” would be a correctness/coverage judgment. Not assigned. Process label is context misuse of rank-1 SS13 §6.
- **H05:** does not fit one plan category; context misuse + instruction-conflict + retrieval miss + an extra **cross-lingual generation artifact**.

## What this table is not

- Not scholar-validated correctness.
- Not a claim that “enough context” equals a unique gold_answer.
- Not a reason to change retrieval or generation code in this pass.
