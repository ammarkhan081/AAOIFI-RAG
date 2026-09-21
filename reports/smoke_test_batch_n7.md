# n=7 Colab batch smoke test (H01–H07)

**Source of every number below:** `reports/e2e_batch_smoke_results.json` (Colab `/content/e2e_batch_smoke_results.json`, copied into this folder). Not re-derived.

**Script:** `scripts/colab_e2e_batch_test.py`  
**Label in the JSON:** 7-item Colab smoke test, not the n=25–30+ pilot evaluation. Counts are descriptive only. No statistical significance is claimed.

**Hard-set verification:** all seven items `status=reviewed`, `verification_basis=corpus_cross_reference`. That is **not** qualified Shari'ah scholar review. `gold_answer_leak_guard_passed` is `true` on all seven.

**JSON field `aggregates` (copied as stored):** `n_abstained` 4, `n_answered` 3, `n_any_gold_clause_in_top5` 7, `n_all_gold_clauses_in_top5` 1 (`4/7`, `3/7`, `7/7`, `1/7`). Item-level `response_class` counts match those four/three. All seven have `any_gold_clause_in_top5=true`. Only **H07** has `all_gold_clauses_in_top5=true`.

**Publication:** `top5` holds chunk/clause ids, ranks, and reranker scores only (no clause `text` field). **`model_response` for H01, H03, and H05 quotes or restates retrieved clause wording.** Treat the JSON as private. It is Git-ignored. This markdown table does not copy those clause bodies; full strings stay in the ignored JSON.

`response_class` is exact `.strip()` match to `I cannot answer from the given context` → `abstained`, else `answered`. H05 is `answered` under that rule (mixed string). `reranker_top1` is `items[].top5[0].reranker_score` as stored.

**Generation block (JSON):** `inception42/Jais-2-8B-Chat`, `hub_revision` `da0e1639cd92b508b24120f7f77f5270a8465dc4`, `do_sample=false`, `jais2_load_seconds` 1085.7, `vram_model_gb` 5.66.

## Summary table (from the JSON)

| item_id | n_gold_clauses | n_gold_clauses_in_top5 | any_gold | all_golds | response_class | reranker_top1 | generation_seconds | new_tokens | excerpt (no clause bodies) |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| H01 | 3 | 2 | true | false | answered | 10.099871635437012 | 34.21 | 277 | Script-`answered`. Restates retrieved SS8 §2/4/4 (rank 1) and §2/4/2 (rank 5), then a Yes. Gold §2/4/1 not in top-5. |
| H02 | 4 | 2 | true | false | abstained | 5.180835723876953 | 5.6 | 8 | Exact abstention line. Rank 1 is `clause:SS9:8:occurrence:0` (section heading). Golds in top-5: §8/1 occ 0 rank 3; §8/2 rank 5. |
| H03 | 2 | 1 | true | false | answered | 7.9021100997924805 | 8.01 | 25 | Script-`answered`. 25-token fragment of rank-1 SS13 §6 only. Gold §3/2 not in top-5. |
| H04 | 2 | 1 | true | false | abstained | 7.276059150695801 | 4.5 | 8 | Exact abstention line. Gold §5/2/2 at rank 1. Gold §5/1/8/7 not in top-5. |
| H05 | 3 | 1 | true | false | answered | 4.981423854827881 | 17.18 | 117 | Script-`answered`. Mixed: partial English plus trailing abstention line; Arabic tokens `وأخرجه`, `تهنئة`. Gold §4/2 at rank 1; §2 and §5/5 not in top-5. |
| H06 | 2 | 1 | true | false | abstained | 7.337094783782959 | 5.91 | 8 | Exact abstention line. Gold §5/6 at rank 1. Gold §5/8 not in top-5. |
| H07 | 1 | 1 | true | **true** | abstained | 3.780673027038574 | 5.47 | 8 | Exact abstention line. Sole gold §8/5 at rank 2. Rank 1 is SS9 §3/2 (not a heading). |

## Findings (n=7 descriptive; see `docs/known_limitations.md`)

1. **Garbled / cross-lingual tokens (H05):** `model_response` contains `وأخرجه` and `تهنئة`.
2. **Mixed completion (H05):** same `model_response` ends with `I cannot answer from the given context` after other text; `response_class` remains `answered`.
3. **Heading at rank 1:** H02 only (`clause_id` `8`). Other six items’ rank-1 `clause_id` values: `2/4/4`, `6`, `5/2/2`, `4/2`, `5/6`, `3/2`.
4. **`answered` is not one quality:** H01 `new_tokens` 277 vs H03 `new_tokens` 25, both `answered`.

## Raw JSON

Kept at `reports/e2e_batch_smoke_results.json` (Git-ignored). Do not commit it.
