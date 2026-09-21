# Open questions and decision status

**Updated:** 2026-08-29

## Resolved from the original Phase 0 list

1. ~~What may be committed publicly, and under what code licence?~~ Resolved: public GitHub repository; MIT for code and permitted metadata; no raw AAOIFI text in public repositories.
2. ~~Which Stage 1 language scope applies?~~ Resolved: English only.
3. ~~Is SAHM acceptable for this project?~~ Resolved: yes, for non-commercial research under CC BY-NC 4.0, with the licence note retained wherever it is used.
4. ~~Are scholar validation and disagreement labels required in Stage 1?~~ Resolved: no; only a nullable future mechanism is permitted, with no placeholder data.
5. ~~What compute may be assumed?~~ Resolved: free-tier Google Colab or Kaggle, T4-class GPU, 16 GB VRAM maximum.
6. ~~Should Phase 1 assume a Hugging Face token or accepted model licence?~~ Resolved: no; the researcher will provide them later. The target model identifier has been independently verified as current.
7. ~~Should the frontier baseline be scaffolded now?~~ Resolved: no; it is deferred outside Phase 1 and Phase 2.
8. ~~Are additional data-security controls or institutional ethics review required?~~ Resolved: no, beyond the Git exclusions and public-artifact boundary already recorded.
9. ~~Which tracking system should be used?~~ Resolved: JSONL-only; no MLflow or W&B.
10. ~~Which timeline governs the first milestone?~~ Resolved: four weeks; three months is the broader program outer bound.
11. ~~Which orchestration approach should be used?~~ Resolved: a plain explicit Python state machine; LangGraph is excluded.

## Still pending from the original Phase 0 list

1. ~~What are the canonical official AAOIFI URLs for SS 8, SS 9, SS 13, SS 17, and SS 26?~~ Resolved: all five URLs are recorded in the source manifest. The edition/publication-date status is intentionally `unconfirmed`; SS 9 records its cover's exact `(Revised Standard)` wording without inferring a year from its URL.
2. ~~What SHA-256 hash and local add date apply to each real PDF?~~ Resolved: all five PDFs were recorded in the source manifest on 2026-08-26.
3. What immutable commit/revision of `inception42/Jais-2-8B-Chat` should be recorded for future experiments? **Hub HEAD SHA recorded** on 2026-08-29 via unauthenticated `huggingface_hub.HfApi().model_info("inception42/Jais-2-8B-Chat")`: `da0e1639cd92b508b24120f7f77f5270a8465dc4` (`lastModified` 2026-08-18 09:59:48+00:00). Details and caveats: `docs/verified_environment.md`. A Hugging Face token was **not** required for this metadata lookup. **Still open:** whether Colab/`from_pretrained` should **pin** that SHA (`revision=`) so later Hub updates cannot silently change the snapshot. The 2026-08-29 Colab loads did not pass `revision=`.

## Genuinely new questions from Phase 1

None. The remaining items above are unresolved metadata or access dependencies from the original list, not new design questions.
