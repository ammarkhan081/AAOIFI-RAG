# Data Card - AAOIFI Grounded Compliance Prototype

## Status

Phase 1 (manifest, schema, data card, private corpus receipt) is complete. Phase 2 lexical and dense retrieval artifacts have been privately built from the five official English AAOIFI PDFs. All source-derived records, chunks, indexes, logs, vector stores, and retrieval outputs remain only in Git-ignored `data/private/`. The Stage 1 hard set has **seven** items (H01–H07), all `reviewed` / `corpus_cross_reference` (not qualified Shari'ah scholar review). An **n=7** Colab T4 end-to-end smoke test (`scripts/colab_e2e_batch_test.py`) is recorded in `reports/e2e_batch_smoke_results.json` (Git-ignored). JSON `aggregates` as stored: abstained **4/7**, answered **3/7**, any gold in top-5 **7/7**, all golds in top-5 **1/7**. Findings: `docs/known_limitations.md` and `reports/smoke_test_batch_n7.md`. That run is **not** the planned pilot evaluation (**n=25–30+**, still future work and still requiring expert validation).

## Purpose and intended use

The planned private corpus will support research on abstention-aware, clause-grounded retrieval for five AAOIFI Shari'ah Standards. It is for research assistance and evaluation only; it is not a fatwa engine, a Shari'ah ruling system, or a financial-product approval service.

## Scope

- **Standards:** SS 8 (Murabahah), SS 9 (Ijarah and Ijarah Muntahia Bittamleek), SS 13 (Mudarabah), SS 17 (Investment Sukuk), and SS 26 (Islamic Insurance).
- **Language:** English only for Stage 1.
- **Source:** Official AAOIFI English PDFs, manually downloaded and placed in `data/private/` by the researcher on 2026-08-26. Filenames and SHA-256 hashes are recorded in `data/manifests/source_manifest.json`.
- **Edition/date:** Unconfirmed for SS 8, SS 13, SS 17, and SS 26. SS 9's cover states `(Revised Standard)`, but its definitive edition date is unconfirmed. The manifest must not infer an edition year from the SS 9 upload path.
- **Official URL:** Complete - canonical AAOIFI URLs are recorded in `data/manifests/source_manifest.json`. SS 9 uses AAOIFI's direct PDF URL; SS 13 uses AAOIFI's canonical `s-13` path.
- **Date added:** 2026-08-26.
- **Added by:** Researcher.
- **Integrity hash:** Complete - SHA-256 hashes are recorded in the source manifest.

## Licence and access status

- **AAOIFI source text:** Publicly accessible according to the project decision, but raw text and PDF files are excluded from public repositories unless redistribution rights are separately confirmed. Exact edition-specific licence or access terms are TBD - pending review of the supplied PDFs.
- **SAHM auxiliary data:** CC BY-NC 4.0 is accepted for this non-commercial research prototype. No SAHM data has been acquired or included. Any future file or documentation that uses SAHM must retain this licence note.
- **Code and non-copyrighted project metadata:** Intended public repository licence is MIT.

## Collection and processing

- **Collection method:** The researcher manually obtained the official PDFs; the project did not scrape, fetch, or download AAOIFI standards.
- **Private storage location:** `data/private/`, which is ignored by Git.
- **Clause extraction and normalisation:** Completed privately for the Stage 1 pilot corpus. The parser removes Contents pages and back matter, handles known wrapped cross-references, and preserves source-faithful duplicate clause identifiers through `occurrence_index`. No synthetic stand-in data has been created.
- **Expert disagreement labels:** Deferred. The schema permits `null` only in Stage 1; no expert-reviewed, simulated, or placeholder tag data exists.

## Excluded material

- All raw AAOIFI standard text and source PDF files from the public repository.
- Any clause excerpts, OCR output, derived chunks, embeddings, vector indexes, and run traces that reproduce protected source text.
- Arabic-language standards and Arabic-language evaluation in Stage 1.
- SAHM data until it is separately acquired under the stated CC BY-NC 4.0 constraint.
- Expert-review labels and disagreement annotations until a qualified review process is approved for a later stage.

## Known Limitations

- The source URLs and file hashes are recorded, but the manifest does not establish that any particular edition is the latest; the edition/date fields are deliberately recorded as unconfirmed.
- The Stage 1 corpus excludes Contents pages, adoption history, appendices, and other back matter by design. Those materials are not available to Stage 1 retrieval and would need a separately approved treatment in Stage 2.
- PDF extraction remains layout-dependent. The Stage 1 parser is validated against known failure modes, but no automated procedure can establish semantic equivalence to the visual PDF; source-page citations and human review remain necessary for high-stakes use.
- Some source clause identifiers are genuinely repeated. `occurrence_index` uniquely addresses such records without rewriting the printed identifier.
- A small number of valid source records are short structural headings (for example, `Indemnity`), rather than standalone normative propositions. They are retained for source fidelity; downstream retrieval and human review should use their surrounding source context before treating them as substantive guidance.
- The source PDFs include legitimate typographic Unicode (for example, curly apostrophes) and occasional source wording or typography that has not been editorially corrected.
- Smoke-test observations from `reports/e2e_batch_smoke_results.json` (n=7, descriptive): abstained 4/7, answered 3/7, any gold in top-5 7/7, all golds in top-5 1/7. Also: garbled/mixed generation on H05; H01 277 vs H03 25 tokens both `answered`; heading at rank 1 only on H02 in this batch. **Frontier baseline comparison (2026-08-30):** A manual test using Claude Sonnet 5 (Extended Thinking mode) with the full 362-clause corpus answered all 7 items correctly (100% answered, 0% abstained). This demonstrates that Jais-2's abstentions are **not attributable to insufficient retrieved evidence** — the leading candidate explanation is Jais-2-specific abstention calibration, not a proven cause (model capability is a confound). See `reports/frontier_baseline_comparison.md` for full details. Candidate questions for a future larger pilot, not conclusions. See `docs/known_limitations.md`.

## Dependencies and deferred work

- No retrieval effectiveness, generation, or pilot-evaluation **claim** has been made. Implementation checks include three throwaway sanity queries, two single-item Colab smoke tests (H07, H02), and one n=7 batch smoke test. See `docs/known_limitations.md`. The n=25–30+ pilot with expert validation remains deferred.
- Dense embeddings and reranking were executed in a private Google Colab T4 runtime because the local coding sandbox cannot transfer the necessary large model binaries. The local private artifacts record the actual run metadata and outputs.
- The required long-context and proprietary frontier baselines remain in scope for the broader research plan; their deferral from Phase 2 does not remove them.
