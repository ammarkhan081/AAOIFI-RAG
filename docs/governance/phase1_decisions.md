# Phase 1 decision log

**Recorded:** 2026-08-26  
**Status:** Active for the four-week Stage 1 prototype scope.

1. AAOIFI source PDFs will be manually downloaded by the researcher from the official AAOIFI site and stored only under `data/private/`. The project must never scrape, fetch, or publicly commit them. Public artifacts are limited to manifests, hashes, scripts, schemas, and other allowed metadata.
2. Stage 1 uses current official English AAOIFI editions. Exact edition and publication-date metadata will be entered only after the researcher supplies each PDF.
3. Stage 1 is English-only; Arabic inputs and evaluation are deferred.
4. The intended repository is public GitHub under the MIT licence for code. This does not permit publication of raw AAOIFI text.
5. SAHM is acceptable only for this non-commercial prototype under CC BY-NC 4.0. Its licence note must remain visible wherever SAHM is later used.
6. Scholar validation and real disagreement labels are outside Stage 1. The clause schema reserves a nullable field for future qualified review, but no simulated, expert-reviewed, or placeholder tags may be created.
7. Compute is limited to free-tier Google Colab or Kaggle with a T4-class GPU and a 16 GB VRAM ceiling. No greater resource is assumed.
8. The target model identifier is [`inception42/Jais-2-8B-Chat`](https://huggingface.co/inception42/Jais-2-8B-Chat). Its public Hugging Face page was checked on 2026-08-26 and the identifier exists, but access requires acceptance of its conditions. The exact immutable model commit/revision remains TBD - pending the researcher's token and licence acceptance; it must be recorded before any experiment. The page also contains stale usage examples naming `inceptionai`, so implementation must use the verified repository identifier rather than copying those examples.
9. The proprietary frontier baseline is deferred outside Phase 1 and Phase 2. No provider choice, API credential, or integration scaffold is created now.
10. No special data-security control is required beyond the existing Git exclusions and the no-public-raw-text boundary.
11. Tracking is JSONL-only. MLflow and Weights & Biases are not part of the dependency setup.
12. No ethics or IRB review is required for this independent Stage 1 prototype.
13. The operative target is the four-week Stage 1 scope. The plan's three-month figure is the broader program's outer bound, not a current blocker.
14. Orchestration will be a plain, explicit Python state machine. LangGraph is intentionally excluded because the committed workflow is single-pass answer-or-escalate and does not presently need graph-runtime features.

## Implementation boundary

This decision log authorizes Phase 1 setup only. It does not authorize retrieval, chunking, embedding, generation, routing, evaluation, model download, or ingestion of source text before the researcher adds the official PDFs.
