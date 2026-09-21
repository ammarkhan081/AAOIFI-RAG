# Pilot hard-set authoring guide

The hard set is a private, human-authored pilot artifact. It is not synthetic expert review, a model-generated dataset, or a source of publishable claims. The Stage 1 pilot target is 25–30 items and is exploratory only.

## Before writing an item

1. Work from the received private source PDFs and their extracted clause records, not from memory, web summaries, or invented fact patterns.
2. Locate every clause or sub-clause that the item relies on. Record its source-faithful `standard_id`, `clause_id`, and `occurrence_index`; include `sub_clause_id` when applicable.
3. Keep all hard-set content in `data/private/hard_set.jsonl`. Do not put questions, answers, or clause excerpts in a public repository.

## Choosing items

Use a deliberate mix:

- Straightforward factual items whose answer is directly supported by one clause or a small, clearly connected set of clauses.
- A smaller number of genuinely ambiguous items, where the supplied text does not by itself support a safe direct answer.
- A smaller number of disagreement-prone items, where qualified human interpretation or review is the appropriate next step.

Avoid questions based on imagined transactions, unstated facts, external law, or assumptions not present in the cited source material. Do not force every item to be answerable: abstention and escalation cases are required parts of the research design.

## Classifying and routing items

- Use `answerable` with expected behavior `answer` only when the recorded clauses directly support the requested response.
- Use `genuinely_ambiguous` with `abstain` when the source basis is insufficient to answer safely.
- Use `disagreement_prone` with `escalate` when a human scholar's review is required rather than an autonomous system conclusion.

The labels describe the expected prototype behavior for research evaluation. They are not fatwas, Shari'ah rulings, or substitutes for qualified review.

## Quality check before adding a line

1. Confirm the item has a unique `item_id` and validates against `data/manifests/hard_set_schema.json`.
2. Confirm every gold clause reference resolves to a private extracted record, including the correct duplicate `occurrence_index`.
3. Confirm the notes explain why the item belongs in the pilot and why its expected behavior is appropriate.
4. Keep the pilot provisional: do not report its composition or outcomes as final, benchmark, or publication-grade evidence.
