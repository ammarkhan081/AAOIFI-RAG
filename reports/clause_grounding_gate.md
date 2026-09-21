# Clause-Reference Grounding Gate: A Mechanically Certain Check, and Why It Ships Enabled

**Date**: 2026-09-06
**Code under test**: `src/aaoifi_rag/reliability/grounding.py`
**Wired into**: `src/aaoifi_rag/reliability/policy.py` (gate `clause_grounding`, stage RESPONSE, routes ESCALATE)
**Config switch**: `configs/reliability/gates_v1.json` → `enable_clause_grounding_gate: true`
**Tests**: `tests/test_grounding.py` (65 total; 7 corpus-gated under `requires_private_data`)
**Corpus**: 362 clause chunks across SS 8, 9, 13, 17, 26 (Git-ignored, `data/private/`)

**Headline**: unlike the lexical-anchoring gate, this one has nothing to calibrate.
It answers a decidable question — *did the answer cite a clause path it was actually
shown?* — with no threshold and no score. It is the prose-citation companion to
`citation_integrity`, which checks only the `[n]` rank markers. It fires on **0 of the
7** stored items, so enabling it moves no published decision; what was measured instead
is the *extractor's* precision, and that is what this report records.

> This gate performs no Shari'ah reasoning. It is a mechanical string check over clause
> identifiers the model wrote versus clause identifiers retrieval returned. Nothing here
> is a fatwa, a ruling, or a compliance approval. This report contains no AAOIFI clause
> prose.

---

## 1. What the gate decides

An answer may name a clause path in running prose — "as required by SS 8 §2/4/1" —
without ever emitting a `[n]` citation marker for it. `citation_integrity` cannot see
that reference; it audits only the bracketed rank markers and their attribution. The
grounding gate closes that gap. It extracts every clause path the response names, binds
each to a standard, and asks whether that clause identity is present in the retrieved
context.

Resolution is layered, strictest first:

- **exact** — the exact `(standard, clause_path)` appears among the retrieved records.
- **ancestor** — a retrieved child clause grounds its named parent (one-directional:
  a retrieved parent does *not* ground a named child).
- **verbatim_in_context** — the path string occurs in the clause **body** text of the
  context (headers excluded, so a provenance line printing a clause's own path cannot
  self-ground a different standard).
- **corpus_exact** — resolvable only against the full corpus index, i.e. a real clause
  that was *not* retrieved. Diagnostic; does not ground.
- **no_match / standard_not_in_corpus** — unresolvable, or a standard the corpus does
  not contain.

`FAILING_STATUSES` = `{OUTSIDE_CONTEXT, UNKNOWN_STANDARD, UNRESOLVABLE}` — every
non-grounded status. The gate **fails closed to ESCALATE**: a reference the system
cannot show it was given is a reference a human should look at, not one the machine
should serve.

---

## 2. Why there is no threshold

The anchoring gate needed a calibration report because "how much lexical overlap is
enough" is a continuous judgement, and the measurement found the judgement worthless
(AUC 0.513). Grounding has no such knob. A clause path either matches a retrieved
identity or it does not; the verdict is set-membership, not a cutoff. The one design
decision that *could* have gone wrong — whether the optional corpus index changes
outcomes — was tested directly (§5) and does not.

`MIN_PATH_DEPTH = 2`: a bare depth-1 reference ("item 8") is not extracted. Measured
cost of that choice on this corpus and on the gold answers: **zero** (§4).

---

## 3. Extractor precision on the corpus — 36/36

`test_the_extractor_takes_exactly_the_measured_number_of_corpus_references`

Run over all **362** clause texts, the extractor takes exactly **36** genuine
cross-references and nothing else. Every one is a real clause cross-reference; there is
no over-extraction — in particular, no Mudarabah profit-ratio fragment ("60/40") is
mistaken for a clause path.

`test_no_slash_numeral_in_the_corpus_falls_outside_an_extracted_span` confirms the
other direction: **no** slash-numeral anywhere in the 362 texts is missed by an
extracted span (`missed == []`). So on the corpus the extractor is exact both ways —
36 taken, none of them spurious, none of the genuine ones dropped.

---

## 4. Depth-1 references cost nothing

`test_depth_one_references_cost_nothing_on_this_corpus`

With `MIN_PATH_DEPTH = 2`, depth-1 references are deliberately not extracted. On both
the corpus and the gold answers the set of depth-1 hits that this excludes is empty
(`corpus_hits == []`, `gold_hits == []`). The minimum-depth rule removes a class of
false positives at no measured cost to true coverage.

---

## 5. The corpus index refines the diagnosis but never changes the verdict

`test_the_verdict_is_index_independent_on_every_real_text`

For every real text — 7 gold answers plus the 7 stored model responses — the gate's
pass/fail verdict is identical whether or not a corpus index is supplied
(`with_index.passed == without.passed` on all 14). The index only sharpens the *label*
on a failing reference (turning a generic `UNRESOLVABLE` into the more specific
`corpus_exact`-adjacent diagnosis). This is the property that makes the index optional,
and it is asserted, not assumed.

This distinction was a genuine design correction during implementation: an early draft
let the index flip a SS 12 reference from failing (no index) to passing
`UNKNOWN_STANDARD` (with index). That conflated "was this shown?" — decidable from
context alone — with "does this exist in AAOIFI?" — which needs the corpus. Making
`UNKNOWN_STANDARD` a failing status restored index-independence.

---

## 6. Behaviour on the stored run — fires on nothing

`test_the_stored_model_run_makes_two_grounded_references_and_no_others`

| Item | Raw refs extracted | Distinct | Failing |
|------|-------------------:|---------:|--------:|
| H01  | 7 | 2 | 0 |
| H02–H07 | 0 | 0 | 0 |

Only H01 names any clause path (7 raw mentions collapsing to 2 distinct references),
and **both** distinct references are grounded in its retrieved context. `flagged == []`:
enabling this gate moves **no** published routing decision. That is why the n=7 replay
regenerated cleanly with 0 of 7 decisions changed.

The error rate on context-only served answers is therefore 0/2 — uninformative, and
recorded as such rather than dressed up as a validated false-positive rate.

---

## 7. The gold answers fail — correctly

`test_every_gold_answer_reference_resolves_to_a_real_clause`

Across the 7 gold answers: **12** references grounded in context, **6** outside context,
**0** unresolvable, **0** unknown-standard. **6 of 7** gold answers carry at least one
reference that would trip the gate.

This is not a false-positive rate. Every one of the 6 out-of-context references resolves
to a *real* clause via `corpus_exact` — a clause that retrieval did not return. The gold
answers were authored with the full standards open; they cite clauses the retrieval
stage never surfaced. The gate flagging them is the correct verdict on a context-only
system: those answers are **not** context-only answers, and the flag says exactly that.
H07 is the sole gold answer that stays entirely within its retrieved context.

The `no_match` / cross-standard idiom is real in the wild:
`test_the_cross_standard_prose_idiom_occurs_and_names_standards_outside_the_corpus`
finds the "Shari'ah Standard No. (N)" prose form occurring **7** times, naming standards
outside the 5-standard corpus — exactly the reference class the gate is built to catch.

---

## 8. Status

Shipped **enabled** in `gates_v1`. It is the tenth enabled gate (2 retrieval + 8
response), leaving 3 gates registered-but-disabled (`lexical_anchoring`,
`reranker_top_1`, `reranker_margin`). Because it fires on none of the 7 stored items,
the published traces and the sensitivity/ablation artefacts were regenerated with the
gate live and every routing decision unchanged; it appears in the ablation lattice as an
**unexercised** gate at n=7 — present in the enumeration, load-bearing on zero items.

**Known limitation.** "Fires on 0/7, error rate 0/2 on context-only answers" is a
statement about seven items, not a guarantee. The gate's *correctness* is mechanical and
does not depend on n; its *usefulness* — how often a real deployed model cites a clause
it was not shown — is unmeasured beyond this hard set and awaits a larger evaluation.
