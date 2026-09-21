# Lexical Anchoring Gate: Calibration Against a Negative Class — and Its Rejection

**Date**: 2026-09-04
**Script**: `scripts/calibrate_anchoring_gate.py` (GPU-free, fully local, re-runnable)
**Machine-readable output**: `reports/anchoring_gate_calibration.json`
**Code under test**: `src/aaoifi_rag/reliability/anchoring.py`
**Positive class**: the 7 answerable items in `data/private/hard_set.jsonl` (Git-ignored)
**Negative class**: the 25 probes in `data/probes/unanswerable_probes.jsonl` (tracked)
**Corpus**: 362 clause chunks; vocabulary 1,491 folded terms at the default spec

**Headline**: the gate does not work. IDF-weighted query-to-context anchor coverage
separates the 7 answerable items from the 11 non-circular unanswerable probes at
**AUC 0.513** — chance. The gate is implemented, unit-testable and **shipped
disabled**. This report records the measurement that disabled it.

---

## 1. Why the gate was built at all

`retrieval_non_empty` and `retrieval_normative`, the two retrieval-stage gates that
already shipped, **cannot fire on a question the corpus does not cover**. BM25 and a
dense index both return *k* records for *any* query: a question about capital
adequacy returns the five least-bad clauses about something else, with a low score
but a structurally complete context block. So before this work the retrieval layer
contributed *nothing* to abstention on out-of-corpus questions, and every abstention
had to come from the model's own hedging.

Lexical anchoring was the cheapest candidate fix. An *anchor* is a query term the
corpus treats as discriminative — IDF ≥ 2.0 against 362 chunks, not a function word,
≥ 4 characters. Coverage is the fraction of a
question's anchors that appear anywhere on the retrieved context surface, weighted by
each anchor's IDF. Low coverage was hypothesised to mean "the retriever did not find
what was asked about".

## 2. Read the three evidence classes before reading any number

| Class | n | Status |
|---|---|---|
| Answerable hard-set items | 7 | **The number that matters.** Authored independently against real clauses. Any firing is a false positive. |
| `standard_absent_from_corpus` + `clause_absent_from_standard` probes | 11 | **Valid true positives.** Selected by standard/clause membership, a criterion unrelated to term frequency. |
| `term_absent_from_corpus` probes | 8 | **Circular. Not evidence.** Selected *by having a term with 0 corpus occurrences*; a detector keyed on rare terms is guaranteed to fire. Printed only so it is never mistaken for a finding. |
| `cross_standard_comparison` probes | 6 | Expect *escalate*, not abstain. Reported separately; their label rests on a stipulated definition. |

Any statement of the form "the gate catches 82% of unanswerable questions" that is
computed over the 8 term probes is a tautology about how those probes were built. The
honest true-positive rate for this gate is the one on the 11 valid probes.

---

## 3. Threshold sweep, default spec, BM25 top-5 context

A gate fires when IDF-weighted coverage falls *below* the threshold.

| Threshold | Answerable n=7 (false positives) | Valid probes n=11 (true positives) | Term probes n=8 (circular) | Cross-standard n=6 |
|---|---|---|---|---|
| 0.10 | 0/7 — 0.000 | 0/11 — 0.000 | 0/8 — 0.000 | 0/6 — 0.000 |
| 0.20 | 0/7 — 0.000 | 0/11 — 0.000 | 1/8 — 0.125 | 0/6 — 0.000 |
| **0.25** | **0/7 — 0.000** | **0/11 — 0.000** | 2/8 — 0.250 | 0/6 — 0.000 |
| 0.30 | 1/7 — 0.143 | 0/11 — 0.000 | 3/8 — 0.375 | 1/6 — 0.167 |
| 0.34 | 1/7 — 0.143 | 1/11 — 0.091 | 3/8 — 0.375 | 2/6 — 0.333 |
| 0.40 | 1/7 — 0.143 | 1/11 — 0.091 | 5/8 — 0.625 | 4/6 — 0.667 |
| 0.50 | 2/7 — 0.286 | 4/11 — 0.364 | 6/8 — 0.750 | 4/6 — 0.667 |
| 0.60 | 2/7 — 0.286 | 4/11 — 0.364 | 7/8 — 0.875 | 5/6 — 0.833 |
| 0.67 | 3/7 — 0.429 | 4/11 — 0.364 | 8/8 — 1.000 | 5/6 — 0.833 |
| 0.75 | 3/7 — 0.429 | 7/11 — 0.636 | 8/8 — 1.000 | 5/6 — 0.833 |

**Read the bolded row.** 0.25 is the largest threshold with a zero false-positive rate
on the answerable items — and at 0.25 the true-positive rate on the valid probes is
*also* zero. There is no threshold at which this gate detects a genuine out-of-corpus
question without also abstaining on an answerable one. Every row where detection
begins (0.34 and above) is a row where the gate has already sacrificed at least one of
seven answerable items.

## 4. Separability, all three specs

Mann-Whitney AUC, computed by hand in the calibration script (no scipy dependency).
Ties count a half. 0.500 is chance. The probe should score *lower* than an answerable
item, so AUC is P(probe < answerable).

| Spec | Signal | vs valid probes (n=11) | vs cross-standard (n=6) | vs term probes (circular) |
|---|---|---|---|---|
| **default** (fold=T, frame-drop=T) | IDF-weighted coverage | **0.513** | 0.738 | *0.821* |
| default | unweighted coverage | 0.513 | 0.702 | *0.812* |
| default | OOV ratio (inverted) | 0.643 | 0.643 | *0.920* |
| fold=T, frame-drop=**F** | IDF-weighted coverage | 0.597 | 0.738 | *0.821* |
| fold=T, frame-drop=**F** | OOV ratio (inverted) | **0.792** | 0.607 | *0.830* |
| fold=**F**, frame-drop=T | IDF-weighted coverage | 0.513 | 0.810 | *0.804* |
| fold=**F**, frame-drop=T | OOV ratio (inverted) | 0.610 | 0.619 | *0.768* |

*Italicised columns are the circular comparison and are not evidence.*

At n=7 vs n=11 the sampling error on an AUC is enormous; these figures are reported to
make a **lack** of separation unmistakable, not to claim a small one. 0.513 needs no
confidence interval to be uninformative.

---

## 5. A tempting result that was rejected

The `drop_frame_terms=False` row above is the one a less careful write-up would have
led with: **OOV ratio reaches AUC 0.792** against the valid probes, which looks like a
working detector.

It is an artefact of my own authoring habits. Every one of the six
`standard_absent_from_corpus` probe questions begins "What does AAOIFI Shari'ah
Standard No. *N* …", because I wrote all six from one template. The token `aaoifi`
appears in `SYSTEM_PROMPT` for every query and essentially never in clause prose, so
it is out-of-vocabulary by construction — and it appears more often per question in the
probes than in the hard set purely because of that template. The 0.792 is measuring the
stylistic regularity of the probe author, not corpus absence.

`FRAME_TERMS` (which includes `aaoifi` and the question-frame directives *summarise*,
*compare*, *explain*, *state*, …) was therefore added to the default spec, and every
term in it was verified to be uncovered in **both** classes before inclusion, so
removing them cannot pull the measurement toward either label. With the artefact
removed the honest figure is 0.643, and the signal the gate actually reads is 0.513.

**This variant is retained in the report and in the script's variant list precisely so
the rejection is auditable.** It is not a fallback to enable later.

## 6. Per-item coverage

### Answerable items, n=7, BM25 top-5

Anchor terms are withheld here for the same reason `redact_for_publication` withholds
them from the JSON: `reports/` is tracked and hard-set question text is Git-ignored.

| Item | IDF-weighted coverage | Unweighted | OOV ratio | # anchors |
|---|---|---|---|---|
| H07 | 0.266 | 0.375 | 0.312 | 16 |
| H02 | 0.424 | 0.500 | 0.071 | 14 |
| H05 | 0.615 | 0.692 | 0.231 | 13 |
| H03 | 0.801 | 0.875 | 0.250 | 8 |
| H06 | 0.837 | 0.867 | 0.000 | 15 |
| H01 | 0.877 | 0.889 | 0.056 | 18 |
| H04 | 1.000 | 1.000 | 0.100 | 10 |

### Probes, n=25, BM25 top-5 (terms shown — probe questions are published)

All 25, default spec, ascending coverage.

| Probe | Coverage | Basis | # anchors | Uncovered anchors |
|---|---|---|---|---|
| U-TERM-05 | 0.169 | term | 5 | control, record, maintain, blockchain |
| U-TERM-03 | 0.234 | term | 6 | organis, tawarruq, liquidity, instrument |
| U-TERM-07 | 0.294 | term | 5 | adequacy, ratio, exposure |
| E-CROSS-01 | 0.297 | cross | 7 | treat, institution', pass, client |
| E-CROSS-04 | 0.314 | cross | 8 | financ, work, index, murabahah, ground |
| U-STD-06 | 0.315 | standard | 7 | permit, arrangement, preced, paragraph |
| U-TERM-06 | 0.367 | term | 4 | recognis, measur |
| E-CROSS-06 | 0.368 | cross | 14 | operator, want, head, office, acros, relevant, constrain |
| E-CROSS-02 | 0.372 | cross | 8 | based, structur, standard', requirement |
| U-TERM-02 | 0.389 | term | 4 | waqf, endowment |
| U-STD-02 | 0.411 | standard | 4 | permissibility, condition |
| U-STD-03 | 0.428 | standard | 5 | disclosur, client |
| U-TERM-01 | 0.429 | term | 4 | zakat, calculat |
| U-STD-04 | 0.473 | standard | 6 | govern, early, termination |
| E-CROSS-03 | 0.518 | cross | 13 | structur, standard', interact, return |
| U-TERM-08 | 0.547 | term | 3 | disclosur |
| U-TERM-04 | 0.631 | term | 4 | cryptocurrency |
| U-CLAUSE-03 | 0.671 | clause | 4 | allocat |
| U-CLAUSE-04 | 0.683 | clause | 7 | impose, issuer |
| U-CLAUSE-01 | 0.698 | clause | 4 | require |
| U-STD-05 | 0.760 | standard | 4 | provide |
| E-CROSS-05 | 0.850 | cross | 9 | apply |
| U-STD-01 | 1.000 | standard | 4 | — |
| U-CLAUSE-02 | 1.000 | clause | 4 | — |
| U-CLAUSE-05 | 1.000 | clause | 5 | — |

Stems are the output of `fold_suffix`, which is deliberately crude (`organised → organis`,
`cities → citi`, and `fees → fees` because stripping would leave only three characters).
The stem need not be a word: both sides of every comparison go
through the same function.

---

## 7. Diagnosis: what the signal actually measures

Put the two tables side by side and the failure is legible. Three probes whose subject
matter is **provably absent** from the corpus score ≥ 0.67 — higher than five of the
seven answerable items. Two answerable items score below every probe except three.

The signal is measuring **question register and vocabulary match, not corpus coverage.**

1. **Answerable questions written as scenarios are punished.** H07 and H02, the two
   worst-scoring answerable items, are narrative questions about a concrete situation.
   They carry ordinary English scenario verbs and nouns that no clause happens to use,
   and each of those counts against coverage exactly as a genuinely absent topic term
   would. H07 has 16 anchors and 6 uncovered; the corpus contains its subject matter in
   full — it was the *only* item at n=7 with complete gold retrieval
   (`reports/reliability_signal_analysis_n7.md`).
2. **Synonymy is invisible to a lexical measure.** Common-usage vocabulary for an
   entity the standards name differently registers as out-of-vocabulary. The measure
   has no way to know the two words denote the same thing.
3. **Terse probes that reuse clause words are rewarded.** A probe that names a real
   standard and a plausible-sounding clause number in formal register reuses corpus
   vocabulary almost entirely, so it scores near 1.000. `U-CLAUSE-05` scores exactly
   1.000 while asking about a clause that does not exist in SS26 — every anchor it
   carries is present in the context, because the context is *from SS26*.
4. **The provenance-header fix removed the only thing that was working by accident.**
   The first calibration run computed coverage against `record["text"]` alone, so a
   standard id such as `SS17` counted as uncovered for **every** item. That depressed
   coverage uniformly and, more importantly, was simply wrong:
   `aaoifi_rag.generation.prompt.build_context_text` renders a provenance header
   (`SS8 §2/4/2 occ=0 page=7 id=clause:SS8:…`) above every excerpt, so the identifiers
   *are* on the surface the model sees. Fixing the measurement removed a spurious
   uniform penalty; it did not create separation, because the penalty was uniform
   across both classes. It also does **not** recover clause *numbers* — see §12.

The honest conclusion is structural, not a tuning problem: **the discriminating variable
between an answerable and an unanswerable question in this corpus is semantic, and a
bag-of-stems overlap cannot see it.** No threshold, stop-list or weighting scheme
recovers a signal that is not there.

## 8. Retriever sensitivity

The finding is not an artefact of using BM25 to retrieve for the probes (the probes can
only be retrieved locally, without a GPU, by BM25). The false-positive curve on the
seven answerable items under the **stored hybrid+rerank top-5 that Jais-2 actually saw**
is the same shape:

| Threshold | BM25 top-5 | Stored hybrid+rerank top-5 |
|---|---|---|
| 0.25 | 0/7 | 0/7 |
| 0.30 | 1/7 | 1/7 |
| 0.50 | 2/7 | 1/7 |
| 0.67 | 3/7 | 3/7 |
| 0.75 | 3/7 | 3/7 |

A better retriever does not rescue the gate. It cannot: the uncovered anchors on the
worst items are words that appear nowhere in the corpus, so no ranking over that corpus
can cover them.

---

## 9. Decision: implemented, registered, shipped disabled

This is the second signal this project has measured and rejected. The first was reranker
top-1 score (`reports/reliability_signal_analysis_n7.md`). The handling is deliberately
identical, so that a reader can see the rejection in the shipped artefacts rather than
only in prose:

| Where | What it says |
|---|---|
| `src/aaoifi_rag/reliability/anchoring.py` | The signal, with the circularity warning and the three honest limits stated before any number in the module docstring. |
| `src/aaoifi_rag/reliability/policy.py` | `EvidentiaryBasis.REJECTED_PROBES = "rejected_by_probe_calibration"`; the `lexical_anchoring` gate registered at `GateStage.RETRIEVAL` with `RouteDecision.ABSTAIN`; `min_anchor_coverage = None` and `enable_anchoring_gate = False`, with AUC 0.513 cited in the field comment. |
| `configs/reliability/gates_v1.json` | `"min_anchor_coverage": null`, `"enable_anchoring_gate": false`, with the reason and the 0.25 starting point recorded in `_min_anchor_coverage`. |
| `src/aaoifi_rag/reliability/signals.py` | `RetrievalSignals.anchoring`, recorded on every item when a corpus vocabulary is supplied, emitted as counts and ratios only, with `anchoring_status` naming this report. |

The gate is *registered* rather than deleted for two reasons. It is measured
infrastructure — re-enabling it at larger *n* is a config change, not a re-implementation
— and a disabled gate in `describe_gates()` output is a standing, machine-readable record
that the idea was tried and failed. `describe_gates(PolicyConfig())` currently reports
**9 enabled of 12 registered**.

**Verified inert.** `python scripts/replay_n7_router.py --no-write` after registration
reproduces the n=7 routing exactly: `answer 1 / abstain 4 / escalate 2`, coverage 0.143,
routing agreement 1/7, clause recall@5 macro 0.571429 / micro 0.529412 (9/17),
over-abstention 6.

## 10. What would change the verdict

Stated so a future run has a falsifiable target rather than a vague aspiration:

- **A semantic anchoring signal.** Maximum cosine similarity between the query embedding
  and each retrieved chunk embedding, from the BGE-M3 index that already exists. This
  measures topical presence rather than string overlap and is the natural successor.
  Needs the GPU path, so it is out of scope here.
- **A larger, more varied negative class.** 11 valid probes cannot distinguish AUC 0.51
  from AUC 0.65. Probes authored by more than one person would also break the template
  regularity that produced the spurious 0.792.
- **Answerable items in probe register, and probes in scenario register.** The confound
  identified in §7 is testable directly: if register is doing the work, paraphrasing a
  probe into narrative form should *raise* its coverage without changing its label.

## 11. Limits of this report

- n=7 answerable and n=11 valid probes. Nothing here is a statistically significant
  finding, and no interval is quoted for that reason.
- The negative class is mechanically labelled. `mechanical_corpus_absence` and
  `stipulated_definition` are both strictly weaker than `corpus_cross_reference` and
  neither is `qualified_scholar_review`. No probe label is a Shari'ah judgement.
- The six `cross_standard_comparison` probes expect *escalate* under a **stipulated**
  definition of what requires human review. Reject the stipulation and the
  cross-standard column disappears; nothing else in this report does.
- The measurement is lexical only, and says nothing about whether a question the gate
  passes is actually answerable from the retrieved clauses.

---

## 12. A tokeniser defect found while writing the tests — and why it changes nothing

`tests/test_anchoring.py` was written to pin the tokeniser against
`BM25_TOKEN_RE`, and doing so surfaced a defect in the shipped module *comment*, not
in the arithmetic above.

`ANCHOR_TOKEN_RE` is `[\w]+(?:['’][\w]+)?|\d+(?:/\d+)*`. Python's `|` is **ordered**,
and `[\w]+` matches at every position where `\d+` would, so **the slash-numeral branch
is unreachable**:

```
anchor_tokens("clause 2/4/2 applies") -> ['clause', '2', '4', '2', 'applies']
```

Each of those digits is then dropped by the bare-digit rule in `_is_candidate`. So a
question citing a clause number contributes **no anchor at all** from that citation, and
`AnchorSpec.keep_slash_numerals` is dead code through the public path. `BM25_TOKEN_RE`
has the identical alternation order, so the two tokenisers *are* genuinely in sync —
which is the whole point of the duplication. It is only the claim that *either* keeps
`9/2` whole that was false. §1 of this report and three comments in
`anchoring.py` asserted it; all four are corrected (§13).

### The counterfactual was measured, not assumed

`scripts/diagnose_slash_tokeniser.py` re-runs the default-spec calibration twice —
once as shipped, once with the branches swapped so `2/4/2` survives as one token — and
diffs them. Re-runnable, GPU-free, writes nothing.

| | shipped | slash-first |
|---|---|---|
| Corpus vocabulary | 1,491 folded terms | 1,511 |
| **AUC, answerable n=7 vs valid probes n=11** | **0.513** | **0.656** |
| False positives @ 0.25 / 0.30 / 0.50 / 0.67 / 0.75 | 0 / 1 / 2 / 3 / 3 of 7 | **identical** |
| True positives @ 0.25 / 0.30 / 0.50 / 0.67 / 0.75 | 0 / 0 / 4 / 4 / 7 of 11 | 0 / 0 / 4 / **8** / **9** |

AUC by probe basis, and the exact set of items that move:

| Basis | AUC shipped → slash-first | Probes that move |
|---|---|---|
| `clause_absent_from_standard` (n=5) | 0.371 → **0.657** | all five |
| `standard_absent_from_corpus` (n=6) | 0.631 → 0.655 | `U-STD-05` only |
| `term_absent_from_corpus` (n=8, circular) | 0.821 → 0.821 | none |
| `cross_standard_comparison` (n=6) | 0.738 → 0.738 | none |
| **Answerable hard-set items (n=7)** | — | **none, on either BM25 or the stored hybrid+rerank context** |

Six of 25 probes move, and every one of them is a probe whose question cites a clause
number: `U-CLAUSE-01` 0.698→0.509, `U-CLAUSE-02` 1.000→0.733, `U-CLAUSE-03`
0.671→0.505, `U-CLAUSE-04` 0.684→0.547, `U-CLAUSE-05` 1.000→0.766, `U-STD-05`
0.760→0.545. Not one answerable item changes by any amount.

### Three reasons the regex is left exactly as it is

1. **The verdict does not move.** After the patch the only zero-false-positive
   threshold is still 0.25, and at 0.25 the true-positive rate on the valid probes is
   still **0/11**. There is still no threshold that detects a genuine out-of-corpus
   question without abstaining on an answerable one. AUC 0.656 at n=7 vs n=11 is not a
   working detector either; it is a slightly less flat chance line.
2. **The gain is a different signal in disguise, and its false-positive rate is
   *unmeasured*, not zero.** The entire improvement is "the clause number you cited is
   not on the retrieved surface". That fires on none of the seven answerable items only
   because none of their questions cites a clause number — so n=0 for the positive
   class. Gold retrieval was *incomplete on 3 of 7 items*
   (`reports/reliability_signal_analysis_n7.md`), so an answerable question that cited a
   real clause the retriever missed would fire it. Reporting 0.656 as an improvement
   would be quoting a true-positive rate with no false-positive rate to set against it —
   the same error §2 of this report exists to prevent.
3. **It would break the BM25 sync and invalidate the index.** `BM25_TOKEN_RE` built the
   persisted index in `data/private/extracted/bm25_clause_level.pkl`. Changing anchoring
   alone desynchronises the two; changing both requires rebuilding the index and
   re-measuring every retrieval number in `reports/retrieval_ablation_n7.md`. Neither is
   justified by a change that does not alter a conclusion.

**Where the clause-reference evidence belongs instead.** The correct home for "did the
model cite a clause it was actually shown?" is a response-stage grounding check on the
*answer's* citations, where the reference set is explicit and the false-positive rate is
directly measurable — not a repair to a bag-of-stems retrieval-stage measure. That is
implemented separately and is not this gate.

---

## 13. Corrections to this report

Both were found by re-deriving published numbers from the tracked JSON and from live
code while writing `tests/test_anchoring.py`. Neither changes a conclusion. Recorded
here rather than silently edited.

**2026-09-04 — §4, AUC table, one cell.** The row `fold=F, frame-drop=T` /
IDF-weighted coverage / *vs term probes (circular)* was published as **0.762** and is
**0.804**. 0.7619 is that variant's *unweighted*-coverage-vs-*cross-standard* AUC — a
cell this report does not print — so the figure had been transcribed from the wrong
computed line. Corrected in place. The cell sits in the column this report itself marks
"not evidence", and the six other rows of that table reproduce exactly to three
decimals, which is how the outlier was found. `tests/test_anchoring.py` now pins all
seven rows against the tracked per-item coverages, so a future transcription slip fails
a test.

**2026-09-04 — §6, `fold_suffix` examples.** The example `policies → polic` was wrong:
the `ies` branch re-appends a `y`, so the output is `policy`. Replaced with two measured
examples — `cities → citi`, and `fees → fees` because stripping would leave a
three-character stem, below `min_stem=4`. The same wrong example was in
`anchoring.py`'s docstring and is corrected there too. No coverage number changes: the
function's behaviour was always `policy` and every published figure was measured under
it.

**2026-09-04 — §1 and §7.4, slash-numeral claims.** §1 listed "or a slash-numeral
clause reference" as a way to qualify as an anchor, and §7.4 said `SS17` *and* `2/4/2`
had counted as uncovered before the provenance-header fix. A clause number is never an
anchor on either side of the comparison (§12), so the first was wrong and the second was
half wrong — the header fix recovers the standard id plus the literal words `clause` and
`occurrence` from `chunk_id`, and never the clause number. Both corrected. The
`context_metadata_keys` fix itself is unaffected and still changes coverage on exactly
three of the seven answerable items (H03 0.801→0.602, H04 1.000→0.836, H05 0.615→0.505
when the metadata surface is removed; H01, H02, H06 and H07 are unchanged by it).





