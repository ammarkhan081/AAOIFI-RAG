# The Retrieval-Stage Blind Spot: Where This Architecture Can and Cannot Abstain

**Date**: 2026-09-04
**Script**: `scripts/audit_retrieval_stage_blind_spot.py` (GPU-free, fully local, re-runnable)
**Machine-readable output**: `reports/retrieval_stage_blind_spot.json`
**Code under test**: `src/aaoifi_rag/reliability/policy.py`, `signals.py`, `text_checks.py`
**Items**: 7 answerable (`data/private/hard_set.jsonl`, Git-ignored) + 25 probes
(`data/probes/unanswerable_probes.jsonl`, tracked) = 32
**Retriever**: BM25 clause-level, top-5 (the probes cannot be retrieved locally with the
dense index — no GPU in this environment; §7 addresses whether that matters)

**Headline**: the shipped retrieval-stage gate battery abstains on **0 of 19**
mechanically-unanswerable questions. Exact 95% CI on its true abstention recall:
**0.000–0.176**. This is the one interval in the project that is informative by its own
stated width criterion, and it is informative because the count is zero at a hard
boundary, *not* because n is large.

The finding is structural, not statistical. §2 gives the mechanism: neither enabled
retrieval-stage gate has a firing condition that a corpus-absent question can satisfy.
The measurement in §3 confirms the mechanism operates as described; it is not the
evidence the claim rests on.

---

## 1. What this report is for, and the one caveat that governs every number in it

The research problem is asymmetric risk: a confidently-wrong answer about a Shari'ah
compliance question costs more than a refusal. An abstention-aware system therefore needs
to know **where in its own pipeline** the capacity to abstain actually lives. This report
localises it, and the answer is narrow: not at the retrieval stage.

> **Stage caveat.** Throughout this report, `decision == "answer"` means only *"no
> retrieval-stage gate objected; proceed to generation"*. It does **not** mean an answer
> was served to a user. Nothing in this audit calls a model — the response-stage gates and
> the end-to-end decision need a GPU and are out of scope here. An item in the
> "unsafe-proceed" cell has reached the generator unchallenged; whether it is ultimately
> answered or refused depends on the model and on the response-stage gates, which is
> precisely the dependency this report exists to expose.

That caveat is not a hedge. It is the finding restated: after this audit, everything
protecting the user on these 19 items sits downstream of the generator.

## 2. The mechanism: both enabled gates have unreachable firing conditions

`describe_gates(PolicyConfig())` reports 9 enabled gates of 12 registered. Five are
registered at `GateStage.RETRIEVAL` and only **two of those five are enabled**. Each enabled
one fails a condition a corpus-absent question does not satisfy:

| Gate | Fails when | Why a corpus-absent question does not trigger it |
|---|---|---|
| `retrieval_non_empty` | `retrieved_count == 0` | BM25 and a dense index both return *k* records for **any** non-empty query. At `top_k=5` over 362 chunks, `retrieved_count` was **5 for all 32 items** — a single distinct value across both classes. The firing condition is unreachable by construction, for every query, not just for these probes. |
| `retrieval_normative` | *every* retrieved record is heading-like | A question about a standard the corpus does not contain still retrieves five ordinary normative clauses — about something else. Absence of the topic does not make the returned prose stop being prose. |

The three disabled retrieval-stage gates are the more telling half of that registry, because
the retrieval stage is where every candidate signal this project has measured has failed:

| Disabled gate | Why | Recorded in |
|---|---|---|
| `lexical_anchoring` | Calibrated against this exact negative class and **rejected at AUC 0.513** — chance. §5 runs the counterfactual anyway. | `reports/anchoring_gate_calibration.md` |
| `reranker_top_1` | Measured and **rejected**: reranker top-1 score does not predict retrieval completeness or response class at n=7. | `reports/reliability_signal_analysis_n7.md` |
| `reranker_margin` | Computed and recorded on every trace but **never analysed against outcomes**. Gating on it would be a guess dressed as a signal. | `configs/reliability/gates_v1.json` |

All 7 response-stage gates, by contrast, are enabled: `response_present`,
`model_did_not_abstain`, `no_self_contradiction`, `citation_integrity`, `script_integrity`,
`no_degenerate_decoding`, `substantive_answer`.

This is why the measurement below is a demonstration rather than a discovery. A gate whose
predicate is `retrieved_count == 0` cannot abstain on a question whose only defect is that
the corpus does not discuss it, because "the corpus does not discuss it" is not a fact the
retriever reports. It returns its five best guesses with no signal that they are bad.

## 3. Shipped policy, 32 items

```
probes n=25:      proceed-to-generation 25, abstained 0, escalated 0
answerable n=7:   proceed-to-generation  7, abstained 0   (any abstention here is a false positive)
triggering gates observed across all 32 items: {None}
```

Gate-by-gate, on the 25 probes:

| Condition | Probes satisfying it |
|---|---|
| `retrieved_count == 0` | **0 / 25** |
| all retrieved records heading-like | **0 / 25** |
| top-1 record heading-like | **0 / 25** |

### Selective risk

Abstention and escalation are computed over **disjoint** partitions and are never pooled;
the 6 `cross_standard_comparison` probes expect *escalate* and are excluded from both the
numerator and the denominator of the abstention figures.

| Cell | Value | Exact 95% CI (Clopper–Pearson) | Width | Informative? |
|---|---|---|---|---|
| **Unsafe-proceed rate** — unanswerable items that reached generation unchallenged | **19 / 19 = 1.000** | 0.824 – 1.000 | 0.176 | **yes** |
| **Abstention recall** | **0 / 19 = 0.000** | 0.000 – 0.176 | 0.176 | **yes** |
| Abstention precision | `None` — undefined, no abstentions issued | — | — | — |
| False abstentions on the answerable class | 0 / 7 | — | — | — |
| Escalation recall (**stipulated**, never pooled) | 0 / 6 = 0.000 | — | — | — |

Read the second row as: *0 of 19 is consistent with a true retrieval-stage abstention recall
anywhere from 0.0% to 17.6%.* The upper bound is what makes this usable. It rules out the
possibility that the retrieval stage catches, say, a quarter of out-of-corpus questions and
this sample simply missed it.

**Why this interval is informative when nothing else in the project is.** Every other
interval this project computes at n=7 spans most of the unit interval and is reported as
uninformative (`ExactInterval.is_informative` is `False`, and `wilson_interval` refuses
outright below n=30). This one is tight for two reasons that have nothing to do with
statistical power: the denominator is 19 rather than 7, because the negative class is larger
than the hard set; and the numerator is **0**, a boundary at which a Clopper–Pearson interval
is one-sided and collapses to `1 − 0.025^(1/19)`. A zero count at a boundary is the cheapest
informative measurement in statistics. Claiming power here would be the error; the honest
statement is that a *bounded* upper limit was purchased by a zero count, and the mechanism in
§2 explains why the count is zero.

## 4. Corroboration from the other direction: where the n=7 abstentions actually came from

The probe audit says the retrieval stage never abstains on unanswerable questions. The n=7
replay of the answerable items says something independently consistent — the abstentions this
system does produce are **all** response-stage:

| Item | Decision | Expected | Triggering gate | Stage |
|---|---|---|---|---|
| H01 | answer | answer | — | — |
| H02 | abstain | answer | `model_did_not_abstain` | **response** |
| H03 | escalate | answer | `substantive_answer` | **response** |
| H04 | abstain | answer | `model_did_not_abstain` | **response** |
| H05 | escalate | answer | `no_self_contradiction` | **response** |
| H06 | abstain | answer | `model_did_not_abstain` | **response** |
| H07 | abstain | answer | `model_did_not_abstain` | **response** |

(`.venv/Scripts/python.exe scripts/replay_n7_router.py --no-write`; gates are named for the
condition they assert, so `model_did_not_abstain` *failing* means the model **did** refuse.)

Zero of the seven items were routed by a retrieval-stage gate. All four abstentions are the
router honouring a refusal the model produced on its own, detected by reading its output.
Two independent measurements over disjoint item classes therefore agree: **100% of this
system's realised abstention capability is model-mediated and response-audited.**

That is not a favourable result, and it should not be read as one. Every one of those four
abstentions is a **false** abstention — all four items expected `answer`. So the only
abstention mechanism this architecture demonstrably has is also the one producing an
over-abstention count of 6 at coverage 0.143. The mechanism is load-bearing and
miscalibrated at the same time, and §6 draws the consequence.

## 5. Counterfactual: would the rejected anchoring gate have closed the blind spot?

`reports/anchoring_gate_calibration.md` establishes 0.25 as the largest IDF-weighted
coverage threshold with a zero false-positive rate on the answerable items. Forcing the
rejected gate on at exactly that threshold answers the counterfactual with a number rather
than a guess:

| | Shipped | Counterfactual (`enable_anchoring_gate=True`, `min_anchor_coverage=0.25`) |
|---|---|---|
| Probes abstained | 0 / 25 | **2 / 25** |
| False abstentions (answerable) | 0 / 7 | 0 / 7 |
| Abstention recall | 0.000 [0.000–0.176] **informative** | 0.105 [0.013–0.331] *uninformative* |
| Abstention precision | `None` | 1.000 [0.158–1.000] *uninformative* |
| Unsafe-proceed | 19 / 19 [0.824–1.000] **informative** | 17 / 19 [0.669–0.987] *uninformative* |

The recall of 0.105 is worth **nothing**, and the reason is visible in the identity of the
two probes that fired:

| Probe | Basis | Coverage | Gate |
|---|---|---|---|
| U-TERM-05 | `term_absent_from_corpus` | 0.169 | `lexical_anchoring` |
| U-TERM-03 | `term_absent_from_corpus` | 0.234 | `lexical_anchoring` |

**Both catches are circular — 2 of 2.** `term_absent_from_corpus` probes were selected *by
having a query term with zero corpus occurrences*, so a detector keyed on rare query terms
is guaranteed to fire on some of them. That is a tautology about how the probes were built,
not a detection. On the 11 non-circular probes — selected by standard or clause membership,
a criterion unrelated to term frequency — the counterfactual gate's recall is **0/11**,
exactly as the calibration sweep predicted.

So the counterfactual does not close the blind spot; it converts an informative zero into an
uninformative small number while adding a gate that would abstain on an answerable item at
any threshold high enough to detect anything (calibration report §3). Enabling it would make
the report look better and the system worse. It stays disabled.

## 6. What follows for the architecture

Three consequences, in decreasing order of confidence.

1. **The reliability layer's job on out-of-corpus questions is auditing a model decision, not
   making its own.** This is now measured rather than assumed. It reframes what the layer is
   for: the 7 response-stage gates (`response_present`, `model_did_not_abstain`,
   `no_self_contradiction`, `citation_integrity`, `script_integrity`,
   `no_degenerate_decoding`, `substantive_answer`) are not a second line of defence behind a
   retrieval filter — on these 19 items they are the *only* line, and they operate on text the
   generator has already produced.
2. **The decisive open measurement is a model-behaviour question, not a retrieval question.**
   Whether this system is safe on corpus-absent questions reduces almost entirely to: *does
   Jais-2-8B-Chat refuse when the retrieved context does not contain the answer?* That needs
   the GPU path and is unmeasured. It is the highest-value single experiment remaining, and
   the probe set now exists to run it against. Until it is run, the honest statement about
   end-to-end abstention on out-of-corpus questions is that it is **unknown**, with a
   measured lower bound of zero contribution from retrieval.
3. **Retrieval-stage abstention in this design would need a semantic signal, and lexical
   overlap is not a cheap substitute for one.** §2 shows the two shipped gates test
   *structural* properties of the context block (is it non-empty, is it prose). Neither can
   test *aboutness*. The one lexical attempt at aboutness failed at chance. The natural
   successor is maximum query-to-chunk cosine similarity from the BGE-M3 index that already
   exists — a topical-presence measure rather than a string-overlap one. Also GPU-gated.

The over-abstention noted in §4 sharpens point 2 rather than softening it. If the model's
refusal behaviour is the only abstention mechanism, then its calibration *is* the system's
calibration, in both directions: 4 false abstentions on 7 answerable items is the same
mechanism seen from the other side. A single GPU run over the 25 probes plus the 7 hard-set
items would characterise both error directions at once, which is why `scripts/colab_probe_eval.py`
is the next artifact and why it must report both cells, not just the favourable one.

## 7. Is the BM25-only retrieval a confound?

The probes have no stored dense/rerank contexts — those were generated on Colab for the 7
hard-set items only, and re-running retrieval for 25 new questions needs the GPU. So this
audit uses BM25 top-5 for both classes. Two reasons the substantive finding does not depend
on that choice:

- **The mechanism is retriever-independent.** `retrieval_non_empty` fails only at
  `retrieved_count == 0`. A dense index, a hybrid, and a reranked hybrid all return *k*
  records for any non-empty query — the same *k*. Improving the ranking changes *which* five
  clauses come back, never *how many*. No retriever over this corpus can make the gate fire
  on a question the corpus does not cover, because no retriever reports absence.
- **The rejected gate's curve is retriever-stable.** The one gate whose behaviour *could*
  have shifted was measured both ways on the answerable items, and the false-positive curve
  under the stored hybrid+rerank top-5 that Jais-2 actually saw has the same shape as under
  BM25 (calibration report §8: 0/7, 1/7, 3/7 at 0.25/0.30/0.67 under both).

What BM25-only *does* limit: the exact identity of the two circular counterfactual firings in
§5, and the per-probe coverage values. Neither is load-bearing for any claim here.

## 8. Limits of this report

- **Pre-generation only.** No model was called. The stage caveat in §1 governs every cell:
  "proceeded to generation" is not "answered". The end-to-end abstention rate on these probes
  is not measured here and is not implied by anything here.
- **The negative class is mechanically labelled**, at verification tier
  `mechanical_corpus_absence`. That tier is strictly weaker than `corpus_cross_reference` and
  is not `qualified_scholar_review`. No probe label is a Shari'ah judgement. What each probe
  asserts is narrow and machine-checkable — a standard is not in the corpus, a clause number
  is not in a standard, a term has zero occurrences — and a test re-verifies each assertion
  against the live corpus rather than trusting the stored label.
- **The 6 escalation probes rest on a stipulated definition** (`cross_standard_comparison_v1`).
  This project decided that cross-standard comparisons need human review; that is not a
  scholar's determination that they do. Reject the stipulation and the escalation row in §3
  disappears. Nothing else in this report changes, because the abstention figures exclude
  those 6 items from both numerator and denominator by construction.
- **n = 19 for the abstention denominator, of which 8 are circular for any lexical signal.**
  The informative interval in §3 is informative for the *shipped* configuration, where no
  gate reads query terms and the circularity is therefore inert. It should not be reused to
  evaluate a lexical gate; §5 is the correct comparison there, and it is uninformative.
- **The two answerable/unanswerable classes were authored by one person** and differ in
  register as well as in answerability (calibration report §5, §7). That confound is fatal to
  a lexical detector and irrelevant to the structural claim in §2, which does not read the
  question text at all.
- **`residual_error_rate` remains uncomputable.** Nothing here distinguishes a fluent,
  well-cited wrong answer from a right one; that needs a correctness judgement the mechanical
  gates cannot make.

## 9. What would change the verdict

Falsifiable targets, so a future run has something to aim at:

1. **Run the probes through the generator** (`scripts/colab_probe_eval.py`, GPU). If Jais-2
   refuses on most of the 19, the architecture is defensible as designed and this report
   documents a deliberate division of labour. If it answers them fluently, the system has a
   real safety gap on out-of-corpus questions and the response-stage gates are the only
   candidate fix. Either outcome is publishable; the current state — unknown — is not.
2. **A semantic retrieval-stage signal.** Maximum query-to-chunk cosine from the existing
   BGE-M3 index, calibrated on the same 32 items with the same circularity discipline. This
   is the only identified route to a retrieval-stage gate that could fire on absence. If its
   AUC against the 11 non-circular probes clears chance by a margin the sample can support,
   §2's structural claim stops being the whole story.
3. **A larger and multi-authored negative class.** 11 non-circular probes cannot distinguish
   AUC 0.51 from 0.65, and probes written by more than one person would break the template
   regularity that produced the spurious 0.792 in the calibration report.
4. **A register-swap experiment.** Paraphrase probes into scenario form and hard-set questions
   into probe register. If register is doing the work, coverage moves without the label
   changing — a direct test of the §8 confound.

---

## Reproducing

```bash
.venv/Scripts/python.exe scripts/audit_retrieval_stage_blind_spot.py
```

Requires the licensed corpus and the persisted BM25 index under `data/private/extracted/`
(both Git-ignored — see `data/README.md`) and `data/private/hard_set.jsonl`. The script exits
2 with a pointer if either is absent. `--no-write` prints without touching
`reports/retrieval_stage_blind_spot.json`.

No AAOIFI clause prose appears in this report or in the JSON. Hard-set question text is
Git-ignored, so answerable items are identified by ID only and no anchor terms are printed
for them; the 25 probe questions are tracked and may be quoted freely.
