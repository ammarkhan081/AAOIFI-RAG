# Gate Sensitivity and Ablation at n=7: What the Battery Actually Does

**Date**: 2026-09-04
**Script**: `scripts/audit_gate_sensitivity.py` (GPU-free, fully local, re-runnable)
**Machine-readable output**: `reports/gate_sensitivity_n7.json`
**Code under test**: `src/aaoifi_rag/reliability/sensitivity.py`, `policy.py`
**Evidence**: the stored n=7 Colab run, replayed with **frozen** `ReliabilitySignals` —
identical inputs to `scripts/replay_n7_router.py`, so the baseline routing matches exactly
(`H01=answer, H02=abstain, H03=escalate, H04=abstain, H05=escalate, H06=abstain, H07=abstain`)

**Two headlines.**

1. `max_echo_ratio` can move across **46% of its entire range** — anywhere in
   [0.443, 0.900] — without changing a single decision. Only **3 of 7** items are
   evaluable by it at all, because only three produced an answer attempt. The threshold is
   not calibrated; it is *placed*, and this report measures how much room it was placed in.
2. **7 of 9 enabled gates are unexercised at n=7.** Removing any of them changes nothing.
   Three of those seven *do fire* — all on H05, all routing to the same destination — so
   they are masked rather than idle, and H05's escalation is consequently the most robust
   decision the system makes: it survives the removal of any single gate.

Neither headline says the battery is badly built. Both say the same thing about the
evidence: seven items, four of which the model refused, cannot exercise nine gates.

---

## 1. Method, and why the sweep is exact rather than sampled

Both analyses re-route **frozen signals**. Signals are computed once per item from the
stored context and the stored model response; then thresholds are varied and gates are
removed without recomputing anything. No model is called, no retrieval is re-run, and
every number is therefore reproducible on a laptop.

A threshold gate over a fixed signal value is a *step function* of the threshold.
`substantive_answer` passes iff `echo_ratio < max_echo_ratio`, so for an item with ratio
*r* the gate's state changes at exactly `threshold == r` and nowhere else. The complete
set of transition points for *n* items is therefore the set of *n* observed values.
`sweep_threshold` visits each observed value and each value plus one epsilon, so it
**enumerates every distinct routing the policy can produce** rather than sampling a grid.
A grid would be slower and could miss a flip between its points.

## 2. `max_echo_ratio`: the whole reachable range

Three items produced an answer attempt and so have an echo ratio the gate can read:
**H01 0.443038, H05 0.717391, H03 0.900000**. The other four refused, and
`_check_substantive_answer` short-circuits to *passed* on a non-answer — a refusal has
nothing to transcribe. Those four contribute no transition point, and including them would
invent a flip the threshold cannot cause.

| Threshold | answer | abstain | escalate | agrees with reference labels |
|---|---|---|---|---|
| 0.000000000 | 0 | 4 | 3 | 0 / 7 |
| 0.443037975 | 0 | 4 | 3 | 0 / 7 |
| 0.443037976 | 1 | 4 | 2 | 1 / 7 |
| 0.717391304 | 1 | 4 | 2 | 1 / 7 |
| 0.717391305 | 1 | 4 | 2 | 1 / 7 |
| **0.850000000 (shipped)** | **1** | **4** | **2** | **1 / 7** |
| 0.900000000 | 1 | 4 | 2 | 1 / 7 |
| 0.900000001 | 2 | 4 | 1 | 2 / 7 |
| 1.000000000 | 2 | 4 | 1 | 2 / 7 |

Nine rows is the complete enumeration — the entire space of behaviours this threshold can
produce on this evidence. Three distinct routings exist across [0, 1].

**Stable interval around the shipped value: [0.443038, 0.900000], width 0.456962.**

Only two decisions ever flip:

| Item | Flip | At threshold |
|---|---|---|
| H01 | escalate → answer | 0.443038 |
| H03 | escalate → answer | 0.900000 |

### What that means, stated plainly

The `configs/reliability/gates_v1.json` comment says 0.85 "sits in the gap between the only
three observed answer attempts … Three points. Provisional and the single most likely
threshold to move once n grows." This report quantifies "provisional": the value could have
been 0.45, 0.60, 0.75 or 0.89 and every decision on every item would be identical. It is
constrained from below by H01 (0.443) and from above by H03 (0.900), and *nothing* in
between distinguishes any choice. 0.85 is defensible as a midpoint-ish placement in a gap
of width 0.457; it is not a calibration, and this table is the reason the config comment
says so.

### The uncomfortable column

Read the agreement column again. At any threshold above 0.900 — that is, with the echo gate
effectively off — agreement with the reference labels is **2/7**. At the shipped 0.85 it is
**1/7**. The echo gate's *only* effect at n=7 is to convert H03 from `answer` to
`escalate`, and H03's reference label is `answer`. On this evidence the gate costs one
agreement point and buys nothing measurable.

That is a genuine finding and it is reported here rather than buried, but it does not by
itself condemn the gate, because the two readings are not decidable at n=3:

- **Against the gate**: it fires once, on an item a human labelled answerable, and the
  reference label is the closest thing to ground truth available.
- **For the gate**: H03's answer is 90.0% verbatim eight-word shingles from the retrieved
  context. The plan's own position is that transcription is not answering, and an
  escalation on a 90%-verbatim response is the gate doing exactly what it was specified to
  do. The reference label says the *question* was answerable, not that *this* response
  answered it well.

The honest conclusion is that the disagreement is between a mechanical property of the
response and a label about the question, and that one item cannot adjudicate it. What
would: any item whose response is heavily verbatim *and* wrong (vindicating the gate), or a
second heavily-verbatim response that is a perfectly good answer (indicting it). Neither
exists at n=7. The threshold stays at 0.85 and this section stays in the report.

## 3. Gate ablation: all 512 subsets

Nine gates are enabled, so all 2^9 = 512 subsets were evaluated over the 7 items. Disabled
gates (`lexical_anchoring`, `reranker_top_1`, `reranker_margin`) are held present in every
subset because they always report *passed* and cannot affect a decision. Gate precedence
order is preserved in every subset, because which gate is named as the cause — and
therefore which `decision_if_failed` wins — depends on it.

**8 distinct routings are reachable from 512 subsets.** The battery has low effective
dimensionality on this evidence.

| Gate | Stage | Fires on | Decisions changed if removed | Shift |
|---|---|---|---|---|
| `retrieval_non_empty` | retrieval | 0 | 0 | — |
| `retrieval_normative` | retrieval | 0 | 0 | — |
| `response_present` | response | 0 | 0 | — |
| **`model_did_not_abstain`** | response | **4** | **4** | H02, H04, H06, H07: abstain → answer |
| `no_self_contradiction` | response | 1 | **0** | — (masked) |
| `citation_integrity` | response | 1 | **0** | — (masked) |
| `script_integrity` | response | 1 | **0** | — (masked) |
| `no_degenerate_decoding` | response | 0 | 0 | — |
| **`substantive_answer`** | response | 1 | **1** | H03: escalate → answer |

Two columns, deliberately separate. "Fires on" counts items where the gate's condition
failed. "Decisions changed if removed" counts items whose final decision differs when the
gate is deleted. A gate can fire and change nothing.

### Masked, not idle: the H05 trio

`no_self_contradiction`, `citation_integrity` and `script_integrity` each fire on exactly
one item — the same one — and all three route to `ESCALATE`:

| Gate on H05 | Detail |
|---|---|
| `no_self_contradiction` (named as cause) | `response_class=mixed residue_words=88` |
| `citation_integrity` | `[2]` misattributed — its text matches rank 4 at 0.87 better than the rank it cites |
| `script_integrity` | Arabic-script characters present that are absent from the English context |

Removing any one leaves the other two, and the decision is unchanged. That is defence in
depth behaving exactly as intended, and it makes **H05's escalation the most robust decision
in the run**: three mechanically independent checks, on three unrelated properties of the
text, agree. No other item has redundant grounds for its routing.

It also means "0 decisions changed" must not be read as "contributed nothing". The correct
reading of a masked gate is *this item would still have been escalated without it* — which
is a property of H05, not a verdict on the gate.

### Minimal sufficient subsets

The smallest subset of gates that reproduces the shipped routing on all 7 items has size
**3**, and there are exactly **3 such subsets**:

```
{model_did_not_abstain, no_self_contradiction, substantive_answer}
{model_did_not_abstain, citation_integrity,    substantive_answer}
{model_did_not_abstain, script_integrity,      substantive_answer}
```

The structure is legible: you need `model_did_not_abstain` (the only gate touching four
items), `substantive_answer` (the only gate touching H03), and **any one** of the three
H05-catching gates. The enumeration recovers the redundancy structure of §3 automatically —
it was not asserted.

## 4. Cross-check against the retrieval-stage blind spot

`retrieval_non_empty` and `retrieval_normative` fire on **0 of 7** answerable items here.
`reports/retrieval_stage_blind_spot.md` independently found they fire on **0 of 25** probes.
Two disjoint item classes, one conclusion: the retrieval-stage gates fire on nothing
available to this project. All 32 items retrieved exactly 5 records and none returned an
all-heading context block.

That is the same finding from the opposite direction, and it is why `model_did_not_abstain`
— a gate that reads the model's own output — is the single most load-bearing component in
the routing layer. Four of the seven decisions at n=7 are it, and it is the only mechanism
by which this architecture has been observed to abstain at all.

## 5. What this does and does not license

**Licensed by this report:**

- The claim that `max_echo_ratio` is unconstrained across [0.443, 0.900] on the available
  evidence. This is exact, not estimated.
- The claim that 7 of 9 enabled gates are unexercised at n=7, and the finer claim that 3 of
  those 7 fire but are masked.
- The claim that H05's escalation is over-determined and H03's is not.
- The claim that the echo gate costs one point of label agreement at n=7.

**Not licensed:**

- That any unexercised gate is redundant, removable, or badly designed. An unexercised gate
  is one whose failure mode did not occur in seven items. `no_degenerate_decoding` in
  particular is a prompt-contract guard against a failure greedy decoding did not produce —
  its zero is the expected result, not a disappointment.
- That 0.85 is a good threshold, or that any other value in the stable interval is better.
  Choosing the midpoint of the stable interval would be tuning a hyperparameter on the same
  seven items the system is evaluated on. `sensitivity.py` deliberately does not recommend a
  value.
- Any statistical claim. n=7, of which 3 are evaluable by the echo gate and 1 exercises it.
  No interval is quoted because none would be informative.

## 6. What would change these numbers

1. **More answer attempts.** The echo sweep has three points because the model refused four
   of seven questions. Items the model actually answers are what constrain this threshold,
   and the negative-control probes (`data/probes/unanswerable_probes.jsonl`) will supply
   more of them only if the model answers those too — which is itself the open question in
   `reports/retrieval_stage_blind_spot.md` §9.
2. **Any item with an empty or heading-only retrieval.** The two retrieval-stage gates have
   never been observed firing. A synthetic item constructed to trip them would at least
   confirm they work; a unit test is the cheaper form of that confirmation and belongs in
   `tests/`, not in an evaluation set.
3. **A second heavily-verbatim response.** One point at 0.900 is why §2 cannot be resolved.
   A second such response, with a known-good or known-bad answer attached, decides it.
4. **A run with sampling enabled.** `no_degenerate_decoding` guards a failure mode greedy
   decoding cannot produce. Its zero is uninformative by construction under the shipped
   decoding config.

## 7. Limits

- Every figure is conditional on the frozen signals supplied. A gate ablation cannot
  discover failure modes absent from the item set, and at n=7 most gates are unexercised by
  construction rather than by any property of the gates.
- The reference labels are `expected_behavior` from `data/private/hard_set.jsonl`, authored
  before the run. All seven expect `answer`, so the agreement column is measuring
  over-abstention only and cannot detect the opposite error.
- Gate precedence determines which gate is *named*, and the leave-one-out analysis inherits
  that ordering. A different declaration order would produce different "named cause"
  attributions on H05 while producing the identical decision.
- No AAOIFI clause prose appears in this report or in the JSON. Gate `detail` strings are
  omitted from the emitted payload; they can contain fragments of the model response.

---

## Reproducing

```bash
.venv/Scripts/python.exe scripts/audit_gate_sensitivity.py
```

Reads the stored n=7 run and the clause corpus under `data/private/` (Git-ignored — see
`data/README.md`) and writes `reports/gate_sensitivity_n7.json`. `--no-write` prints only.
The baseline routing printed on the first line must match `scripts/replay_n7_router.py`
exactly; a divergence means the frozen signals and the pipeline have drifted apart.
