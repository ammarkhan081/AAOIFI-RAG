"""Metrics: exact intervals, the selective-risk cross-tab, and trace reproducibility.

Three properties are worth more than the rest of this file put together, and each is
pinned against something outside the module under test:

* **The exact intervals are checked against closed forms, not against themselves.** For
  ``x = 0``, ``x = n``, ``x = 1`` and ``x = n-1`` the Clopper-Pearson bound has an
  algebraic solution, so those cases are asserted against a formula written out
  independently in :func:`_closed_form_bounds`. Interior cases are checked against the
  *defining property* of the interval - that the exact binomial tail at the bound equals
  ``alpha/2`` - rather than against a number copied out of a previous run. A bisection
  that converged to the wrong root would pass a self-comparison and fail both of these.
* **Escalation is never pooled with abstention.** The escalation numbers rest on a
  stipulated definition this project authored; the abstention numbers rest on mechanical
  corpus absence. :data:`ESCALATION_KEYS` exists to keep them apart and the tests below
  assert the separation structurally, not by convention.
* **The published n=7 numbers are reproducible from the published traces.**
  ``reports/replay_n7_public_traces.jsonl`` carries no clause text, no prompt and no
  response - only hashes. :func:`metrics_from_traces` over that file must reproduce
  ``reports/replay_n7_metrics.json`` field for field. That is the reproducibility claim
  in the module docstring, tested rather than asserted in prose.

No AAOIFI clause prose appears here. The two report artifacts this file reads are the
tracked, redacted ones; the ``item_id`` values (H01-H07) are identifiers, not content.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from aaoifi_rag.reporting.evaluation import (
    EvaluationLabel,
    GoldClauseHit,
    RetrievalScore,
)
from aaoifi_rag.reporting.metrics import (
    ESCALATION_KEYS,
    EXPECTED_BEHAVIORS,
    MAX_INFORMATIVE_WIDTH,
    MIN_N_FOR_INTERVALS,
    NOT_COMPUTABLE,
    InsufficientSampleError,
    RoutingMetrics,
    clopper_pearson_interval,
    compute_routing_metrics,
    compute_selective_risk,
    metrics_from_traces,
    wilson_interval,
)

#: 95% two-sided, so each tail carries 2.5%.
TAIL = 0.025

#: The shipped blind-spot audit: 25 probes + 7 answerable items, every one of which the
#: retrieval-stage gates let through. Source: ``reports/retrieval_stage_blind_spot.json``.
BLIND_SPOT_ABSTAIN_EXPECTED = 19
BLIND_SPOT_ESCALATE_EXPECTED = 6
BLIND_SPOT_ANSWER_EXPECTED = 7


def _closed_form_bounds(successes: int, n: int) -> tuple[float | None, float | None]:
    """Algebraic Clopper-Pearson bounds for the four cases that have one.

    Derived here rather than imported so that the assertion has an independent source:

    * ``P(X >= 1 | p) = 1 - (1-p)^n = tail``  ->  lower for ``x = 1``.
    * ``P(X <= n-1 | p) = 1 - p^n = tail``    ->  upper for ``x = n-1``.
    * ``x = 0`` and ``x = n`` are the degenerate ends of the same two identities.

    Returns ``(lower, upper)`` with ``None`` where no closed form applies.
    """
    lower = upper = None
    if successes == 0:
        lower, upper = 0.0, 1.0 - TAIL ** (1.0 / n)
    elif successes == n:
        lower, upper = TAIL ** (1.0 / n), 1.0
    if successes == 1:
        lower = 1.0 - (1.0 - TAIL) ** (1.0 / n)
    if successes == n - 1:
        upper = (1.0 - TAIL) ** (1.0 / n)
    return lower, upper


def _binomial_tail(n: int, p: float, lo: int, hi: int) -> float:
    """Independent reimplementation of the exact binomial sum used for verification."""
    return sum(
        math.comb(n, i) * p**i * (1.0 - p) ** (n - i) for i in range(lo, hi + 1)
    )


def _label(
    item_id: str,
    expected: str,
    decision: str,
    *,
    retrieval: RetrievalScore | None = None,
) -> EvaluationLabel:
    return EvaluationLabel(
        item_id=item_id,
        expected_behavior=expected,
        answerability="answerable" if expected == "answer" else "unanswerable_from_corpus",
        verification_basis="corpus_cross_reference",
        status="draft",
        decision=decision,
        retrieval=retrieval,
    )


def _score(*in_context: bool, context_size: int = 5) -> RetrievalScore:
    """A retrieval score with one gold clause per flag."""
    return RetrievalScore(
        hits=tuple(
            GoldClauseHit(
                standard_id="XX",
                clause_id=f"1/{index}",
                occurrence_index=0,
                sub_clause_id=None,
                in_context=flag,
                rank=index + 1 if flag else None,
            )
            for index, flag in enumerate(in_context)
        ),
        context_size=context_size,
    )


# --------------------------------------------------------------------------------------
# Wilson: the interval the project refuses to compute
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 7, 25, 29])
def test_wilson_refuses_below_the_minimum_sample(n: int) -> None:
    """Refusal is the feature. n=7 supports counts, not rates.

    ``InsufficientSampleError`` subclasses ``ValueError`` so a caller that only guards
    ``ValueError`` still cannot proceed on a fabricated interval.
    """
    with pytest.raises(InsufficientSampleError, match=f"n={n} is below min_n=30"):
        wilson_interval(0, n)
    assert issubclass(InsufficientSampleError, ValueError)


def test_wilson_computes_at_the_threshold_and_above() -> None:
    assert MIN_N_FOR_INTERVALS == 30
    lower, upper = wilson_interval(15, MIN_N_FOR_INTERVALS)
    assert 0.0 < lower < 0.5 < upper < 1.0
    # A symmetric count gives an interval centred on 0.5.
    assert lower + upper == pytest.approx(1.0, abs=1e-12)


def test_wilson_is_mirror_symmetric() -> None:
    """``k/n`` and ``(n-k)/n`` must give mirrored intervals.

    A genuine property of the Wilson formula and a cheap check on the algebra: an error
    in the continuity term or the denominator breaks the mirror.
    """
    low_a, high_a = wilson_interval(7, 30)
    low_b, high_b = wilson_interval(23, 30)
    assert low_a == pytest.approx(1.0 - high_b, abs=1e-12)
    assert high_a == pytest.approx(1.0 - low_b, abs=1e-12)


def test_wilson_brackets_the_point_estimate_and_stays_in_range() -> None:
    for successes in range(0, 31):
        lower, upper = wilson_interval(successes, 30)
        assert 0.0 <= lower <= successes / 30 <= upper <= 1.0


@pytest.mark.parametrize(
    ("successes", "n", "match"),
    [
        (31, 30, "out of range"),
        (-1, 30, "out of range"),
        (0, 0, "n must be positive"),
        (0, -5, "n must be positive"),
    ],
)
def test_wilson_rejects_impossible_inputs(successes: int, n: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        wilson_interval(successes, n)


def test_wilson_min_n_is_an_explicit_override_not_a_default() -> None:
    """The guard can be lifted, but only by naming ``min_n`` at the call site.

    Kept deliberately possible: a future n=30+ evaluation should not have to edit the
    module, and a reader can see in the caller that the refusal was overridden.
    """
    lower, upper = wilson_interval(1, 7, min_n=1)
    assert 0.0 < lower < 1.0 / 7 < upper < 1.0


# --------------------------------------------------------------------------------------
# Clopper-Pearson: checked against closed forms and the defining property
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [7, 19, 25, 32])
@pytest.mark.parametrize("offset", [0, 1])
def test_exact_interval_matches_the_closed_form_at_the_ends(n: int, offset: int) -> None:
    """``x`` in ``{0, 1, n-1, n}``: assert the algebraic solution, not a stored number."""
    for successes in (offset, n - offset):
        interval = clopper_pearson_interval(successes, n)
        lower, upper = _closed_form_bounds(successes, n)
        assert lower is not None or upper is not None, "no closed form for this case"
        if lower is not None:
            assert interval.lower == pytest.approx(lower, abs=1e-12)
        if upper is not None:
            assert interval.upper == pytest.approx(upper, abs=1e-12)


@pytest.mark.parametrize(("successes", "n"), [(2, 25), (3, 7), (9, 17), (10, 32)])
def test_exact_interval_satisfies_its_defining_property(successes: int, n: int) -> None:
    """At the lower bound ``P(X >= x) = alpha/2``; at the upper, ``P(X <= x) = alpha/2``.

    This is what "exact" means. Checking it directly catches a bisection that converged
    on the wrong tail, which comparing against the module's own previous output cannot.
    """
    interval = clopper_pearson_interval(successes, n)
    assert _binomial_tail(n, interval.lower, successes, n) == pytest.approx(
        TAIL, abs=1e-9
    )
    assert _binomial_tail(n, interval.upper, 0, successes) == pytest.approx(
        TAIL, abs=1e-9
    )
    assert interval.lower < successes / n < interval.upper


def test_exact_interval_reproduces_the_two_published_blind_spot_intervals() -> None:
    """The two figures ``reports/retrieval_stage_blind_spot.md`` quotes.

    ``0/19`` and ``19/19`` are the abstention-recall and unsafe-answer-rate intervals of
    the shipped policy. Both are informative *because* they sit against a boundary: a
    rate of zero out of nineteen excludes everything above 17.6%, which is enough to
    support the report's claim without any appeal to significance.
    """
    recall = clopper_pearson_interval(0, 19).as_dict()
    assert (recall["lower"], recall["upper"], recall["width"]) == (0.0, 0.176467, 0.176467)
    assert recall["is_informative"] is True
    assert recall["method"] == "clopper_pearson_exact"

    unsafe = clopper_pearson_interval(19, 19).as_dict()
    assert (unsafe["lower"], unsafe["upper"], unsafe["width"]) == (0.823533, 1.0, 0.176467)
    assert unsafe["is_informative"] is True
    # The two bounds are the same number reflected, which is why the widths match.
    assert recall["upper"] == pytest.approx(1.0 - unsafe["lower"], abs=1e-6)


def test_exact_interval_at_one_of_seven_reports_itself_uninformative() -> None:
    """The case the module docstring describes as "roughly 0.4% to 58%".

    An interval spanning more than half the unit line cannot separate "rarely" from
    "usually", so ``is_informative`` is False and ``as_dict`` says in words that the
    point estimate must not be quoted as a rate. That sentence is the guard rail against
    a reader turning 1/7 into "14%".
    """
    payload = clopper_pearson_interval(1, 7).as_dict()
    assert payload["lower"] == pytest.approx(0.003610, abs=1e-6)
    assert payload["upper"] == pytest.approx(0.578723, abs=1e-6)
    assert payload["is_informative"] is False
    assert payload["width"] > MAX_INFORMATIVE_WIDTH
    assert "must not be quoted as a rate" in payload["interpretation"]


def test_informativeness_is_a_width_test_at_the_stated_threshold() -> None:
    """``width <= max_informative_width``, inclusive, and overridable per call."""
    assert MAX_INFORMATIVE_WIDTH == 0.20
    interval = clopper_pearson_interval(0, 19)
    assert interval.width == pytest.approx(0.176467, abs=1e-6)
    assert clopper_pearson_interval(0, 19, max_informative_width=0.17).is_informative is False
    assert clopper_pearson_interval(0, 19, max_informative_width=interval.width).is_informative


def test_exact_interval_narrows_as_n_grows_at_a_fixed_proportion() -> None:
    """Half of n successes: the interval must shrink monotonically in n.

    Nothing subtle, but it is the property that makes the n=7 refusal a statement about
    sample size rather than about the estimator.
    """
    widths = [clopper_pearson_interval(n // 2, n).width for n in (8, 20, 40, 100, 400)]
    assert widths == sorted(widths, reverse=True)
    assert widths[0] > MAX_INFORMATIVE_WIDTH > widths[-1]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"successes": 0, "n": 0}, "n must be positive"),
        ({"successes": 8, "n": 7}, "out of range"),
        ({"successes": -1, "n": 7}, "out of range"),
    ],
)
def test_exact_interval_rejects_impossible_inputs(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        clopper_pearson_interval(**kwargs)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5])
def test_exact_interval_rejects_confidence_outside_the_open_unit_interval(
    confidence: float,
) -> None:
    with pytest.raises(ValueError, match="confidence must be in"):
        clopper_pearson_interval(1, 7, confidence=confidence)


def test_higher_confidence_gives_a_wider_interval() -> None:
    widths = [clopper_pearson_interval(3, 25, confidence=c).width for c in (0.80, 0.95, 0.99)]
    assert widths == sorted(widths)


def test_exact_interval_needs_no_scientific_stack() -> None:
    """No scipy, no numpy: the reporting layer must import on a bare interpreter.

    ``test_import_boundaries.py`` covers the heavy ML dependencies; this covers the
    quieter temptation, since an exact binomial interval is the one place a reporting
    module would reach for ``scipy.stats``.
    """
    import sys

    import aaoifi_rag.reporting.metrics as module

    clopper_pearson_interval(3, 11)
    assert "scipy" not in sys.modules
    assert "numpy" not in sys.modules
    assert not hasattr(module, "scipy")


# --------------------------------------------------------------------------------------
# Routing metrics
# --------------------------------------------------------------------------------------


def test_routing_metrics_count_decisions_and_rates() -> None:
    labels = [
        _label("A1", "answer", "answer"),
        _label("A2", "answer", "abstain"),
        _label("A3", "answer", "escalate"),
        _label("P1", "abstain", "abstain"),
    ]
    metrics = compute_routing_metrics(labels)
    assert (metrics.n_items, metrics.n_answer, metrics.n_abstain, metrics.n_escalate) == (
        4,
        1,
        2,
        1,
    )
    assert metrics.coverage == 0.25
    assert metrics.abstention_rate == 0.5
    assert metrics.escalation_rate == 0.25
    assert metrics.n_decision_matches_expected == 2
    assert metrics.routing_agreement == 0.5
    # Over-abstention counts answerable items that did not answer, whether they were
    # abstained or escalated: both withhold an answer from a question the corpus covers.
    assert metrics.n_over_abstention == 2


def test_triggering_gate_counts_sum_to_n_items_with_none_bucketed() -> None:
    """A passing item has no triggering gate; ``None`` becomes ``"none"``, not a drop."""
    labels = [_label(f"A{i}", "answer", "answer") for i in range(4)]
    metrics = compute_routing_metrics(
        labels, triggering_gates=[None, "citation_integrity", None, "script_integrity"]
    )
    assert metrics.triggering_gates == {
        "none": 2,
        "citation_integrity": 1,
        "script_integrity": 1,
    }
    assert sum(metrics.triggering_gates.values()) == metrics.n_items


def test_misaligned_triggering_gates_raise() -> None:
    """Positional alignment is the only thing tying a gate to an item here.

    Silently zipping to the shorter sequence would attribute gates to the wrong items and
    still produce a plausible-looking table.
    """
    labels = [_label("A1", "answer", "answer"), _label("A2", "answer", "abstain")]
    with pytest.raises(ValueError, match="must be positionally aligned"):
        compute_routing_metrics(labels, triggering_gates=["citation_integrity"])


def test_recall_is_none_when_no_label_carries_a_retrieval_score() -> None:
    """Unmeasured recall and zero recall are different claims.

    ``0.0`` in a results table reads as "retrieved nothing relevant". ``None`` reads as
    "not scored". Conflating them would have made the probe rows - which have no gold
    clauses by construction - look like total retrieval failure.
    """
    metrics = compute_routing_metrics([_label("P1", "abstain", "answer")])
    assert metrics.macro_recall is None
    assert metrics.micro_recall is None
    assert metrics.n_gold_total == 0
    assert metrics.as_dict()["clause_recall"]["macro"] is None


def test_macro_and_micro_recall_are_computed_separately() -> None:
    """They differ whenever items carry different numbers of gold clauses.

    The stored n=7 run is exactly this case - macro 0.571429 against micro 0.529412 - so
    an aggregate that reported one number would be ambiguous about which. Here: one item
    with 1/1 and one with 1/3 give macro (1.0 + 0.333)/2 = 0.667 but micro 2/4 = 0.5.
    """
    labels = [
        _label("A1", "answer", "answer", retrieval=_score(True)),
        _label("A2", "answer", "answer", retrieval=_score(True, False, False)),
    ]
    metrics = compute_routing_metrics(labels)
    assert metrics.macro_recall == pytest.approx((1.0 + 1 / 3) / 2)
    assert metrics.micro_recall == pytest.approx(0.5)
    assert metrics.macro_recall != metrics.micro_recall
    assert (metrics.n_gold_total, metrics.n_gold_in_context_total) == (4, 2)
    assert (metrics.n_all_gold_in_context, metrics.n_any_gold_in_context) == (1, 2)


def test_as_dict_states_that_intervals_are_omitted_at_small_n() -> None:
    """The dict is what lands in a report, so the refusal has to be visible in it."""
    small = compute_routing_metrics([_label("A1", "answer", "answer")]).as_dict()
    assert small["intervals"] == f"omitted: n=1 < min_n={MIN_N_FOR_INTERVALS}"
    big = compute_routing_metrics(
        [_label(f"A{i}", "answer", "answer") for i in range(MIN_N_FOR_INTERVALS)]
    ).as_dict()
    assert big["intervals"] == "computable; call wilson_interval explicitly"


def test_as_dict_carries_the_not_computable_register() -> None:
    """Metrics the plan asks for and this evidence cannot support, named in the output.

    Listing them beside the computed ones is what stops a reader inferring that an absent
    metric was simply forgotten.
    """
    payload = compute_routing_metrics([_label("A1", "answer", "answer")]).as_dict()
    assert payload["not_computable"] == NOT_COMPUTABLE
    assert set(payload["not_computable"]) == {
        "residual_error_rate",
        "escalation_precision",
        "escalation_recall",
    }
    assert "qualified_scholar_review" in payload["verification_note"]
    assert json.loads(json.dumps(payload)) == payload


def test_empty_label_set_gives_zero_rates_not_a_division_error() -> None:
    metrics = compute_routing_metrics([])
    assert metrics.n_items == 0
    assert (metrics.coverage, metrics.abstention_rate, metrics.routing_agreement) == (
        0.0,
        0.0,
        0.0,
    )
    assert metrics.macro_recall is None


# --------------------------------------------------------------------------------------
# Reproducing the published n=7 numbers from the published traces
# --------------------------------------------------------------------------------------


def test_public_traces_reproduce_the_published_metrics_exactly(repo_root) -> None:
    """The reproducibility claim, tested.

    ``replay_n7_public_traces.jsonl`` is the redacted view: ``query_text`` is null, the
    prompt and response are SHA-256 prefixes, and every clause is a hash. If
    :func:`metrics_from_traces` over that file reproduces
    ``replay_n7_metrics.json`` field for field, then a reader without the licensed corpus
    can check every routing and recall number in the n=7 report. Any mismatch means the
    published table and the published traces have drifted apart, and the table is the one
    to distrust.
    """
    traces_path = repo_root / "reports" / "replay_n7_public_traces.jsonl"
    metrics_path = repo_root / "reports" / "replay_n7_metrics.json"
    traces = [
        json.loads(line)
        for line in traces_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(traces) == 7
    stored = json.loads(metrics_path.read_text(encoding="utf-8"))

    recomputed = metrics_from_traces(traces).as_dict()
    mismatches = {
        key: (value, stored.get(key))
        for key, value in recomputed.items()
        if stored.get(key) != value
    }
    assert mismatches == {}, f"published metrics disagree with published traces: {mismatches}"

    # The specific figures the reports quote, so a silent co-drift of both files fails too.
    assert (stored["n_answer"], stored["n_abstain"], stored["n_escalate"]) == (1, 4, 2)
    assert stored["clause_recall"]["macro"] == 0.571429
    assert stored["clause_recall"]["micro"] == 0.529412
    assert stored["clause_recall"]["n_gold_in_context_total"] == 9
    assert stored["clause_recall"]["n_gold_total"] == 17
    assert stored["triggering_gates"] == {
        "none": 1,
        "model_did_not_abstain": 4,
        "substantive_answer": 1,
        "no_self_contradiction": 1,
    }


def test_public_traces_carry_no_text_to_reproduce_those_numbers_from(repo_root) -> None:
    """The other half of the claim: the reproducing input is publication-safe.

    Checked here rather than only in ``test_trace.py`` because it is what makes the test
    above meaningful - reproducing the numbers from a file that still held clause text
    would prove nothing about a reader who lacks the corpus.
    """
    traces_path = repo_root / "reports" / "replay_n7_public_traces.jsonl"
    # Keys are checked rather than searched for the substring "text": the redacted
    # record legitimately contains ``text_sha256_16`` and ``text_char_count``, and a
    # substring test would flag those while missing a differently-named text field.
    allowed_record_keys = {
        "rank",
        "chunk_id",
        "standard_id",
        "clause_id",
        "sub_clause_id",
        "occurrence_index",
        "source_page",
        "reranker_score",
        "retrieval_sources",
        "text_sha256_16",
        "text_char_count",
    }
    for line in traces_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        trace = json.loads(line)
        assert trace["view"] == "public_no_clause_text"
        assert trace["query_text"] is None
        assert set(trace["response"]) == {"sha256_16", "char_count"}
        assert set(trace["prompt"]) == {
            "prompt_version",
            "system_prompt_sha256_16",
            "user_prompt_char_count",
            "user_prompt_sha256_16",
        }
        for record in trace["retrieval"]["records"]:
            assert record["text_sha256_16"]
            assert set(record) <= allowed_record_keys, sorted(
                set(record) - allowed_record_keys
            )
            assert "text" not in record


def test_metrics_from_traces_refuses_a_trace_with_no_evaluation_block() -> None:
    """A decision with no reference label cannot contribute to agreement.

    Defaulting it to "not matched" would silently depress routing agreement; defaulting
    it to matched would inflate it. Neither is defensible, so it raises.
    """
    with pytest.raises(ValueError, match="has no evaluation block"):
        metrics_from_traces([{"item_id": "X1", "routing": {"decision": "answer"}}])


def test_metrics_from_traces_round_trips_a_constructed_trace() -> None:
    """Rebuilding a ``RetrievalScore`` from its dict form must preserve the recall."""
    label = _label("A1", "answer", "answer", retrieval=_score(True, False))
    trace = {
        "item_id": "A1",
        "routing": {"decision": "answer", "triggering_gate": None},
        "evaluation": label.as_dict(),
    }
    metrics = metrics_from_traces([trace])
    assert metrics.macro_recall == pytest.approx(0.5)
    assert metrics.micro_recall == pytest.approx(0.5)
    assert metrics.triggering_gates == {"none": 1}
    assert isinstance(metrics, RoutingMetrics)


# --------------------------------------------------------------------------------------
# Selective risk: the cross-tabulation the research problem is about
# --------------------------------------------------------------------------------------


def _cross_tab(**cells: int) -> list[dict[str, str]]:
    """Build items from ``expected_decision=count`` pairs, e.g. ``abstain_answer=19``."""
    items: list[dict[str, str]] = []
    for key, count in cells.items():
        expected, decision = key.split("_")
        items.extend(
            {"expected_behavior": expected, "decision": decision} for _ in range(count)
        )
    return items


def test_every_cell_of_the_three_by_three_lands_in_its_own_field() -> None:
    """Nine distinct counts, so no two cells can be confused for one another.

    Each cell gets a different count, which means a transposition or an off-by-one in the
    cross-tab shows up as a specific wrong number rather than a plausible total.
    """
    items = _cross_tab(
        abstain_abstain=1,
        abstain_answer=2,
        abstain_escalate=3,
        answer_abstain=4,
        answer_answer=5,
        answer_escalate=6,
        escalate_abstain=7,
        escalate_answer=8,
        escalate_escalate=9,
    )
    risk = compute_selective_risk(items)
    assert risk.n_items == 45
    assert risk.n_abstain_expected == 6  # 1 + 2 + 3
    assert risk.n_answer_expected == 15  # 4 + 5 + 6
    assert risk.n_escalation_expected == 24  # 7 + 8 + 9
    assert risk.n_abstention_scope == 21
    assert risk.n_true_abstain == 1
    assert risk.n_unsafe_answers == 2
    assert risk.n_abstain_expected_escalated == 3
    assert risk.n_false_abstain == 4
    assert risk.n_true_escalate == 9
    # Every escalate *decision*, whatever was expected: 3 + 6 + 9.
    assert risk.n_decided_escalate == 18
    assert risk.n_decided_abstain_in_scope == 5  # 1 + 4, in scope only


def test_escalate_expected_items_are_excluded_from_the_abstention_scope() -> None:
    """The structural separation, asserted rather than trusted.

    An ``escalate``-expected probe routed ABSTAIN is *not* counted as a wrong abstention.
    That is deliberate: the escalate label is this project's stipulation, so scoring
    abstention against it would import the stipulation into the one block that rests on
    mechanical corpus absence. Reject the stipulation and the abstention block must be
    untouched - which is only true if these items never enter it.
    """
    without = compute_selective_risk(_cross_tab(abstain_abstain=2, answer_answer=3))
    with_escalate = compute_selective_risk(
        _cross_tab(
            abstain_abstain=2, answer_answer=3, escalate_abstain=5, escalate_answer=4
        )
    )
    for field in (
        "n_abstention_scope",
        "n_abstain_expected",
        "n_answer_expected",
        "n_true_abstain",
        "n_false_abstain",
        "n_unsafe_answers",
    ):
        assert getattr(with_escalate, field) == getattr(without, field), field
    assert with_escalate.abstention_precision == without.abstention_precision
    assert with_escalate.abstention_recall == without.abstention_recall
    assert with_escalate.n_items == 14 and without.n_items == 5


def test_unsafe_answer_cell_is_the_named_asymmetric_risk() -> None:
    """Expected abstain, decided answer: the confidently-wrong cell.

    Escalation is not this cell. An unanswerable item routed ESCALATE is wrong but safe -
    it reaches a human - so it is counted apart in ``n_abstain_expected_escalated`` and
    excluded from the unsafe count.
    """
    risk = compute_selective_risk(
        _cross_tab(abstain_answer=3, abstain_escalate=2, abstain_abstain=1)
    )
    assert risk.n_unsafe_answers == 3
    assert risk.n_abstain_expected_escalated == 2
    assert risk.unsafe_answer_rate == pytest.approx(0.5)
    assert risk.abstention_recall == pytest.approx(1 / 6)
    payload = risk.as_dict()["asymmetric_risk"]
    assert payload["n_unsafe_answers"] == 3
    assert "the failure mode the research problem is about" in payload["definition"]
    assert payload["interval"]["n"] == 6


def test_empty_denominators_give_none_not_a_flattering_number() -> None:
    """No abstentions decided means precision is undefined, not perfect.

    ``1.0`` here would read as "every abstention this system made was correct" on a run
    that abstained zero times. The shipped blind-spot audit is exactly that run.
    """
    risk = compute_selective_risk(_cross_tab(answer_answer=7))
    assert risk.abstention_precision is None
    assert risk.abstention_recall is None
    assert risk.unsafe_answer_rate is None
    assert risk.escalation_precision is None
    assert risk.escalation_recall is None
    payload = risk.as_dict()
    assert payload["abstention"]["precision"] is None
    assert payload["abstention"]["precision_interval"] is None
    assert payload["asymmetric_risk"]["interval"] is None


def test_zero_and_undefined_are_distinguished_in_the_same_block() -> None:
    """``recall = 0.0`` with ``precision = None``: different claims, both reported.

    From the shipped audit: 19 items should have abstained and none did, so recall is a
    measured zero. Nothing was routed abstain at all, so precision has no denominator.
    A schema that collapsed both to ``None`` - or both to ``0.0`` - would lose the fact
    that the recall figure is real evidence and the precision figure is not.
    """
    risk = compute_selective_risk(
        _cross_tab(abstain_answer=19, answer_answer=7, escalate_answer=6)
    )
    assert risk.abstention_recall == 0.0
    assert risk.abstention_precision is None
    assert risk.escalation_recall == 0.0
    assert risk.escalation_precision is None


def test_reproduces_the_shipped_blind_spot_selective_risk_block(repo_root) -> None:
    """The published numbers, recomputed from the cross-tab they describe.

    ``reports/retrieval_stage_blind_spot.json`` records the shipped policy routing all 32
    items (25 probes + 7 answerable) to ANSWER at the retrieval stage. Reconstructing that
    cross-tab here must reproduce the stored block exactly, including both intervals and
    the null precisions.
    """
    stored = json.loads(
        (repo_root / "reports" / "retrieval_stage_blind_spot.json").read_text(
            encoding="utf-8"
        )
    )["shipped"]["summary"]["selective_risk"]

    risk = compute_selective_risk(
        _cross_tab(
            abstain_answer=BLIND_SPOT_ABSTAIN_EXPECTED,
            answer_answer=BLIND_SPOT_ANSWER_EXPECTED,
            escalate_answer=BLIND_SPOT_ESCALATE_EXPECTED,
        )
    )
    assert risk.as_dict() == stored

    assert stored["n_items"] == 32
    assert stored["asymmetric_risk"]["n_unsafe_answers"] == 19
    assert stored["asymmetric_risk"]["unsafe_answer_rate"] == 1.0
    assert stored["abstention"]["recall"] == 0.0
    assert stored["abstention"]["precision"] is None
    assert stored["abstention"]["recall_interval"]["upper"] == 0.176467
    assert stored["escalation_stipulated"]["n_escalation_expected"] == 6


@pytest.mark.parametrize(
    ("item", "match"),
    [
        ({"expected_behavior": "ANSWER", "decision": "answer"}, "expected_behavior"),
        ({"expected_behavior": "answer", "decision": "defer"}, "decision"),
        ({"expected_behavior": "refuse", "decision": "answer"}, "expected_behavior"),
    ],
)
def test_unrecognised_labels_raise_rather_than_bucketing(item, match: str) -> None:
    """A typo must not land in an "other" bucket.

    ``"ANSWER"`` differing only in case would otherwise be silently dropped from
    ``n_answer_expected`` while still counting towards ``n_items``, understating exactly
    the cell this function exists to surface.
    """
    with pytest.raises(ValueError, match=f"unrecognised {match}"):
        compute_selective_risk([item])


def test_alternate_key_names_are_honoured() -> None:
    """Probes and hard-set items can be scored without renaming their fields first."""
    risk = compute_selective_risk(
        [{"want": "abstain", "got": "answer"}], expected_key="want", decision_key="got"
    )
    assert risk.n_unsafe_answers == 1


def test_missing_key_raises_a_key_error_rather_than_defaulting() -> None:
    with pytest.raises(KeyError):
        compute_selective_risk([{"expected_behavior": "answer"}])


# --------------------------------------------------------------------------------------
# The escalation / abstention firewall
# --------------------------------------------------------------------------------------


def test_escalation_lives_in_its_own_block_and_carries_its_own_warning() -> None:
    """A reader who rejects the stipulation must be able to delete one block and stop.

    So: the stipulated numbers appear under ``escalation_stipulated`` and nowhere else,
    that block names its ``verification_basis`` and its stipulation id, and the abstention
    block separately states its own basis. Nothing anywhere pools the two.
    """
    payload = compute_selective_risk(
        _cross_tab(abstain_answer=19, answer_answer=7, escalate_escalate=6)
    ).as_dict()

    escalation = payload["escalation_stipulated"]
    assert escalation["verification_basis"] == "stipulated_definition"
    assert "cross_standard_comparison_v1" in escalation["warning"]
    assert "NOT a scholar judgement" in escalation["warning"]
    assert "Never pool these figures with the abstention block." in escalation["warning"]

    abstention = payload["abstention"]
    assert abstention["verification_basis"] == "mechanical_corpus_absence"
    assert "excluded from both the numerator and the denominator" in abstention["scope_note"]
    assert not set(abstention) & set(ESCALATION_KEYS)
    assert not set(payload["asymmetric_risk"]) & set(ESCALATION_KEYS)


def test_no_escalation_figure_appears_outside_the_escalation_block() -> None:
    """Structural, so it cannot be defeated by adding a field somewhere else.

    Two payloads are built from identical abstention cells and *different* escalation
    cells. With the escalation block removed, they must be equal - if any abstention or
    asymmetric-risk figure moved, an escalation count has leaked into a block a reader
    might keep after rejecting the stipulation. Comparing payloads rather than searching
    for a digit string avoids matching the ``0.975`` in an interval by coincidence.

    ``n_items`` is the one field that legitimately moves: it is the total number of items
    scored, not an escalation metric. The stipulation-free denominator is
    ``abstention.n_scope``, and that must not move.
    """
    common = {"abstain_abstain": 1, "abstain_answer": 3, "answer_answer": 2}
    few = compute_selective_risk(_cross_tab(**common, escalate_escalate=1)).as_dict()
    many = compute_selective_risk(_cross_tab(**common, escalate_escalate=97)).as_dict()

    assert few["escalation_stipulated"] != many["escalation_stipulated"], "vacuous test"
    assert many["escalation_stipulated"]["n_true_escalate"] == 97
    assert (few.pop("n_items"), many.pop("n_items")) == (7, 103)
    del few["escalation_stipulated"], many["escalation_stipulated"]
    assert few == many
    assert few["abstention"]["n_scope"] == 6

    for key in ESCALATION_KEYS:
        assert key not in json.dumps(few)


def test_selective_risk_payload_is_json_serialisable_and_scholar_free() -> None:
    payload = compute_selective_risk(
        _cross_tab(abstain_answer=19, answer_answer=7, escalate_answer=6)
    ).as_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["label"] == "selective_risk_v1"
    assert "No label here is qualified_scholar_review" in payload["verification_note"]
    assert "qualified_scholar_review" not in json.dumps(
        {k: v for k, v in payload.items() if k != "verification_note"}
    )


def test_expected_behaviors_match_the_route_decision_enum() -> None:
    """One vocabulary across policy, schema and metrics, so no mapping table exists."""
    from aaoifi_rag.reliability import RouteDecision

    assert EXPECTED_BEHAVIORS == ("answer", "abstain", "escalate")
    assert set(EXPECTED_BEHAVIORS) == {decision.value for decision in RouteDecision}


# --------------------------------------------------------------------------------------
# End-to-end over the real hard set, when it is present
# --------------------------------------------------------------------------------------


@pytest.mark.requires_private_data
def test_hard_set_has_no_negatives_which_is_why_probes_exist(hard_set) -> None:
    """The gap the probe file fills, measured on the real hard set.

    Every one of the seven items is ``expected_behavior = answer``, so on the hard set
    alone abstention recall is ``None`` (no positives) and every abstention is
    over-abstention by construction. This is the state ``NOT_COMPUTABLE`` describes, and
    it is a property of the authored set, not of the metrics code.
    """
    assert len(hard_set) == 7
    assert {item["expected_behavior"] for item in hard_set} == {"answer"}

    items: list[dict[str, Any]] = [
        {"expected_behavior": item["expected_behavior"], "decision": "abstain"}
        for item in hard_set
    ]
    risk = compute_selective_risk(items)
    assert risk.n_abstain_expected == 0
    assert risk.abstention_recall is None
    assert risk.abstention_precision == 0.0, "seven false abstentions, zero true ones"
    assert risk.n_false_abstain == 7
    assert NOT_COMPUTABLE["escalation_precision"].startswith(
        "no hard-set item has expected_behavior='escalate'"
    )
