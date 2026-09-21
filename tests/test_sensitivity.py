"""Threshold sweeps and the gate ablation lattice.

The two claims this module makes about its own outputs are the ones worth testing, and
both are structural rather than numeric:

* **The sweep is a complete enumeration, not a grid.** Because every threshold gate is a
  step function of the threshold, the transition points are exactly the observed signal
  values. The tests below check that the candidate set is precisely
  ``{bounds} | {shipped} | observed | observed + eps``, that every flip sits on one of
  those steps, and that an item whose gate is inert contributes no transition point at
  all. A sampled sweep would pass none of these.
* **"Changes no decision" is not "removable".** Leave-one-out reports three gates on the
  n=7 run as changing nothing, yet the 2^k lattice shows that no sufficient subset drops
  all three: they redundantly cover the same item, so each masks the others. A reader who
  pruned the battery on the leave-one-out column alone would release that item. This is
  the sharpest finding in ``reports/gate_sensitivity_n7.md`` and it is pinned twice -
  once on a synthetic set that reproduces the structure without the corpus, and once
  against the stored run.

The synthetic four-item set below reproduces the real run's entire lattice structure -
same load-bearing gates, same unexercised list, same three minimal sufficient subsets of
size 3 - from invented text. That is a convenience, not a coincidence: the structure
follows from which gates co-fire, and the fixture is built to co-fire the same way.

No AAOIFI clause prose appears here.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest

from aaoifi_rag.generation.prompt import ABSTENTION_LINE
from aaoifi_rag.reliability import (
    GATES,
    PolicyConfig,
    SelectivePredictionPolicy,
    compute_signals,
)
from aaoifi_rag.reliability.sensitivity import (
    MAX_LATTICE_GATES,
    THRESHOLD_SIGNALS,
    SensitivityItem,
    compute_ablation_lattice,
    compute_gate_contributions,
    sweep_threshold,
)

#: The shipped echo threshold. Sits in the gap between the three observed values at n=7.
SHIPPED_ECHO = 0.85

#: Epsilon the sweep steps by. Mirrored from ``sensitivity._EPSILON`` deliberately: if
#: that constant changes, the assertions on candidate thresholds should fail loudly.
EPSILON = 1e-9

CLEAN = (
    "The institution must enter the widget at the agreed value and disclose it in "
    "the notes [1]."
)

#: Half the shingles match rank 1, half do not. Measures echo_ratio exactly 0.5.
HALF_ECHO = (
    "The institution shall record the widget number 1 in its books at the value "
    "agreed between the parties, and must also apply prudence when unrelated matters "
    "arise in other periods [1]."
)

#: Rank 1 quoted back with a citation. Measures 0.962963.
TRANSCRIBED = (
    "The institution shall record the widget number 1 in its books at the value "
    "agreed between the parties, and shall disclose that value in the notes to the "
    "financial statements for period 1. [1]"
)

#: Answer text plus the abstention line plus an Arabic character and no citation:
#: fails no_self_contradiction, citation_integrity and script_integrity together. This
#: is H05's profile, and the reason three gates mask each other.
MULTI_DEFECT = f"The institution must record it. {ABSTENTION_LINE} و"

#: The three gates that co-fire on ``MULTI_DEFECT``. Exactly one is needed; which one is
#: named as the cause follows declaration order, so ``no_self_contradiction`` wins.
MASKING_GATES = ("no_self_contradiction", "citation_integrity", "script_integrity")


def _item(item_id: str, context: list[dict[str, Any]], text: str) -> SensitivityItem:
    return SensitivityItem(
        item_id=item_id,
        signals=compute_signals(context, text),
        expected_behavior="answer",
    )


@pytest.fixture
def sweep_items(synthetic_context) -> list[SensitivityItem]:
    """Three answer attempts at echo 0.0 / 0.5 / 0.962963, plus one abstention.

    The abstention is the important one: ``substantive_answer`` short-circuits to passed
    on a non-answer, so that item must contribute no transition point.
    """
    return [
        _item("S1", synthetic_context, CLEAN),
        _item("S2", synthetic_context, HALF_ECHO),
        _item("S3", synthetic_context, TRANSCRIBED),
        _item("S4", synthetic_context, ABSTENTION_LINE),
    ]


@pytest.fixture
def lattice_items(synthetic_context) -> list[SensitivityItem]:
    """One clean answer, one abstention, one transcription, one multi-defect item."""
    return [
        _item("S1", synthetic_context, CLEAN),
        _item("S2", synthetic_context, ABSTENTION_LINE),
        _item("S3", synthetic_context, TRANSCRIBED),
        _item("S4", synthetic_context, MULTI_DEFECT),
    ]


# --------------------------------------------------------------------------------------
# The fixture's own signal values, measured before anything is built on them
# --------------------------------------------------------------------------------------


def test_fixture_signals_are_what_the_sweep_assertions_assume(sweep_items) -> None:
    """Assert the inputs before asserting the outputs.

    Every threshold in the tests below is one of these four numbers. If the fixture text
    or the shingle width changed, this test names the cause instead of leaving three
    downstream failures to be diagnosed.
    """
    measured = {
        item.item_id: (
            item.signals.response.is_answer_attempt,
            item.signals.response.echo_ratio,
        )
        for item in sweep_items
    }
    assert measured["S1"][0] is True and measured["S1"][1] == 0.0
    assert measured["S2"][0] is True and measured["S2"][1] == pytest.approx(0.5)
    assert measured["S3"][0] is True
    assert measured["S3"][1] == pytest.approx(0.962963, abs=1e-6)
    assert measured["S4"][0] is False, "the abstention line is not an answer attempt"


# --------------------------------------------------------------------------------------
# Threshold sweeps
# --------------------------------------------------------------------------------------


def test_candidate_thresholds_are_exactly_the_steps_and_the_bounds(sweep_items) -> None:
    """The completeness argument, checked as a set identity.

    ``{0.0, 1.0, shipped} | observed | {v + eps}``. Nothing else is visited, and nothing
    in that set is skipped - which is what makes the sweep an enumeration of every
    reachable routing rather than a sample of some of them.
    """
    sweep = sweep_threshold(sweep_items)
    observed = set(sweep.observed_values)
    expected = {0.0, 1.0, SHIPPED_ECHO} | observed | {v + EPSILON for v in observed}
    assert {point.threshold for point in sweep.points} == expected
    assert [point.threshold for point in sweep.points] == sorted(expected)
    assert len(sweep.points) == 8


def test_a_non_answer_contributes_no_transition_point(sweep_items) -> None:
    """S4 abstains, so ``substantive_answer`` is inert on it and it has no step.

    Including a non-evaluable item's echo ratio would invent a flip the threshold cannot
    cause: the gate short-circuits before ever comparing it.
    """
    sweep = sweep_threshold(sweep_items)
    assert sweep.n_items == 4
    assert sweep.n_items_evaluable == 3
    assert len(sweep.observed_values) == 3
    assert all(flip.item_id != "S4" for flip in sweep.flips)
    # S4 is routed the same way at every threshold in the sweep.
    assert {dict(point.decisions)["S4"] for point in sweep.points} == {"abstain"}


def test_every_flip_sits_on_an_observed_value_and_crosses_upward(sweep_items) -> None:
    """A flip's lower endpoint is an observed value; its upper is that value plus eps.

    That is the signature of a strict ``<`` comparison against a frozen signal. All three
    flips go escalate -> answer as the threshold rises, because a higher tolerance for
    verbatim overlap releases more answers.
    """
    sweep = sweep_threshold(sweep_items)
    observed = set(sweep.observed_values)
    assert len(sweep.flips) == 3
    for flip in sweep.flips:
        assert flip.threshold_below in observed
        assert flip.threshold_at_or_above == pytest.approx(
            flip.threshold_below + EPSILON, abs=1e-15
        )
        assert (flip.decision_below, flip.decision_at_or_above) == ("escalate", "answer")
    assert {flip.item_id for flip in sweep.flips} == {"S1", "S2", "S3"}


def test_routing_is_monotone_in_the_echo_threshold(sweep_items) -> None:
    """Raising the tolerance can only release answers, never withdraw them.

    Monotonicity is what makes a single stable interval the right summary. If a higher
    threshold could withdraw an answer, the routing would not be a step function of one
    variable and the interval would be meaningless.
    """
    sweep = sweep_threshold(sweep_items)
    counts = [point.n_answer for point in sweep.points]
    assert counts == sorted(counts)
    assert counts == [0, 1, 1, 2, 2, 2, 3, 3]
    assert [point.n_abstain for point in sweep.points] == [1] * 8


def test_stable_interval_is_the_contiguous_run_around_the_shipped_value(
    sweep_items,
) -> None:
    """Bounds are the adjacent steps: ``(0.5 + eps, 0.962963)``.

    Both endpoints are inclusive of the shipped routing, and the interval is contiguous
    by construction. An identical routing recurring at a distant threshold must not
    widen it - see the next test.
    """
    sweep = sweep_threshold(sweep_items)
    lower, upper = sweep.stable_interval
    assert lower == pytest.approx(0.5 + EPSILON, abs=1e-15)
    assert upper == pytest.approx(0.962963, abs=1e-6)
    assert sweep.stable_interval_width == pytest.approx(upper - lower)
    assert lower < SHIPPED_ECHO < upper

    shipped_points = [point for point in sweep.points if point.is_shipped_value]
    assert len(shipped_points) == 1
    target = shipped_points[0].routing_vector
    inside = [
        point.routing_vector
        for point in sweep.points
        if lower <= point.threshold <= upper
    ]
    assert set(inside) == {target}


def test_stable_interval_excludes_a_routing_that_only_recurs_later(sweep_items) -> None:
    """Contiguity, tested where it bites.

    At threshold 0.0 the routing is ``(escalate, escalate, escalate, abstain)`` and at
    ``0.5`` it is ``(answer, escalate, escalate, abstain)``. Neither equals the shipped
    routing, so the interval must stop below them even though thresholds further out
    would be numerically closer to a bound.
    """
    sweep = sweep_threshold(sweep_items)
    lower, _ = sweep.stable_interval
    below = [point for point in sweep.points if point.threshold < lower]
    assert below, "the fixture must have points below the stable interval"
    shipped = next(p for p in sweep.points if p.is_shipped_value).routing_vector
    assert all(point.routing_vector != shipped for point in below)


def test_distinct_routings_are_counted_over_the_whole_sweep(sweep_items) -> None:
    sweep = sweep_threshold(sweep_items)
    assert sweep.n_distinct_routings == 4
    assert sweep.is_decision_relevant is True
    assert sweep.n_distinct_routings == len(
        {point.routing_vector for point in sweep.points}
    )


def test_agreement_is_counted_only_against_authored_expectations(
    synthetic_context,
) -> None:
    """``n_agree`` is ``None`` when no item carries an ``expected_behavior``.

    Zero agreement and unmeasured agreement are different claims, the same distinction
    ``compute_routing_metrics`` draws between ``0.0`` and ``None`` recall.
    """
    unlabelled = [
        SensitivityItem(
            item_id="U1",
            signals=compute_signals(synthetic_context, CLEAN),
            expected_behavior=None,
        )
    ]
    sweep = sweep_threshold(unlabelled)
    assert all(point.n_agree is None for point in sweep.points)

    labelled = sweep_threshold([_item("S1", synthetic_context, CLEAN)])
    assert {point.n_agree for point in labelled.points} == {0, 1}


def test_a_threshold_on_an_unmeasured_signal_reports_itself_irrelevant(
    sweep_items,
) -> None:
    """Zero observations gives a full-width "stable" interval, and says so.

    The synthetic contexts carry no reranker signal, so ``min_reranker_top_1`` has no
    observed values, one distinct routing and a stable interval of ``[0.0, 1.0]``. That
    width is the reductio of reading interval width as confidence: it is widest exactly
    where the evidence is thinnest, which is why ``as_dict`` carries the ``limitation``
    string rather than leaving the number to speak for itself.
    """
    sweep = sweep_threshold(sweep_items, field_name="min_reranker_top_1")
    assert sweep.observed_values == ()
    assert sweep.n_items_evaluable == 0
    assert sweep.n_distinct_routings == 1
    assert sweep.is_decision_relevant is False
    assert sweep.stable_interval == (0.0, 1.0)
    assert sweep.stable_interval_width == 1.0
    payload = sweep.as_dict()
    assert "0 observed" in payload["limitation"]
    assert "NOT evidence that the threshold is well chosen" in payload["limitation"]


def test_a_null_shipped_threshold_sweeps_from_the_upper_bound(sweep_items) -> None:
    """``min_reranker_top_1`` is ``None`` in the shipped config: the gate is off.

    ``None`` is treated as the upper bound so the sweep still has a reference point, and
    exactly one swept threshold is flagged as the shipped value.
    """
    assert PolicyConfig().min_reranker_top_1 is None
    sweep = sweep_threshold(sweep_items, field_name="min_reranker_top_1")
    assert sweep.shipped_value == 1.0
    assert sum(1 for point in sweep.points if point.is_shipped_value) == 1


def test_sweep_bounds_are_configurable(sweep_items) -> None:
    sweep = sweep_threshold(sweep_items, lower_bound=0.4, upper_bound=0.95)
    thresholds = [point.threshold for point in sweep.points]
    assert min(thresholds) == 0.0, "observed values are always included, bounds or not"
    assert 0.4 in thresholds and 0.95 in thresholds


@pytest.mark.parametrize("field_name", sorted(THRESHOLD_SIGNALS))
def test_every_declared_threshold_field_can_be_swept(field_name, sweep_items) -> None:
    """The registry and the sweep must not drift apart.

    A field listed in ``THRESHOLD_SIGNALS`` whose extractor raised, or which no config
    attribute backed, would only surface when someone swept it.
    """
    sweep = sweep_threshold(sweep_items, field_name=field_name)
    assert sweep.field_name == field_name
    assert sweep.points
    assert json.loads(json.dumps(sweep.as_dict())) == sweep.as_dict()


def test_sweeping_an_unknown_field_raises_and_lists_the_known_ones(sweep_items) -> None:
    """Including the plausible typo, which must not silently sweep nothing."""
    with pytest.raises(ValueError, match="not a numeric threshold field"):
        sweep_threshold(sweep_items, field_name="max_eco_ratio")
    with pytest.raises(ValueError, match="max_echo_ratio"):
        sweep_threshold(sweep_items, field_name="enable_echo_gate")


def test_sweeping_nothing_raises() -> None:
    with pytest.raises(ValueError, match="needs at least one item"):
        sweep_threshold([])


def test_sweep_payload_records_its_method_and_its_limitation(sweep_items) -> None:
    payload = sweep_threshold(sweep_items).as_dict()
    assert payload["method"] == "exact_enumeration_over_observed_signal_values"
    assert payload["stable_interval"]["width"] == pytest.approx(
        payload["stable_interval"]["upper"] - payload["stable_interval"]["lower"]
    )
    assert "exact, not sampled" in payload["stable_interval"]["meaning"]
    assert len(payload["points"]) == len(sweep_threshold(sweep_items).points)


# --------------------------------------------------------------------------------------
# Leave-one-out and the ablation lattice
# --------------------------------------------------------------------------------------


def test_lattice_enumerates_subsets_of_the_enabled_gates_only(lattice_items) -> None:
    """2^10, not 2^13. Disabled gates ride along in every subset.

    They always report ``passed``, so including them in the enumeration would multiply
    the work by eight and change no routing. The count is asserted because it is the
    cheapest evidence that the partition is by *enabled*, not by registration.
    """
    lattice = compute_ablation_lattice(lattice_items)
    assert len(lattice.enabled_gate_names) == 10
    assert len(lattice.disabled_gate_names) == 3
    assert lattice.n_subsets == 1024 == 2**10
    assert set(lattice.enabled_gate_names) | set(lattice.disabled_gate_names) == {
        gate.name for gate in GATES
    }
    assert not set(lattice.enabled_gate_names) & set(lattice.disabled_gate_names)


def test_enabled_gate_names_keep_declaration_order(lattice_items) -> None:
    """Precedence order, because which gate is *named* as the cause depends on it."""
    lattice = compute_ablation_lattice(lattice_items)
    declared = [gate.name for gate in GATES]
    assert list(lattice.enabled_gate_names) == [
        name for name in declared if name in set(lattice.enabled_gate_names)
    ]


def test_full_routing_matches_the_policy_run_directly(lattice_items) -> None:
    """The lattice's reference routing is the shipped policy's, not a re-derivation."""
    policy = SelectivePredictionPolicy()
    direct = tuple(
        (item.item_id, policy.decide(item.signals).decision.value)
        for item in lattice_items
    )
    lattice = compute_ablation_lattice(lattice_items)
    assert lattice.full_routing == direct
    assert dict(direct) == {
        "S1": "answer",
        "S2": "abstain",
        "S3": "escalate",
        "S4": "escalate",
    }


def test_leave_one_out_separates_firing_from_changing_a_decision(lattice_items) -> None:
    """Four gates fire; two change a decision. The gap is the whole point.

    ``no_self_contradiction``, ``citation_integrity`` and ``script_integrity`` each fire
    on S4 and each change nothing when removed, because the other two still catch it.
    Reporting only "fires on N items" would call three gates load-bearing; reporting only
    "changes N decisions" would call them idle. Both columns are kept.
    """
    contributions = {c.gate_name: c for c in compute_gate_contributions(lattice_items)}
    fired = {name for name, c in contributions.items() if c.n_items_triggering}
    changed = {name for name, c in contributions.items() if c.n_items_changed}
    assert fired == {"model_did_not_abstain", "substantive_answer", *MASKING_GATES}
    assert changed == {"model_did_not_abstain", "substantive_answer"}
    for name in MASKING_GATES:
        assert contributions[name].n_items_triggering == 1
        assert contributions[name].n_items_changed == 0
        assert contributions[name].is_load_bearing is False
        assert contributions[name].changed_items == ()


def test_load_bearing_gates_name_the_decisions_they_move(lattice_items) -> None:
    contributions = {c.gate_name: c for c in compute_gate_contributions(lattice_items)}
    assert contributions["model_did_not_abstain"].decision_shifts == (
        "S2: abstain -> answer",
    )
    assert contributions["substantive_answer"].decision_shifts == (
        "S3: escalate -> answer",
    )
    assert contributions["model_did_not_abstain"].is_load_bearing is True
    assert contributions["substantive_answer"].evidentiary_basis == "observed_in_n7_run"


def test_masked_gates_are_individually_removable_but_not_jointly(lattice_items) -> None:
    """The finding that a leave-one-out table cannot express.

    Every minimal sufficient subset contains exactly one of the three masking gates. So
    each is droppable alone and none is droppable together with the other two: pruning
    the battery on the leave-one-out column would release S4, the multi-defect item.
    """
    lattice = compute_ablation_lattice(lattice_items)
    assert lattice.minimal_sufficient_size == 3
    assert len(lattice.minimal_sufficient_subsets) == 3
    for subset in lattice.minimal_sufficient_subsets:
        assert set(subset) & {"model_did_not_abstain", "substantive_answer"} == {
            "model_did_not_abstain",
            "substantive_answer",
        }
        assert len(set(subset) & set(MASKING_GATES)) == 1
    covered = {
        name for subset in lattice.minimal_sufficient_subsets for name in subset
    }
    assert covered == {"model_did_not_abstain", "substantive_answer", *MASKING_GATES}

    # Dropping all three at once does change the routing, which is what "not jointly
    # removable" means. Verified by running the policy without them.
    without = SelectivePredictionPolicy(
        PolicyConfig(), tuple(g for g in GATES if g.name not in MASKING_GATES)
    )
    assert without.decide(lattice_items[3].signals).decision.value == "answer"
    assert lattice.load_bearing_gates == (
        "model_did_not_abstain",
        "substantive_answer",
    )


def test_unexercised_is_reported_separately_from_redundant(lattice_items) -> None:
    """Seven gates change nothing here; the payload refuses to call them redundant."""
    lattice = compute_ablation_lattice(lattice_items)
    assert set(lattice.unexercised_gates) == set(lattice.enabled_gate_names) - set(
        lattice.load_bearing_gates
    )
    assert len(lattice.unexercised_gates) == 8
    payload = lattice.as_dict()
    assert "must not be described as redundant" in payload["interpretation"]
    assert "cannot discover failure modes absent from the item set" in payload["limitation"]
    assert json.loads(json.dumps(payload)) == payload


def test_minimal_sufficient_subsets_do_reproduce_the_full_routing(lattice_items) -> None:
    """Verified by re-running each subset, not taken from the lattice's own bookkeeping."""
    lattice = compute_ablation_lattice(lattice_items)
    disabled = set(lattice.disabled_gate_names)
    full = dict(lattice.full_routing)
    for subset in lattice.minimal_sufficient_subsets:
        keep = set(subset) | disabled
        policy = SelectivePredictionPolicy(
            PolicyConfig(), tuple(g for g in GATES if g.name in keep)
        )
        routing = {
            item.item_id: policy.decide(item.signals).decision.value
            for item in lattice_items
        }
        assert routing == full, subset
        # And that it is minimal: dropping any one member must break it.
        for dropped in subset:
            smaller = keep - {dropped}
            reduced = SelectivePredictionPolicy(
                PolicyConfig(), tuple(g for g in GATES if g.name in smaller)
            )
            assert {
                item.item_id: reduced.decide(item.signals).decision.value
                for item in lattice_items
            } != full, f"{subset} minus {dropped} still reproduces the routing"


def test_lattice_refuses_an_intractable_enumeration() -> None:
    """The ceiling turns a future combinatorial blow-up into an error, not a hang."""
    assert MAX_LATTICE_GATES == 16
    oversized = SelectivePredictionPolicy(PolicyConfig(), GATES + GATES)
    enabled = sum(1 for gate in oversized.gates if gate.is_enabled(oversized.config))
    assert enabled == 20 > MAX_LATTICE_GATES
    items = [
        SensitivityItem(
            item_id="S1",
            signals=compute_signals([], ""),
            expected_behavior="answer",
        )
    ]
    with pytest.raises(ValueError, match=r"2\*\*20 subsets; the ceiling is 16"):
        compute_ablation_lattice(items, oversized)


def test_lattice_needs_at_least_one_item() -> None:
    with pytest.raises(ValueError, match="needs at least one item"):
        compute_ablation_lattice([])


def test_distinct_routings_never_exceed_the_subset_count(lattice_items) -> None:
    lattice = compute_ablation_lattice(lattice_items)
    assert 1 <= lattice.n_distinct_routings <= lattice.n_subsets
    assert lattice.n_distinct_routings == 8


# --------------------------------------------------------------------------------------
# Regression against the stored n=7 audit
# --------------------------------------------------------------------------------------


def _build_stored_items() -> list[SensitivityItem]:
    """Frozen signals for the seven stored items, via the audit script's own builder."""
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from audit_gate_sensitivity import build_items  # noqa: PLC0415

    return build_items()


@pytest.mark.requires_private_data
def test_published_sensitivity_report_is_reproducible(repo_root, stored_run) -> None:
    """Recompute ``reports/gate_sensitivity_n7.json`` and diff it field for field.

    ``stored_run`` is requested purely so this skips cleanly without the Git-ignored run
    artifact. What it establishes is that the published sweep and lattice are the current
    code's output on the current config - not a snapshot that has drifted from either.
    """
    stored = json.loads(
        (repo_root / "reports" / "gate_sensitivity_n7.json").read_text(encoding="utf-8")
    )
    items = _build_stored_items()
    assert [item.item_id for item in items] == [f"H0{i}" for i in range(1, 8)]

    sweep = sweep_threshold(items).as_dict()
    assert sweep == stored["threshold_sensitivity"]["max_echo_ratio"]

    lattice = compute_ablation_lattice(items).as_dict()
    assert lattice == stored["ablation_lattice"]


@pytest.mark.requires_private_data
def test_the_three_observed_echo_values_are_the_only_transition_points(
    stored_run,
) -> None:
    """H01 0.443038, H05 0.717391, H03 0.900000, and nothing else.

    Four of the seven items abstained, so they are not evaluable and contribute no step.
    A threshold set from three points is what ``reports/gate_sensitivity_n7.md`` reports,
    and this is the assertion behind that sentence.
    """
    sweep = sweep_threshold(_build_stored_items())
    assert sweep.n_items == 7
    assert sweep.n_items_evaluable == 3
    assert [round(value, 6) for value in sweep.observed_values] == [
        0.443038,
        0.717391,
        0.900000,
    ]
    assert sweep.shipped_value == SHIPPED_ECHO
    assert [flip.item_id for flip in sweep.flips] == ["H01", "H03"]
    # 0.85 sits between H05's 0.717 and H03's 0.900, so H05 is released and H03 is not.
    shipped = next(point for point in sweep.points if point.is_shipped_value)
    decisions = dict(shipped.decisions)
    assert decisions["H03"] == "escalate"
    assert decisions["H01"] == "answer"


@pytest.mark.requires_private_data
def test_the_masking_finding_holds_on_the_real_run(stored_run) -> None:
    """The synthetic structure above, confirmed on the stored evidence.

    Same two load-bearing gates, same minimal sufficient size, same three masking gates -
    on the real n=7 signals rather than invented text. The synthetic fixture is a
    convenience for running without the corpus, not the basis of the published claim.
    """
    lattice = compute_ablation_lattice(_build_stored_items())
    assert lattice.load_bearing_gates == (
        "model_did_not_abstain",
        "substantive_answer",
    )
    assert lattice.minimal_sufficient_size == 3
    assert len(lattice.minimal_sufficient_subsets) == 3
    for subset in lattice.minimal_sufficient_subsets:
        assert len(set(subset) & set(MASKING_GATES)) == 1
    assert len(lattice.unexercised_gates) == 8
    assert lattice.n_subsets == 1024
