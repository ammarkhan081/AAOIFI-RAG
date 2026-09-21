"""The routing policy: gate registry, precedence, and the three-way decision.

The policy is the only place in the system where a decision is made, so its behaviour is
pinned here rather than inferred from the n=7 replay. Every archetype below was checked
against the live code before being written down as an assertion; none of the expected
values in this file is a guess.

Two properties get the most attention because the reports lean on them:

* **Precedence.** Declaration order in :data:`GATES` decides which gate is *named* as the
  cause, and therefore which ``decision_if_failed`` wins when several gates fail at once.
  ``reports/gate_sensitivity_n7.md`` §3 says the ablation lattice inherits that ordering;
  these tests are what make that claim checkable.
* **The enabled/registered split.** Five gates are registered at the retrieval stage and
  only two are enabled. That distinction was mis-stated once in a draft report, so it is
  now asserted from ``describe_gates`` rather than remembered.

Synthetic contexts only. The records are invented normative-*looking* sentences; no
AAOIFI clause prose appears in this file, and none is needed, because every gate here
measures a surface property of text rather than its content.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from aaoifi_rag.generation.prompt import ABSTENTION_LINE
from aaoifi_rag.reliability import (
    GATES,
    EvidentiaryBasis,
    GateStage,
    PolicyConfig,
    RouteDecision,
    SelectivePredictionPolicy,
    compute_signals,
    describe_gates,
)

#: Declaration order is precedence order, so this list is the policy's precedence.
EXPECTED_GATE_ORDER = (
    "retrieval_non_empty",
    "retrieval_normative",
    "lexical_anchoring",
    "response_present",
    "model_did_not_abstain",
    "no_self_contradiction",
    "citation_integrity",
    "clause_grounding",
    "script_integrity",
    "no_degenerate_decoding",
    "substantive_answer",
    "reranker_top_1",
    "reranker_margin",
)

#: Off under the shipped config, each for a different and documented reason.
DISABLED_BY_DEFAULT = {
    "lexical_anchoring": EvidentiaryBasis.REJECTED_PROBES,
    "reranker_top_1": EvidentiaryBasis.REJECTED_N7,
    "reranker_margin": EvidentiaryBasis.UNANALYSED,
}


@pytest.fixture
def policy() -> SelectivePredictionPolicy:
    return SelectivePredictionPolicy()


def _answer_signals(context: list[dict[str, Any]], text: str):
    """Signals for a completed generation over ``context``."""
    return compute_signals(context, text)


CLEAN_ANSWER = (
    "The institution must enter the widget at the agreed value and disclose it in "
    "the notes [1]."
)

#: Rank 1 of ``synthetic_context``, quoted back verbatim with a citation. Measures
#: echo_ratio 0.962963 against the assembled context block.
TRANSCRIBED_ANSWER = (
    "The institution shall record the widget number 1 in its books at the value "
    "agreed between the parties, and shall disclose that value in the notes to the "
    "financial statements for period 1. [1]"
)


# --------------------------------------------------------------------------------------
# The gate registry
# --------------------------------------------------------------------------------------


def test_gate_declaration_order_is_precedence_order() -> None:
    assert tuple(gate.name for gate in GATES) == EXPECTED_GATE_ORDER
    names = [gate.name for gate in GATES]
    assert len(set(names)) == len(names), "duplicate gate name"


def test_absence_gates_precede_defect_gates() -> None:
    """Nothing-to-serve must outrank something-flawed-to-serve.

    If an ESCALATE gate were declared before an ABSTAIN gate, an item with no retrieved
    evidence at all could be routed to a human reviewer as though an answer existed.
    The two disabled reranker gates are ESCALATE gates declared last, so the claim is
    made about the enabled battery.
    """
    enabled = [
        (index, gate.decision_if_failed)
        for index, gate in enumerate(GATES)
        if gate.name not in DISABLED_BY_DEFAULT
    ]
    abstains = [index for index, d in enabled if d is RouteDecision.ABSTAIN]
    escalates = [index for index, d in enabled if d is RouteDecision.ESCALATE]
    assert abstains and escalates
    assert max(abstains) < min(escalates), (
        f"an ESCALATE gate is declared before an ABSTAIN gate: "
        f"abstain indices {abstains}, escalate indices {escalates}"
    )


def test_registered_and_enabled_counts_by_stage() -> None:
    """Five retrieval gates registered, two enabled. Eight response gates, all enabled.

    Asserted rather than remembered: a draft of
    ``reports/retrieval_stage_blind_spot.md`` claimed two *registered* retrieval gates,
    which understated how much of the retrieval battery is switched off. The blind-spot
    finding depends on this split, so it is pinned here.
    """
    described = describe_gates()
    assert len(described) == len(GATES) == 13

    by_stage: dict[str, list[dict[str, Any]]] = {"retrieval": [], "response": []}
    for entry in described:
        by_stage[entry["stage"]].append(entry)

    assert len(by_stage["retrieval"]) == 5
    assert sum(1 for e in by_stage["retrieval"] if e["enabled"]) == 2
    assert len(by_stage["response"]) == 8
    assert sum(1 for e in by_stage["response"] if e["enabled"]) == 8
    assert sum(1 for e in described if e["enabled"]) == 10


def test_disabled_gates_each_carry_their_rejection_reason() -> None:
    """A gate is off for a stated reason, and the reason travels in the registry.

    ``lexical_anchoring`` was measured against the probes and rejected;
    ``reranker_top_1`` was measured at n=7 and rejected; ``reranker_margin`` was never
    analysed. Three different epistemic positions, three different bases - collapsing
    them to one "disabled" flag would lose the distinction the reports rely on.
    """
    described = {entry["name"]: entry for entry in describe_gates()}
    off = {name for name, entry in described.items() if not entry["enabled"]}
    assert off == set(DISABLED_BY_DEFAULT)
    for name, basis in DISABLED_BY_DEFAULT.items():
        assert described[name]["evidentiary_basis"] == basis.value


def test_route_decision_values_match_the_hard_set_schema(repo_root) -> None:
    """``decision.value == item['expected_behavior']`` must need no mapping table."""
    schema = json.loads(
        (repo_root / "data" / "manifests" / "hard_set_schema.json").read_text(
            encoding="utf-8"
        )
    )
    allowed = schema["properties"]["expected_behavior"]["enum"]
    assert {d.value for d in RouteDecision} == set(allowed)
    assert list(allowed) == ["answer", "abstain", "escalate"]


# --------------------------------------------------------------------------------------
# Routing archetypes. Every expected value below was measured against the live policy.
# --------------------------------------------------------------------------------------


def test_clean_answer_routes_to_answer(policy, synthetic_context) -> None:
    outcome = policy.decide(_answer_signals(synthetic_context, CLEAN_ANSWER))
    assert outcome.decision is RouteDecision.ANSWER
    assert outcome.triggering_gate is None
    assert outcome.triggered_gates == ()
    assert outcome.rationale == "all 10 enabled gates passed"
    assert outcome.stage_scope == "all"


@pytest.mark.parametrize(
    ("label", "response", "decision", "gate"),
    [
        (
            "paraphrase with a citation",
            CLEAN_ANSWER,
            RouteDecision.ANSWER,
            None,
        ),
        (
            "answer attempt with no citation marker at all",
            "The institution must enter the widget at the agreed value and disclose it.",
            RouteDecision.ESCALATE,
            "citation_integrity",
        ),
        (
            "the fixed abstention line alone",
            ABSTENTION_LINE,
            RouteDecision.ABSTAIN,
            "model_did_not_abstain",
        ),
        (
            "answer plus the abstention line: self-contradiction, H05's defect",
            f"The institution must record it [1]. {ABSTENTION_LINE}",
            RouteDecision.ESCALATE,
            "no_self_contradiction",
        ),
        (
            "degenerate decoding",
            "The institution must record it [1] aaaaaa.",
            RouteDecision.ESCALATE,
            "no_degenerate_decoding",
        ),
        (
            "out-of-script characters, H05's other defect",
            "The institution must record it [1] و.",
            RouteDecision.ESCALATE,
            "script_integrity",
        ),
        (
            "verbatim transcription of rank 1, H03's defect",
            TRANSCRIBED_ANSWER,
            RouteDecision.ESCALATE,
            "substantive_answer",
        ),
    ],
)
def test_response_archetypes_route_as_specified(
    policy, synthetic_context, label, response, decision, gate
) -> None:
    outcome = policy.decide(_answer_signals(synthetic_context, response))
    assert outcome.decision is decision, f"{label}: {outcome.rationale}"
    assert outcome.triggering_gate == gate, label


def test_retrieval_archetypes(policy, heading_context) -> None:
    """The two enabled retrieval-stage gates, exercised on constructed inputs.

    Neither has ever fired on real data - 0 of 7 answerable items and 0 of 25 probes,
    per ``reports/gate_sensitivity_n7.md`` §4. That is precisely why they are tested
    synthetically: an unexercised gate whose code was never run is a different and worse
    thing than an unexercised gate that provably works.
    """
    heading_only = policy.decide(_answer_signals(heading_context, "The scope is defined [1]."))
    assert heading_only.decision is RouteDecision.ABSTAIN
    assert heading_only.triggering_gate == "retrieval_normative"

    empty = policy.decide(_answer_signals([], "Something [1]."))
    assert empty.decision is RouteDecision.ABSTAIN
    assert empty.triggering_gate == "retrieval_non_empty"


def test_precedence_when_several_gates_fail_at_once(policy) -> None:
    """Empty retrieval fails two gates with two different decisions; ABSTAIN wins.

    ``retrieval_non_empty`` (ABSTAIN) and ``citation_integrity`` (ESCALATE) both trigger
    on an empty context with a cited answer. The decision follows the earlier gate, and
    the later one is still recorded in the rationale rather than dropped.
    """
    outcome = policy.decide(_answer_signals([], "Something [1]."))
    assert [result.name for result in outcome.triggered_gates] == [
        "retrieval_non_empty",
        "citation_integrity",
    ]
    assert outcome.decision is RouteDecision.ABSTAIN
    assert outcome.triggering_gate == "retrieval_non_empty"
    assert "also failed: citation_integrity" in outcome.rationale


def test_all_gates_are_evaluated_even_after_one_triggers(policy, synthetic_context) -> None:
    """The trace records the whole picture, not just the first failure.

    ``reports/gate_sensitivity_n7.md`` separates "fires on" from "changes a decision".
    That distinction is only measurable because evaluation does not short-circuit.
    """
    outcome = policy.decide(_answer_signals(synthetic_context, ABSTENTION_LINE))
    assert len(outcome.gate_results) == len(GATES)
    assert {result.name for result in outcome.gate_results} == set(EXPECTED_GATE_ORDER)
    assert [result.name for result in outcome.gate_results] == list(EXPECTED_GATE_ORDER)


def test_pre_generation_scope_evaluates_only_retrieval_gates(policy) -> None:
    """The pre-generation check exists to skip a GPU call, so its scope must be narrow.

    ``ANSWER`` here means "no retrieval-stage gate objects; proceed to generation" - not
    "this answer is servable". The blind-spot report carries that caveat as a blockquote
    because the distinction is easy to misread in a results table.
    """
    outcome = policy.decide_pre_generation(_answer_signals([], "Something [1]."))
    assert outcome.stage_scope == "retrieval"
    assert len(outcome.gate_results) == 5
    assert all(r.stage is GateStage.RETRIEVAL for r in outcome.gate_results)
    assert outcome.decision is RouteDecision.ABSTAIN
    assert outcome.triggering_gate == "retrieval_non_empty"


def test_pre_generation_passes_on_usable_evidence(policy, synthetic_context) -> None:
    outcome = policy.decide_pre_generation(compute_signals(synthetic_context))
    assert outcome.decision is RouteDecision.ANSWER
    assert outcome.triggering_gate is None
    assert outcome.rationale == "all 2 enabled retrieval-stage gates passed"


# --------------------------------------------------------------------------------------
# Switches and thresholds
# --------------------------------------------------------------------------------------


def test_disabled_gate_reports_passed_without_being_triggered(policy, synthetic_context) -> None:
    """A disabled gate must be inert, and visibly so.

    ``sensitivity.py`` holds disabled gates present in every ablation subset on exactly
    this basis: they always report ``passed`` and so cannot affect a decision.
    """
    outcome = policy.decide(_answer_signals(synthetic_context, CLEAN_ANSWER))
    results = {result.name: result for result in outcome.gate_results}
    for name in DISABLED_BY_DEFAULT:
        result = results[name]
        assert result.enabled is False
        assert result.passed is True
        assert result.triggered is False
        assert result.detail == "disabled by configuration"


def test_turning_off_the_echo_gate_releases_a_transcription(policy, synthetic_context) -> None:
    """The one gate whose removal changes a decision at n=7, switched off here.

    Its only effect on the stored run is converting H03 from ``answer`` to ``escalate``;
    ``reports/gate_sensitivity_n7.md`` §2 records that this costs a point of label
    agreement, and keeps the gate anyway. The switch is real either way.
    """
    signals = _answer_signals(synthetic_context, TRANSCRIBED_ANSWER)
    assert policy.decide(signals).decision is RouteDecision.ESCALATE
    relaxed = policy.with_config(enable_echo_gate=False)
    assert relaxed.decide(signals).decision is RouteDecision.ANSWER


def test_echo_threshold_comparison_is_strict(policy, synthetic_context) -> None:
    """``echo_ratio < max_echo_ratio``: a ratio exactly at the threshold fails.

    This is the boundary the whole sweep in ``sensitivity.py`` is built on. That module
    enumerates each observed value *and* each value plus one epsilon precisely because
    the comparison is strict, so the transition sits at ``threshold == ratio`` and not a
    hair either side. If this ever became ``<=``, every stable interval in
    ``reports/gate_sensitivity_n7.md`` would be off by one endpoint.
    """
    signals = _answer_signals(synthetic_context, TRANSCRIBED_ANSWER)
    ratio = signals.response.echo_ratio
    assert policy.with_config(max_echo_ratio=ratio).decide(signals).decision is (
        RouteDecision.ESCALATE
    )
    assert policy.with_config(max_echo_ratio=ratio + 1e-9).decide(signals).decision is (
        RouteDecision.ANSWER
    )


def test_with_config_does_not_mutate_the_original(policy, synthetic_context) -> None:
    original = policy.config
    policy.with_config(max_echo_ratio=0.1, enable_citation_gate=False)
    assert policy.config == original
    assert policy.config.max_echo_ratio == 0.85


# --------------------------------------------------------------------------------------
# PolicyConfig serialisation
# --------------------------------------------------------------------------------------


def test_config_round_trips_through_as_dict() -> None:
    config = PolicyConfig(max_echo_ratio=0.6, enable_script_gate=False)
    assert PolicyConfig.from_dict(config.as_dict()) == config


def test_config_rejects_unknown_keys() -> None:
    """A typo must not leave the default silently in force.

    If ``max_eco_ratio`` were accepted and ignored, the trace would record
    ``policy_version: gates_v1`` for a run that used 0.85 while the operator believed
    they had set 0.60. The recorded version would then describe a policy that never ran.
    """
    with pytest.raises(ValueError, match="unknown PolicyConfig key\\(s\\): max_eco_ratio"):
        PolicyConfig.from_dict({"max_eco_ratio": 0.6})


def test_config_treats_underscore_keys_as_comments() -> None:
    """``configs/reliability/gates_v1.json`` documents itself in ``_``-prefixed keys."""
    config = PolicyConfig.from_dict(
        {"_max_echo_ratio": "prose explaining the value", "max_echo_ratio": 0.6}
    )
    assert config.max_echo_ratio == 0.6


def test_shipped_config_file_equals_the_code_defaults(repo_root) -> None:
    """``gates_v1.json`` is the replay's configuration, written out for reproducibility.

    Asserting equality with the dataclass defaults is what keeps the file honest: if a
    default moved in code and the file did not follow, a reader reproducing a run from
    configuration would get different routing from the one the reports describe.
    """
    path = repo_root / "configs" / "reliability" / "gates_v1.json"
    from_file = PolicyConfig.from_json_file(path)
    assert from_file == PolicyConfig()
    assert from_file.policy_version == "gates_v1"
    # The three rejected/unanalysed gates must be off in the shipped file too, not just
    # in the dataclass defaults.
    assert from_file.enable_anchoring_gate is False
    assert from_file.enable_reranker_top_1_gate is False
    assert from_file.enable_reranker_margin_gate is False
    assert from_file.min_anchor_coverage is None


def test_outcome_as_dict_is_json_serialisable(policy, synthetic_context) -> None:
    """Traces are written as JSON, so every field has to survive the round trip.

    Note what is *not* claimed here: gate ``detail`` strings can contain fragments of the
    model's response (``script_integrity`` embeds the offending characters), which is why
    the report-writing scripts omit ``detail`` from their payloads rather than relying on
    this method to be publication-safe.
    """
    outcome = policy.decide(_answer_signals(synthetic_context, CLEAN_ANSWER))
    payload = outcome.as_dict()
    encoded = json.dumps(payload, ensure_ascii=False)
    assert json.loads(encoded) == payload
    assert payload["decision"] == "answer"
    assert payload["triggering_gate"] is None
    assert payload["stage_scope"] == "all"
    assert len(payload["gates"]) == len(GATES)
    assert payload["policy"]["policy_version"] == "gates_v1"
    assert payload["policy"]["max_echo_ratio"] == 0.85
