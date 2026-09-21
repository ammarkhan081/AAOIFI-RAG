"""Selective-prediction policy and routing decision (plan Layers 4-5 boundary).

The policy is the only place a decision is made. It reads
:class:`aaoifi_rag.reliability.signals.ReliabilitySignals` and returns a
:class:`RouteDecision` plus the full gate-by-gate record behind it. The
orchestrator sequences stages; it does not decide.

Design position, stated plainly
-------------------------------
``reports/reliability_signal_analysis_n7.md`` tested the obvious
selective-prediction signal - reranker top-1 score - and rejected it: at n=7 it
predicted neither retrieval completeness nor whether the model answered. This
module does **not** resurrect it. There is no calibrated confidence score here
and no coverage/risk guarantee is claimed.

What it does instead is gate on conditions that are *mechanically certain* given
the prompt contract, each one traceable to a defect observed in the stored n=7
run:

=========================  ===========================  =========================
Gate                       Fails on                     Evidence at n=7
=========================  ===========================  =========================
``retrieval_non_empty``    zero retrieved records       never fired (principle)
``retrieval_normative``    every record a heading       never fired (principle)
``response_present``       empty generation             never fired (principle)
``model_did_not_abstain``  the fixed abstention line    H02, H04, H06, H07
``no_self_contradiction``  answer *and* that line       H05
``citation_integrity``     bad index / misattribution   H05 (``[2]`` is rank 4)
``clause_grounding``       clause path never retrieved  never fired (principle)
``script_integrity``       out-of-script characters     H05 (Arabic for U+2019)
``no_degenerate_decoding`` 4+ character repetitions     never fired (principle)
``substantive_answer``     answer is transcription      H03 (echo 0.90)
``reranker_margin``        *disabled by default*        never analysed
=========================  ===========================  =========================

Gates marked "principle" have never fired on real data, so they are asserted
from the prompt contract, not from evidence. That distinction is recorded on
every gate in :attr:`Gate.evidentiary_basis` and travels into the trace log, so
a reader of the results never has to take this docstring's word for it.

``clause_grounding`` is a "principle" gate with one qualification worth stating,
because "never fired" is weak evidence on its own and could mean the detector is
simply broken. Its *extractor* was measured before the gate was written: 36 of 36
clause references taken across all 362 corpus clause texts are genuine
cross-references, with zero ratios or fractions mis-extracted and zero
slash-numerals missed against a bare-regex baseline. So the gate not firing at
n=7 is a fact about the seven responses - only H01 names any clause path, and
both of its distinct references are grounded - and not an artefact of a detector
that never detects anything. See :mod:`aaoifi_rag.reliability.grounding`.

Known limitations
-----------------
* Thresholds rest on three answer attempts. They are provisional, not tuned.
* No gate can detect a fluent, correctly-cited answer that is nonetheless wrong.
  Detecting that needs entailment or scholar review, neither of which exists
  here; ``corpus_cross_reference`` is not ``qualified_scholar_review``.
* An abstention is routed ``ABSTAIN`` even when the question was answerable.
  Whether that was over-abstention is an evaluation question, not an inference
  one, and is measured in :mod:`aaoifi_rag.reporting.metrics`.

This module contains no AAOIFI clause prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields, replace
from enum import StrEnum
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .response_class import ResponseClass
from .signals import ReliabilitySignals


class RouteDecision(StrEnum):
    """Terminal routing states.

    The values match the ``expected_behavior`` enum in
    ``data/manifests/hard_set_schema.json`` exactly, so evaluation can compare
    ``decision.value == item["expected_behavior"]`` with no mapping table.
    """

    #: Every gate passed; the answer may be served with its citations.
    ANSWER = "answer"
    #: No usable evidence, or the model declined. Nothing to serve.
    ABSTAIN = "abstain"
    #: An answer exists but failed a gate, or the signals conflict. Needs a human.
    ESCALATE = "escalate"


class GateStage(StrEnum):
    """When a gate can be evaluated.

    ``RETRIEVAL`` gates read only retrieval signals, so the orchestrator can run
    them before generation and skip the model call entirely when there is no
    usable evidence. ``RESPONSE`` gates need generated text.
    """

    RETRIEVAL = "retrieval"
    RESPONSE = "response"


class EvidentiaryBasis(StrEnum):
    """Why a gate exists. Recorded on every gate and carried into traces."""

    #: Fired on at least one item in the stored n=7 run.
    OBSERVED_N7 = "observed_in_n7_run"
    #: Follows from the prompt contract; never fired on real data yet.
    PROMPT_CONTRACT = "prompt_contract_untested_at_n7"
    #: Tested at n=7 and found not predictive. Off by default.
    REJECTED_N7 = "rejected_by_n7_analysis"
    #: Tested against the 25-probe negative class and found not separable from the
    #: answerable items. Off by default. See ``reports/anchoring_gate_calibration.md``.
    REJECTED_PROBES = "rejected_by_probe_calibration"
    #: Never analysed against outcomes. Off by default.
    UNANALYSED = "unanalysed"


@dataclass(frozen=True)
class PolicyConfig:
    """Thresholds and switches. Serialised into every trace."""

    #: Echo ratio at or above which an "answer" is treated as transcription.
    #: 0.85 sits in the gap between the three observed answer attempts
    #: (H01 0.443, H05 0.717, H03 0.900). Three points; provisional.
    max_echo_ratio: float = 0.85
    enable_echo_gate: bool = True

    enable_heading_only_gate: bool = True
    enable_citation_gate: bool = True
    enable_script_gate: bool = True
    enable_degenerate_decoding_gate: bool = True

    #: On. Unlike the reranker and anchoring switches below, this gate has no
    #: threshold and nothing calibrated: it asks whether a clause path named in
    #: the answer corresponds to a clause actually retrieved, which is decidable.
    #: It fires on none of the seven stored items (only H01 names a clause path,
    #: and both of its references are grounded), so enabling it moves no n=7
    #: decision; it is on because a mechanically-certain contract check should
    #: not ship inert. See ``src/aaoifi_rag/reliability/grounding.py``.
    enable_clause_grounding_gate: bool = True

    #: Off. ``reports/reliability_signal_analysis_n7.md`` rejected reranker
    #: top-1 as a selective-prediction signal; turning this on would contradict
    #: the project's own finding. Kept as an explicit switch so the rejection is
    #: visible in configuration rather than only in prose.
    min_reranker_top_1: float | None = None
    enable_reranker_top_1_gate: bool = False

    #: Off. The margin is computed by ``RerankerSignal`` and recorded, but it has
    #: never been analysed against outcomes, so gating on it would be a guess.
    min_reranker_margin: float | None = None
    enable_reranker_margin_gate: bool = False

    #: Off. ``reports/anchoring_gate_calibration.md`` measured IDF-weighted
    #: query-to-context anchor coverage on the seven answerable items and the 25
    #: probes: AUC 0.513 against the non-circular unanswerable probes, which is
    #: chance. The gate is implemented, unit-tested and left disabled, because the
    #: measurement says it would trade false abstentions for almost no detection.
    #: 0.25 is the largest threshold with zero false positives at n=7 and is
    #: recorded as the value a future run would start from, not as a live setting.
    min_anchor_coverage: float | None = None
    enable_anchoring_gate: bool = False

    policy_version: str = "gates_v1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "max_echo_ratio": self.max_echo_ratio,
            "enable_echo_gate": self.enable_echo_gate,
            "enable_heading_only_gate": self.enable_heading_only_gate,
            "enable_citation_gate": self.enable_citation_gate,
            "enable_script_gate": self.enable_script_gate,
            "enable_degenerate_decoding_gate": self.enable_degenerate_decoding_gate,
            "enable_clause_grounding_gate": self.enable_clause_grounding_gate,
            "min_reranker_top_1": self.min_reranker_top_1,
            "enable_reranker_top_1_gate": self.enable_reranker_top_1_gate,
            "min_reranker_margin": self.min_reranker_margin,
            "enable_reranker_margin_gate": self.enable_reranker_margin_gate,
            "min_anchor_coverage": self.min_anchor_coverage,
            "enable_anchoring_gate": self.enable_anchoring_gate,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PolicyConfig":
        """Build a config from a stored mapping.

        Unknown keys are rejected rather than ignored: a typo in a threshold name
        would otherwise leave the default silently in force, and the recorded
        ``policy_version`` would then describe a policy that never ran. Keys
        beginning with ``_`` are treated as comments and skipped, which is how
        ``configs/reliability/gates_v1.json`` carries its own documentation.
        """
        fields = {field.name for field in dataclass_fields(cls)}
        supplied = {
            key: value
            for key, value in payload.items()
            if not key.startswith("_")
        }
        unknown = sorted(set(supplied) - fields)
        if unknown:
            raise ValueError(
                f"unknown PolicyConfig key(s): {', '.join(unknown)}; "
                f"known keys are {', '.join(sorted(fields))}"
            )
        return cls(**supplied)

    @classmethod
    def from_json_file(cls, path: "Path | str") -> "PolicyConfig":
        """Load a config from a JSON file such as ``configs/reliability/gates_v1.json``."""
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))


@dataclass(frozen=True)
class GateResult:
    """Outcome of one gate on one item."""

    name: str
    enabled: bool
    passed: bool
    decision_if_failed: RouteDecision
    evidentiary_basis: EvidentiaryBasis
    detail: str
    stage: GateStage = GateStage.RESPONSE

    @property
    def triggered(self) -> bool:
        return self.enabled and not self.passed

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "passed": self.passed,
            "triggered": self.triggered,
            "stage": self.stage.value,
            "decision_if_failed": self.decision_if_failed.value,
            "evidentiary_basis": self.evidentiary_basis.value,
            "detail": self.detail,
        }


#: A gate check returns ``(passed, human-readable detail)``.
GateCheck = Callable[[ReliabilitySignals, PolicyConfig], "tuple[bool, str]"]


@dataclass(frozen=True)
class Gate:
    """A single mechanical condition, plus what failing it means."""

    name: str
    check: GateCheck
    decision_if_failed: RouteDecision
    evidentiary_basis: EvidentiaryBasis
    is_enabled: Callable[[PolicyConfig], bool] = lambda config: True
    stage: GateStage = GateStage.RESPONSE

    def evaluate(
        self, signals: ReliabilitySignals, config: PolicyConfig
    ) -> GateResult:
        enabled = self.is_enabled(config)
        if not enabled:
            return GateResult(
                name=self.name,
                enabled=False,
                passed=True,
                decision_if_failed=self.decision_if_failed,
                evidentiary_basis=self.evidentiary_basis,
                detail="disabled by configuration",
                stage=self.stage,
            )
        passed, detail = self.check(signals, config)
        return GateResult(
            name=self.name,
            enabled=True,
            passed=passed,
            decision_if_failed=self.decision_if_failed,
            evidentiary_basis=self.evidentiary_basis,
            detail=detail,
            stage=self.stage,
        )


def _check_retrieval_non_empty(
    signals: ReliabilitySignals, config: PolicyConfig
) -> tuple[bool, str]:
    count = signals.retrieval.retrieved_count
    return count > 0, f"retrieved_count={count}"


def _check_retrieval_normative(
    signals: ReliabilitySignals, config: PolicyConfig
) -> tuple[bool, str]:
    retrieval = signals.retrieval
    if retrieval.is_empty:
        return True, "no records to judge; earlier gate owns this case"
    detail = (
        f"heading_like_ranks={list(retrieval.heading_like_ranks)} "
        f"of {retrieval.retrieved_count}"
    )
    return not retrieval.all_heading_like, detail


def _check_response_present(
    signals: ReliabilitySignals, config: PolicyConfig
) -> tuple[bool, str]:
    response = signals.response
    if response is None:
        return False, "generation was not attempted"
    is_empty = response.response_class is ResponseClass.EMPTY
    return not is_empty, f"response_class={response.response_class.value}"


def _check_model_did_not_abstain(
    signals: ReliabilitySignals, config: PolicyConfig
) -> tuple[bool, str]:
    response = signals.response
    if response is None:
        return True, "no response; earlier gate owns this case"
    abstained = response.response_class is ResponseClass.ABSTAINED
    return (
        not abstained,
        f"response_class={response.response_class.value} "
        f"abstention_line_present={response.classification.abstention_line_present}",
    )


def _check_no_self_contradiction(
    signals: ReliabilitySignals, config: PolicyConfig
) -> tuple[bool, str]:
    """MIXED means the model answered *and* emitted the abstention line (H05)."""
    response = signals.response
    if response is None:
        return True, "no response; earlier gate owns this case"
    mixed = response.response_class is ResponseClass.MIXED
    return (
        not mixed,
        f"response_class={response.response_class.value} "
        f"residue_words={response.classification.residue_word_count}",
    )


def _check_citation_integrity(
    signals: ReliabilitySignals, config: PolicyConfig
) -> tuple[bool, str]:
    response = signals.response
    if response is None:
        return True, "no response; earlier gate owns this case"
    audit = response.citation_audit
    if not response.is_answer_attempt:
        return True, "not an answer attempt; nothing to cite"
    failures = [
        f"[{segment.rank}]:{segment.attribution.value}"
        f"(best=r{segment.best_matching_rank}@{segment.best_overlap:.2f})"
        for segment in audit.failing_segments
    ]
    detail = (
        f"cited_ranks={list(audit.cited_ranks)} context_size={audit.context_size} "
        f"uncited={audit.uncited_answer_attempt} failures={failures}"
    )
    return audit.passed, detail


def _check_clause_grounding(
    signals: ReliabilitySignals, config: PolicyConfig
) -> tuple[bool, str]:
    """Did every clause path the answer names correspond to a retrieved clause?

    The companion to ``citation_integrity``: that gate checks the ``[n]`` rank
    markers the prompt hands the model, this one checks the AAOIFI clause paths
    the model writes in prose (``SS8 §2/4/1``). Both are mechanically certain, so
    both fail closed to ESCALATE rather than feeding a score.

    Silent on an answer that names no clause path - six of the seven stored
    responses. That is deliberate: an unreferenced answer is
    ``citation_integrity``'s and ``substantive_answer``'s business, not this
    gate's, and inventing a failure here would double-count them.
    """
    response = signals.response
    if response is None:
        return True, "no response; earlier gate owns this case"
    audit = response.clause_grounding
    if audit is None:
        # Reachable only if a caller constructs ResponseSignals by hand;
        # compute_response_signals always populates it. Not-measured is not a
        # failure - the same rule the anchoring gate follows.
        return True, "clause grounding not computed; not evaluable"
    if not response.is_answer_attempt:
        return True, "not an answer attempt; nothing to ground"
    failures = [
        f"{reference.standard_id or '?'}:{reference.clause_path}={reference.status.value}"
        for reference in audit.failing_references
    ]
    detail = (
        f"n_references={len(audit.references)} "
        f"distinct={audit.distinct_reference_count} "
        f"context_standards={list(audit.context_standards)} "
        f"corpus_index={audit.corpus_index_used} failures={failures}"
    )
    return audit.passed, detail


def _check_script_integrity(
    signals: ReliabilitySignals, config: PolicyConfig
) -> tuple[bool, str]:
    response = signals.response
    if response is None:
        return True, "no response; earlier gate owns this case"
    found = response.unexpected_script
    return not found, f"unexpected_script={list(found)}"


def _check_no_degenerate_decoding(
    signals: ReliabilitySignals, config: PolicyConfig
) -> tuple[bool, str]:
    response = signals.response
    if response is None:
        return True, "no response; earlier gate owns this case"
    runs = response.repeated_runs
    return not runs, f"repeated_char_runs={len(runs)}"


def _check_substantive_answer(
    signals: ReliabilitySignals, config: PolicyConfig
) -> tuple[bool, str]:
    """An answer that is almost entirely verbatim context is a transcription."""
    response = signals.response
    if response is None or not response.is_answer_attempt:
        return True, "not an answer attempt; nothing to measure"
    ratio = response.echo_ratio
    return (
        ratio < config.max_echo_ratio,
        f"echo_ratio={ratio:.4f} threshold={config.max_echo_ratio}",
    )


def _check_reranker_top_1(
    signals: ReliabilitySignals, config: PolicyConfig
) -> tuple[bool, str]:
    score = signals.retrieval.reranker_top_1
    threshold = config.min_reranker_top_1
    if score is None or threshold is None:
        return True, f"top_1={score} threshold={threshold}; not evaluable"
    return score >= threshold, f"top_1={score:.4f} threshold={threshold}"


def _check_reranker_margin(
    signals: ReliabilitySignals, config: PolicyConfig
) -> tuple[bool, str]:
    margin = signals.retrieval.reranker_margin
    threshold = config.min_reranker_margin
    if margin is None or threshold is None:
        return True, f"margin={margin} threshold={threshold}; not evaluable"
    return margin >= threshold, f"margin={margin:.4f} threshold={threshold}"


def _check_lexical_anchoring(
    signals: ReliabilitySignals, config: PolicyConfig
) -> tuple[bool, str]:
    """Does the retrieved context mention the question's distinctive terms?

    Passes when the signal was not computed, when the query has no anchors, or
    when no threshold is configured. Absence of a measurement is never treated as
    a failed measurement: a question made entirely of function words yields
    ``coverage is None``, and abstaining on that would be inventing evidence.
    """
    anchoring = signals.retrieval.anchoring
    threshold = config.min_anchor_coverage
    if anchoring is None:
        return True, "anchoring signal not computed; not evaluable"
    if threshold is None:
        return True, "no min_anchor_coverage configured; not evaluable"
    coverage = anchoring.idf_weighted_coverage
    if coverage is None:
        return True, f"no anchors in query (n_anchors={anchoring.n_anchors}); not evaluable"
    return (
        coverage >= threshold,
        f"idf_weighted_coverage={coverage:.4f} threshold={threshold} "
        f"n_anchors={anchoring.n_anchors} "
        f"n_uncovered={len(anchoring.uncovered_anchors)}",
    )


#: Declaration order is precedence order. Absence gates (nothing to serve) come
#: before defect gates (something to serve, but flawed), so the first triggered
#: gate is also the correct one to name as the cause.
GATES: tuple[Gate, ...] = (
    Gate(
        "retrieval_non_empty",
        _check_retrieval_non_empty,
        RouteDecision.ABSTAIN,
        EvidentiaryBasis.PROMPT_CONTRACT,
        stage=GateStage.RETRIEVAL,
    ),
    Gate(
        "retrieval_normative",
        _check_retrieval_normative,
        RouteDecision.ABSTAIN,
        EvidentiaryBasis.PROMPT_CONTRACT,
        lambda config: config.enable_heading_only_gate,
        stage=GateStage.RETRIEVAL,
    ),
    Gate(
        "lexical_anchoring",
        _check_lexical_anchoring,
        RouteDecision.ABSTAIN,
        EvidentiaryBasis.REJECTED_PROBES,
        lambda config: config.enable_anchoring_gate,
        stage=GateStage.RETRIEVAL,
    ),
    Gate(
        "response_present",
        _check_response_present,
        RouteDecision.ABSTAIN,
        EvidentiaryBasis.PROMPT_CONTRACT,
    ),
    Gate(
        "model_did_not_abstain",
        _check_model_did_not_abstain,
        RouteDecision.ABSTAIN,
        EvidentiaryBasis.OBSERVED_N7,
    ),
    Gate(
        "no_self_contradiction",
        _check_no_self_contradiction,
        RouteDecision.ESCALATE,
        EvidentiaryBasis.OBSERVED_N7,
    ),
    Gate(
        "citation_integrity",
        _check_citation_integrity,
        RouteDecision.ESCALATE,
        EvidentiaryBasis.OBSERVED_N7,
        lambda config: config.enable_citation_gate,
    ),
    Gate(
        "clause_grounding",
        _check_clause_grounding,
        RouteDecision.ESCALATE,
        EvidentiaryBasis.PROMPT_CONTRACT,
        lambda config: config.enable_clause_grounding_gate,
    ),
    Gate(
        "script_integrity",
        _check_script_integrity,
        RouteDecision.ESCALATE,
        EvidentiaryBasis.OBSERVED_N7,
        lambda config: config.enable_script_gate,
    ),
    Gate(
        "no_degenerate_decoding",
        _check_no_degenerate_decoding,
        RouteDecision.ESCALATE,
        EvidentiaryBasis.PROMPT_CONTRACT,
        lambda config: config.enable_degenerate_decoding_gate,
    ),
    Gate(
        "substantive_answer",
        _check_substantive_answer,
        RouteDecision.ESCALATE,
        EvidentiaryBasis.OBSERVED_N7,
        lambda config: config.enable_echo_gate,
    ),
    Gate(
        "reranker_top_1",
        _check_reranker_top_1,
        RouteDecision.ESCALATE,
        EvidentiaryBasis.REJECTED_N7,
        lambda config: config.enable_reranker_top_1_gate,
        stage=GateStage.RETRIEVAL,
    ),
    Gate(
        "reranker_margin",
        _check_reranker_margin,
        RouteDecision.ESCALATE,
        EvidentiaryBasis.UNANALYSED,
        lambda config: config.enable_reranker_margin_gate,
        stage=GateStage.RETRIEVAL,
    ),
)


@dataclass(frozen=True)
class RoutingOutcome:
    """A decision plus the complete evidence for it."""

    decision: RouteDecision
    triggering_gate: str | None
    gate_results: tuple[GateResult, ...]
    policy: PolicyConfig
    rationale: str
    #: Which gate subset was evaluated: ``"retrieval"`` for the pre-generation
    #: check, ``"all"`` for the full post-generation decision.
    stage_scope: str = "all"

    @property
    def triggered_gates(self) -> tuple[GateResult, ...]:
        return tuple(result for result in self.gate_results if result.triggered)

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "triggering_gate": self.triggering_gate,
            "stage_scope": self.stage_scope,
            "rationale": self.rationale,
            "policy": self.policy.as_dict(),
            "gates": [result.as_dict() for result in self.gate_results],
        }


class SelectivePredictionPolicy:
    """Evaluate every gate, then route on the first one that triggers.

    All gates are evaluated even after one triggers, so the trace records the
    full picture rather than stopping at the first failure. Only the ordering in
    :data:`GATES` determines which gate is named as the cause.
    """

    def __init__(
        self,
        config: PolicyConfig | None = None,
        gates: tuple[Gate, ...] = GATES,
    ) -> None:
        self.config = config or PolicyConfig()
        self.gates = gates

    def with_config(self, **overrides: Any) -> "SelectivePredictionPolicy":
        """Return a copy with individual config fields replaced."""
        return SelectivePredictionPolicy(replace(self.config, **overrides), self.gates)

    def decide(self, signals: ReliabilitySignals) -> RoutingOutcome:
        """Full post-generation decision over every gate."""
        return self._evaluate(signals, self.gates, "all")

    def decide_pre_generation(self, signals: ReliabilitySignals) -> RoutingOutcome:
        """Decide using retrieval-stage gates only.

        Lets the orchestrator abstain before spending a generation call when the
        evidence is unusable. Returns ``ANSWER`` to mean "no retrieval-stage gate
        objects; proceed to generation", not "this answer is servable".
        """
        retrieval_gates = tuple(
            gate for gate in self.gates if gate.stage is GateStage.RETRIEVAL
        )
        return self._evaluate(signals, retrieval_gates, GateStage.RETRIEVAL.value)

    def _evaluate(
        self,
        signals: ReliabilitySignals,
        gates: tuple[Gate, ...],
        stage_scope: str,
    ) -> RoutingOutcome:
        results = tuple(gate.evaluate(signals, self.config) for gate in gates)
        triggered = [result for result in results if result.triggered]
        if not triggered:
            enabled_count = sum(1 for result in results if result.enabled)
            scope_phrase = (
                "gates" if stage_scope == "all" else f"{stage_scope}-stage gates"
            )
            return RoutingOutcome(
                decision=RouteDecision.ANSWER,
                triggering_gate=None,
                gate_results=results,
                policy=self.config,
                rationale=f"all {enabled_count} enabled {scope_phrase} passed",
                stage_scope=stage_scope,
            )
        first = triggered[0]
        also = [result.name for result in triggered[1:]]
        rationale = f"{first.name} failed ({first.detail})"
        if also:
            rationale += f"; also failed: {', '.join(also)}"
        return RoutingOutcome(
            decision=first.decision_if_failed,
            triggering_gate=first.name,
            gate_results=results,
            policy=self.config,
            rationale=rationale,
            stage_scope=stage_scope,
        )


def describe_gates(config: PolicyConfig | None = None) -> list[dict[str, Any]]:
    """Static description of the gate set, for docs and trace headers."""
    resolved = config or PolicyConfig()
    return [
        {
            "name": gate.name,
            "enabled": gate.is_enabled(resolved),
            "stage": gate.stage.value,
            "decision_if_failed": gate.decision_if_failed.value,
            "evidentiary_basis": gate.evidentiary_basis.value,
        }
        for gate in GATES
    ]
