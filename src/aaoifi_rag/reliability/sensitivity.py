"""Sensitivity and ablation analysis for the gate battery (plan Layer 4).

Two questions a reviewer asks about any threshold-based reliability policy, answered
mechanically over **frozen** :class:`~aaoifi_rag.reliability.signals.ReliabilitySignals`
so that neither answer needs a GPU, a model call, or a re-run of retrieval:

1. **How much does the decision depend on a threshold that was set from three points?**
   ``PolicyConfig.max_echo_ratio = 0.85`` was chosen because it sits in the gap between
   the only three observed answer attempts at n=7 (H01 0.443, H05 0.717, H03 0.900).
   Three points is not a calibration. :func:`sweep_threshold` characterises the routing
   as a function of that threshold *exactly* rather than approximately — see below — and
   :class:`ThresholdSensitivity` reports the interval around the shipped value inside
   which no item's decision changes at all.

2. **Which gates actually do anything on the evidence that exists?**
   :func:`compute_ablation_lattice` evaluates all 2^k subsets of the gate battery over
   the frozen signals and reports, per gate, whether removing it changes any decision.
   A gate that never changes a decision at n=7 is *not* thereby useless - it is
   **unexercised**, which is a statement about the evidence, not about the gate. The
   distinction is recorded in :attr:`GateContribution.is_load_bearing` and in the prose
   of :meth:`AblationLattice.as_dict`.

Why sweeping *observed* values is exact, not sampled
----------------------------------------------------
A threshold gate over a fixed signal value is a step function of the threshold: the gate
``substantive_answer`` passes iff ``echo_ratio < max_echo_ratio``, so for an item with
ratio *r* the gate's state changes at exactly ``threshold == r`` and nowhere else. The
complete set of transition points for *n* items is therefore the set of *n* observed
values. Sweeping each observed value plus a point immediately above it visits every
distinct routing the policy can produce, so the sweep is a complete enumeration rather
than a grid approximation. A grid would be both slower and capable of missing a flip.

What this module deliberately does not do
-----------------------------------------
It does not choose a better threshold. Picking the midpoint of the widest stable interval
would be selecting a hyperparameter on the same seven items the system is evaluated on,
and at n=7 the stable interval is wide mostly because the sample is small. The module
reports the sensitivity and leaves the threshold where the evidence left it.

This module contains no AAOIFI clause prose, imports no torch and no rank_bm25, and
computes nothing from question text.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

from .policy import (
    Gate,
    PolicyConfig,
    RouteDecision,
    SelectivePredictionPolicy,
)
from .signals import ReliabilitySignals

#: 2**k subsets are enumerated, so k is capped. 16 gates is 65,536 subsets times the
#: item count - still trivial - and the ceiling exists to turn a future accidental
#: combinatorial blow-up into an error instead of a hang.
MAX_LATTICE_GATES = 16

#: Smallest representable step used to probe just above an observed signal value.
_EPSILON = 1e-9


@dataclass(frozen=True)
class SensitivityItem:
    """One item's frozen signals, plus its identity and reference label.

    ``expected_behavior`` is optional: the sweep is well defined without labels (it
    reports how the routing *distribution* moves), and only the agreement column needs
    them. ``signals`` must already be computed - nothing here calls a model.
    """

    item_id: str
    signals: ReliabilitySignals
    expected_behavior: str | None = None


@dataclass(frozen=True)
class ThresholdSweepPoint:
    """The complete routing at one threshold value."""

    threshold: float
    decisions: tuple[tuple[str, str], ...]
    n_answer: int
    n_abstain: int
    n_escalate: int
    n_agree: int | None
    is_shipped_value: bool = False

    @property
    def routing_vector(self) -> tuple[str, ...]:
        """Decisions in item order - the identity used to detect a flip."""
        return tuple(decision for _, decision in self.decisions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "is_shipped_value": self.is_shipped_value,
            "n_answer": self.n_answer,
            "n_abstain": self.n_abstain,
            "n_escalate": self.n_escalate,
            "n_agree": self.n_agree,
            "decisions": {item_id: decision for item_id, decision in self.decisions},
        }


@dataclass(frozen=True)
class DecisionFlip:
    """One item changing decision as the threshold crosses its signal value."""

    item_id: str
    threshold_below: float
    threshold_at_or_above: float
    decision_below: str
    decision_at_or_above: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "threshold_below": self.threshold_below,
            "threshold_at_or_above": self.threshold_at_or_above,
            "decision_below": self.decision_below,
            "decision_at_or_above": self.decision_at_or_above,
        }


@dataclass(frozen=True)
class ThresholdSensitivity:
    """Exact characterisation of the routing as a function of one threshold."""

    field_name: str
    shipped_value: float
    n_items: int
    observed_values: tuple[float, ...]
    points: tuple[ThresholdSweepPoint, ...]
    flips: tuple[DecisionFlip, ...]
    stable_interval: tuple[float, float]
    n_distinct_routings: int
    n_items_evaluable: int

    @property
    def stable_interval_width(self) -> float:
        low, high = self.stable_interval
        return high - low

    @property
    def is_decision_relevant(self) -> bool:
        """Does this threshold change any decision anywhere in its range?

        ``False`` means the gate is inert across the whole sweep on this evidence:
        every reachable threshold produces identical routing, so the value 0.85 is
        doing no work that could be observed here.
        """
        return self.n_distinct_routings > 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "shipped_value": self.shipped_value,
            "n_items": self.n_items,
            "n_items_evaluable": self.n_items_evaluable,
            "observed_values": list(self.observed_values),
            "n_distinct_routings": self.n_distinct_routings,
            "is_decision_relevant": self.is_decision_relevant,
            "stable_interval": {
                "lower": self.stable_interval[0],
                "upper": self.stable_interval[1],
                "width": self.stable_interval_width,
                "meaning": (
                    f"every threshold in this interval produces routing identical to "
                    f"{self.field_name}={self.shipped_value}; the bounds are the "
                    "adjacent observed signal values, so the interval is exact, not "
                    "sampled"
                ),
            },
            "flips": [flip.as_dict() for flip in self.flips],
            "points": [point.as_dict() for point in self.points],
            "method": "exact_enumeration_over_observed_signal_values",
            "limitation": (
                f"the transition points are the {self.n_items_evaluable} observed "
                "signal values on this evidence set; a wide stable interval reflects "
                "how few values were observed and is NOT evidence that the threshold "
                "is well chosen"
            ),
        }


def _echo_ratio(signals: ReliabilitySignals) -> float | None:
    """The value ``substantive_answer`` compares. ``None`` when the gate is inert.

    The gate short-circuits to *passed* on a non-answer (a refusal has nothing to
    transcribe), so those items are not evaluable and must not contribute a transition
    point - including them would invent a flip the threshold cannot cause.
    """
    response = signals.response
    if response is None or not response.is_answer_attempt:
        return None
    return response.echo_ratio


def _reranker_top_1(signals: ReliabilitySignals) -> float | None:
    return signals.retrieval.reranker_top_1


def _reranker_margin(signals: ReliabilitySignals) -> float | None:
    return signals.retrieval.reranker_margin


def _anchor_coverage(signals: ReliabilitySignals) -> float | None:
    anchoring = signals.retrieval.anchoring
    return None if anchoring is None else anchoring.idf_weighted_coverage


#: Which :class:`~aaoifi_rag.reliability.policy.PolicyConfig` float field reads which
#: signal. Only these four fields are numeric thresholds; everything else in the config
#: is a boolean switch, and switches are the ablation lattice's subject, not the sweep's.
THRESHOLD_SIGNALS: Mapping[str, Any] = {
    "max_echo_ratio": _echo_ratio,
    "min_reranker_top_1": _reranker_top_1,
    "min_reranker_margin": _reranker_margin,
    "min_anchor_coverage": _anchor_coverage,
}


def _decisions_at(
    items: Sequence[SensitivityItem],
    policy: SelectivePredictionPolicy,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (item.item_id, policy.decide(item.signals).decision.value) for item in items
    )


def _point(
    threshold: float,
    items: Sequence[SensitivityItem],
    policy: SelectivePredictionPolicy,
    *,
    is_shipped: bool,
) -> ThresholdSweepPoint:
    decisions = _decisions_at(items, policy)
    by_item = dict(decisions)
    labelled = [item for item in items if item.expected_behavior is not None]
    n_agree = (
        sum(1 for item in labelled if by_item[item.item_id] == item.expected_behavior)
        if labelled
        else None
    )
    return ThresholdSweepPoint(
        threshold=threshold,
        decisions=decisions,
        n_answer=sum(1 for _, d in decisions if d == RouteDecision.ANSWER.value),
        n_abstain=sum(1 for _, d in decisions if d == RouteDecision.ABSTAIN.value),
        n_escalate=sum(1 for _, d in decisions if d == RouteDecision.ESCALATE.value),
        n_agree=n_agree,
        is_shipped_value=is_shipped,
    )


def sweep_threshold(
    items: Sequence[SensitivityItem],
    *,
    field_name: str = "max_echo_ratio",
    policy: SelectivePredictionPolicy | None = None,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> ThresholdSensitivity:
    """Characterise the routing as an exact step function of one threshold.

    The candidate thresholds are the observed signal values, each observed value plus
    one epsilon, the shipped value, and the two range endpoints. That set provably
    contains every threshold at which any decision can change (see the module
    docstring), so the returned sweep is a complete enumeration.

    Raises ``ValueError`` for a field that is not a numeric threshold, rather than
    silently sweeping a value no gate reads.
    """
    if field_name not in THRESHOLD_SIGNALS:
        raise ValueError(
            f"{field_name!r} is not a numeric threshold field; known fields are "
            f"{', '.join(sorted(THRESHOLD_SIGNALS))}"
        )
    if not items:
        raise ValueError("sweep_threshold needs at least one item")
    base = policy or SelectivePredictionPolicy()
    shipped_raw = getattr(base.config, field_name)
    shipped_value = float(shipped_raw) if shipped_raw is not None else upper_bound

    extract = THRESHOLD_SIGNALS[field_name]
    observed = sorted(
        {
            value
            for value in (extract(item.signals) for item in items)
            if value is not None
        }
    )
    candidates = sorted(
        {lower_bound, upper_bound, shipped_value}
        | set(observed)
        | {value + _EPSILON for value in observed}
    )
    points = tuple(
        _point(
            threshold,
            items,
            SelectivePredictionPolicy(
                replace(base.config, **{field_name: threshold}), base.gates
            ),
            is_shipped=threshold == shipped_value,
        )
        for threshold in candidates
    )

    flips: list[DecisionFlip] = []
    for previous, current in zip(points, points[1:]):
        previous_map = dict(previous.decisions)
        for item_id, decision in current.decisions:
            if previous_map[item_id] != decision:
                flips.append(
                    DecisionFlip(
                        item_id=item_id,
                        threshold_below=previous.threshold,
                        threshold_at_or_above=current.threshold,
                        decision_below=previous_map[item_id],
                        decision_at_or_above=decision,
                    )
                )
    return ThresholdSensitivity(
        field_name=field_name,
        shipped_value=shipped_value,
        n_items=len(items),
        observed_values=tuple(observed),
        points=points,
        flips=tuple(flips),
        stable_interval=_stable_interval(points, shipped_value),
        n_distinct_routings=len({point.routing_vector for point in points}),
        n_items_evaluable=len(
            [item for item in items if extract(item.signals) is not None]
        ),
    )


def _stable_interval(
    points: Sequence[ThresholdSweepPoint], shipped_value: float
) -> tuple[float, float]:
    """Widest contiguous threshold run around the shipped value with identical routing.

    Contiguity matters: an identical routing vector recurring at a distant threshold is
    a coincidence of a small sample, not evidence that the intervening range is safe.
    """
    index = next(
        (i for i, point in enumerate(points) if point.threshold == shipped_value),
        None,
    )
    if index is None:
        return (shipped_value, shipped_value)
    target = points[index].routing_vector
    low = index
    while low - 1 >= 0 and points[low - 1].routing_vector == target:
        low -= 1
    high = index
    while high + 1 < len(points) and points[high + 1].routing_vector == target:
        high += 1
    return (points[low].threshold, points[high].threshold)


@dataclass(frozen=True)
class GateContribution:
    """What removing one gate does to the routing, holding everything else fixed."""

    gate_name: str
    stage: str
    decision_if_failed: str
    evidentiary_basis: str
    n_items_changed: int
    changed_items: tuple[str, ...]
    decision_shifts: tuple[str, ...]
    n_items_triggering: int

    @property
    def is_load_bearing(self) -> bool:
        """Does removing this gate change any decision on this evidence?"""
        return self.n_items_changed > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "stage": self.stage,
            "decision_if_failed": self.decision_if_failed,
            "evidentiary_basis": self.evidentiary_basis,
            "n_items_triggering": self.n_items_triggering,
            "n_items_changed_when_removed": self.n_items_changed,
            "changed_items": list(self.changed_items),
            "decision_shifts": list(self.decision_shifts),
            "is_load_bearing": self.is_load_bearing,
            "reading": (
                "load-bearing on this evidence"
                if self.is_load_bearing
                else "UNEXERCISED on this evidence - removing it changes nothing here, "
                "which is a statement about the items available, not about the gate"
            ),
        }


@dataclass(frozen=True)
class AblationLattice:
    """All 2^k gate subsets, evaluated over frozen signals."""

    n_items: int
    enabled_gate_names: tuple[str, ...]
    disabled_gate_names: tuple[str, ...]
    n_subsets: int
    full_routing: tuple[tuple[str, str], ...]
    contributions: tuple[GateContribution, ...]
    minimal_sufficient_subsets: tuple[tuple[str, ...], ...]
    n_distinct_routings: int

    @property
    def load_bearing_gates(self) -> tuple[str, ...]:
        return tuple(c.gate_name for c in self.contributions if c.is_load_bearing)

    @property
    def unexercised_gates(self) -> tuple[str, ...]:
        return tuple(c.gate_name for c in self.contributions if not c.is_load_bearing)

    @property
    def minimal_sufficient_size(self) -> int | None:
        if not self.minimal_sufficient_subsets:
            return None
        return len(self.minimal_sufficient_subsets[0])

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_items": self.n_items,
            "n_enabled_gates": len(self.enabled_gate_names),
            "enabled_gate_names": list(self.enabled_gate_names),
            "disabled_gate_names": list(self.disabled_gate_names),
            "n_subsets_evaluated": self.n_subsets,
            "n_distinct_routings": self.n_distinct_routings,
            "full_routing": {item: decision for item, decision in self.full_routing},
            "load_bearing_gates": list(self.load_bearing_gates),
            "unexercised_gates": list(self.unexercised_gates),
            "minimal_sufficient_size": self.minimal_sufficient_size,
            "minimal_sufficient_subsets": [
                list(subset) for subset in self.minimal_sufficient_subsets
            ],
            "contributions": [c.as_dict() for c in self.contributions],
            "interpretation": (
                f"{len(self.load_bearing_gates)} of "
                f"{len(self.enabled_gate_names)} enabled gates change at least one "
                f"decision on these {self.n_items} items; "
                f"{self.minimal_sufficient_size} gate(s) suffice to reproduce the "
                "shipped routing exactly. The remainder are unexercised by this "
                "evidence and must not be described as redundant: an unexercised gate "
                "is one whose failure mode did not occur in the sample."
            ),
            "limitation": (
                "every figure here is conditional on the frozen signals supplied. A "
                "gate ablation cannot discover failure modes absent from the item set, "
                "and at small n most gates will be unexercised by construction."
            ),
        }


def _routing_with_gates(
    items: Sequence[SensitivityItem],
    config: PolicyConfig,
    gates: tuple[Gate, ...],
) -> tuple[tuple[str, str], ...]:
    policy = SelectivePredictionPolicy(config, gates)
    return _decisions_at(items, policy)


def _partition_gates(
    policy: SelectivePredictionPolicy,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Indices of enabled and disabled gates, in declaration (precedence) order."""
    enabled: list[int] = []
    disabled: list[int] = []
    for index, gate in enumerate(policy.gates):
        (enabled if gate.is_enabled(policy.config) else disabled).append(index)
    return tuple(enabled), tuple(disabled)


def _subset_gates(
    policy: SelectivePredictionPolicy,
    keep_enabled: Iterable[int],
    disabled: Iterable[int],
) -> tuple[Gate, ...]:
    """Rebuild a gate tuple in precedence order from a set of kept indices.

    Disabled gates are retained in every subset because they always report *passed* and
    so cannot affect a decision; excluding them would only inflate the lattice.
    Precedence order is preserved, because which gate is *named* as the cause - and
    therefore which ``decision_if_failed`` wins - depends on it.
    """
    keep = set(keep_enabled) | set(disabled)
    return tuple(gate for index, gate in enumerate(policy.gates) if index in keep)


def compute_gate_contributions(
    items: Sequence[SensitivityItem],
    policy: SelectivePredictionPolicy | None = None,
) -> tuple[GateContribution, ...]:
    """Leave-one-out: what changes when each enabled gate is removed."""
    base = policy or SelectivePredictionPolicy()
    enabled, disabled = _partition_gates(base)
    full = dict(_routing_with_gates(items, base.config, base.gates))

    triggering: dict[str, int] = {}
    for item in items:
        for result in base.decide(item.signals).gate_results:
            if result.triggered:
                triggering[result.name] = triggering.get(result.name, 0) + 1

    contributions: list[GateContribution] = []
    for index in enabled:
        gate = base.gates[index]
        without = _subset_gates(base, (i for i in enabled if i != index), disabled)
        routing = dict(_routing_with_gates(items, base.config, without))
        changed = tuple(
            item_id for item_id in full if routing[item_id] != full[item_id]
        )
        contributions.append(
            GateContribution(
                gate_name=gate.name,
                stage=gate.stage.value,
                decision_if_failed=gate.decision_if_failed.value,
                evidentiary_basis=gate.evidentiary_basis.value,
                n_items_changed=len(changed),
                changed_items=changed,
                decision_shifts=tuple(
                    f"{item_id}: {full[item_id]} -> {routing[item_id]}"
                    for item_id in changed
                ),
                n_items_triggering=triggering.get(gate.name, 0),
            )
        )
    return tuple(contributions)


def compute_ablation_lattice(
    items: Sequence[SensitivityItem],
    policy: SelectivePredictionPolicy | None = None,
) -> AblationLattice:
    """Evaluate all 2^k subsets of the enabled gates over frozen signals.

    Reports the leave-one-out contribution of each gate and the *smallest* subsets that
    reproduce the shipped routing exactly. Raises ``ValueError`` above
    :data:`MAX_LATTICE_GATES` rather than attempting an intractable enumeration.
    """
    base = policy or SelectivePredictionPolicy()
    if not items:
        raise ValueError("compute_ablation_lattice needs at least one item")
    enabled, disabled = _partition_gates(base)
    if len(enabled) > MAX_LATTICE_GATES:
        raise ValueError(
            f"{len(enabled)} enabled gates would require 2**{len(enabled)} subsets; "
            f"the ceiling is {MAX_LATTICE_GATES}. Disable gates or analyse a subset."
        )
    full = _routing_with_gates(items, base.config, base.gates)
    full_map = dict(full)

    routings: set[tuple[str, ...]] = set()
    sufficient: list[tuple[str, ...]] = []
    for size in range(len(enabled) + 1):
        for keep in combinations(enabled, size):
            gates = _subset_gates(base, keep, disabled)
            routing = _routing_with_gates(items, base.config, gates)
            routings.add(tuple(decision for _, decision in routing))
            if dict(routing) == full_map:
                sufficient.append(tuple(base.gates[i].name for i in keep))
    smallest = min((len(subset) for subset in sufficient), default=None)
    minimal = tuple(
        subset for subset in sufficient if smallest is not None and len(subset) == smallest
    )
    return AblationLattice(
        n_items=len(items),
        enabled_gate_names=tuple(base.gates[i].name for i in enabled),
        disabled_gate_names=tuple(base.gates[i].name for i in disabled),
        n_subsets=2 ** len(enabled),
        full_routing=full,
        contributions=compute_gate_contributions(items, base),
        minimal_sufficient_subsets=minimal,
        n_distinct_routings=len(routings),
    )
