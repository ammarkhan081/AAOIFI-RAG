"""Descriptive metrics over routed items (plan Layer 6).

Everything here is descriptive. Nothing here produces a confidence interval,
p-value or significance claim at n=7, and the guard that prevents it is code
rather than a caveat: :func:`wilson_interval` raises below
:data:`MIN_N_FOR_INTERVALS`.

Three metrics the plan lists are **not** computed, and each is reported as
``None`` with a machine-readable reason rather than silently omitted:

``residual_error_rate``
    Needs a correctness judgement on served answers. The gates in
    :mod:`aaoifi_rag.reliability.policy` cannot tell a fluent, well-cited wrong
    answer from a right one; that needs entailment checking or scholar review.
``escalation_precision`` / ``escalation_recall``
    Need items whose ``expected_behavior`` is ``escalate``. The hard set contains
    none - all seven are ``answer`` - so there are no positives to recover and the
    quantities are undefined, not merely small.

Two of those three became computable once a negative class existed. Over a set that
mixes the hard set with ``data/probes/unanswerable_probes.jsonl``,
:func:`compute_selective_risk` reports abstention precision and recall - and, in a
separate block that must never be pooled with them, escalation precision and recall
under the project's own stipulated definition. ``residual_error_rate`` remains
uncomputable and remains listed above, because a negative class does not tell you
whether a *served* answer was right.

:func:`clopper_pearson_interval` is the counterpart guard to
:func:`wilson_interval`. Wilson refuses to compute below
:data:`MIN_N_FOR_INTERVALS`; the exact interval computes at any n and then reports
its own uninformativeness, so "1/7" is answered with "anywhere from 0.4% to 58%"
rather than with a refusal that invites guessing.

Macro and micro recall are both reported and both named. They differ materially
at n=7 and the stored reports print the macro value labelled only "mean".

This module contains no AAOIFI clause prose.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .evaluation import EvaluationLabel

#: Below this n, :func:`wilson_interval` refuses to produce an interval. 30 is a
#: conventional floor, not a plan requirement; the plan's pilot target is
#: n>=100-150. The point is that no interval is emitted for the n=7 prototype.
MIN_N_FOR_INTERVALS = 30


class InsufficientSampleError(ValueError):
    """Raised when an interval is requested below :data:`MIN_N_FOR_INTERVALS`."""


def wilson_interval(
    successes: int,
    n: int,
    *,
    z: float = 1.959963984540054,
    min_n: int = MIN_N_FOR_INTERVALS,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Raises :class:`InsufficientSampleError` when ``n < min_n``. Callers that want
    a descriptive proportion without an interval should read the ``*_rate``
    fields on :class:`RoutingMetrics` instead of lowering ``min_n``.
    """
    if n <= 0:
        raise InsufficientSampleError("n must be positive")
    if n < min_n:
        raise InsufficientSampleError(
            f"n={n} is below min_n={min_n}; refusing to emit an interval. "
            "The n=7 prototype supports descriptive counts only."
        )
    if not 0 <= successes <= n:
        raise ValueError(f"successes={successes} out of range for n={n}")
    proportion = successes / n
    denominator = 1 + z**2 / n
    centre = proportion + z**2 / (2 * n)
    spread = z * math.sqrt(proportion * (1 - proportion) / n + z**2 / (4 * n**2))
    return ((centre - spread) / denominator, (centre + spread) / denominator)


#: An exact interval wider than this is reported with ``is_informative=False``.
#: 0.20 is a stated readability threshold, not a statistical standard: an interval
#: spanning more than 20 percentage points cannot separate the hypotheses this
#: project cares about (is the residual error rate 2% or 30%?). It is a knob, and
#: it is named so that it is visibly a knob.
MAX_INFORMATIVE_WIDTH = 0.20


def _binomial_pmf_sum(n: int, p: float, lo: int, hi: int) -> float:
    """Exact ``sum_{i=lo}^{hi} C(n,i) p^i (1-p)^(n-i)`` via :func:`math.comb`.

    Exact integer binomial coefficients rather than ``lgamma``, because the n here
    is single or double digits and correctness is cheaper than speed.
    """
    return sum(
        math.comb(n, i) * p**i * (1.0 - p) ** (n - i) for i in range(lo, hi + 1)
    )


def _bisect_probability(target: float, monotone_increasing_in_p: Any) -> float:
    """Solve ``f(p) = target`` on [0, 1] for a monotone increasing ``f``."""
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if monotone_increasing_in_p(mid) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


@dataclass(frozen=True)
class ExactInterval:
    """A Clopper-Pearson interval that reports its own uninformativeness.

    This exists to make a *negative* point precisely. The project refuses a Wilson
    interval below :data:`MIN_N_FOR_INTERVALS` (:func:`wilson_interval` raises), but
    refusing to compute anything invites the reply "so how bad is it really?". The
    exact interval answers that question in the only honest way available at n=7:
    by being too wide to act on, and saying so in a machine-readable field.
    """

    successes: int
    n: int
    lower: float
    upper: float
    confidence: float
    max_informative_width: float = MAX_INFORMATIVE_WIDTH

    @property
    def point_estimate(self) -> float:
        return self.successes / self.n if self.n else 0.0

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def is_informative(self) -> bool:
        return self.width <= self.max_informative_width

    def as_dict(self) -> dict[str, Any]:
        return {
            "successes": self.successes,
            "n": self.n,
            "point_estimate": round(self.point_estimate, 6),
            "lower": round(self.lower, 6),
            "upper": round(self.upper, 6),
            "width": round(self.width, 6),
            "confidence": self.confidence,
            "method": "clopper_pearson_exact",
            "is_informative": self.is_informative,
            "max_informative_width": self.max_informative_width,
            "interpretation": (
                f"{self.successes}/{self.n} is consistent with any true rate from "
                f"{self.lower:.1%} to {self.upper:.1%} at {self.confidence:.0%} "
                "confidence"
                + (
                    ""
                    if self.is_informative
                    else "; the interval is too wide to distinguish the hypotheses "
                    "this project cares about, so the point estimate must not be "
                    "quoted as a rate"
                )
            ),
        }


def clopper_pearson_interval(
    successes: int,
    n: int,
    *,
    confidence: float = 0.95,
    max_informative_width: float = MAX_INFORMATIVE_WIDTH,
) -> ExactInterval:
    """Exact binomial interval, computed by bisection on the exact binomial CDF.

    Unlike :func:`wilson_interval` this does **not** raise at small ``n``. It is
    valid at every ``n``; that is the whole point of an exact method. What it will
    not do is pretend the result is usable: at 1/7 the 95% interval spans roughly
    0.4% to 58%, :attr:`ExactInterval.is_informative` is ``False``, and
    :meth:`ExactInterval.as_dict` says in words that the point estimate must not be
    quoted as a rate.

    No scipy dependency: the reporting layer must import without a GPU or a
    scientific stack, matching the constraint on
    :mod:`aaoifi_rag.reliability.anchoring`.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError(f"successes={successes} out of range for n={n}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    tail = (1.0 - confidence) / 2.0

    # P(X >= successes | p) increases in p; the lower bound is where it equals tail.
    lower = (
        0.0
        if successes == 0
        else _bisect_probability(
            tail, lambda p: _binomial_pmf_sum(n, p, successes, n)
        )
    )
    # P(X <= successes | p) decreases in p, so bisect on its complement.
    upper = (
        1.0
        if successes == n
        else _bisect_probability(
            1.0 - tail, lambda p: 1.0 - _binomial_pmf_sum(n, p, 0, successes)
        )
    )
    return ExactInterval(
        successes=successes,
        n=n,
        lower=lower,
        upper=upper,
        confidence=confidence,
        max_informative_width=max_informative_width,
    )


#: Reasons a metric the plan lists is not computable from this evidence.
NOT_COMPUTABLE = {
    "residual_error_rate": (
        "needs a correctness judgement on served answers; the mechanical gates "
        "cannot distinguish a fluent well-cited wrong answer from a right one"
    ),
    "escalation_precision": (
        "no hard-set item has expected_behavior='escalate', so there are no "
        "positives; undefined rather than small"
    ),
    "escalation_recall": (
        "no hard-set item has expected_behavior='escalate', so there are no "
        "positives; undefined rather than small"
    ),
}


@dataclass(frozen=True)
class RoutingMetrics:
    """Descriptive routing and retrieval aggregates for one configuration."""

    n_items: int
    n_answer: int
    n_abstain: int
    n_escalate: int
    n_decision_matches_expected: int
    #: Counts of the gate named as the cause, by gate name.
    triggering_gates: dict[str, int]
    #: Mean of per-item recall. What the stored reports print as "mean".
    macro_recall: float | None
    #: Total gold clauses retrieved / total gold clauses.
    micro_recall: float | None
    n_gold_total: int
    n_gold_in_context_total: int
    n_all_gold_in_context: int
    n_any_gold_in_context: int
    #: Items whose expected_behavior is 'answer' but which did not answer.
    n_over_abstention: int
    label: str = "routing_metrics_v1"

    @property
    def coverage(self) -> float:
        """Fraction of items served an answer. The plan's Coverage."""
        return self.n_answer / self.n_items if self.n_items else 0.0

    @property
    def abstention_rate(self) -> float:
        return self.n_abstain / self.n_items if self.n_items else 0.0

    @property
    def escalation_rate(self) -> float:
        return self.n_escalate / self.n_items if self.n_items else 0.0

    @property
    def routing_agreement(self) -> float:
        return (
            self.n_decision_matches_expected / self.n_items if self.n_items else 0.0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_items": self.n_items,
            "n_answer": self.n_answer,
            "n_abstain": self.n_abstain,
            "n_escalate": self.n_escalate,
            "coverage": round(self.coverage, 6),
            "abstention_rate": round(self.abstention_rate, 6),
            "escalation_rate": round(self.escalation_rate, 6),
            "routing_agreement": round(self.routing_agreement, 6),
            "n_decision_matches_expected": self.n_decision_matches_expected,
            "n_over_abstention": self.n_over_abstention,
            "triggering_gates": dict(self.triggering_gates),
            "clause_recall": {
                "macro": None if self.macro_recall is None else round(self.macro_recall, 6),
                "micro": None if self.micro_recall is None else round(self.micro_recall, 6),
                "n_gold_total": self.n_gold_total,
                "n_gold_in_context_total": self.n_gold_in_context_total,
                "n_all_gold_in_context": self.n_all_gold_in_context,
                "n_any_gold_in_context": self.n_any_gold_in_context,
                "note": (
                    "macro is the mean of per-item recall; micro is total hits over "
                    "total golds. They differ at n=7 and must be labelled."
                ),
            },
            "not_computable": dict(NOT_COMPUTABLE),
            "intervals": (
                f"omitted: n={self.n_items} < min_n={MIN_N_FOR_INTERVALS}"
                if self.n_items < MIN_N_FOR_INTERVALS
                else "computable; call wilson_interval explicitly"
            ),
            "verification_note": (
                "Descriptive only. Labels are corpus_cross_reference, not "
                "qualified_scholar_review. No significance is claimed."
            ),
        }


def compute_routing_metrics(
    labels: Sequence[EvaluationLabel],
    *,
    triggering_gates: Sequence[str | None] | None = None,
) -> RoutingMetrics:
    """Aggregate per-item labels into one configuration's descriptive metrics.

    ``labels`` carry the decision, the authored ``expected_behavior`` and (when
    the caller passed context records to :func:`build_evaluation_label`) the
    per-item :class:`~aaoifi_rag.reporting.evaluation.RetrievalScore`.

    ``triggering_gates`` is positionally aligned with ``labels`` and holds
    :attr:`~aaoifi_rag.reliability.policy.RoutingOutcome.triggering_gate` for
    each item. ``None`` entries mean no gate objected and are counted under the
    key ``"none"`` rather than dropped, so the counts sum to ``n_items``.

    Recall is ``None`` - not ``0.0`` - when no label carries a retrieval score.
    Zero recall and unmeasured recall are different claims.
    """
    if triggering_gates is not None and len(triggering_gates) != len(labels):
        raise ValueError(
            f"triggering_gates has {len(triggering_gates)} entries for "
            f"{len(labels)} labels; they must be positionally aligned"
        )

    gate_counts: dict[str, int] = {}
    for gate in triggering_gates or ():
        key = gate or "none"
        gate_counts[key] = gate_counts.get(key, 0) + 1

    scored = [label.retrieval for label in labels if label.retrieval is not None]
    n_gold_total = sum(score.n_gold for score in scored)
    n_gold_in_context_total = sum(score.n_gold_in_context for score in scored)

    return RoutingMetrics(
        n_items=len(labels),
        n_answer=sum(1 for label in labels if label.decision == "answer"),
        n_abstain=sum(1 for label in labels if label.decision == "abstain"),
        n_escalate=sum(1 for label in labels if label.decision == "escalate"),
        n_decision_matches_expected=sum(
            1 for label in labels if label.decision_matches_expected
        ),
        triggering_gates=gate_counts,
        macro_recall=(
            sum(score.recall for score in scored) / len(scored) if scored else None
        ),
        micro_recall=(
            n_gold_in_context_total / n_gold_total if n_gold_total else None
        ),
        n_gold_total=n_gold_total,
        n_gold_in_context_total=n_gold_in_context_total,
        n_all_gold_in_context=sum(1 for score in scored if score.all_gold_in_context),
        n_any_gold_in_context=sum(1 for score in scored if score.any_gold_in_context),
        n_over_abstention=sum(
            1
            for label in labels
            if label.expected_behavior == "answer" and label.decision != "answer"
        ),
    )


def metrics_from_traces(traces: Sequence[Mapping[str, Any]]) -> RoutingMetrics:
    """Re-derive the same aggregates from stored trace payloads.

    Works on either trace view, because every field it reads
    (``routing.decision``, ``routing.triggering_gate``, ``evaluation``) is
    present in the public view as well as the full one. That is the point: a
    reader holding only the published, clause-text-free traces can reproduce
    every number in :class:`RoutingMetrics` without the licensed corpus.

    Raises :class:`ValueError` on a trace with no ``evaluation`` block, since a
    routing decision with no reference label cannot be scored for agreement.
    """
    labels: list[EvaluationLabel] = []
    gates: list[str | None] = []
    for trace in traces:
        evaluation = trace.get("evaluation")
        if not evaluation:
            raise ValueError(
                f"trace {trace.get('item_id')!r} has no evaluation block; "
                "cannot aggregate routing agreement without a reference label"
            )
        retrieval = evaluation.get("retrieval")
        labels.append(
            EvaluationLabel(
                item_id=str(evaluation["item_id"]),
                expected_behavior=str(evaluation["expected_behavior"]),
                answerability=str(evaluation["answerability"]),
                verification_basis=str(evaluation["verification_basis"]),
                status=str(evaluation["status"]),
                decision=str(evaluation["decision"]),
                retrieval=_retrieval_score_from_dict(retrieval),
            )
        )
        gates.append(trace.get("routing", {}).get("triggering_gate"))
    return compute_routing_metrics(labels, triggering_gates=gates)


#: Fields of :class:`SelectiveRiskMetrics` that must never be pooled with the
#: abstention fields. The cross-standard escalation probes carry
#: ``verification_basis == "stipulated_definition"``: this project *defined* what
#: requires human review, and no scholar agreed to that definition. Reject the
#: stipulation and every escalation number below disappears; the abstention numbers,
#: which rest on mechanical corpus absence, do not.
ESCALATION_KEYS = ("escalation_precision", "escalation_recall", "n_escalation_expected")


@dataclass(frozen=True)
class SelectiveRiskMetrics:
    """Abstention precision/recall over a set that contains real negatives.

    Computable only because :mod:`aaoifi_rag.reporting.probes` supplies a negative
    class. With the seven hard-set items alone every ``expected_behavior`` is
    ``answer``, so abstention has no positives and the quantity is undefined - which
    is exactly what :data:`NOT_COMPUTABLE` recorded before the probes existed.

    Two deliberate structural choices:

    1. **Escalation is a separate partition, not a third label folded into
       abstention.** Items whose ``expected_behavior`` is ``escalate`` are excluded
       from the abstention numerator and denominator entirely. See
       :data:`ESCALATION_KEYS`.
    2. **The asymmetric-risk cell is named and counted.** ``n_unsafe_answers`` is
       items the corpus provably cannot support that were nonetheless served an
       answer. That single count is the quantity the whole research problem is
       about, and it is reported before any precision or recall.
    """

    n_items: int
    #: Items whose expected_behavior is 'abstain' or 'answer'. Escalation excluded.
    n_abstention_scope: int
    n_abstain_expected: int
    n_answer_expected: int
    n_escalation_expected: int
    #: expected abstain, decided abstain.
    n_true_abstain: int
    #: expected answer, decided abstain. An over-abstention on a real question.
    n_false_abstain: int
    #: expected abstain, decided answer. The dangerous cell.
    n_unsafe_answers: int
    #: expected abstain, decided escalate. Not correct, but not unsafe either:
    #: the item reaches a human. Counted apart from both cells above.
    n_abstain_expected_escalated: int
    n_true_escalate: int
    n_decided_escalate: int
    label: str = "selective_risk_v1"

    @property
    def n_decided_abstain_in_scope(self) -> int:
        return self.n_true_abstain + self.n_false_abstain

    @property
    def abstention_precision(self) -> float | None:
        """Of items routed ABSTAIN, the fraction that should have been.

        ``None`` when nothing was routed abstain - undefined, not 1.0.
        """
        if not self.n_decided_abstain_in_scope:
            return None
        return self.n_true_abstain / self.n_decided_abstain_in_scope

    @property
    def abstention_recall(self) -> float | None:
        """Of items that should abstain, the fraction that did.

        ``None`` when the set has no unanswerable items, which is the pre-probe
        situation and the reason this class exists.
        """
        if not self.n_abstain_expected:
            return None
        return self.n_true_abstain / self.n_abstain_expected

    @property
    def unsafe_answer_rate(self) -> float | None:
        """Fraction of provably-unanswerable items served an answer anyway."""
        if not self.n_abstain_expected:
            return None
        return self.n_unsafe_answers / self.n_abstain_expected

    @property
    def escalation_precision(self) -> float | None:
        if not self.n_decided_escalate:
            return None
        return self.n_true_escalate / self.n_decided_escalate

    @property
    def escalation_recall(self) -> float | None:
        if not self.n_escalation_expected:
            return None
        return self.n_true_escalate / self.n_escalation_expected

    def as_dict(self, *, confidence: float = 0.95) -> dict[str, Any]:
        """Counts, rates, and an exact interval on every rate that has a denominator.

        The intervals are :func:`clopper_pearson_interval`, not Wilson, and each one
        carries its own ``is_informative`` flag. At the sizes this project works with
        they will almost all be ``False``; that is the honest reading, not a defect.
        """

        def interval(successes: int | None, n: int) -> dict[str, Any] | None:
            if successes is None or n <= 0:
                return None
            return clopper_pearson_interval(
                successes, n, confidence=confidence
            ).as_dict()

        return {
            "label": self.label,
            "n_items": self.n_items,
            "asymmetric_risk": {
                "n_unsafe_answers": self.n_unsafe_answers,
                "unsafe_answer_rate": (
                    None
                    if self.unsafe_answer_rate is None
                    else round(self.unsafe_answer_rate, 6)
                ),
                "interval": interval(self.n_unsafe_answers, self.n_abstain_expected),
                "definition": (
                    "items whose expected_behavior is 'abstain' on mechanical corpus "
                    "absence that were nonetheless served an answer; the failure mode "
                    "the research problem is about"
                ),
            },
            "abstention": {
                "n_scope": self.n_abstention_scope,
                "n_abstain_expected": self.n_abstain_expected,
                "n_answer_expected": self.n_answer_expected,
                "n_true_abstain": self.n_true_abstain,
                "n_false_abstain": self.n_false_abstain,
                "n_abstain_expected_escalated": self.n_abstain_expected_escalated,
                "precision": (
                    None
                    if self.abstention_precision is None
                    else round(self.abstention_precision, 6)
                ),
                "recall": (
                    None
                    if self.abstention_recall is None
                    else round(self.abstention_recall, 6)
                ),
                "precision_interval": interval(
                    self.n_true_abstain, self.n_decided_abstain_in_scope
                ),
                "recall_interval": interval(
                    self.n_true_abstain, self.n_abstain_expected
                ),
                "scope_note": (
                    "expected_behavior 'escalate' items are excluded from both the "
                    "numerator and the denominator"
                ),
                "verification_basis": "mechanical_corpus_absence",
            },
            "escalation_stipulated": {
                "n_escalation_expected": self.n_escalation_expected,
                "n_decided_escalate": self.n_decided_escalate,
                "n_true_escalate": self.n_true_escalate,
                "precision": (
                    None
                    if self.escalation_precision is None
                    else round(self.escalation_precision, 6)
                ),
                "recall": (
                    None
                    if self.escalation_recall is None
                    else round(self.escalation_recall, 6)
                ),
                "verification_basis": "stipulated_definition",
                "warning": (
                    "NOT a scholar judgement that human review is required. This "
                    "project defined which questions count as needing escalation "
                    "(stipulation cross_standard_comparison_v1). Reject the "
                    "definition and these numbers go away; the abstention block, "
                    "which rests on mechanical corpus absence, is unaffected. Never "
                    "pool these figures with the abstention block."
                ),
            },
            "verification_note": (
                "Descriptive. No label here is qualified_scholar_review and no "
                "output is a Shari'ah ruling."
            ),
        }


#: Accepted values of ``expected_behavior`` across both classes. ``answer`` comes
#: from ``hard_set_schema.json``; ``abstain`` and ``escalate`` from
#: ``probe_schema.json``. Anything else is a schema drift and raises rather than
#: being silently bucketed.
EXPECTED_BEHAVIORS = ("answer", "abstain", "escalate")


def compute_selective_risk(
    items: Sequence[Mapping[str, Any]],
    *,
    expected_key: str = "expected_behavior",
    decision_key: str = "decision",
) -> SelectiveRiskMetrics:
    """Cross-tabulate expected behaviour against routing decision.

    ``items`` may mix hard-set items and probes freely - that is the point - but the
    two classes keep their own labels, so nothing here silently pools a scholar-
    authored judgement with a mechanically derived one.

    Raises on an unrecognised ``expected_behavior`` or ``decision``. A typo that
    quietly landed in an "other" bucket would understate exactly the risk cell this
    function exists to surface.
    """
    counts = {
        (expected, decision): 0
        for expected in EXPECTED_BEHAVIORS
        for decision in EXPECTED_BEHAVIORS
    }
    for item in items:
        expected = str(item[expected_key])
        decision = str(item[decision_key])
        if expected not in EXPECTED_BEHAVIORS:
            raise ValueError(
                f"unrecognised expected_behavior {expected!r}; "
                f"allowed: {EXPECTED_BEHAVIORS}"
            )
        if decision not in EXPECTED_BEHAVIORS:
            raise ValueError(
                f"unrecognised decision {decision!r}; allowed: {EXPECTED_BEHAVIORS}"
            )
        counts[(expected, decision)] += 1

    def total(expected: str) -> int:
        return sum(counts[(expected, d)] for d in EXPECTED_BEHAVIORS)

    n_abstain_expected = total("abstain")
    n_answer_expected = total("answer")
    n_escalation_expected = total("escalate")
    return SelectiveRiskMetrics(
        n_items=len(items),
        n_abstention_scope=n_abstain_expected + n_answer_expected,
        n_abstain_expected=n_abstain_expected,
        n_answer_expected=n_answer_expected,
        n_escalation_expected=n_escalation_expected,
        n_true_abstain=counts[("abstain", "abstain")],
        n_false_abstain=counts[("answer", "abstain")],
        n_unsafe_answers=counts[("abstain", "answer")],
        n_abstain_expected_escalated=counts[("abstain", "escalate")],
        n_true_escalate=counts[("escalate", "escalate")],
        n_decided_escalate=sum(
            counts[(e, "escalate")] for e in EXPECTED_BEHAVIORS
        ),
    )


def _retrieval_score_from_dict(payload: Mapping[str, Any] | None) -> Any:
    """Rebuild a :class:`RetrievalScore` from its ``as_dict`` form, or ``None``."""
    if not payload:
        return None
    from .evaluation import GoldClauseHit, RetrievalScore

    return RetrievalScore(
        hits=tuple(
            GoldClauseHit(
                standard_id=str(hit["standard_id"]),
                clause_id=str(hit["clause_id"]),
                occurrence_index=int(hit.get("occurrence_index", 0) or 0),
                sub_clause_id=hit.get("sub_clause_id"),
                in_context=bool(hit["in_context"]),
                rank=hit.get("rank"),
            )
            for hit in payload.get("hits", ())
        ),
        context_size=int(payload.get("context_size", 0) or 0),
    )
