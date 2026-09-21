"""Evaluation labels and gold-clause scoring (plan Layer 6).

Two things live here that must not drift apart from the rest of the project:

* **Clause identity.** A clause is identified by the 4-tuple
  ``(standard_id, clause_id, occurrence_index, sub_clause_id)``. This is a port of
  ``gold_clause_key`` / ``retrieved_clause_key`` in
  ``scripts/colab_e2e_batch_test.py``, and ``tests/test_evaluation.py`` checks the port
  by recomputing every per-item verdict the Colab run recorded for itself.

  The 4-tuple is the right identity because it is the only variant that is *unique
  across the corpus*: ``SS17:3/6/2`` genuinely appears twice in the source, on
  consecutive pages with different text, and only ``occurrence_index`` separates those
  two records. But an earlier version of this docstring went further and called
  ``occurrence_index`` load-bearing for the published numbers, which measurement does
  not support. ``SS17:3/6/2`` is not gold for any of the seven items, so its merging is
  invisible to Config A recall. What the two disambiguators actually protect at n=7 is
  H02, which cites both ``SS9:8/1`` occurrence 0 and ``SS9:8/1`` occurrence 3
  sub-clause ``c)`` while only the parent is retrieved - and either field alone tells
  those apart (``0`` vs ``3``, or ``None`` vs ``c)``). So the two are **mutually
  redundant** on this corpus: dropping either one changes nothing, and dropping both
  inflates micro recall from 9/17 to 10/17. The reason they overlap is that the
  extractor numbered lettered sub-clauses ``1, 2, 3`` instead of restarting at ``0``,
  so 16 of the 17 non-zero occurrence indices are really sub-clause labels in
  disguise.

* **Macro vs micro recall.** The stored reports print the *macro* mean - the mean
  of per-item recall - and label it simply "mean". Both are computed here and
  named, because they differ materially at n=7: Config A is macro 0.571 (4.0/7)
  and micro 0.529 (9/17). Anyone comparing against a paper's number needs to know
  which one they are holding.

Verification discipline: ``verification_basis`` is carried through untouched.
``corpus_cross_reference`` is **not** ``qualified_scholar_review`` and this module
never upgrades one to the other. No output here is a Shari'ah ruling.

This module contains no AAOIFI clause prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

#: Clause identity: standard, clause, occurrence, sub-clause.
ClauseKey = tuple[str, str, int, str | None]


def normalise_sub_clause_id(value: Any) -> str | None:
    """Empty strings and whitespace collapse to ``None``; other values stringify."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def clause_key(record: Mapping[str, Any]) -> ClauseKey:
    """Build the 4-tuple identity for a gold reference or a retrieved record."""
    return (
        str(record["standard_id"]),
        str(record["clause_id"]),
        int(record.get("occurrence_index", 0) or 0),
        normalise_sub_clause_id(record.get("sub_clause_id")),
    )


def clause_keys(records: Iterable[Mapping[str, Any]]) -> list[ClauseKey]:
    return [clause_key(record) for record in records]


@dataclass(frozen=True)
class GoldClauseHit:
    """Whether one gold clause was retrieved, and at what rank."""

    standard_id: str
    clause_id: str
    occurrence_index: int
    sub_clause_id: str | None
    in_context: bool
    rank: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "standard_id": self.standard_id,
            "clause_id": self.clause_id,
            "occurrence_index": self.occurrence_index,
            "sub_clause_id": self.sub_clause_id,
            "in_context": self.in_context,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class RetrievalScore:
    """Per-item gold-clause recall over the final context block."""

    hits: tuple[GoldClauseHit, ...]
    context_size: int

    @property
    def n_gold(self) -> int:
        return len(self.hits)

    @property
    def n_gold_in_context(self) -> int:
        return sum(1 for hit in self.hits if hit.in_context)

    @property
    def any_gold_in_context(self) -> bool:
        return self.n_gold_in_context > 0

    @property
    def all_gold_in_context(self) -> bool:
        return self.n_gold > 0 and self.n_gold_in_context == self.n_gold

    @property
    def recall(self) -> float:
        """Per-item recall. This is the quantity a *macro* mean averages."""
        return self.n_gold_in_context / self.n_gold if self.n_gold else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "context_size": self.context_size,
            "n_gold": self.n_gold,
            "n_gold_in_context": self.n_gold_in_context,
            "any_gold_in_context": self.any_gold_in_context,
            "all_gold_in_context": self.all_gold_in_context,
            "recall": round(self.recall, 6),
            "hits": [hit.as_dict() for hit in self.hits],
        }


def score_retrieval(
    gold_clause_ids: Sequence[Mapping[str, Any]],
    context_records: Sequence[Mapping[str, Any]],
) -> RetrievalScore:
    """Match gold references against the ranked context block by clause identity."""
    rank_by_key: dict[ClauseKey, int] = {}
    for rank, record in enumerate(context_records, start=1):
        rank_by_key.setdefault(clause_key(record), rank)

    hits: list[GoldClauseHit] = []
    for gold in gold_clause_ids:
        key = clause_key(gold)
        rank = rank_by_key.get(key)
        hits.append(
            GoldClauseHit(
                standard_id=key[0],
                clause_id=key[1],
                occurrence_index=key[2],
                sub_clause_id=key[3],
                in_context=rank is not None,
                rank=rank,
            )
        )
    return RetrievalScore(hits=tuple(hits), context_size=len(context_records))


#: Vocabulary from ``data/manifests/hard_set_schema.json``, repeated here so a
#: trace can be validated without reading the manifest.
EXPECTED_BEHAVIOURS = ("answer", "abstain", "escalate")

#: The last two admit :mod:`aaoifi_rag.reporting.probes` items. Both are strictly
#: weaker than ``corpus_cross_reference``, not stronger, and neither is scholar
#: review: ``mechanical_corpus_absence`` means a script re-derived a count from the
#: corpus, and ``stipulated_definition`` means this project *defined* the label.
#: :attr:`EvaluationLabel.is_scholar_validated` stays gated on
#: ``qualified_scholar_review`` alone, so widening this tuple cannot upgrade any
#: item's evidentiary standing.
VERIFICATION_BASES = (
    "unverified",
    "mechanical_corpus_absence",
    "stipulated_definition",
    "corpus_cross_reference",
    "qualified_scholar_review",
)



@dataclass(frozen=True)
class EvaluationLabel:
    """The reference labels for one item, and how the decision compared.

    ``decision_matches_expected`` is a comparison against an *authored*
    ``expected_behavior``, not against a Shari'ah ruling.

    Caveat that must travel with any aggregate built from these labels: all seven
    hard-set items are ``answerability = answerable`` with
    ``expected_behavior = answer``. There is not one ``abstain`` or ``escalate``
    item in the set. So every abstention is over-abstention *by construction*,
    and escalation precision/recall cannot be computed at all - there are no
    positives to recover. ``data/manifests/hard_set_schema.json`` permits both
    missing classes; their absence is an authoring gap, not a schema limit.
    Fixing it means adding items, which is Stage 2 work.
    """

    item_id: str
    expected_behavior: str
    answerability: str
    verification_basis: str
    status: str
    decision: str
    retrieval: RetrievalScore | None = None

    def __post_init__(self) -> None:
        if self.expected_behavior not in EXPECTED_BEHAVIOURS:
            raise ValueError(
                f"{self.item_id}: expected_behavior {self.expected_behavior!r} "
                f"not in {EXPECTED_BEHAVIOURS}"
            )
        if self.verification_basis not in VERIFICATION_BASES:
            raise ValueError(
                f"{self.item_id}: verification_basis {self.verification_basis!r} "
                f"not in {VERIFICATION_BASES}"
            )

    @property
    def decision_matches_expected(self) -> bool:
        return self.decision == self.expected_behavior

    @property
    def is_scholar_validated(self) -> bool:
        """False for every Stage 1 item. Never infer this from ``status`` alone."""
        return self.verification_basis == "qualified_scholar_review"

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "expected_behavior": self.expected_behavior,
            "answerability": self.answerability,
            "verification_basis": self.verification_basis,
            "status": self.status,
            "decision": self.decision,
            "decision_matches_expected": self.decision_matches_expected,
            "is_scholar_validated": self.is_scholar_validated,
            "retrieval": self.retrieval.as_dict() if self.retrieval else None,
        }


def build_evaluation_label(
    item: Mapping[str, Any],
    decision: str,
    context_records: Sequence[Mapping[str, Any]] | None = None,
) -> EvaluationLabel:
    """Build a label from a hard-set item and a routing decision."""
    retrieval = (
        score_retrieval(item.get("gold_clause_ids", ()), context_records)
        if context_records is not None
        else None
    )
    return EvaluationLabel(
        item_id=str(item["item_id"]),
        expected_behavior=str(item["expected_behavior"]),
        answerability=str(item["answerability"]),
        verification_basis=str(item["verification_basis"]),
        status=str(item["status"]),
        decision=decision,
        retrieval=retrieval,
    )
