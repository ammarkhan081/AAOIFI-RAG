"""Clause identity, gold-clause recall, and the evaluation label (plan Layer 6).

This module is where a published retrieval number is either right or quietly wrong, so
the tests are split into a mechanical half and a closing-the-loop half.

The mechanical half pins four things that a plausible reimplementation would get
differently, each of which would move a number:

* **Identity is a 4-tuple and both disambiguators are permissive on input.**
  ``occurrence_index`` coerces a missing key, ``None`` and ``""`` all to ``0``;
  ``sub_clause_id`` coerces ``""`` and whitespace to ``None`` but *not* ``0``. The
  asymmetry is real and asserted, because a gold file written by hand will contain
  every one of those spellings.
* **The first rank wins.** ``score_retrieval`` uses ``setdefault``, so when a context
  block contains the same clause twice the earlier rank is reported. Reversing that
  would inflate no recall but would silently change every published rank.
* **Recall is 0.0 at zero gold, and ``all_gold_in_context`` is False there.** The
  degenerate case is a probe, which by construction has no gold clause. Returning
  ``1.0`` for "all zero of them were retrieved" would let the negative class contribute
  perfect scores to a macro mean.
* **Widening ``VERIFICATION_BASES`` cannot upgrade an item.** The tuple is a superset of
  the schema enum by two probe-only values, and ``is_scholar_validated`` is gated on
  ``qualified_scholar_review`` alone, never on ``status``.

The corpus-gated half re-derives the published Config A retrieval numbers from the raw
private data - hard set plus corpus plus stored run - through live code, and confirms
they agree with the aggregate fields the Colab run recorded for itself. It then runs the
identity counterfactual that the module docstring's "load-bearing" claim rests on, and
finds that claim needs qualifying: see
``test_neither_disambiguator_is_load_bearing_alone_but_one_of_them_must_survive``.

No AAOIFI clause prose appears here. Clause *identifiers* do - they are published in
``reports/smoke_test_batch_n7.md`` and ``reports/error_taxonomy_n7.md`` already - but no
clause text, and no hard-set question text.
"""

from __future__ import annotations

from collections import Counter
import json
from typing import Any, Mapping, Sequence

import pytest

from aaoifi_rag.reporting.evaluation import (
    EXPECTED_BEHAVIOURS,
    VERIFICATION_BASES,
    EvaluationLabel,
    GoldClauseHit,
    RetrievalScore,
    build_evaluation_label,
    build_evaluation_label as _build,  # alias keeps long call sites readable
    clause_key,
    clause_keys,
    normalise_sub_clause_id,
    score_retrieval,
)


def record(
    standard_id: str = "XX",
    clause_id: str = "1/1",
    occurrence_index: Any = 0,
    sub_clause_id: Any = None,
) -> dict[str, Any]:
    """An identity-bearing record in the shape retrieval and the gold file both use."""
    return {
        "standard_id": standard_id,
        "clause_id": clause_id,
        "occurrence_index": occurrence_index,
        "sub_clause_id": sub_clause_id,
    }


def item(
    item_id: str = "T01",
    expected_behavior: str = "answer",
    answerability: str = "answerable",
    verification_basis: str = "corpus_cross_reference",
    status: str = "reviewed",
    gold: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "expected_behavior": expected_behavior,
        "answerability": answerability,
        "verification_basis": verification_basis,
        "status": status,
        "gold_clause_ids": list(gold),
    }


# --------------------------------------------------------------------------------------
# normalise_sub_clause_id
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("\t\n", None),
        ("c)", "c)"),
        ("  c)  ", "c)"),
        ("a)", "a)"),
    ],
)
def test_a_blank_sub_clause_is_the_same_as_no_sub_clause(raw, expected) -> None:
    assert normalise_sub_clause_id(raw) == expected


def test_a_numeric_zero_sub_clause_is_not_blanked() -> None:
    """The one place the two coercions disagree, and it is deliberate.

    ``occurrence_index`` runs ``int(value or 0)``, so a falsy value becomes ``0``.
    ``sub_clause_id`` runs ``str(value).strip() or None``, and ``str(0)`` is ``"0"``,
    which is truthy - so an integer ``0`` survives as the string ``"0"`` rather than
    collapsing to ``None``. Harmless on this corpus, whose only sub-clause labels are
    ``a)``, ``b)`` and ``c)``, but a gold file that used ``0`` for "no sub-clause" would
    fail to match and the miss would look like a retrieval failure.
    """
    assert normalise_sub_clause_id(0) == "0"
    assert normalise_sub_clause_id(1) == "1"
    assert normalise_sub_clause_id(False) == "False"


# --------------------------------------------------------------------------------------
# clause_key
# --------------------------------------------------------------------------------------


def test_identity_is_the_four_tuple_in_a_fixed_order() -> None:
    key = clause_key(record("SS9", "8/1", 3, "c)"))
    assert key == ("SS9", "8/1", 3, "c)")
    assert len(key) == 4, "the order is depended on positionally by GoldClauseHit"


@pytest.mark.parametrize(
    "raw",
    [
        {"standard_id": "SS9", "clause_id": "8/1"},
        {"standard_id": "SS9", "clause_id": "8/1", "occurrence_index": None},
        {"standard_id": "SS9", "clause_id": "8/1", "occurrence_index": ""},
        {"standard_id": "SS9", "clause_id": "8/1", "occurrence_index": 0},
        {"standard_id": "SS9", "clause_id": "8/1", "occurrence_index": "0"},
    ],
)
def test_a_missing_or_empty_occurrence_index_is_zero_not_a_distinct_value(raw) -> None:
    """A gold reference omitting the field must match an occurrence-0 record.

    All five spellings appear in hand-written reference data at some point. If any of
    them produced a key distinct from ``0``, the gold clause would be reported as not
    retrieved even when it sat at rank 1.
    """
    assert clause_key(raw) == ("SS9", "8/1", 0, None)


def test_the_occurrence_index_is_coerced_through_int() -> None:
    """Strings are accepted; a float truncates; nonsense raises rather than defaulting.

    Raising matters more than the coercion: a silent fallback to ``0`` on unparseable
    input would turn a corrupt gold file into a plausible-looking recall number.
    """
    assert clause_key(record(occurrence_index="3"))[2] == 3
    assert clause_key(record(occurrence_index=" 3 "))[2] == 3
    assert clause_key(record(occurrence_index=3.9))[2] == 3
    with pytest.raises(ValueError):
        clause_key(record(occurrence_index="x"))


def test_the_standard_and_clause_are_stringified() -> None:
    """So an integer-valued id in JSON cannot fork the key space."""
    assert clause_key({"standard_id": 9, "clause_id": 8}) == ("9", "8", 0, None)


def test_a_missing_standard_or_clause_is_an_error_not_a_default() -> None:
    """Both are subscripted directly. Identity without them is meaningless."""
    with pytest.raises(KeyError):
        clause_key({"clause_id": "8/1"})
    with pytest.raises(KeyError):
        clause_key({"standard_id": "SS9"})


def test_the_key_ignores_every_other_field() -> None:
    """Text, page, score and rank are not identity. Two records differing only in
    ``reranker_score`` are the same clause and must not both occupy the rank map."""
    bare = record("SS9", "8/1")
    decorated = {
        **bare,
        "chunk_id": "clause:SS9:8/1:occurrence:0",
        "text": "invented",
        "source_page": 17,
        "reranker_score": 1.57,
        "rank": 3,
    }
    assert clause_key(bare) == clause_key(decorated)


def test_clause_keys_preserves_order_and_duplicates() -> None:
    """It is a plain map, not a set: the caller decides what to do about repeats."""
    records = [record(clause_id="1/1"), record(clause_id="1/2"), record(clause_id="1/1")]
    assert clause_keys(records) == [
        ("XX", "1/1", 0, None),
        ("XX", "1/2", 0, None),
        ("XX", "1/1", 0, None),
    ]
    assert clause_keys([]) == []


# --------------------------------------------------------------------------------------
# score_retrieval
# --------------------------------------------------------------------------------------


def test_ranks_are_one_based() -> None:
    """Because they are printed next to prompt labels, which start at 1.

    ``reports/error_taxonomy_n7.md`` reads "SS8 §2/4/4 rank 1"; a 0-based rank here
    would make every one of those cross-references off by one.
    """
    context = [record(clause_id=f"1/{n}") for n in range(1, 6)]
    score = score_retrieval([record(clause_id="1/1"), record(clause_id="1/5")], context)
    assert [hit.rank for hit in score.hits] == [1, 5]


def test_a_gold_clause_absent_from_the_context_has_rank_none() -> None:
    context = [record(clause_id="1/1")]
    score = score_retrieval([record(clause_id="9/9")], context)
    assert score.hits[0].in_context is False
    assert score.hits[0].rank is None


def test_the_first_rank_wins_when_a_clause_appears_twice() -> None:
    """``setdefault``, not assignment. Asserted because both are one line.

    Retrieval can legitimately emit the same clause twice - BM25 and dense fusion did,
    before dedup - and the reported rank should be the best one, not the last one.
    """
    context = [
        record(clause_id="1/1"),
        record(clause_id="1/2"),
        record(clause_id="1/1"),
    ]
    score = score_retrieval([record(clause_id="1/1")], context)
    assert score.hits[0].rank == 1
    assert score.context_size == 3, "context_size counts records, not distinct clauses"


def test_a_duplicated_gold_reference_is_scored_twice() -> None:
    """Gold refs are not deduplicated, so a repeated ref double-counts in micro recall.

    Not a bug to fix silently: it is a property of the input file, and the seven items
    contain no duplicate. Pinned so that if one is ever added the effect is visible.
    """
    context = [record(clause_id="1/1")]
    score = score_retrieval([record(clause_id="1/1"), record(clause_id="1/1")], context)
    assert score.n_gold == 2
    assert score.n_gold_in_context == 2
    assert score.recall == 1.0


def test_the_hit_carries_the_normalised_identity_not_the_raw_gold_fields() -> None:
    """So a trace records what was actually matched on."""
    score = score_retrieval([{"standard_id": "SS9", "clause_id": "8/1"}], [])
    hit = score.hits[0]
    assert (hit.standard_id, hit.clause_id, hit.occurrence_index, hit.sub_clause_id) == (
        "SS9",
        "8/1",
        0,
        None,
    )


def test_a_sub_clause_reference_does_not_match_its_parent_clause() -> None:
    """The distinction that carries the whole 4-tuple.

    ``SS9 8/1`` and ``SS9 8/1 c)`` are separate records in the corpus with separate
    text. Retrieving the parent is not retrieving the sub-clause, and the published
    9/17 depends on that being true - see the corpus-gated section.
    """
    context = [record("SS9", "8/1", 0, None)]
    parent = score_retrieval([record("SS9", "8/1", 0, None)], context)
    child = score_retrieval([record("SS9", "8/1", 3, "c)")], context)
    assert parent.n_gold_in_context == 1
    assert child.n_gold_in_context == 0


def test_an_empty_context_scores_every_gold_as_missed() -> None:
    """The zero-retrieval path, which the router's short-circuit produces."""
    score = score_retrieval([record(clause_id="1/1"), record(clause_id="1/2")], [])
    assert (score.n_gold, score.n_gold_in_context, score.context_size) == (2, 0, 0)
    assert score.recall == 0.0
    assert score.any_gold_in_context is False


def test_no_gold_references_yields_an_empty_score_not_an_error() -> None:
    """Which is the probe case. See the property tests below for what recall means."""
    score = score_retrieval([], [record(clause_id="1/1")])
    assert score.hits == ()
    assert score.n_gold == 0
    assert score.context_size == 1


# --------------------------------------------------------------------------------------
# RetrievalScore properties
# --------------------------------------------------------------------------------------


def test_recall_is_zero_at_zero_gold_rather_than_dividing_or_returning_one() -> None:
    """The single most consequential degenerate case in this module.

    A probe has no gold clause. "All zero of its gold clauses were retrieved" is
    vacuously true, so an implementation returning ``1.0`` here is defensible in the
    abstract and wrong in context: probes would contribute perfect recall to a macro
    mean over a mixed set, and the aggregate would improve as the negative class grew.
    ``probes.Probe.as_evaluation_item`` documents that callers should pass
    ``context_records=None`` for exactly this reason; this test is the second line of
    defence if one does not.
    """
    empty = RetrievalScore(hits=(), context_size=5)
    assert empty.recall == 0.0
    assert empty.n_gold == 0
    assert empty.n_gold_in_context == 0


def test_all_gold_in_context_is_false_at_zero_gold() -> None:
    """``n_gold > 0`` is an explicit guard in the property, not an accident of ``==``.

    Without it, ``0 == 0`` would make an item with no gold clauses count towards
    ``n_all_gold_clauses_in_top5``, which the stored run reports as 1/7.
    """
    assert RetrievalScore(hits=(), context_size=5).all_gold_in_context is False
    assert RetrievalScore(hits=(), context_size=0).any_gold_in_context is False


def test_any_and_all_separate_partial_from_complete_retrieval() -> None:
    """The two aggregates the stored run reports as 7/7 and 1/7 respectively."""
    context = [record(clause_id="1/1")]
    partial = score_retrieval(
        [record(clause_id="1/1"), record(clause_id="9/9")], context
    )
    complete = score_retrieval([record(clause_id="1/1")], context)
    assert (partial.any_gold_in_context, partial.all_gold_in_context) == (True, False)
    assert (complete.any_gold_in_context, complete.all_gold_in_context) == (True, True)


def test_recall_is_the_per_item_quantity_a_macro_mean_averages() -> None:
    """One third, not one of three. The distinction between macro and micro is here."""
    context = [record(clause_id="1/1")]
    score = score_retrieval(
        [record(clause_id="1/1"), record(clause_id="9/8"), record(clause_id="9/9")],
        context,
    )
    assert score.recall == pytest.approx(1 / 3)


def test_the_score_dict_rounds_recall_to_six_places_and_keeps_the_hits() -> None:
    """Six places is what the reports print; the hits are what makes a miss auditable."""
    context = [record(clause_id="1/1")]
    score = score_retrieval(
        [record(clause_id="1/1"), record(clause_id="9/8"), record(clause_id="9/9")],
        context,
    )
    payload = score.as_dict()
    assert payload["recall"] == 0.333333
    assert payload["n_gold"] == 3
    assert payload["n_gold_in_context"] == 1
    assert payload["context_size"] == 1
    assert payload["any_gold_in_context"] is True
    assert payload["all_gold_in_context"] is False
    assert [hit["rank"] for hit in payload["hits"]] == [1, None, None]
    assert len(payload["hits"]) == payload["n_gold"], "one hit row per gold reference"


def test_the_score_dict_carries_no_clause_text() -> None:
    """A retrieval score is publishable; the records it was computed from are not."""
    context = [{**record(clause_id="1/1"), "text": "SENTINEL invented clause prose"}]
    payload = score_retrieval([record(clause_id="1/1")], context).as_dict()
    assert "SENTINEL" not in json.dumps(payload)
    assert "text" not in payload["hits"][0]


def test_the_score_is_frozen() -> None:
    score = RetrievalScore(hits=(), context_size=0)
    with pytest.raises(Exception):
        score.context_size = 3  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# GoldClauseHit
# --------------------------------------------------------------------------------------


def test_the_hit_dict_has_the_six_fields_a_trace_needs() -> None:
    hit = GoldClauseHit(
        standard_id="SS9",
        clause_id="8/1",
        occurrence_index=3,
        sub_clause_id="c)",
        in_context=False,
        rank=None,
    )
    assert hit.as_dict() == {
        "standard_id": "SS9",
        "clause_id": "8/1",
        "occurrence_index": 3,
        "sub_clause_id": "c)",
        "in_context": False,
        "rank": None,
    }


# --------------------------------------------------------------------------------------
# EvaluationLabel
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("behaviour", EXPECTED_BEHAVIOURS)
def test_every_expected_behaviour_in_the_vocabulary_constructs(behaviour) -> None:
    label = EvaluationLabel(
        item_id="T01",
        expected_behavior=behaviour,
        answerability="answerable",
        verification_basis="unverified",
        status="draft",
        decision=behaviour,
    )
    assert label.decision_matches_expected is True


@pytest.mark.parametrize("bad", ["Answer", "answered", "refuse", "", None, "route"])
def test_an_unknown_expected_behaviour_raises_at_construction(bad) -> None:
    """Validation is in ``__post_init__``, so a bad label cannot exist to be written.

    ``"answered"`` is in the list on purpose: it is the *response class* vocabulary,
    and confusing the two is the likeliest way this field gets a wrong value.
    """
    with pytest.raises(ValueError, match="expected_behavior"):
        EvaluationLabel(
            item_id="T01",
            expected_behavior=bad,  # type: ignore[arg-type]
            answerability="answerable",
            verification_basis="unverified",
            status="draft",
            decision="answer",
        )


@pytest.mark.parametrize("basis", VERIFICATION_BASES)
def test_every_verification_basis_in_the_vocabulary_constructs(basis) -> None:
    label = EvaluationLabel(
        item_id="T01",
        expected_behavior="answer",
        answerability="answerable",
        verification_basis=basis,
        status="draft",
        decision="answer",
    )
    assert label.verification_basis == basis


@pytest.mark.parametrize(
    "bad",
    ["scholar_review", "corpus", "Verified", "", None, "qualified_scholar"],
)
def test_an_unknown_verification_basis_raises_at_construction(bad) -> None:
    with pytest.raises(ValueError, match="verification_basis"):
        EvaluationLabel(
            item_id="T01",
            expected_behavior="answer",
            answerability="answerable",
            verification_basis=bad,  # type: ignore[arg-type]
            status="draft",
            decision="answer",
        )


def test_the_error_names_the_item_so_a_batch_failure_is_locatable() -> None:
    with pytest.raises(ValueError, match="H04"):
        EvaluationLabel(
            item_id="H04",
            expected_behavior="nonsense",
            answerability="answerable",
            verification_basis="unverified",
            status="draft",
            decision="answer",
        )


def test_answerability_and_status_are_not_validated() -> None:
    """A deliberate looseness, recorded rather than defended.

    ``probes.Probe.as_evaluation_item`` emits ``answerability`` values
    (``unanswerable_from_corpus``, ``requires_multiple_standards``) and a ``status``
    (``generated``) that are outside ``hard_set_schema.json``'s enums, because a probe
    is not a hard-set item. Validating them here would reject the negative class. The
    cost is that a typo in either field passes silently.
    """
    label = EvaluationLabel(
        item_id="P01",
        expected_behavior="abstain",
        answerability="unanswerable_from_corpus",
        verification_basis="mechanical_corpus_absence",
        status="generated",
        decision="abstain",
    )
    assert label.answerability == "unanswerable_from_corpus"
    assert label.status == "generated"


@pytest.mark.parametrize(
    ("expected", "decision", "matches"),
    [
        ("answer", "answer", True),
        ("answer", "abstain", False),
        ("answer", "escalate", False),
        ("abstain", "abstain", True),
        ("escalate", "escalate", True),
        ("escalate", "abstain", False),
    ],
)
def test_the_match_is_exact_string_equality_against_the_authored_label(
    expected, decision, matches
) -> None:
    """Which is why ``RouteDecision`` is a ``StrEnum``: ``decision.value`` slots in here.

    Nothing about this comparison is a Shari'ah judgement. It says the router did what
    the person who wrote the item said it should do.
    """
    label = EvaluationLabel(
        item_id="T01",
        expected_behavior=expected,
        answerability="answerable",
        verification_basis="unverified",
        status="draft",
        decision=decision,
    )
    assert label.decision_matches_expected is matches


def test_an_unvalidated_decision_string_simply_fails_to_match() -> None:
    """``decision`` is not checked against the vocabulary - only ``expected_behavior``
    is. A junk decision therefore reads as a mismatch rather than raising, which is the
    right failure mode for a value that comes from the router at runtime."""
    label = EvaluationLabel(
        item_id="T01",
        expected_behavior="answer",
        answerability="answerable",
        verification_basis="unverified",
        status="draft",
        decision="ANSWER",
    )
    assert label.decision_matches_expected is False, "case-sensitive, and StrEnum is lower"


# --------------------------------------------------------------------------------------
# is_scholar_validated
# --------------------------------------------------------------------------------------


def test_only_qualified_scholar_review_counts_as_scholar_validated() -> None:
    """The one assertion in this file that is a governance requirement, not a design one.

    ``docs/governance/`` and ``data/README.md`` both say ``corpus_cross_reference`` is
    not scholar review. All seven hard-set items carry ``corpus_cross_reference``, so a
    property that returned True for it would make every Stage 1 answer look
    scholar-validated.
    """
    validated = [
        basis
        for basis in VERIFICATION_BASES
        if EvaluationLabel(
            item_id="T01",
            expected_behavior="answer",
            answerability="answerable",
            verification_basis=basis,
            status="draft",
            decision="answer",
        ).is_scholar_validated
    ]
    assert validated == ["qualified_scholar_review"]


def test_a_scholar_validated_status_does_not_confer_scholar_validation() -> None:
    """``status`` is authoring workflow; ``verification_basis`` is evidence.

    ``hard_set_schema.json`` permits ``status = "scholar_validated"``, and an author
    could set it before any scholar has looked at the item. The property must not read
    it.
    """
    label = EvaluationLabel(
        item_id="T01",
        expected_behavior="answer",
        answerability="answerable",
        verification_basis="corpus_cross_reference",
        status="scholar_validated",
        decision="answer",
    )
    assert label.is_scholar_validated is False


def test_the_label_dict_exposes_both_derived_flags_and_no_prose() -> None:
    context = [{**record("SS9", "8/1"), "text": "SENTINEL invented clause prose"}]
    payload = build_evaluation_label(
        item(gold=[record("SS9", "8/1")]), "answer", context
    ).as_dict()
    assert payload["decision_matches_expected"] is True
    assert payload["is_scholar_validated"] is False
    assert payload["retrieval"]["n_gold_in_context"] == 1
    assert "SENTINEL" not in json.dumps(payload)


def test_the_label_is_frozen() -> None:
    label = build_evaluation_label(item(), "abstain")
    with pytest.raises(Exception):
        label.decision = "answer"  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# VERIFICATION_BASES against the schema and the probe module
# --------------------------------------------------------------------------------------


def test_the_bases_are_ordered_weakest_to_strongest_with_no_duplicates() -> None:
    """The order is semantic, not alphabetical, and is read by nothing at runtime.

    Recorded because the tuple's *comment* claims the two probe-only values are
    "strictly weaker than ``corpus_cross_reference``", and a reader checking that claim
    should be able to check it against a test rather than against prose.
    """
    assert VERIFICATION_BASES == (
        "unverified",
        "mechanical_corpus_absence",
        "stipulated_definition",
        "corpus_cross_reference",
        "qualified_scholar_review",
    )
    assert len(set(VERIFICATION_BASES)) == len(VERIFICATION_BASES)
    assert VERIFICATION_BASES.index("mechanical_corpus_absence") < VERIFICATION_BASES.index(
        "corpus_cross_reference"
    )
    assert VERIFICATION_BASES.index("stipulated_definition") < VERIFICATION_BASES.index(
        "corpus_cross_reference"
    )
    assert VERIFICATION_BASES[-1] == "qualified_scholar_review"


def test_the_bases_are_a_superset_of_the_schema_enum_by_exactly_two_probe_values(
    repo_root,
) -> None:
    """``hard_set_schema.json`` governs the *hard set*; this tuple also governs probes.

    Tracked file, so no skip. The two extra values exist because probes flow through
    the same ``EvaluationLabel``, and ``__post_init__`` would reject them otherwise. If
    a third ever appears without a schema change, this test says so.
    """
    schema = json.loads(
        (repo_root / "data" / "manifests" / "hard_set_schema.json").read_text(
            encoding="utf-8"
        )
    )
    enum = tuple(schema["properties"]["verification_basis"]["enum"])
    assert enum == ("unverified", "corpus_cross_reference", "qualified_scholar_review")
    assert set(enum) < set(VERIFICATION_BASES)
    assert set(VERIFICATION_BASES) - set(enum) == {
        "mechanical_corpus_absence",
        "stipulated_definition",
    }
    # Relative order of the shared values is preserved, so the superset is a refinement.
    assert [b for b in VERIFICATION_BASES if b in enum] == list(enum)


def test_the_expected_behaviours_match_the_schema_enum_exactly(repo_root) -> None:
    """No superset here: a probe's ``expected_behavior`` is one of the same three."""
    schema = json.loads(
        (repo_root / "data" / "manifests" / "hard_set_schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert EXPECTED_BEHAVIOURS == tuple(
        schema["properties"]["expected_behavior"]["enum"]
    )


def test_every_basis_the_probe_module_emits_is_accepted_here() -> None:
    """The seam between the negative class and the label. Both directions asserted.

    Imported inside the test so a probe-module import error surfaces as this test
    failing rather than as a collection error across the whole file.
    """
    from aaoifi_rag.reporting import probes

    emitted_bases = set(probes.VERIFICATION_BASIS_BY_BASIS.values())
    assert emitted_bases == {"mechanical_corpus_absence", "stipulated_definition"}
    assert emitted_bases <= set(VERIFICATION_BASES)
    assert set(probes.EXPECTED_BEHAVIOR_BY_BASIS.values()) <= set(EXPECTED_BEHAVIOURS)
    # And no probe basis is ever scholar review or corpus cross-reference.
    assert "qualified_scholar_review" not in emitted_bases
    assert "corpus_cross_reference" not in emitted_bases


def test_a_probe_shaped_item_builds_a_label_that_is_not_scholar_validated() -> None:
    """End to end across the seam, on an invented probe rather than a stored one."""
    from aaoifi_rag.reporting import probes

    probe = probes.Probe(
        probe_id="TP01",
        question_text="Does the corpus define a widget reserve?",
        basis=probes.UnanswerableBasis.TERM_ABSENT,
        basis_evidence={"term": "widget reserve", "corpus_occurrences": 0},
    )
    label = build_evaluation_label(probe.as_evaluation_item(), "abstain")
    assert label.expected_behavior == "abstain"
    assert label.verification_basis == "mechanical_corpus_absence"
    assert label.decision_matches_expected is True
    assert label.is_scholar_validated is False
    assert label.retrieval is None


def test_a_cross_standard_probe_expects_escalation_on_a_stipulated_basis() -> None:
    from aaoifi_rag.reporting import probes

    probe = probes.Probe(
        probe_id="TP02",
        question_text="Which of two standards governs a widget lease?",
        basis=probes.UnanswerableBasis.CROSS_STANDARD,
        basis_evidence={"standard_ids": ["SS8", "SS9"]},
    )
    label = build_evaluation_label(probe.as_evaluation_item(), "escalate")
    assert label.expected_behavior == "escalate"
    assert label.verification_basis == "stipulated_definition"
    assert label.decision_matches_expected is True
    assert label.is_scholar_validated is False


# --------------------------------------------------------------------------------------
# build_evaluation_label
# --------------------------------------------------------------------------------------


def test_omitting_the_context_leaves_retrieval_none_rather_than_an_empty_score() -> None:
    """``None`` and "scored against nothing" are different facts and must stay distinct.

    A probe has no gold clause, so its retrieval score is undefined, not zero; an
    answerable item whose retrieval returned nothing scored a real 0.0. Collapsing the
    two would put an undefined value into a macro mean.
    """
    assert _build(item(), "abstain").retrieval is None
    assert _build(item(), "abstain", None).retrieval is None
    scored = _build(item(gold=[record()]), "answer", [])
    assert scored.retrieval is not None
    assert scored.retrieval.recall == 0.0
    assert scored.retrieval.context_size == 0


def test_a_missing_gold_key_scores_zero_gold_rather_than_raising() -> None:
    """``item.get("gold_clause_ids", ())`` - a probe-shaped item has none."""
    bare = {
        "item_id": "T01",
        "expected_behavior": "abstain",
        "answerability": "answerable",
        "verification_basis": "unverified",
        "status": "draft",
    }
    label = _build(bare, "abstain", [record()])
    assert label.retrieval is not None
    assert label.retrieval.n_gold == 0
    assert label.retrieval.context_size == 1


def test_the_builder_stringifies_the_item_fields() -> None:
    label = _build({**item(item_id=4), "gold_clause_ids": []}, "answer")
    assert label.item_id == "4"


def test_the_builder_propagates_the_items_own_validation_failure() -> None:
    with pytest.raises(ValueError, match="verification_basis"):
        _build(item(verification_basis="scholar_review"), "answer")


def test_the_decision_is_taken_from_the_router_not_from_the_item() -> None:
    """Obvious, and worth an assertion: an implementation that defaulted ``decision`` to
    ``expected_behavior`` would report 7/7 routing agreement instead of the measured 1/7."""
    label = _build(item(expected_behavior="answer"), "abstain")
    assert label.decision == "abstain"
    assert label.decision_matches_expected is False


# --------------------------------------------------------------------------------------
# Corpus-gated: the published Config A retrieval numbers, re-derived from raw data
#
# ``test_metrics.py`` already checks that the macro and micro means are computed
# correctly *from the public traces*. What is checked here is one layer further back:
# that live ``score_retrieval`` over the private hard set and the private corpus
# reproduces those per-item numbers at all. The two together close the loop from
# licensed source data to published figure, and neither is sufficient alone.
# --------------------------------------------------------------------------------------

#: Measured 2026-09-04 from ``data/private/hard_set.jsonl`` +
#: ``data/private/extracted/clause_chunks.jsonl`` + ``reports/e2e_batch_smoke_results.json``.
#: ``item_id -> (n_gold, n_gold_in_context, ranks in gold-file order)``.
PUBLISHED_PER_ITEM: dict[str, tuple[int, int, tuple[int | None, ...]]] = {
    "H01": (3, 2, (None, 5, 1)),
    "H02": (4, 2, (3, None, 5, None)),
    "H03": (2, 1, (1, None)),
    "H04": (2, 1, (1, None)),
    "H05": (3, 1, (None, 1, None)),
    "H06": (2, 1, (1, None)),
    "H07": (1, 1, (2,)),
}

#: ``reports/full_ablation_table.md`` Config A. Macro is the mean the reports print.
PUBLISHED_MICRO = (9, 17)
PUBLISHED_MACRO = 0.571429


def _ranked_context(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return sorted(row["top5"], key=lambda record_: record_["rank"])


@pytest.fixture(scope="session")
def scored_run(hard_set, stored_run) -> dict[str, RetrievalScore]:
    """Live ``score_retrieval`` over the real gold refs and the real stored top-5."""
    by_id = {entry["item_id"]: entry for entry in hard_set}
    return {
        row["item_id"]: score_retrieval(
            by_id[row["item_id"]]["gold_clause_ids"], _ranked_context(row)
        )
        for row in stored_run["items"]
    }


@pytest.mark.requires_private_data
def test_the_corpus_and_hard_set_are_the_ones_every_number_was_measured_against(
    clause_chunks, hard_set
) -> None:
    assert len(clause_chunks) == 362
    assert len(hard_set) == 7
    assert sorted(entry["item_id"] for entry in hard_set) == list(PUBLISHED_PER_ITEM)
    assert sum(len(entry["gold_clause_ids"]) for entry in hard_set) == PUBLISHED_MICRO[1]


@pytest.mark.requires_private_data
def test_every_published_per_item_recall_is_reproduced_from_the_private_gold_file(
    scored_run,
) -> None:
    """Seven items, three numbers each, exact equality.

    Ranks are asserted in *gold-file order*, not sorted, because the order is what a
    trace records and what ``reports/error_taxonomy_n7.md`` quotes. H01's ``(None, 5,
    1)`` is the item where the highest-ranked gold clause is cited third.
    """
    measured = {
        item_id: (score.n_gold, score.n_gold_in_context, tuple(h.rank for h in score.hits))
        for item_id, score in scored_run.items()
    }
    assert measured == PUBLISHED_PER_ITEM


@pytest.mark.requires_private_data
def test_the_micro_and_macro_totals_are_nine_of_seventeen_and_point_five_seven(
    scored_run,
) -> None:
    """The two numbers that get confused. Both derived here, neither read from a file."""
    hit = sum(score.n_gold_in_context for score in scored_run.values())
    gold = sum(score.n_gold for score in scored_run.values())
    assert (hit, gold) == PUBLISHED_MICRO
    assert hit / gold == pytest.approx(0.529412, abs=5e-7)
    macro = sum(score.recall for score in scored_run.values()) / len(scored_run)
    assert macro == pytest.approx(PUBLISHED_MACRO, abs=5e-7)
    assert macro != pytest.approx(hit / gold, abs=1e-3), "they really do differ at n=7"


@pytest.mark.requires_private_data
def test_live_scoring_agrees_with_what_the_colab_run_recorded_for_itself(
    scored_run, stored_run
) -> None:
    """The one test that would catch a port error in ``evaluation.py``.

    ``score_retrieval`` is a port of ``gold_clause_key`` / ``retrieved_clause_key`` in
    ``scripts/colab_e2e_batch_test.py``. That script wrote its own per-item verdicts into
    the run artifact on the GPU box. This recomputes them from the same inputs with the
    library code and compares field for field, so the port is checked against the
    original rather than against a transcription of the original's output.
    """
    for row in stored_run["items"]:
        score = scored_run[row["item_id"]]
        assert score.n_gold == row["n_gold_clauses"], row["item_id"]
        assert score.n_gold_in_context == row["n_gold_clauses_in_top5"], row["item_id"]
        assert score.any_gold_in_context is row["any_gold_clause_in_top5"], row["item_id"]
        assert score.all_gold_in_context is row["all_gold_clauses_in_top5"], row["item_id"]
        assert score.context_size == 5, row["item_id"]

    aggregates = stored_run["aggregates"]
    assert aggregates["n_any_gold_clause_in_top5"] == sum(
        score.any_gold_in_context for score in scored_run.values()
    )
    assert aggregates["n_all_gold_clauses_in_top5"] == sum(
        score.all_gold_in_context for score in scored_run.values()
    )
    assert aggregates["all_gold_in_top5_over_n"] == "1/7", "only H07 is complete"


@pytest.mark.requires_private_data
def test_the_stored_context_records_and_the_corpus_agree_on_every_identity(
    clause_chunks, stored_run
) -> None:
    """Rejoining by ``chunk_id`` must not change any clause key.

    The stored top-5 rows carry their own identity fields, so recall could be computed
    without touching the corpus at all. This asserts the two routes are the same route:
    if the run artifact's fields had drifted from the corpus, the previous test would
    pass while the number meant something else.
    """
    by_chunk = {chunk["chunk_id"]: chunk for chunk in clause_chunks}
    seen = 0
    for row in stored_run["items"]:
        for stored_record in row["top5"]:
            corpus_record = by_chunk[stored_record["chunk_id"]]
            assert clause_key(stored_record) == clause_key(corpus_record)
            seen += 1
    assert seen == 35, "seven items times five records"


@pytest.mark.requires_private_data
def test_the_chunk_id_encodes_the_occurrence_index_it_claims(clause_chunks) -> None:
    """``chunk_id`` is derived, not independent; the suffix must match the field.

    Load-bearing for the previous test and for every report that identifies a record by
    ``chunk_id`` alone.
    """
    for chunk in clause_chunks:
        assert chunk["chunk_id"].endswith(f":occurrence:{chunk['occurrence_index']}")


# --------------------------------------------------------------------------------------
# Corpus-gated: the identity counterfactual behind the module docstring
# --------------------------------------------------------------------------------------


def _degraded_micro(
    hard_set: Sequence[Mapping[str, Any]],
    stored_run: Mapping[str, Any],
    fields: Sequence[str],
) -> tuple[int, int]:
    """Micro recall when clause identity is cut down to ``fields``.

    Deliberately reimplements ``clause_key`` rather than parameterising it: the point is
    to measure what a *different* identity scheme would have produced, and calling the
    real function with a mutilated record would not do that.
    """

    def key(record_: Mapping[str, Any]) -> tuple[Any, ...]:
        parts: list[Any] = [str(record_["standard_id"]), str(record_["clause_id"])]
        if "occurrence_index" in fields:
            parts.append(int(record_.get("occurrence_index", 0) or 0))
        if "sub_clause_id" in fields:
            parts.append(normalise_sub_clause_id(record_.get("sub_clause_id")))
        return tuple(parts)

    by_id = {entry["item_id"]: entry for entry in hard_set}
    hit = gold = 0
    for row in stored_run["items"]:
        present = {key(record_) for record_ in row["top5"]}
        refs = by_id[row["item_id"]]["gold_clause_ids"]
        hit += sum(1 for ref in refs if key(ref) in present)
        gold += len(refs)
    return hit, gold


@pytest.mark.requires_private_data
def test_the_four_tuple_is_unique_across_the_whole_corpus(clause_chunks) -> None:
    """362 records, 362 distinct keys. Nothing in the corpus is ambiguous under it."""
    keys = clause_keys(clause_chunks)
    assert len(set(keys)) == len(clause_chunks) == 362


@pytest.mark.requires_private_data
def test_the_only_clause_appearing_twice_in_the_source_is_ss17_three_six_two(
    clause_chunks,
) -> None:
    """The example the module docstring cites, verified rather than assumed.

    Two records, occurrences 0 and 1, on consecutive pages, with genuinely different
    text - so merging them really would merge two distinct clauses. This is the *corpus
    uniqueness* argument for ``occurrence_index``, and it holds. The next test shows it
    is not the argument that protects the published number.
    """
    triples = Counter(
        (chunk["standard_id"], chunk["clause_id"], normalise_sub_clause_id(chunk["sub_clause_id"]))
        for chunk in clause_chunks
    )
    repeated = {triple: n for triple, n in triples.items() if n > 1}
    assert repeated == {("SS17", "3/6/2", None): 2}

    pair = sorted(
        (chunk for chunk in clause_chunks
         if (chunk["standard_id"], chunk["clause_id"]) == ("SS17", "3/6/2")),
        key=lambda chunk: chunk["occurrence_index"],
    )
    assert [chunk["occurrence_index"] for chunk in pair] == [0, 1]
    assert pair[0]["source_page"] != pair[1]["source_page"]
    assert pair[0]["text"] != pair[1]["text"], "distinct clauses, not a duplicate record"


@pytest.mark.requires_private_data
def test_no_gold_reference_cites_the_duplicated_clause(hard_set) -> None:
    """Which is why the docstring's example cannot be what defends the n=7 number.

    ``SS17 3/6/2`` is not gold for any of the seven items, so whether the two records
    merge is invisible to Config A's recall. Something else is doing the work - see
    below.
    """
    refs = [ref for entry in hard_set for ref in entry["gold_clause_ids"]]
    assert len(refs) == 17
    assert not any(
        (ref["standard_id"], ref["clause_id"]) == ("SS17", "3/6/2") for ref in refs
    )
    assert sorted({ref.get("occurrence_index") for ref in refs}) == [0, 3]
    assert sorted({normalise_sub_clause_id(ref.get("sub_clause_id")) or "-" for ref in refs}) == [
        "-",
        "c)",
    ]


@pytest.mark.requires_private_data
def test_neither_disambiguator_is_load_bearing_alone_but_one_of_them_must_survive(
    hard_set, stored_run
) -> None:
    """A qualification of ``evaluation.py``'s docstring, measured four ways.

    The docstring singles out ``occurrence_index`` as load-bearing. On this corpus and
    at n=7 that is too strong: the two disambiguators are *mutually redundant*, and
    dropping either one on its own changes nothing. Dropping **both** inflates micro
    recall from 9/17 to 10/17 and macro from 0.571 to 0.607.

    The mechanism is H02, not SS17. H02 cites both ``SS9 8/1`` occurrence 0 and
    ``SS9 8/1`` occurrence 3 sub-clause ``c)``; only the first is in the top-5. Under
    ``(standard, clause)`` alone the retrieved parent satisfies both references and H02
    scores 3/4 instead of 2/4. Keeping ``occurrence_index`` tells them apart by ``0``
    versus ``3``; keeping ``sub_clause_id`` tells them apart by ``None`` versus ``c)``.
    Either suffices, so no single-field claim of load-bearingness is defensible - but
    the 4-tuple is still the right identity, because it is the only variant that is also
    unique across the full corpus.
    """
    full = _degraded_micro(hard_set, stored_run, ("occurrence_index", "sub_clause_id"))
    no_occurrence = _degraded_micro(hard_set, stored_run, ("sub_clause_id",))
    no_sub_clause = _degraded_micro(hard_set, stored_run, ("occurrence_index",))
    neither = _degraded_micro(hard_set, stored_run, ())

    assert full == PUBLISHED_MICRO == (9, 17)
    assert no_occurrence == (9, 17), "occurrence_index alone changes no published number"
    assert no_sub_clause == (9, 17), "sub_clause_id alone changes no published number"
    assert neither == (10, 17), "dropping both inflates recall by one spurious hit"
    assert neither[0] / neither[1] == pytest.approx(0.588235, abs=5e-7)


@pytest.mark.requires_private_data
def test_the_spurious_hit_is_h02_and_the_records_that_collide_are_named(
    hard_set, stored_run
) -> None:
    """Names the item and the collision, so the previous test's story is checkable."""
    by_id = {entry["item_id"]: entry for entry in hard_set}
    moved = []
    for row in stored_run["items"]:
        strict = {clause_key(record_) for record_ in row["top5"]}
        loose = {(record_["standard_id"], record_["clause_id"]) for record_ in row["top5"]}
        refs = by_id[row["item_id"]]["gold_clause_ids"]
        strict_hits = sum(1 for ref in refs if clause_key(ref) in strict)
        loose_hits = sum(
            1 for ref in refs if (ref["standard_id"], ref["clause_id"]) in loose
        )
        if strict_hits != loose_hits:
            moved.append((row["item_id"], strict_hits, loose_hits, len(refs)))
    assert moved == [("H02", 2, 3, 4)]

    h02 = by_id["H02"]["gold_clause_ids"]
    colliding = [
        ref for ref in h02 if (ref["standard_id"], ref["clause_id"]) == ("SS9", "8/1")
    ]
    assert len(colliding) == 2, "H02 cites the parent and one sub-clause of SS9 8/1"
    assert sorted(int(ref.get("occurrence_index", 0) or 0) for ref in colliding) == [0, 3]
    assert sorted(
        normalise_sub_clause_id(ref.get("sub_clause_id")) or "-" for ref in colliding
    ) == ["-", "c)"]


@pytest.mark.requires_private_data
def test_the_occurrence_index_mostly_encodes_sub_clauses_not_repeated_clauses(
    clause_chunks,
) -> None:
    """Why the two fields turn out to be redundant: they encode the same thing here.

    Seventeen records carry a non-zero ``occurrence_index``. Sixteen of them have no
    occurrence-0 sibling at all, because the extractor numbered a clause's lettered
    sub-clauses ``1``, ``2``, ``3`` rather than restarting at ``0`` - so the sub-clause
    label and the occurrence index vary together. Only one non-zero record is a genuine
    repetition of an otherwise-identical clause, and that is ``SS17 3/6/2``.
    """
    non_zero = [chunk for chunk in clause_chunks if chunk["occurrence_index"] != 0]
    zero_triples = {
        (chunk["standard_id"], chunk["clause_id"], normalise_sub_clause_id(chunk["sub_clause_id"]))
        for chunk in clause_chunks
        if chunk["occurrence_index"] == 0
    }
    orphans = [
        chunk
        for chunk in non_zero
        if (
            chunk["standard_id"],
            chunk["clause_id"],
            normalise_sub_clause_id(chunk["sub_clause_id"]),
        )
        not in zero_triples
    ]
    assert len(non_zero) == 17
    assert len(orphans) == 16
    assert [chunk["chunk_id"] for chunk in non_zero if chunk not in orphans] == [
        "clause:SS17:3/6/2:occurrence:1"
    ]
    assert dict(sorted(Counter(chunk["occurrence_index"] for chunk in clause_chunks).items())) == {
        0: 345,
        1: 8,
        2: 7,
        3: 2,
    }
    assert sorted(
        {str(normalise_sub_clause_id(chunk["sub_clause_id"])) for chunk in clause_chunks}
    ) == ["None", "a)", "b)", "c)"]


@pytest.mark.requires_private_data
def test_every_hard_set_item_builds_a_label_and_not_one_is_scholar_validated(
    hard_set, stored_run
) -> None:
    """The governance claim, asserted against the real file rather than a fixture.

    All seven items are ``answerable`` / ``answer`` / ``corpus_cross_reference`` /
    ``reviewed``. There is no ``abstain`` or ``escalate`` item in the set, which is the
    authoring gap ``EvaluationLabel``'s docstring records: every abstention the router
    produces on this set is over-abstention by construction, and escalation recall is
    undefined because there are no positives.

    The decision passed in is the *generator's* behaviour, translated out of the
    response-class vocabulary and into the routing one. The translation is written out
    rather than assumed, because ``answered``/``abstained`` and ``answer``/``abstain``
    are two different vocabularies and ``EvaluationLabel`` does not validate ``decision``
    - handing it a raw ``response_class`` would silently score every item as a mismatch.
    This is Config A with no gates in front of it: the model answered three of seven, so
    agreement with the authored ``expected_behavior`` is 3/7 before any reliability layer
    is applied.
    """
    as_decision = {"answered": "answer", "abstained": "abstain"}
    by_id = {entry["item_id"]: entry for entry in hard_set}
    labels = [
        build_evaluation_label(
            by_id[row["item_id"]],
            as_decision[row["response_class"]],
            _ranked_context(row),
        )
        for row in stored_run["items"]
    ]
    assert len(labels) == 7
    assert {label.expected_behavior for label in labels} == {"answer"}
    assert {label.answerability for label in labels} == {"answerable"}
    assert {label.verification_basis for label in labels} == {"corpus_cross_reference"}
    assert {label.status for label in labels} == {"reviewed"}
    assert not any(label.is_scholar_validated for label in labels)
    assert all(label.retrieval is not None for label in labels)

    agreement = sum(label.decision_matches_expected for label in labels)
    assert agreement == 3, "three answered, four abstained; every abstention is a miss"
    assert stored_run["aggregates"]["answered_over_n"] == "3/7"
    assert [label.item_id for label in labels if label.decision_matches_expected] == [
        "H01",
        "H03",
        "H05",
    ]

