"""Tests for clause-reference grounding.

What these tests are for
------------------------
:mod:`aaoifi_rag.reliability.grounding` makes a claim that is unusual for this
project: its decision involves no calibration. So the tests are shaped around the
three things that could still make it wrong.

1. **The extractor could over-extract.** SS13 is Mudarabah and its profit-sharing
   ratios are written ``x/y``, exactly like a clause path. If ``2/3`` in "a ratio
   of 2/3" were taken as a reference, the gate would fire on correct answers. The
   corpus-gated half runs the extractor over all 362 clause texts and checks the
   count against the measurement the module docstring cites.

2. **The corpus index could change the decision.** The module claims the index
   refines the diagnosis and never the verdict. That is asserted directly, both on
   synthetic records and on all fourteen real texts, because it is the reason the
   index is optional.

3. **Leniency could be silently load-bearing in the wrong direction.** An
   unqualified reference is resolved against every retrieved standard, which can
   only turn a failure into a pass. The tests pin that direction, and pin that the
   *reverse* ancestry case does not ground - a reference to ``2/4/4/1`` when only
   ``2/4/4`` was shown points at a sub-clause the model never saw.

The false-positive population, and what it is not
-------------------------------------------------
Six of seventeen references in the seven gold answers resolve outside the
retrieved context. That is not this gate's false-positive rate. Every one of the
six resolves to a real clause in the corpus that was not retrieved, and the gold
answers were authored with the full standards open - ``reports/retrieval_ablation_n7.md``
already records that gold retrieval was incomplete. What 6/17 measures is that
the gold answers are not context-only answers. The gate is *right* about them.

The only genuinely context-only population available locally is the stored model
run, where H01 is the sole item making clause references at all: two distinct
references, both grounded. n=2. The tests pin that number rather than dressing it
up as a validated error rate.

No AAOIFI clause prose is written into this file. The corpus-gated tests read real
prose at run time and count what the extractor does with it; nothing they read is
asserted as a literal.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from aaoifi_rag.reliability.grounding import (
    CLAUSE_REFERENCE_RE,
    FAILING_STATUSES,
    MIN_PATH_DEPTH,
    STANDARD_LOOKBACK,
    STANDARD_NUMBER_RE,
    ClauseGroundingAudit,
    ClauseReference,
    CorpusClauseIndex,
    ReferenceStatus,
    audit_clause_grounding,
    clause_ancestors,
    extract_clause_references,
    normalise_standard,
)

#: Invented prose, positioned where clause text goes. Not AAOIFI's.
INVENTED_CLAUSE = (
    "The institution shall record the widget at the value agreed between the parties "
    "and shall disclose that value in the notes to the financial statements."
)


def record(standard: str, clause: str, *, text: str = INVENTED_CLAUSE) -> dict[str, Any]:
    return {
        "chunk_id": f"clause:{standard}:{clause}:occurrence:0",
        "standard_id": standard,
        "clause_id": clause,
        "sub_clause_id": None,
        "occurrence_index": 0,
        "source_page": 10,
        "text": text,
    }


# ---------------------------------------------------------------------------
# clause_ancestors and normalise_standard
# ---------------------------------------------------------------------------


def test_clause_ancestors_includes_the_path_itself() -> None:
    """Self-inclusion is what makes the ``exact`` and ``ancestor`` tests composable."""
    assert clause_ancestors("2/4/4") == frozenset({"2", "2/4", "2/4/4"})


def test_clause_ancestors_of_a_depth_one_path_is_just_itself() -> None:
    assert clause_ancestors("8") == frozenset({"8"})


def test_clause_ancestors_tolerates_stray_separators() -> None:
    """Defensive: a trailing slash must not produce an empty-string ancestor.

    An empty ancestor would match ``"" in context_blob``, which is always true,
    and would ground every reference in a response.
    """
    assert clause_ancestors("2/4/") == frozenset({"2", "2/4"})
    assert "" not in clause_ancestors("//2//4//")


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("SS8", "SS8"),
        ("SS 8", "SS8"),
        ("ss8", "SS8"),
        ("SS08", "SS8"),
        ("12", "SS12"),
        ("(5)", "SS5"),
        (None, None),
        ("", None),
        ("SS", None),
    ],
)
def test_normalise_standard(token: str | None, expected: str | None) -> None:
    """``SS08`` folding to ``SS8`` matters: the corpus spells them unpadded."""
    assert normalise_standard(token) == expected


# ---------------------------------------------------------------------------
# CorpusClauseIndex
# ---------------------------------------------------------------------------


def test_the_index_stores_ancestors_so_a_shallow_reference_resolves() -> None:
    index = CorpusClauseIndex.from_records([record("SS8", "2/4/4")])
    assert index.contains("SS8", "2/4/4")
    assert index.contains("SS8", "2/4"), "ancestor of a real clause"
    assert index.contains("SS8", "2")
    assert not index.contains("SS8", "2/4/4/1"), "deeper than anything present"
    assert not index.contains("SS9", "2/4/4"), "wrong standard"


def test_the_index_skips_records_missing_either_identifier() -> None:
    """A malformed record must not create a ``None``-keyed standard."""
    index = CorpusClauseIndex.from_records(
        [
            record("SS8", "2/4/4"),
            {"standard_id": "SS9", "clause_id": None},
            {"standard_id": None, "clause_id": "3/1"},
            {},
        ]
    )
    assert index.standards == frozenset({"SS8"})


def test_the_index_carries_identifiers_and_not_clause_text() -> None:
    """The index is publishable; the corpus it is built from is not."""
    index = CorpusClauseIndex.from_records([record("SS8", "2/4/4")])
    assert INVENTED_CLAUSE not in repr(index)
    assert index.as_dict() == {"standards": ["SS8"], "n_paths": {"SS8": 3}}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_a_depth_one_reference_is_not_extracted() -> None:
    """``item 8`` cannot be told from "8 years" without judgement.

    Measured cost on this corpus: zero occurrences, asserted in the corpus-gated
    section. Recorded here as the deliberate choice it is.
    """
    assert extract_clause_references("as required by item 8 of the standard") == []
    assert MIN_PATH_DEPTH == 2


@pytest.mark.parametrize(
    ("text", "path"),
    [
        ("see item 2/4", "2/4"),
        ("see item 2/4/4", "2/4/4"),
        ("see item 5/1/8/7", "5/1/8/7"),
        ("under §2/4/4 the rule", "2/4/4"),
        ("under clause 2/4/4", "2/4/4"),
        ("under paragraph 2/4/4", "2/4/4"),
        ("under para. 2/4/4", "2/4/4"),
        ("under section 2/4/4", "2/4/4"),
        ("under item no. 2/4/4", "2/4/4"),
        ("a bare 2/4/4 with no cue", "2/4/4"),
    ],
)
def test_the_cue_forms_that_appear_in_this_corpus_are_all_extracted(
    text: str, path: str
) -> None:
    found = extract_clause_references(text)
    assert [reference[0] for reference in found] == [path]


def test_the_offset_points_at_the_path_not_at_the_cue_word() -> None:
    """The offset is used to slice diagnostic windows, so it must be the path's.

    ``CLAUSE_REFERENCE_RE`` consumes the cue word, so ``match.start()`` would sit
    on "item" and a window built from it would be shifted.
    """
    text = "as laid down in item 2/5/3."
    (_, _, _, offset), = extract_clause_references(text)
    assert text[offset:].startswith("2/5/3")


def test_an_adjacent_standard_token_qualifies_the_reference() -> None:
    (path, standard, qualified, _), = extract_clause_references("see SS8 §2/4/1 which")
    assert (path, standard, qualified) == ("2/4/1", "SS8", True)


def test_an_unqualified_reference_binds_to_the_nearest_preceding_standard() -> None:
    """The gold answers run ``SS8 §2/4/4 ... §2/4/2`` across one sentence."""
    found = extract_clause_references("SS8 §2/4/4 differs from §2/4/2 here")
    assert [(reference[0], reference[1]) for reference in found] == [
        ("2/4/4", "SS8"),
        ("2/4/2", "SS8"),
    ]


def test_a_standard_further_back_than_the_lookback_window_does_not_bind() -> None:
    """The window is a bounded heuristic, so its boundary is pinned.

    Without a bound, an ``SS8`` in the first sentence would qualify every
    reference in a long answer, including ones about a different standard.
    """
    near = "SS26 " + "x" * (STANDARD_LOOKBACK - 20) + " see item 4/2"
    far = "SS26 " + "x" * (STANDARD_LOOKBACK + 20) + " see item 4/2"
    assert extract_clause_references(near)[0][1] == "SS26"
    assert extract_clause_references(far)[0][1] is None
    assert extract_clause_references(far)[0][2] is False, "reported unqualified"


def test_a_forward_shariah_standard_number_binds_the_reference() -> None:
    """AAOIFI's idiom for a *different* standard, 7 occurrences in this corpus.

    Without this rule the reference would be unqualified and resolved leniently
    against whatever standards were retrieved, which is a wrong diagnosis even
    though it produces the same verdict.
    """
    text = "taking into account item 3/1/4/3 of Shari'ah Standard No. (12) on Sukuk"
    (path, standard, qualified, _), = extract_clause_references(text)
    assert (path, standard, qualified) == ("3/1/4/3", "SS12", True)


def test_the_curly_apostrophe_spelling_also_binds() -> None:
    text = "see item 6/4 of Shari’ah Standard No. (5) on Guarantees"
    assert extract_clause_references(text)[0][1] == "SS5"


def test_an_explicit_standard_token_beats_a_forward_standard_name() -> None:
    """Precedence is asserted, not assumed: rule 1 before rule 2."""
    text = "SS17 §5/1/9 of Shari'ah Standard No. (12)"
    assert extract_clause_references(text)[0][1] == "SS17"


def test_a_forward_standard_name_beats_a_stale_preceding_token() -> None:
    """Rule 2 before rule 3, because the forward name is the stronger signal."""
    text = "SS17 says much; see item 4/1/2/4 of Shari'ah Standard No. (12) instead"
    assert extract_clause_references(text)[0][1] == "SS12"


def test_repeats_are_returned_rather_than_deduplicated_by_the_extractor() -> None:
    """Deduplication is the audit's job; a repeat at a new offset is still evidence."""
    found = extract_clause_references("SS8 §2/4/4 and again SS8 §2/4/4")
    assert len(found) == 2
    assert found[0][3] != found[1][3]


def test_blank_and_none_input_extract_nothing() -> None:
    assert extract_clause_references("") == []
    assert extract_clause_references(None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def audit(text: str, records: list[dict[str, Any]], **kwargs: Any) -> ClauseGroundingAudit:
    return audit_clause_grounding(text, records, **kwargs)


def test_a_reference_to_a_retrieved_clause_is_grounded_exactly() -> None:
    result = audit("see SS8 §2/4/4", [record("SS8", "2/4/4")])
    assert result.references[0].status is ReferenceStatus.GROUNDED_IN_CONTEXT
    assert result.references[0].resolved_via == "exact"
    assert result.passed


def test_a_parent_reference_is_grounded_by_a_retrieved_child() -> None:
    """``2/4`` was shown, in the sense that ``2/4/4`` was."""
    result = audit("see SS8 §2/4", [record("SS8", "2/4/4")])
    assert result.references[0].resolved_via == "ancestor"
    assert result.passed


def test_a_deeper_reference_is_not_grounded_by_a_retrieved_parent() -> None:
    """The one-directional half of the ancestry rule.

    A response citing ``2/4/4/1`` when only ``2/4/4`` was retrieved is pointing at
    a sub-clause it never saw. Grounding that would make the gate unable to catch
    the most plausible over-specific citation error there is.
    """
    index = CorpusClauseIndex.from_records([record("SS8", "2/4/4")])
    result = audit("see SS8 §2/4/4/1", [record("SS8", "2/4/4")], corpus_index=index)
    assert result.references[0].status is ReferenceStatus.UNRESOLVABLE
    assert not result.passed


def test_a_path_quoted_inside_the_retrieved_text_is_grounded_verbatim() -> None:
    """AAOIFI clauses cross-reference each other, and those paths are in front of
    the model even though they are not retrieved clause identities."""
    quoting = record("SS8", "3/1/3", text="In such a case the provisions of item 9/9/9 apply.")
    result = audit("as item 9/9/9 requires", [quoting])
    assert result.references[0].resolved_via == "verbatim_in_context"
    assert result.passed


def test_the_provenance_header_is_grounded_by_the_exact_rule_not_verbatim() -> None:
    """The stored H01 response transcribes the header the prompt printed.

    That echo must be grounded - it is a quotation of the context - but by the
    ``exact`` rule, which checks the standard as well as the path. The verbatim
    rule deliberately does not see the header: it repeats every retrieved clause's
    own path, so scanning it would ground a reference to any standard that merely
    shares a path shape with something retrieved.
    """
    result = audit("[1] SS8 §2/4/4 occ=0 page=10", [record("SS8", "2/4/4")])
    assert result.passed
    assert result.references[0].resolved_via == "exact"


def test_a_path_only_in_the_header_does_not_ground_a_different_standard() -> None:
    """The false negative the body-text scoping removes.

    The header for a retrieved ``SS17 §3/1`` prints "3/1". A response referring to
    ``SS8 §3/1`` has not been shown that clause, and must not be grounded by a
    coincidence of path shape across standards.
    """
    index = CorpusClauseIndex.from_records([record("SS17", "3/1"), record("SS8", "3/1")])
    result = audit("see SS8 §3/1", [record("SS17", "3/1")], corpus_index=index)
    assert result.references[0].status is ReferenceStatus.OUTSIDE_CONTEXT
    assert not result.passed


def test_a_real_clause_that_was_not_retrieved_is_outside_the_context() -> None:
    """The measured shape of all six gold-answer failures."""
    index = CorpusClauseIndex.from_records(
        [record("SS8", "2/4/4"), record("SS8", "2/4/1")]
    )
    result = audit("see SS8 §2/4/1", [record("SS8", "2/4/4")], corpus_index=index)
    assert result.references[0].status is ReferenceStatus.OUTSIDE_CONTEXT
    assert result.references[0].resolved_via == "corpus_exact"
    assert not result.passed


def test_a_standard_outside_the_corpus_gates_but_is_reported_as_such() -> None:
    """``SS12`` is not in this five-standard corpus - but it was still not shown.

    An earlier draft excused this status as "unadjudicable" and let it pass. That
    conflated two questions. *Was this clause shown to the model?* is decidable
    from the context block: no. *Does the clause exist in AAOIFI?* is what needs
    the corpus, and it is not what the gate asks. So the reference fails, and the
    status records only that its existence could not be checked here.
    """
    index = CorpusClauseIndex.from_records([record("SS8", "2/4/4")])
    result = audit(
        "see item 3/1/4/3 of Shari'ah Standard No. (12)",
        [record("SS8", "2/4/4")],
        corpus_index=index,
    )
    assert result.references[0].status is ReferenceStatus.UNKNOWN_STANDARD
    assert result.references[0].resolved_via == "standard_not_in_corpus"
    assert not result.passed
    assert ReferenceStatus.UNKNOWN_STANDARD in FAILING_STATUSES


def test_forward_binding_prevents_grounding_against_an_unrelated_standard() -> None:
    """The forward binding changes the verdict, in the strict direction.

    ``3/1/4/3`` is an SS17-shaped path. A reference to *SS12's* ``3/1/4/3`` must
    not be grounded by a retrieved SS17 clause that happens to share the path, and
    without :data:`STANDARD_NUMBER_RE` the reference would be unqualified and
    resolved leniently against exactly that clause.
    """
    context = [record("SS17", "3/1/4/3")]
    index = CorpusClauseIndex.from_records(context)
    bound = audit("item 3/1/4/3 of Shari'ah Standard No. (12) applies", context, corpus_index=index)
    assert bound.references[0].standard_id == "SS12"
    assert not bound.passed

    unbound = audit("item 3/1/4/3 applies", context, corpus_index=index)
    assert unbound.references[0].standard_id is None
    assert unbound.passed, "the lenient path grounds it; forward binding is what stops that"


def test_context_checks_run_before_the_unknown_standard_check() -> None:
    """A retrieved clause may itself quote another standard's item.

    If ``UNKNOWN_STANDARD`` were tested first, a path genuinely printed in the
    context would be reported unadjudicable instead of grounded.
    """
    quoting = record(
        "SS26", "6/1", text="See Shari'ah Standard No. (5) on Guarantees - item 6/4 here."
    )
    index = CorpusClauseIndex.from_records([quoting])
    result = audit("as item 6/4 of Shari'ah Standard No. (5) says", [quoting], corpus_index=index)
    assert result.references[0].status is ReferenceStatus.GROUNDED_IN_CONTEXT
    assert result.references[0].resolved_via == "verbatim_in_context"


def test_an_unqualified_reference_is_tried_against_every_retrieved_standard() -> None:
    """Leniency, in the direction that can only turn a failure into a pass."""
    result = audit(
        "see item 4/2", [record("SS8", "9/9"), record("SS26", "4/2")]
    )
    reference = result.references[0]
    assert reference.qualified is False
    assert reference.status is ReferenceStatus.GROUNDED_IN_CONTEXT
    assert reference.candidate_standards == ("SS26", "SS8"), "sorted, and recorded"


def test_a_qualified_reference_records_no_candidate_list() -> None:
    """``candidate_standards`` is only meaningful for the lenient path."""
    result = audit("see SS8 §2/4/4", [record("SS8", "2/4/4")])
    assert result.references[0].candidate_standards == ()


def test_without_a_corpus_index_a_non_grounded_reference_still_fails() -> None:
    result = audit("see SS8 §2/4/1", [record("SS8", "2/4/4")])
    assert result.references[0].status is ReferenceStatus.OUTSIDE_CONTEXT
    assert result.references[0].resolved_via == "no_corpus_index"
    assert result.corpus_index_used is False
    assert not result.passed


def test_the_corpus_index_never_changes_the_verdict_only_the_diagnosis() -> None:
    """The property that makes the index optional, asserted on every status.

    Each text below lands on a different status with the index present. Without
    it, the two failing statuses collapse into one - but ``passed`` is unchanged
    in all four cases, so a caller that cannot load the corpus gets the same
    routing decision with a coarser explanation.
    """
    context = [record("SS8", "2/4/4")]
    index = CorpusClauseIndex.from_records(
        [record("SS8", "2/4/4"), record("SS8", "2/4/1")]
    )
    texts = {
        "grounded": "see SS8 §2/4/4",
        "outside_context": "see SS8 §2/4/1",
        "unresolvable": "see SS8 §9/9/9",
        "unknown_standard": "see item 1/1 of Shari'ah Standard No. (12)",
    }
    for label, text in texts.items():
        with_index = audit(text, context, corpus_index=index)
        without = audit(text, context)
        assert with_index.passed == without.passed, label


# ---------------------------------------------------------------------------
# The audit object
# ---------------------------------------------------------------------------


def test_distinct_references_deduplicate_by_resolved_identity() -> None:
    """The H01 shape: a header echo and a prose claim are one reference.

    Seven raw slash-numerals in the stored response reduce to two distinct
    references. Reporting seven would describe the answer as more heavily
    referenced than it is.
    """
    text = "[1] SS8 §2/4/4 occ=0 ... as SS8 §2/4/4 states ... and SS8 §2/4/2 too"
    result = audit(text, [record("SS8", "2/4/4"), record("SS8", "2/4/2")])
    assert len(result.references) == 3
    assert result.distinct_reference_count == 2


def test_an_answer_with_no_clause_references_passes() -> None:
    """Six of seven stored responses are in this position.

    The gate is silent on an answer that cites no clause path. Whether such an
    answer is grounded at all is ``citation_integrity``'s question, not this one.
    """
    result = audit("The institution must disclose the value [1].", [record("SS8", "2/4/4")])
    assert result.references == ()
    assert result.passed
    assert result.distinct_reference_count == 0


def test_an_empty_or_none_response_passes_vacuously() -> None:
    for text in ("", None):
        result = audit(text, [record("SS8", "2/4/4")])  # type: ignore[arg-type]
        assert result.passed


def test_an_empty_context_grounds_nothing() -> None:
    index = CorpusClauseIndex.from_records([record("SS8", "2/4/4")])
    result = audit("see SS8 §2/4/4", [], corpus_index=index)
    assert result.references[0].status is ReferenceStatus.OUTSIDE_CONTEXT
    assert result.context_standards == ()


def test_the_audit_dict_carries_no_prose(  # noqa: D103
) -> None:
    result = audit(
        f"see SS8 §2/4/4 because {INVENTED_CLAUSE}", [record("SS8", "2/4/4")]
    )
    payload = result.as_dict()
    import json

    blob = json.dumps(payload)
    assert INVENTED_CLAUSE not in blob
    assert "widget" not in blob
    assert set(payload) == {
        "n_references",
        "n_distinct_references",
        "context_standards",
        "corpus_index_used",
        "min_path_depth",
        "passed",
        "status_counts",
        "references",
    }
    assert set(payload["references"][0]) == {
        "clause_path",
        "standard_id",
        "qualified",
        "char_offset",
        "status",
        "resolved_via",
        "candidate_standards",
    }


def test_status_counts_enumerate_every_status_including_the_zeroes() -> None:
    """A zero must be printed, not absent, so a report cannot silently omit a tier."""
    result = audit("see SS8 §2/4/4", [record("SS8", "2/4/4")])
    assert set(result.counts()) == {status.value for status in ReferenceStatus}
    assert result.counts()["unresolvable"] == 0


def test_failing_statuses_are_every_status_except_grounded() -> None:
    """Grounding is decided from the context alone, so anything else fails.

    The three failing statuses differ in how precisely the failure can be
    described, not in whether it is one.
    """
    assert FAILING_STATUSES == frozenset(ReferenceStatus) - {
        ReferenceStatus.GROUNDED_IN_CONTEXT
    }


def test_a_reference_knows_whether_it_failed() -> None:
    grounded = ClauseReference(
        clause_path="2/4", standard_id="SS8", qualified=True, char_offset=0,
        status=ReferenceStatus.GROUNDED_IN_CONTEXT, resolved_via="exact",
    )
    outside = ClauseReference(
        clause_path="2/5", standard_id="SS8", qualified=True, char_offset=0,
        status=ReferenceStatus.OUTSIDE_CONTEXT, resolved_via="corpus_exact",
    )
    assert not grounded.failed and outside.failed


# ---------------------------------------------------------------------------
# Corpus-gated: the measurements the module docstring cites
# ---------------------------------------------------------------------------


@pytest.mark.requires_private_data
def test_the_extractor_takes_exactly_the_measured_number_of_corpus_references(
    clause_chunks: list[dict[str, Any]],
) -> None:
    """36 references over 362 clause texts, and every one a genuine cross-reference.

    The count is pinned because it is the evidence for the over-extraction claim.
    A change here means either the corpus moved or the extractor's precision did,
    and both need looking at rather than a silently updated number.
    """
    assert len(clause_chunks) == 362
    total = sum(
        len(extract_clause_references(chunk.get("text") or "")) for chunk in clause_chunks
    )
    assert total == 36


@pytest.mark.requires_private_data
def test_no_slash_numeral_in_the_corpus_falls_outside_an_extracted_span(
    clause_chunks: list[dict[str, Any]],
) -> None:
    """Under-extraction check against a bare baseline.

    Compares *spans*, not match starts: ``CLAUSE_REFERENCE_RE`` consumes the cue
    word, so comparing start offsets reports every cue-marked reference as missed.
    """
    crude = re.compile(r"\b\d+(?:/\d+){1,3}\b")
    missed = []
    for chunk in clause_chunks:
        text = chunk.get("text") or ""
        spans = [
            (match.start("path"), match.end("path"))
            for match in CLAUSE_REFERENCE_RE.finditer(text)
        ]
        missed += [
            match.group(0)
            for match in crude.finditer(text)
            if not any(start <= match.start() and match.end() <= end for start, end in spans)
        ]
    assert missed == []


@pytest.mark.requires_private_data
def test_depth_one_references_cost_nothing_on_this_corpus(
    clause_chunks: list[dict[str, Any]], hard_set: list[dict[str, Any]]
) -> None:
    """The measured price of ``MIN_PATH_DEPTH == 2``: zero missed references.

    If AAOIFI prose used ``item 8`` this would be a real blind spot. It does not,
    on these five standards, and neither do the gold answers.
    """
    depth_one = re.compile(r"(?i)\b(?:items?|clauses?|paras?\.?|paragraphs?)\s+(\d{1,2})\b(?!/)")
    corpus_hits = [
        match.group(0)
        for chunk in clause_chunks
        for match in depth_one.finditer(chunk.get("text") or "")
    ]
    gold_hits = [
        match.group(0)
        for item in hard_set
        for match in depth_one.finditer(item.get("gold_answer") or "")
    ]
    assert corpus_hits == []
    assert gold_hits == []


@pytest.mark.requires_private_data
def test_the_cross_standard_prose_idiom_occurs_and_names_standards_outside_the_corpus(
    clause_chunks: list[dict[str, Any]],
) -> None:
    """Why :data:`STANDARD_NUMBER_RE` exists, measured.

    Seven occurrences naming standards 5, 9, 12 and 13. Two of those are in the
    corpus and two are not, which is exactly the case that needs the
    ``UNKNOWN_STANDARD`` tier rather than a lenient resolution.
    """
    named = [
        match.group(1)
        for chunk in clause_chunks
        for match in STANDARD_NUMBER_RE.finditer(chunk.get("text") or "")
    ]
    assert len(named) == 7
    assert sorted({normalise_standard(number) for number in named}) == [
        "SS12",
        "SS13",
        "SS5",
        "SS9",
    ]


@pytest.mark.requires_private_data
def test_every_gold_answer_reference_resolves_to_a_real_clause(
    clause_chunks: list[dict[str, Any]],
    hard_set: list[dict[str, Any]],
    stored_run: dict[str, Any],
) -> None:
    """The core false-positive measurement, and its correct interpretation.

    17 references across the seven gold answers: 11 grounded in the retrieved
    context, 6 outside it, **0 unresolvable and 0 unknown-standard**. Every one of
    the six resolves to a real corpus clause via ``corpus_exact`` - so none is an
    extractor error, and none is a fabricated reference.

    Six of seven gold answers therefore contain at least one reference the gate
    would flag. That is not a false-positive rate: the gold answers were authored
    with the full standards open, and ``reports/retrieval_ablation_n7.md`` records
    that gold retrieval was incomplete. The gate is correct about all six. What
    this measures is that a gold answer is not a context-only answer.
    """
    chunks = {chunk["chunk_id"]: chunk for chunk in clause_chunks}
    index = CorpusClauseIndex.from_records(clause_chunks)
    contexts = {
        row["item_id"]: [
            chunks[entry["chunk_id"]]
            for entry in sorted(row["top5"], key=lambda entry: entry["rank"])
        ]
        for row in stored_run["items"]
    }

    totals = {status.value: 0 for status in ReferenceStatus}
    flagged_items = []
    for item in hard_set:
        result = audit_clause_grounding(
            item["gold_answer"], contexts[item["item_id"]], corpus_index=index
        )
        for status, count in result.counts().items():
            totals[status] += count
        if not result.passed:
            flagged_items.append(item["item_id"])
        assert all(
            reference.resolved_via == "corpus_exact"
            for reference in result.failing_references
        ), "a failure that is not a real corpus clause would be an extractor defect"

    assert totals["grounded_in_context"] == 12
    assert totals["outside_context"] == 6
    assert totals["unresolvable"] == 0
    assert totals["unknown_standard"] == 0
    assert len(flagged_items) == 6, "H07 is the only gold answer that stays in context"


@pytest.mark.requires_private_data
def test_the_stored_model_run_makes_two_grounded_references_and_no_others(
    clause_chunks: list[dict[str, Any]], stored_run: dict[str, Any]
) -> None:
    """The only context-only population available locally, and its size.

    H01 is the sole stored response containing any clause reference: seven raw
    slash-numerals, two distinct references, both grounded. The other six make
    none. So the gate does not fire on any item of the published run, and the
    measured error rate on context-only answers is 0 out of 2 - which is
    uninformative, and is recorded as such rather than as a validated rate.
    """
    chunks = {chunk["chunk_id"]: chunk for chunk in clause_chunks}
    index = CorpusClauseIndex.from_records(clause_chunks)

    raw, distinct, flagged = {}, {}, []
    for row in stored_run["items"]:
        context = [
            chunks[entry["chunk_id"]]
            for entry in sorted(row["top5"], key=lambda entry: entry["rank"])
        ]
        result = audit_clause_grounding(
            row.get("model_response") or "", context, corpus_index=index
        )
        raw[row["item_id"]] = len(result.references)
        distinct[row["item_id"]] = result.distinct_reference_count
        if not result.passed:
            flagged.append(row["item_id"])

    assert raw == {"H01": 7, "H02": 0, "H03": 0, "H04": 0, "H05": 0, "H06": 0, "H07": 0}
    assert distinct["H01"] == 2
    assert flagged == [], "enabling this gate would move no published decision"


@pytest.mark.requires_private_data
def test_the_verdict_is_index_independent_on_every_real_text(
    clause_chunks: list[dict[str, Any]],
    hard_set: list[dict[str, Any]],
    stored_run: dict[str, Any],
) -> None:
    """The optionality claim, checked on all fourteen real texts rather than in the
    abstract - seven gold answers and seven model responses."""
    chunks = {chunk["chunk_id"]: chunk for chunk in clause_chunks}
    index = CorpusClauseIndex.from_records(clause_chunks)
    contexts = {
        row["item_id"]: [
            chunks[entry["chunk_id"]]
            for entry in sorted(row["top5"], key=lambda entry: entry["rank"])
        ]
        for row in stored_run["items"]
    }
    texts = [(item["item_id"], item["gold_answer"]) for item in hard_set]
    texts += [(row["item_id"], row.get("model_response") or "") for row in stored_run["items"]]

    for item_id, text in texts:
        with_index = audit_clause_grounding(text, contexts[item_id], corpus_index=index)
        without = audit_clause_grounding(text, contexts[item_id])
        assert with_index.passed == without.passed, item_id
