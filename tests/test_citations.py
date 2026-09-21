"""Mechanical citation auditing: what it can prove, and what it deliberately will not say.

The context block numbers its excerpts ``[1]``..``[k]`` and those are the only citation
targets the model has, so citation *indices* are checkable without judgement. Three
design choices here are easy to get wrong in a re-implementation, and each is asserted:

* **Segmentation is marker-driven, not quotation-driven.** Six of the seven stored
  responses contain no quotation marks at all, so an extractor keyed on quotes would have
  found nothing to audit in H05 - the one item with a real citation defect. Each ``[n]``
  owns the text up to the next marker.
* **Attribution is ordered, and the order is load-bearing.** ``supported`` is tested
  before ``corrupted``, so a lightly garbled transcription of the *right* excerpt is
  reported supported with its script characters recorded rather than as corruption. On
  the stored run that is exactly what happens to H05's first segment, which is why the
  run reports ``corrupted: 0`` while carrying ten distinct Arabic substitutions.
* **"Not verbatim" is not "unsupported".** A segment that matches nothing is
  ``unverified`` and does not gate. Separating paraphrase from fabrication needs
  entailment checking, which this layer does not do; ``reports/citation_entailment_check_n7.md``
  called H05's segment invented content, and the audit shows it is genuine corpus text
  cited under the wrong index.

No AAOIFI clause prose appears here. The invented records below are deliberately
dissimilar to each other so that cross-rank overlap is unambiguous.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from aaoifi_rag.generation.prompt import clause_label
from aaoifi_rag.reliability.citations import (
    CITATION_MARKER_RE,
    FAILING_ATTRIBUTIONS,
    MIN_SEGMENT_WORDS,
    MIN_VERBATIM_OVERLAP,
    Attribution,
    audit_citations,
    rendered_rank_texts,
    split_citation_segments,
)

#: U+0648, the character H05 emitted in place of an apostrophe.
ARABIC_WAW = "و"

RANK_1 = (
    "The institution shall record the widget at the value agreed between the parties "
    "and disclose that value in the notes."
)
RANK_2 = (
    "The trustee shall submit an annual statement of the fund to the supervisory board "
    "before the end of the period."
)
RANK_3 = "Scope And Definitions"


def _record(index: int, text: str) -> dict[str, Any]:
    return {
        "chunk_id": f"clause:XX{index}:1/{index}:occurrence:1",
        "standard_id": f"XX{index}",
        "clause_id": f"1/{index}",
        "sub_clause_id": None,
        "occurrence_index": 1,
        "source_page": 10 + index,
        "text": text,
    }


CONTEXT = [_record(1, RANK_1), _record(2, RANK_2), _record(3, RANK_3)]


def _only(text: str, **kwargs: Any) -> Any:
    """Audit ``text`` against :data:`CONTEXT` and return its single segment."""
    audit = audit_citations(text, CONTEXT, **kwargs)
    assert len(audit.segments) == 1, "helper is for single-citation responses"
    return audit.segments[0]


# --------------------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------------------


def test_a_marker_owns_the_text_up_to_the_next_marker() -> None:
    assert split_citation_segments("[1] first bit [2] second bit") == [
        (1, 0, " first bit "),
        (2, 14, " second bit"),
    ]


def test_text_before_the_first_marker_belongs_to_no_citation() -> None:
    """Uncited lead-in is not attributed to anything.

    It is not lost: the whole-response echo ratio in
    :mod:`aaoifi_rag.reliability.signals` measures the response including its lead-in, so
    a response that transcribes a clause *before* citing it still registers as echo.
    """
    assert split_citation_segments("lead in [1] owned") == [(1, 8, " owned")]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[1] a [12] b [0] c", ["1", "12", "0"]),
        ("[123] not a marker", []),
        ("[ 1 ] spaced", []),
        ("no markers here", []),
    ],
)
def test_only_one_or_two_digit_bracketed_integers_are_markers(text, expected) -> None:
    """``[123]`` is not a citation. Two digits is already far past any top-k used here."""
    assert CITATION_MARKER_RE.findall(text) == expected


def test_a_response_with_no_marker_produces_no_segments() -> None:
    assert split_citation_segments("The institution shall record the widget.") == []
    assert split_citation_segments("") == []
    assert split_citation_segments(None) == []


def test_marker_offsets_are_recorded_so_a_segment_can_be_located() -> None:
    audit = audit_citations(f"[1] {RANK_1} [2] {RANK_2}", CONTEXT)
    assert [segment.marker_start for segment in audit.segments] == [0, 122]


# --------------------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------------------


def test_a_verbatim_segment_is_supported() -> None:
    segment = _only(f"[1] {RANK_1}")
    assert segment.attribution is Attribution.SUPPORTED
    assert segment.overlap_with_cited_rank == 1.0
    assert segment.best_matching_rank == 1
    assert segment.failed is False


def test_the_same_text_under_the_wrong_rank_is_misattributed() -> None:
    """H05's defect, on invented text.

    The mechanically certain part is the index: the words are in the context, just not at
    the rank claimed. Nothing here says the content is wrong - only that a reader
    following the citation would land on the wrong excerpt.
    """
    segment = _only(f"[2] {RANK_1}")
    assert segment.attribution is Attribution.MISATTRIBUTED
    assert segment.overlap_with_cited_rank == 0.0
    assert segment.best_matching_rank == 1
    assert segment.best_overlap == 1.0
    assert segment.failed is True


@pytest.mark.parametrize("rank", [0, 4, 9, 99])
def test_an_out_of_range_rank_fails_before_any_text_is_compared(rank) -> None:
    """The only check that needs no corpus at all, and the only certain one.

    ``[0]`` matters: the context block is 1-indexed, so a zero is a real off-by-one and
    must be caught rather than silently treated as rank 1.
    """
    audit = audit_citations(f"[{rank}] {RANK_1}", CONTEXT)
    segment = audit.segments[0]
    assert segment.attribution is Attribution.UNKNOWN_RANK
    assert segment.overlap_with_cited_rank == 0.0
    assert audit.out_of_range_ranks == (rank,)
    assert audit.passed is False


def test_the_segment_length_floor_is_exactly_six_words() -> None:
    """Below the floor no attribution is claimed in either direction."""
    assert MIN_SEGMENT_WORDS == 6
    assert _only("[1] The institution shall record the widget").attribution is (
        Attribution.SUPPORTED
    )
    assert _only("[1] The institution shall record the").attribution is Attribution.TOO_SHORT


def test_a_short_segment_records_an_overlap_it_does_not_act_on() -> None:
    """The length floor is checked before the overlap, and that ordering is deliberate.

    Since ``verbatim_overlap_ratio`` compares a sub-window answer at its own width, a
    four-word verbatim fragment now scores 1.0. The floor is what stops that from being
    read as a supported citation: four words of a clause are not evidence that the
    citation is right, because four words match too easily. The score is still recorded so
    the label can be inspected.
    """
    segment = _only("[1] The institution shall record")
    assert segment.attribution is Attribution.TOO_SHORT
    assert segment.overlap_with_cited_rank == 1.0
    assert segment.failed is False


def test_prose_matching_nothing_is_unverified_rather_than_unsupported() -> None:
    """The limitation the module is explicit about, asserted rather than described."""
    segment = _only(
        "[1] Something entirely different is asserted here about matters nobody wrote."
    )
    assert segment.attribution is Attribution.UNVERIFIED
    assert segment.best_matching_rank is None
    assert segment.best_overlap == 0.0
    assert segment.failed is False


def test_only_index_and_corruption_failures_gate() -> None:
    """``unverified`` and ``too_short`` are recorded and never fail the audit.

    A gate that failed on ``unverified`` would be an entailment gate wearing a citation
    gate's name, and it would have failed H01 - whose second segment is legitimate
    synthesis at 0.19 overlap.
    """
    assert FAILING_ATTRIBUTIONS == {
        Attribution.UNKNOWN_RANK,
        Attribution.MISATTRIBUTED,
        Attribution.CORRUPTED,
    }
    assert Attribution.UNVERIFIED not in FAILING_ATTRIBUTIONS
    assert Attribution.TOO_SHORT not in FAILING_ATTRIBUTIONS
    assert Attribution.SUPPORTED not in FAILING_ATTRIBUTIONS


def test_the_overlap_threshold_is_a_parameter_and_travels_with_the_result() -> None:
    """A stored audit is uninterpretable without the threshold that produced it."""
    assert MIN_VERBATIM_OVERLAP == 0.5
    text = f"[1] {RANK_1}"
    default = audit_citations(text, CONTEXT)
    assert default.min_verbatim_overlap == 0.5
    assert default.as_dict()["min_verbatim_overlap"] == 0.5
    strict = audit_citations(f"[1] {RANK_1[:60]} and something else entirely different",
                            CONTEXT, min_verbatim_overlap=0.99)
    assert strict.segments[0].attribution is Attribution.UNVERIFIED
    assert strict.min_verbatim_overlap == 0.99


def test_the_provenance_label_is_part_of_the_matchable_rank_text() -> None:
    """H01 transcribes the header line, so the header has to be matchable.

    Matching against ``record["text"]`` alone would have scored H01's first segment lower
    than it deserves and could have turned a supported citation into an unverified one.
    """
    rendered = rendered_rank_texts(CONTEXT)
    assert rendered[0].startswith(clause_label(CONTEXT[0]))
    assert rendered[0].endswith(RANK_1)
    segment = _only(f"[1] {clause_label(CONTEXT[0])}")
    assert segment.attribution is Attribution.SUPPORTED


def test_an_empty_context_leaves_every_rank_unknown() -> None:
    """The zero-retrieval case: nothing can be cited, so every marker is out of range."""
    audit = audit_citations(f"[1] {RANK_1}", [])
    assert audit.context_size == 0
    assert audit.segments[0].attribution is Attribution.UNKNOWN_RANK
    assert audit.segments[0].best_matching_rank is None


# --------------------------------------------------------------------------------------
# Script corruption
# --------------------------------------------------------------------------------------


def test_light_corruption_of_the_right_excerpt_stays_supported() -> None:
    """One substitution breaks only the windows containing it.

    So the script characters are recorded on a *supported* segment. Any reading of the
    attribution counts that treats ``corrupted: 0`` as "no script corruption" is wrong;
    ``unexpected_script`` is where that lives.
    """
    segment = _only(f"[1] {RANK_1.replace('institution', 'institution' + ARABIC_WAW)}")
    assert segment.attribution is Attribution.SUPPORTED
    assert segment.overlap_with_cited_rank == pytest.approx(0.8462, abs=5e-5)
    assert segment.unexpected_script == (ARABIC_WAW,)
    assert segment.failed is False


def test_dense_corruption_is_reported_corrupted_even_under_the_wrong_rank() -> None:
    """And therefore ``misattributed`` is a lower bound when script is corrupt.

    This segment transcribes rank 1 while citing rank 2, so both defects are present. The
    unstripped text matches nothing at 0.5, so the misattribution is never reached and the
    label is ``corrupted``. Both are failures, so no decision is lost - but a table of
    attribution counts under-reports misattribution by however many segments are also
    garbled.
    """
    dense = RANK_1
    for word in ("institution", "widget", "value", "between", "parties", "disclose"):
        dense = dense.replace(word, word + ARABIC_WAW)
    segment = _only(f"[2] {dense}")
    assert segment.attribution is Attribution.CORRUPTED
    assert segment.overlap_with_cited_rank == 0.0
    assert segment.best_overlap == 0.0
    assert segment.failed is True


# --------------------------------------------------------------------------------------
# The audit as a whole
# --------------------------------------------------------------------------------------


def test_an_answer_attempt_that_cites_nothing_fails() -> None:
    """Enforced on principle, not on evidence.

    All three n=7 answer attempts cited at least one rank, so this rule is untested by the
    stored run. It ships because an uncitable answer cannot be checked at all - the
    weakest possible justification, and stated as such here and in the module.
    """
    audit = audit_citations(RANK_1, CONTEXT, is_answer_attempt=True)
    assert audit.segments == ()
    assert audit.uncited_answer_attempt is True
    assert audit.passed is False


def test_an_abstention_that_cites_nothing_is_not_a_citation_failure() -> None:
    """Otherwise every abstention would fail the citation gate."""
    audit = audit_citations("I cannot answer from the given context", CONTEXT,
                            is_answer_attempt=False)
    assert audit.uncited_answer_attempt is False
    assert audit.passed is True


def test_cited_ranks_are_deduplicated_in_first_seen_order() -> None:
    audit = audit_citations(f"[2] {RANK_2} [1] {RANK_1} [2] {RANK_2}", CONTEXT)
    assert audit.cited_ranks == (2, 1)
    assert len(audit.segments) == 3


def test_counts_enumerate_every_attribution_including_the_zeroes() -> None:
    """A missing key and a zero read differently in a report table."""
    audit = audit_citations(f"[1] {RANK_1} [9] {RANK_2}", CONTEXT)
    assert audit.counts() == {
        "unknown_rank": 1,
        "supported": 1,
        "misattributed": 0,
        "corrupted": 0,
        "unverified": 0,
        "too_short": 0,
    }
    assert set(audit.counts()) == {member.value for member in Attribution}


def test_the_serialised_audit_carries_scores_and_ranks_but_no_text() -> None:
    """The publication boundary, asserted at the point where text could escape."""
    audit = audit_citations(f"[1] {RANK_1} [2] {RANK_1}", CONTEXT)
    payload = audit.as_dict()
    serialised = json.dumps(payload, ensure_ascii=False)
    for fragment in ("institution", "widget", "trustee", "Scope"):
        assert fragment not in serialised
    assert payload["n_segments"] == 2
    assert payload["attribution_counts"]["misattributed"] == 1
    assert payload["passed"] is False
    assert set(payload["segments"][0]) == {
        "rank",
        "marker_start",
        "word_count",
        "attribution",
        "overlap_with_cited_rank",
        "best_overlap",
        "best_matching_rank",
        "unexpected_script",
    }


def test_the_audit_and_its_segments_are_frozen() -> None:
    audit = audit_citations(f"[1] {RANK_1}", CONTEXT)
    with pytest.raises(dataclasses.FrozenInstanceError):
        audit.segments = ()  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        audit.segments[0].rank = 2  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# Against the stored run
# --------------------------------------------------------------------------------------


def _stored_audits(repo_root: Path, stored_run: dict[str, Any]) -> dict[str, Any]:
    """Re-derive each item's audit through the same rejoin the replay script uses."""
    scripts = repo_root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from replay_n7_router import read_jsonl, rejoin_context  # noqa: PLC0415

    chunks = {
        record["chunk_id"]: record
        for record in read_jsonl(
            repo_root / "data" / "private" / "extracted" / "clause_chunks.jsonl"
        )
    }
    from aaoifi_rag.reliability.response_class import classify_response  # noqa: PLC0415

    audits = {}
    for item in stored_run["items"]:
        context = rejoin_context(item["top5"], chunks)
        response = item["model_response"] or ""
        audits[item["item_id"]] = audit_citations(
            response,
            context,
            is_answer_attempt=classify_response(response).is_answer_attempt,
        )
    return audits


def _published_audits(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "reports" / "replay_n7_public_traces.jsonl"
    published = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            trace = json.loads(line)
            response = (trace.get("signals") or {}).get("response")
            if response:
                published[trace["item_id"]] = response["citation_audit"]
    return published


@pytest.mark.requires_private_data
def test_the_published_citation_audits_are_reproducible(repo_root, stored_run) -> None:
    """Field for field, from the current module over the real contexts.

    This is the claim that ``reports/replay_n7_public_traces.jsonl`` is a derived artifact
    and not a snapshot: if the threshold, the segmentation or the attribution order
    changed, the published numbers would stop matching here.
    """
    derived = _stored_audits(repo_root, stored_run)
    published = _published_audits(repo_root)
    assert set(published) == {f"H0{index}" for index in range(1, 8)}
    for item_id, expected in published.items():
        assert derived[item_id].as_dict() == expected, item_id
    uncited = {
        item_id for item_id, audit in derived.items() if audit.uncited_answer_attempt
    }
    assert uncited == set(), "every answer attempt on the stored run cited something"


@pytest.mark.requires_private_data
def test_h05_cites_rank_two_for_text_that_lives_at_rank_four(repo_root, stored_run) -> None:
    """The one mechanically certain citation defect in the run.

    ``reports/citation_entailment_check_n7.md`` recorded this segment as invented content.
    It is not: 62 words at 0.873 overlap with rank 4 and 0.0 with the rank it cites. The
    correction matters because "the model fabricated a clause" and "the model mislabelled
    a real clause" call for different mitigations.
    """
    audit = _stored_audits(repo_root, stored_run)["H05"]
    misattributed = [
        segment
        for segment in audit.segments
        if segment.attribution is Attribution.MISATTRIBUTED
    ]
    assert len(misattributed) == 1
    segment = misattributed[0]
    assert (segment.rank, segment.best_matching_rank) == (2, 4)
    assert segment.overlap_with_cited_rank == 0.0
    assert segment.best_overlap == pytest.approx(0.8727, abs=5e-5)
    assert segment.word_count == 62
    assert audit.passed is False


@pytest.mark.requires_private_data
def test_the_run_reports_no_corrupted_segment_despite_dense_arabic_substitution(
    repo_root, stored_run
) -> None:
    """The caveat that makes ``corrupted: 0`` readable.

    H05's first segment carries ten distinct Arabic characters and still matches its cited
    rank at 0.64, so it is labelled supported. Script corruption on that run is visible
    only through ``unexpected_script`` and the separate ``script_integrity`` gate - never
    through the attribution counts.
    """
    audits = _stored_audits(repo_root, stored_run)
    assert all(audit.counts()["corrupted"] == 0 for audit in audits.values())
    first = audits["H05"].segments[0]
    assert first.attribution is Attribution.SUPPORTED
    assert first.overlap_with_cited_rank == pytest.approx(0.6429, abs=5e-5)
    assert len(first.unexpected_script) == 10
    assert ARABIC_WAW in first.unexpected_script


@pytest.mark.requires_private_data
def test_no_stored_segment_is_short_enough_for_the_window_rule_to_reach(
    repo_root, stored_run
) -> None:
    """Why narrowing the shingle window for short answers moved no published number.

    Every segment in the run is at least 16 words, well past both the 6-word attribution
    floor and the 8-word shingle width, so the sub-window comparison rule in
    ``text_checks`` cannot have applied to any of them.
    """
    lengths = sorted(
        segment.word_count
        for audit in _stored_audits(repo_root, stored_run).values()
        for segment in audit.segments
    )
    assert lengths == [16, 35, 52, 62, 140]
    assert min(lengths) >= 8
