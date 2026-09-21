"""The negative-control probe set must stay true against the live corpus.

Every probe in ``data/probes/unanswerable_probes.jsonl`` carries machine-checkable
``basis_evidence``. These tests re-derive that evidence rather than trusting it. The
governing rule, and the reason this module exists at all:

    **A probe whose term later appears in the corpus is a FAILING test, not a silent
    stale label.**

A wrong label in the negative class does not produce an obvious error. It inflates
abstention precision, and it does so invisibly, because the probe still looks
unanswerable in the file. So the file is treated as a set of claims and this module
is the auditor.

Two groups of tests here:

* Corpus-dependent (``requires_private_data``): re-verify all 25 stored probes against
  the real 362-chunk corpus. These skip when ``data/private/`` is absent, never pass
  vacuously and never substitute synthetic text for licensed content.
* Mechanism tests: exercise the verification logic itself on small synthetic corpora
  built here. Their text is obviously invented; none of it is AAOIFI prose. These run
  everywhere and are what prove the auditor can actually fail.

No AAOIFI clause prose appears in this file.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from aaoifi_rag.reporting.probes import (
    ESCALATION_STIPULATION,
    EXPECTED_BEHAVIOR_BY_BASIS,
    PROBE_SCHEMA_VERSION,
    VERIFICATION_BASIS_BY_BASIS,
    Probe,
    ProbeVerificationError,
    UnanswerableBasis,
    assert_chunks_are_single_standard,
    corpus_clause_ids,
    corpus_standards,
    load_probes,
    term_occurrences,
    verify_probe,
    verify_probes,
)

#: The distribution documented in reports/retrieval_stage_blind_spot.md and
#: reports/probe_set_construction.md. Pinned so silently dropping or duplicating a
#: probe fails here rather than quietly changing every downstream denominator.
EXPECTED_BASIS_COUNTS = {
    UnanswerableBasis.TERM_ABSENT: 8,
    UnanswerableBasis.STANDARD_ABSENT: 6,
    UnanswerableBasis.CLAUSE_ABSENT: 5,
    UnanswerableBasis.CROSS_STANDARD: 6,
}

EXPECTED_PROBE_COUNT = 25

#: The five verification tiers the project recognises. A probe may only ever claim the
#: two that need no scholar; the others exist for hard-set items.
SCHOLAR_FREE_TIERS = {"mechanical_corpus_absence", "stipulated_definition"}


def _chunk(index: int, text: str, standard: str = "XX", clause: str = "1") -> dict[str, Any]:
    """One synthetic corpus record. Invented text, real shape."""
    return {
        "chunk_id": f"clause:{standard}:{clause}:occurrence:{index}",
        "standard_id": standard,
        "clause_id": clause,
        "sub_clause_id": None,
        "occurrence_index": index,
        "page_number": index,
        "text": text,
    }


def _synthetic_corpus(*texts: str) -> list[dict[str, Any]]:
    return [_chunk(i + 1, text, clause=str(i + 1)) for i, text in enumerate(texts)]


PLAIN_CORPUS = _synthetic_corpus(
    "The institution shall record the agreed value in its books.",
    "The lessee shall maintain the asset during the term of the lease.",
    "Disclosure shall be made in the notes to the financial statements.",
)


def _term_probe(term: str, corpus: list[dict[str, Any]], **overrides: Any) -> Probe:
    """A term-absent probe whose stored evidence matches ``corpus`` by construction.

    Built here rather than borrowed from the real file so a mechanism test cannot fail
    for the unrelated reason that a synthetic corpus is not 362 chunks long.
    """
    evidence: dict[str, Any] = {
        "term": term,
        "corpus_occurrences": 0,
        "of_chunks": len(corpus),
        "match_rule": "whole word, after normalise_for_match on both sides",
    }
    evidence.update(overrides)
    return Probe(
        probe_id="U-SYNTH-01",
        question_text=f"Is {term} addressed by the indexed standards?",
        basis=UnanswerableBasis.TERM_ABSENT,
        basis_evidence=evidence,
    )


@pytest.fixture(scope="module")
def probes(probes_path) -> list[Probe]:
    return load_probes(probes_path)


@pytest.fixture(scope="module")
def raw_probes(probes_path) -> list[dict[str, Any]]:
    """The file as written, so the on-disk payload is checked and not just the model."""
    with probes_path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


# --------------------------------------------------------------------------------------
# The probe file as shipped
# --------------------------------------------------------------------------------------


def test_probe_file_loads_with_unique_ids(probes: list[Probe]) -> None:
    assert len(probes) == EXPECTED_PROBE_COUNT
    ids = [probe.probe_id for probe in probes]
    assert len(set(ids)) == len(ids), "duplicate probe_id in the shipped file"
    assert all(probe.schema_version == PROBE_SCHEMA_VERSION for probe in probes)


def test_basis_distribution_is_pinned(probes: list[Probe]) -> None:
    """Denominators in three reports depend on these counts; drift must be loud."""
    counts: dict[UnanswerableBasis, int] = {}
    for probe in probes:
        counts[probe.basis] = counts.get(probe.basis, 0) + 1
    assert counts == EXPECTED_BASIS_COUNTS
    assert sum(counts.values()) == EXPECTED_PROBE_COUNT


def test_expected_behavior_is_derived_not_authored(raw_probes: list[dict[str, Any]]) -> None:
    """The stored label must equal the table's label for that basis.

    ``expected_behavior`` is a *function* of the basis. A row where someone hand-edited
    it to something else would make the routing evaluation measure a label no rule
    produced.
    """
    for row in raw_probes:
        basis = UnanswerableBasis(row["unanswerable_basis"])
        assert row["expected_behavior"] == EXPECTED_BEHAVIOR_BY_BASIS[basis], (
            f"{row['probe_id']}: stored {row['expected_behavior']!r} but basis "
            f"{basis.value} implies {EXPECTED_BEHAVIOR_BY_BASIS[basis]!r}"
        )
        assert row["verification_basis"] == VERIFICATION_BASIS_BY_BASIS[basis]
        assert row["verification_basis"] in SCHOLAR_FREE_TIERS
        assert row["requires_scholar"] is False


def test_only_cross_standard_probes_are_stipulated(raw_probes: list[dict[str, Any]]) -> None:
    """The stipulation must travel with the file, and only on the stipulated class.

    A reader who rejects the cross-standard definition has to be able to find every
    item that depends on it. If the warning were only in the module docstring, a copy
    of the JSONL would look like 25 equally factual labels.
    """
    for row in raw_probes:
        is_cross = row["unanswerable_basis"] == UnanswerableBasis.CROSS_STANDARD.value
        assert ("stipulation" in row) is is_cross, (
            f"{row['probe_id']}: stipulation block present={('stipulation' in row)} "
            f"but cross_standard={is_cross}"
        )
        if not is_cross:
            assert row["verification_basis"] == "mechanical_corpus_absence"
            continue
        stipulation = row["stipulation"]
        assert stipulation == ESCALATION_STIPULATION
        warning = stipulation["warning"].lower()
        assert "stipulated" in warning
        assert "not a qualified scholar judgement" in warning
        assert "stipulated" in row["notes"].lower()


def test_abstain_and_escalate_classes_are_disjoint_and_complete(probes: list[Probe]) -> None:
    """19 abstain + 6 escalate = 25, with no third value.

    ``metrics.py`` reports escalation under a separate key and never pools it with
    abstention. That is only sound if the two label sets partition the probe set.
    """
    abstain = {p.probe_id for p in probes if p.expected_behavior == "abstain"}
    escalate = {p.probe_id for p in probes if p.expected_behavior == "escalate"}
    assert abstain & escalate == set()
    assert len(abstain) == 19
    assert len(escalate) == 6
    assert len(abstain | escalate) == EXPECTED_PROBE_COUNT


def test_round_trip_through_as_dict(probes: list[Probe]) -> None:
    """``from_dict(as_dict(p))`` must reproduce every probe exactly."""
    for probe in probes:
        assert Probe.from_dict(probe.as_dict()) == probe


def test_evaluation_items_carry_no_gold_clauses(probes: list[Probe]) -> None:
    """A probe has no gold clause, so per-item recall must be undefined, not 0.0.

    Emitting ``gold_clause_ids: []`` is what keeps a 0/0 out of the macro recall mean.
    """
    for probe in probes:
        item = probe.as_evaluation_item()
        assert item["gold_clause_ids"] == []
        assert item["gold_answer"] is None
        assert item["item_id"] == probe.probe_id
        assert item["expected_behavior"] == probe.expected_behavior
        assert item["verification_basis"] == probe.verification_basis


# --------------------------------------------------------------------------------------
# Against the live corpus. These skip when data/private/ is absent; they never fake it.
# --------------------------------------------------------------------------------------


@pytest.mark.requires_private_data
def test_all_probes_verify_against_live_corpus(
    probes: list[Probe], clause_chunks: list[dict[str, Any]]
) -> None:
    """The decisive test. Every stored justification re-derived from the real corpus.

    ``verify_probes`` raises with the full failure list, so a regression names the
    probe, the term and the new occurrence count rather than just failing.
    """
    verify_probes(probes, clause_chunks)


@pytest.mark.requires_private_data
def test_corpus_shape_is_what_the_probe_evidence_claims(
    clause_chunks: list[dict[str, Any]]
) -> None:
    """362 chunks over 5 standards. Every ``of_chunks`` field asserts the first number."""
    assert len(clause_chunks) == 362
    assert corpus_standards(clause_chunks) == {"SS8", "SS9", "SS13", "SS17", "SS26"}
    assert_chunks_are_single_standard(clause_chunks)


@pytest.mark.requires_private_data
def test_every_term_absent_probe_still_has_zero_occurrences(
    probes: list[Probe], clause_chunks: list[dict[str, Any]]
) -> None:
    """Per-probe form of the rule, so a failure points at one term, not the batch."""
    checked = 0
    for probe in probes:
        if probe.basis is not UnanswerableBasis.TERM_ABSENT:
            continue
        term = str(probe.basis_evidence["term"])
        found = term_occurrences(term, clause_chunks)
        assert found == 0, (
            f"{probe.probe_id}: {term!r} now occurs in {found} of "
            f"{len(clause_chunks)} chunks - the probe's label is stale and its "
            f"expected_behavior 'abstain' is no longer a fact about this corpus"
        )
        assert int(probe.basis_evidence["of_chunks"]) == len(clause_chunks)
        checked += 1
    assert checked == EXPECTED_BASIS_COUNTS[UnanswerableBasis.TERM_ABSENT]


@pytest.mark.requires_private_data
def test_standard_and_clause_absence_probes_are_still_absent(
    probes: list[Probe], clause_chunks: list[dict[str, Any]]
) -> None:
    """The two structural bases, checked as set membership against the real index."""
    standards = corpus_standards(clause_chunks)
    n_standard = n_clause = 0
    for probe in probes:
        evidence = probe.basis_evidence
        if probe.basis is UnanswerableBasis.STANDARD_ABSENT:
            named = str(evidence["named_standard"])
            assert named not in standards, f"{probe.probe_id}: {named} IS indexed"
            assert set(evidence["indexed_standards"]) == standards, (
                f"{probe.probe_id}: stored indexed_standards drifted from the corpus"
            )
            n_standard += 1
        elif probe.basis is UnanswerableBasis.CLAUSE_ABSENT:
            standard = str(evidence["standard_id"])
            clause = str(evidence["clause_id"])
            assert standard in standards, (
                f"{probe.probe_id}: names unindexed {standard}; this probe should be "
                "standard_absent_from_corpus instead"
            )
            assert clause not in corpus_clause_ids(clause_chunks, standard), (
                f"{probe.probe_id}: clause {standard}:{clause} EXISTS - answerable"
            )
            n_clause += 1
    assert n_standard == EXPECTED_BASIS_COUNTS[UnanswerableBasis.STANDARD_ABSENT]
    assert n_clause == EXPECTED_BASIS_COUNTS[UnanswerableBasis.CLAUSE_ABSENT]


@pytest.mark.requires_private_data
def test_cross_standard_probes_span_two_indexed_standards(
    probes: list[Probe], clause_chunks: list[dict[str, Any]]
) -> None:
    """The stipulation only bites if the standards it names are actually indexed.

    A cross-standard probe naming an absent standard would be plainly unanswerable and
    would inflate escalation recall with an item that belongs in the abstain class.
    """
    standards = corpus_standards(clause_chunks)
    checked = 0
    for probe in probes:
        if probe.basis is not UnanswerableBasis.CROSS_STANDARD:
            continue
        named = [str(value) for value in probe.basis_evidence["standard_ids"]]
        assert len(set(named)) >= 2, f"{probe.probe_id}: names {named}"
        assert int(probe.basis_evidence["n_standards_required"]) == len(set(named))
        assert set(named) <= standards, f"{probe.probe_id}: {set(named) - standards}"
        checked += 1
    assert checked == EXPECTED_BASIS_COUNTS[UnanswerableBasis.CROSS_STANDARD]


# --------------------------------------------------------------------------------------
# Mechanism: can the auditor actually fail? Synthetic corpora, invented text.
# --------------------------------------------------------------------------------------


def test_clean_probe_verifies_with_no_failures() -> None:
    probe = _term_probe("widgetronic", PLAIN_CORPUS)
    assert verify_probe(probe, PLAIN_CORPUS) == []


def test_stale_term_label_is_a_failure() -> None:
    """The governing rule of this module, executed.

    A probe asserting a term is absent must fail the moment the term appears. The
    failure string has to name the new count, because the interesting question when
    this fires is whether the corpus grew or the label was wrong to begin with.
    """
    corpus = PLAIN_CORPUS + _synthetic_corpus(
        "A widgetronic arrangement shall be disclosed by the institution."
    )
    probe = _term_probe("widgetronic", PLAIN_CORPUS)  # evidence still claims 0 of 3
    failures = verify_probe(probe, corpus)
    assert failures, "term is present but verification passed - the auditor is blind"
    assert any("no longer unanswerable" in failure for failure in failures)
    assert any("occurs in 1 chunk" in failure for failure in failures)
    with pytest.raises(ProbeVerificationError, match="widgetronic"):
        verify_probes([probe], corpus)


def test_stored_occurrence_count_drift_is_a_failure() -> None:
    """Evidence that disagrees with recomputation fails even when both are non-zero.

    Guards the case where someone updates ``corpus_occurrences`` to a non-zero value to
    silence the previous test instead of retiring the probe.
    """
    probe = _term_probe("widgetronic", PLAIN_CORPUS, corpus_occurrences=3)
    failures = verify_probe(probe, PLAIN_CORPUS)
    assert any("stored corpus_occurrences=3 but recomputed 0" in f for f in failures)


def test_corpus_size_drift_is_a_failure() -> None:
    """``of_chunks`` pins the denominator; a resized corpus invalidates the claim."""
    probe = _term_probe("widgetronic", PLAIN_CORPUS)
    grown = PLAIN_CORPUS + _synthetic_corpus("An unrelated provision about leases.")
    failures = verify_probe(probe, grown)
    assert any("of_chunks=3 but corpus has 4" in f for f in failures)


def test_term_absent_probe_without_a_term_fails_fast() -> None:
    probe = Probe(
        probe_id="U-SYNTH-02",
        question_text="Is anything addressed?",
        basis=UnanswerableBasis.TERM_ABSENT,
        basis_evidence={},
    )
    assert verify_probe(probe, PLAIN_CORPUS) == [
        "U-SYNTH-02: term_absent probe has no 'term' in evidence"
    ]


# --------------------------------------------------------------------------------------
# term_occurrences: the exact matching semantics the labels depend on
# --------------------------------------------------------------------------------------

MATCH_CORPUS = _synthetic_corpus(
    "The widgetronic structure is permitted.",
    "Widgetronics in the plural are a different word.",
    "A WIDGETRONIC in capitals, followed by punctuation.",
    "The semi-widgetronic hybrid appears here.",
)


@pytest.mark.parametrize(
    ("term", "expected", "why"),
    [
        ("widgetronic", 3, "matches chunks 1, 3 and 4; not the plural in chunk 2"),
        ("widgetronics", 1, "exact plural only - there is no stemming"),
        ("WiDgEtRoNiC", 3, "case is folded on both sides"),
        ("semi-widgetronic", 1, "a hyphen is preserved and matches literally"),
        ("semi", 1, "a hyphen acts as a word boundary, so the prefix is a whole word"),
        ("widgetron", 0, "a prefix that is not a whole word does not match"),
        ("nonexistentterm", 0, "absent terms return zero, which is what a probe claims"),
    ],
)
def test_term_occurrences_semantics(term: str, expected: int, why: str) -> None:
    """Pinned because every ``term_absent_from_corpus`` label is exactly this function.

    Note what is *not* folded: ``normalise_for_match`` lowercases and NFKC-normalises but
    does not strip punctuation or stem. So a probe term must be written the way the
    corpus writes it, and ``semi-widgetronic`` and ``widgetronic`` are distinct claims
    even though the hyphen makes the second match the first's suffix.
    """
    assert term_occurrences(term, MATCH_CORPUS) == expected, why


def test_term_occurrences_counts_chunks_not_hits() -> None:
    """Stability under re-chunking: two hits in one chunk still count once."""
    corpus = _synthetic_corpus("widgetronic and widgetronic again in one chunk")
    assert term_occurrences("widgetronic", corpus) == 1


def test_term_occurrences_rejects_an_empty_term() -> None:
    """An empty needle would match every chunk and label every probe answerable."""
    with pytest.raises(ValueError, match="empty after normalisation"):
        term_occurrences("   ", MATCH_CORPUS)


# --------------------------------------------------------------------------------------
# The structural bases, and the fact the stipulation rests on
# --------------------------------------------------------------------------------------

TWO_STANDARD_CORPUS = [
    _chunk(1, "A provision of the first standard.", standard="AA", clause="1"),
    _chunk(2, "A provision of the second standard.", standard="BB", clause="1"),
    _chunk(3, "A further provision of the first standard.", standard="AA", clause="2"),
]


def test_standard_absent_probe_fails_when_the_standard_is_indexed() -> None:
    probe = Probe(
        probe_id="U-SYNTH-03",
        question_text="What does the AA standard require?",
        basis=UnanswerableBasis.STANDARD_ABSENT,
        basis_evidence={"named_standard": "AA", "indexed_standards": ["AA", "BB"]},
    )
    failures = verify_probe(probe, TWO_STANDARD_CORPUS)
    assert any("IS indexed" in failure for failure in failures)


def test_clause_absent_probe_fails_when_the_clause_exists() -> None:
    probe = Probe(
        probe_id="U-SYNTH-04",
        question_text="What does clause 2 of AA require?",
        basis=UnanswerableBasis.CLAUSE_ABSENT,
        basis_evidence={"standard_id": "AA", "clause_id": "2"},
    )
    assert any("EXISTS" in f for f in verify_probe(probe, TWO_STANDARD_CORPUS))


def test_clause_absent_probe_on_an_unindexed_standard_is_the_wrong_basis() -> None:
    """Mislabelling matters: the two bases have different evidentiary content.

    A clause-absence claim about a standard that is not indexed at all is trivially
    true and says nothing, so the auditor redirects it rather than passing it.
    """
    probe = Probe(
        probe_id="U-SYNTH-05",
        question_text="What does clause 9 of ZZ require?",
        basis=UnanswerableBasis.CLAUSE_ABSENT,
        basis_evidence={"standard_id": "ZZ", "clause_id": "9"},
    )
    failures = verify_probe(probe, TWO_STANDARD_CORPUS)
    assert any("use standard_absent_from_corpus instead" in f for f in failures)


def test_cross_standard_probe_needs_two_distinct_indexed_standards() -> None:
    one = Probe(
        probe_id="E-SYNTH-01",
        question_text="Compare AA with AA.",
        basis=UnanswerableBasis.CROSS_STANDARD,
        basis_evidence={"standard_ids": ["AA", "AA"], "n_standards_required": 2},
    )
    assert any("at least two distinct" in f for f in verify_probe(one, TWO_STANDARD_CORPUS))

    absent = Probe(
        probe_id="E-SYNTH-02",
        question_text="Compare AA with ZZ.",
        basis=UnanswerableBasis.CROSS_STANDARD,
        basis_evidence={"standard_ids": ["AA", "ZZ"], "n_standards_required": 2},
    )
    failures = verify_probe(absent, TWO_STANDARD_CORPUS)
    assert any("simply unanswerable instead" in f for f in failures)


def test_single_standard_assertion_rejects_a_chunk_without_a_standard() -> None:
    """The escalation stipulation is unsound if a chunk can span standards.

    ``cross_standard_comparison`` means "no single retrieved clause can settle this"
    *because* each chunk belongs to one standard. That premise is checked, not assumed,
    so a future re-chunking that merged standards would break the tests rather than
    quietly invalidate six escalation labels.
    """
    broken = TWO_STANDARD_CORPUS + [
        {"chunk_id": "clause:?:1:occurrence:1", "standard_id": "", "text": "Merged."}
    ]
    with pytest.raises(ProbeVerificationError, match="stipulation is unsound"):
        assert_chunks_are_single_standard(broken)


def test_verify_probes_reports_duplicate_ids() -> None:
    """Two probes with one id would double-count one item in every rate."""
    probe = _term_probe("widgetronic", PLAIN_CORPUS)
    with pytest.raises(ProbeVerificationError, match="duplicate probe_id"):
        verify_probes([probe, probe], PLAIN_CORPUS)


def test_verify_probes_aggregates_every_failure() -> None:
    """One raise, all failures. Fixing probes one exception at a time is slow."""
    corpus = PLAIN_CORPUS + _synthetic_corpus("A widgetronic and a sprocketal appear.")
    stale_a = _term_probe("widgetronic", PLAIN_CORPUS)
    stale_b = Probe(
        probe_id="U-SYNTH-06",
        question_text="Is sprocketal treatment addressed?",
        basis=UnanswerableBasis.TERM_ABSENT,
        basis_evidence={"term": "sprocketal", "corpus_occurrences": 0, "of_chunks": 4},
    )
    with pytest.raises(ProbeVerificationError) as excinfo:
        verify_probes([stale_a, stale_b], corpus)
    message = str(excinfo.value)
    assert "U-SYNTH-01" in message and "U-SYNTH-06" in message
    # Five, and each one is a distinct claim rather than the same fault restated:
    # U-SYNTH-01 fails the occurrence check, the stored-count check and the of_chunks
    # check (its evidence was written against a 3-chunk corpus); U-SYNTH-06 fails the
    # first two but has the right denominator.
    assert "5 probe verification failure(s)" in message
    assert message.count("no longer unanswerable") == 2
    assert message.count("of_chunks=") == 1
