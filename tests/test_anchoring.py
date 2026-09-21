"""Lexical anchoring: the tokeniser contract, the two independent filters, and the
arithmetic behind a rejected gate.

``anchoring.py`` is the one reliability module whose measurements are *published as a
rejection* (``reports/anchoring_gate_calibration.md``: AUC 0.513, gate shipped disabled).
That makes its tests unusual in purpose. They are not here to show the gate works - it
does not - they are here so the numbers that disabled it cannot drift unnoticed, and so
the mechanism producing them is pinned rather than described.

Four things a reader needs before trusting any assertion below:

* **The tokeniser does not keep slash numerals, and the module comment claiming it does
  is wrong.** ``ANCHOR_TOKEN_RE`` puts ``[\\w]+`` before ``\\d+(?:/\\d+)*`` in its
  alternation, so ``2/4/2`` is matched as ``2``, ``4``, ``2`` - and each of those is then
  dropped by the bare-digit rule. A question citing a clause number therefore contributes
  **no anchor at all** from that citation. ``BM25_TOKEN_RE`` has the same alternation
  order, so the two are genuinely in sync, which is what the duplication exists for; the
  claim that *either* keeps ``9/2`` whole is what fails.
  :func:`test_no_token_from_the_public_path_ever_contains_a_slash` pins the real
  behaviour and :func:`test_the_slash_branch_is_reachable_only_by_calling_it_directly`
  records that ``AnchorSpec.keep_slash_numerals`` is consequently unreachable through
  :func:`anchor_tokens`.
* **The stop list and the IDF floor are not redundant.** ``shall`` has IDF 2.46 against
  the real corpus and would pass the floor; ``institution`` has 1.53 and would pass any
  plausible stop list. Each filter catches what the other misses, which is why removing
  either is not a simplification.
* **Coverage is ``None``, not ``0.0``, for a query with no anchors.** A gate reading it
  must treat that as "no measurement", and every arithmetic property here returns ``None``
  in that state rather than a number that would compare as very low.
* **IDF weighting is what stops a covered question frame from masking an absent topic.**
  The unweighted and weighted numbers are asserted on the same signal so the gap is
  visible: 0.750 unweighted against 0.499 weighted, on a query whose only uncovered
  anchor is its subject.

The two sections at the end do different jobs and close a loop. One needs no private
data: it recomputes the report's threshold sweep and every AUC in its §4 table from the
per-item coverages in the tracked JSON, so the report's *arithmetic* is checked even on a
clone without the corpus. The other is corpus-gated: it re-derives all 21 published
``stored_hybrid_rerank`` rows from live code over the real hard set, so the *measurement*
behind those coverages is checked too - and it pins the counterfactual behind §12 of the
report, namely that swapping the alternation order so ``2/4/2`` survives moves six probes
and not one of the seven answerable items.

No AAOIFI clause prose appears here. Every constructed string is invented.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import re
from typing import Any

import pytest

from aaoifi_rag.reliability import anchoring as anchoring_module
from aaoifi_rag.reliability.anchoring import (
    ANCHOR_TOKEN_RE,
    FRAME_TERMS,
    STOPWORDS,
    AnchoringSignal,
    AnchorSpec,
    CorpusVocabulary,
    _is_candidate,
    anchor_tokens,
    compute_anchoring_signal,
    context_surface_terms,
    fold_suffix,
    select_anchors,
)

#: A vocabulary in which *every* term is out of vocabulary, so ``idf`` is the constant
#: ``log(100) = 4.61`` and clears ``min_idf`` for anything. Shape rules can then be tested
#: on their own: whatever this vocabulary rejects was rejected by ``_is_candidate``, not by
#: the IDF floor. Isolating the two is the whole point - against the real corpus a failure
#: could come from either and the assertion would not say which.
FLAT_VOCABULARY = CorpusVocabulary(df={}, n_documents=100, spec=AnchorSpec())

#: An invented context record in the shape retrieval emits, provenance fields included.
CONTEXT = [
    {
        "chunk_id": "clause:XX:1/1:occurrence:1",
        "standard_id": "XX",
        "clause_id": "1/1",
        "sub_clause_id": None,
        "text": "the trustee shall record disclose amount in the register",
    }
]


def _weighted_vocabulary() -> CorpusVocabulary:
    """1000 documents, three terms at df=100 and everything else out of vocabulary.

    Sized so the IDF spread is legible: ``log(1000/101) = 2.293`` for the common terms
    against ``log(1000/1) = 6.908`` for an unseen one, a factor of three. At corpus scale
    (N=362) the spread is smaller but the direction is the same.
    """
    documents = [{"text": "record disclose amount"}] * 100 + [{"text": "filler prose"}] * 900
    return CorpusVocabulary.from_chunks(documents, AnchorSpec())


def _bm25_token_pattern(repo_root: Path) -> str:
    """The ``BM25_TOKEN_RE`` literal, read out of the source with :mod:`ast`.

    Imported by parsing rather than by ``import``: ``bm25_baseline`` pulls in
    ``rank_bm25`` and ``tiktoken`` at module scope, and ``tests/conftest.py`` rule 1
    forbids any test from depending on either. Parsing keeps the sync check honest -
    it reads the real shipped literal - without importing the dependency it is
    checked against.
    """
    source = (repo_root / "src" / "aaoifi_rag" / "retrieval" / "bm25_baseline.py").read_text(
        encoding="utf-8"
    )
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if "BM25_TOKEN_RE" in names:
            return node.value.args[0].value
    raise AssertionError("BM25_TOKEN_RE is no longer a module-level re.compile(...) call")


# --------------------------------------------------------------------------------------
# The tokeniser, and its contract with BM25
# --------------------------------------------------------------------------------------

#: Strings chosen to exercise every alternative in the pattern: an apostrophe form of both
#: kinds, a clause reference, a provenance header, an underscore, an alphanumeric token.
TOKENISER_SAMPLES = [
    "clause 2/4/2 applies",
    "the institution's books",
    "the institution’s books",
    "XX17 and XX8 §2/4/4 occ=0 page=10",
    "a_b c-d 5 10.10 [1]",
    "widget 9/2 gadget",
    "What does the Standard No. 26 require?",
    "",
    "x9 y7 co-op",
]


@pytest.mark.parametrize("sample", TOKENISER_SAMPLES)
def test_the_anchor_tokeniser_agrees_with_bm25_token_for_token(sample, repo_root) -> None:
    """Behavioural sync, which is the claim the duplication actually needs.

    The two pattern *strings* are not equal - ``bm25_baseline`` writes the apostrophe
    class as ``[’']`` and anchoring as ``['’']``, and only the former passes
    ``re.UNICODE`` - so a string comparison would fail while the tokenisers agreed. What
    matters is that a term is an anchor candidate exactly when BM25 would have scored it,
    and that is asserted directly.
    """
    bm25 = re.compile(_bm25_token_pattern(repo_root), re.UNICODE)
    assert ANCHOR_TOKEN_RE.findall(sample) == bm25.findall(sample)


def test_the_two_patterns_differ_only_in_ways_that_cannot_change_a_match(repo_root) -> None:
    """Named so a future reader does not "fix" the difference and think it was a bug."""
    bm25 = _bm25_token_pattern(repo_root)
    assert bm25 != ANCHOR_TOKEN_RE.pattern
    assert sorted(bm25) == sorted(ANCHOR_TOKEN_RE.pattern), "same characters, reordered"
    assert bm25.replace("’'", "'’") == ANCHOR_TOKEN_RE.pattern, "only the class order"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("clause 2/4/2 applies", ["clause", "2", "4", "2", "applies"]),
        ("XX17 clause 9/2", ["xx17", "clause", "9", "2"]),
        ("section 2/4/2", ["section", "2", "4", "2"]),
    ],
)
def test_no_token_from_the_public_path_ever_contains_a_slash(text, expected) -> None:
    """The module comment on ``ANCHOR_TOKEN_RE`` is wrong, and this is the counter-example.

    It states that ``9/2`` stays one token "which matters because a clause reference is
    usually the most diagnostic thing in a question". The alternation is
    ``[\\w]+(?:['’][\\w]+)?|\\d+(?:/\\d+)*`` and Python's ``|`` is ordered, so at the
    ``2`` of ``2/4/2`` the *first* branch matches and consumes just ``2``. The second
    branch is only ever tried where the first cannot match, and any position where
    ``\\d+`` matches is a position where ``[\\w]+`` matches too - so it is unreachable.
    BM25 inherits this, so retrieval scores clause references as loose digits as well.

    Not corrected in code: ``ANCHOR_TOKEN_RE`` exists to mirror the tokeniser the shipped
    BM25 index was built with, and every published coverage number was measured under this
    behaviour. Changing it would break the sync the comment is about and silently
    invalidate ``reports/anchoring_gate_calibration.json``.
    """
    assert anchor_tokens(text) == expected
    assert not any("/" in token for token in anchor_tokens(text))


def test_the_slash_branch_is_reachable_only_by_calling_it_directly() -> None:
    """``keep_slash_numerals`` is live code on a path nothing currently walks.

    ``_is_candidate`` accepts a slash token, and *exempts it from the length floor* -
    ``9/2`` is three characters and would otherwise be dropped. But no token reaching it
    from :func:`anchor_tokens` can contain a slash, so the switch has no effect on any
    measured coverage. Asserted rather than deleted for two reasons: the helper is
    callable by anything, and a future tokeniser fix would make the branch matter again
    without touching it. Deleting it would make that fix silently change more than it
    looked like it changed.
    """
    default = AnchorSpec()
    assert _is_candidate("2/4/2", default) is True
    assert _is_candidate("9/2", default) is True, "slash tokens skip min_term_length"
    assert _is_candidate("9/2", AnchorSpec(keep_slash_numerals=False)) is False
    # ...and the path that would deliver one never does.
    for sample in TOKENISER_SAMPLES:
        assert not any("/" in token for token in anchor_tokens(sample))


def test_the_tokeniser_normalises_before_splitting() -> None:
    """So an anchor and a corpus term are compared in the same form, not two forms."""
    assert anchor_tokens("The Widget") == ["the", "widget"]
    assert anchor_tokens("the widget’s value") == anchor_tokens("the widget's value")
    assert anchor_tokens("") == []


# --------------------------------------------------------------------------------------
# fold_suffix
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        ("assets", "asset"),
        ("asset", "asset"),
        ("leasing", "leas"),
        ("organised", "organis"),
        ("discloses", "disclos"),
        ("ratios", "ratio"),
        # "ies" is the one suffix that puts a character back, so it is not a pure strip.
        ("policies", "policy"),
        # ...but only when four characters survive the strip. "cities" loses "ies" -> 3,
        # so it falls through to "es" and yields a stem that is not a word.
        ("cities", "citi"),
        # The min_stem guard, which is why several ordinary plurals do not fold at all.
        ("fees", "fees"),
        ("uses", "uses"),
        ("goes", "goes"),
        ("does", "does"),
        ("bus", "bus"),
        ("is", "is"),
    ],
)
def test_the_fold_table_is_crude_and_exact(term, expected) -> None:
    """The module docstring's third example is wrong; the code is right.

    It reads ``policies -> polic``, and ``reports/anchoring_gate_calibration.md`` §6
    repeats it. The implementation appends ``y`` when the matched suffix is ``ies``, so the
    real output is ``policy``. Both texts are corrected; the behaviour is untouched,
    because ``policy`` is what every published number was measured with.
    """
    assert fold_suffix(term) == expected


def test_the_stem_floor_is_what_stops_short_plurals_folding() -> None:
    """Asserted on both sides of the boundary, and shown to be a parameter.

    ``fees`` does not fold because ``fee`` is three characters. That is a real
    consequence: ``fee`` and ``fees`` are two vocabulary entries and a question using one
    is not anchored by a clause using the other. Lowering the floor fixes that pair and
    starts folding things that should not fold, which is why it was left where it is.
    """
    assert fold_suffix("fees") == "fees"
    assert fold_suffix("fees", min_stem=3) == "fee"
    assert fold_suffix("costs") == "cost", "four characters survive, so this one folds"


def test_folding_is_applied_to_both_sides_of_every_comparison() -> None:
    """The property that makes an ugly stem harmless.

    ``leas`` is not a word. It does not need to be: the query term and the corpus term go
    through the same function, so they meet at the same stem.
    """
    vocabulary = CorpusVocabulary.from_chunks([{"text": "the leasing arrangement"}] * 3)
    assert vocabulary.document_frequency("leases") == 3
    assert vocabulary.document_frequency("leasing") == 3
    assert fold_suffix("leases") == fold_suffix("leasing") == "leas"


def test_folding_off_keeps_the_surface_forms_apart() -> None:
    unfolded = AnchorSpec(fold_suffixes=False)
    vocabulary = CorpusVocabulary.from_chunks([{"text": "the leasing arrangement"}] * 3, unfolded)
    assert vocabulary.document_frequency("leasing") == 3
    assert vocabulary.document_frequency("leases") == 0


# --------------------------------------------------------------------------------------
# Shape rules, isolated from the corpus
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "expected", "reason"),
    [
        ("widget", True, "ordinary six-letter term"),
        ("cost", True, "exactly at min_term_length"),
        ("fee", False, "one below min_term_length"),
        ("xx17", True, "a standard identifier is alphanumeric, not a bare number"),
        ("x9", False, "two characters"),
        ("5", False, "bare digit"),
        ("10", False, "bare digit"),
        ("the", False, "stopword"),
        ("shall", False, "stopword, and long enough to pass the length floor"),
        ("cannot", False, "stopword"),
        ("aaoifi", False, "frame term: fixed prompt boilerplate"),
        ("summarise", False, "frame term: a directive about how to answer"),
        ("a_b", False, "underscore is a word character, but the token is three long"),
    ],
)
def test_the_shape_rules_are_independent_of_the_corpus(token, expected, reason) -> None:
    """Run against a vocabulary where every term clears the IDF floor.

    ``FLAT_VOCABULARY`` gives ``log(100) = 4.61`` to everything, so any rejection here is
    a shape rejection. Against the real corpus a failure could come from either filter and
    the assertion would not say which.
    """
    assert _is_candidate(token, AnchorSpec()) is expected, reason
    anchors, _ = select_anchors(token, FLAT_VOCABULARY)
    assert bool(anchors) is expected, f"{reason} (via select_anchors)"


def test_frame_terms_are_dropped_only_when_the_switch_is_on() -> None:
    """The switch exists so ``reports/anchoring_gate_calibration.md`` §5 can show its work.

    Leaving frame terms in produces the report's most flattering number - OOV ratio at AUC
    0.792 against the valid probes - and that number is an artefact of the probe author
    writing six questions from one template. Keeping the variant runnable is what makes the
    rejection auditable instead of a claim.
    """
    keep = AnchorSpec(drop_frame_terms=False)
    assert _is_candidate("aaoifi", keep) is True
    assert _is_candidate("summarise", keep) is True
    assert _is_candidate("the", keep) is False, "stopwords are not switchable"


def test_the_stop_list_carries_no_domain_judgement() -> None:
    """Nothing in it is an Islamic-finance term, by design.

    Domain vocabulary is meant to be removed by the corpus IDF floor, which is a
    measurement, not by a hand-written list, which would be an opinion about which
    concepts matter. Spot-checked here; the floor's half of the job is asserted against
    the real corpus further down.
    """
    domain = {
        "murabahah", "ijarah", "mudarabah", "sukuk", "takaful", "shari'ah", "riba",
        "wakalah", "profit", "lessee", "lessor", "asset", "institution", "contract",
    }
    assert domain.isdisjoint(STOPWORDS)
    assert domain.isdisjoint(FRAME_TERMS)
    assert STOPWORDS.isdisjoint(FRAME_TERMS), "no term is filtered by two rules"


def test_the_length_floor_and_the_stop_list_are_both_parameters_of_the_spec() -> None:
    """A recorded spec is what makes a stored coverage number re-derivable."""
    assert _is_candidate("fee", AnchorSpec(min_term_length=3)) is True
    assert _is_candidate("widget", AnchorSpec(min_term_length=99)) is False
    assert AnchorSpec().as_dict()["spec_version"] == "anchor_spec_v1"
    assert AnchorSpec().as_dict()["context_metadata_keys"] == [
        "standard_id",
        "clause_id",
        "sub_clause_id",
        "chunk_id",
    ]


# --------------------------------------------------------------------------------------
# The vocabulary and its IDF
# --------------------------------------------------------------------------------------


def test_idf_is_log_n_over_one_plus_df() -> None:
    """Exact formula, asserted so a stored ``max_uncovered_idf`` stays interpretable.

    The ``1 +`` is what keeps an unseen term finite, so out-of-vocabulary can be compared
    against ``min_idf`` rather than special-cased with an infinity.
    """
    vocabulary = CorpusVocabulary(df={"widget": 9, "gadget": 1}, n_documents=100, spec=AnchorSpec())
    assert vocabulary.idf("widget") == pytest.approx(math.log(100 / 10))
    assert vocabulary.idf("gadget") == pytest.approx(math.log(100 / 2))
    assert vocabulary.idf("absent") == pytest.approx(math.log(100 / 1))


def test_an_unseen_term_takes_the_maximum_idf_in_the_vocabulary() -> None:
    """Which is the property that makes IDF weighting a corpus-absence signal at all."""
    vocabulary = CorpusVocabulary(
        df={"a": 1, "b": 5, "c": 50}, n_documents=362, spec=AnchorSpec()
    )
    assert vocabulary.idf("unseen") == pytest.approx(math.log(362))
    assert vocabulary.idf("unseen") > max(vocabulary.idf(term) for term in "abc")


@pytest.mark.parametrize("df", [0, 1, 2, 10, 100])
def test_idf_is_monotone_decreasing_in_document_frequency(df) -> None:
    vocabulary = CorpusVocabulary(df={"t": df}, n_documents=1000, spec=AnchorSpec())
    rarer = CorpusVocabulary(df={"t": df // 2}, n_documents=1000, spec=AnchorSpec())
    assert rarer.idf("t") >= vocabulary.idf("t")


def test_a_term_in_every_document_gets_a_negative_idf_and_can_never_anchor() -> None:
    """``log(N/(N+1)) < 0``, so the floor rejects it without a special case.

    Worth pinning because a negative weight would be a bug if such a term could reach
    :attr:`AnchoringSignal.anchor_idf`: it would *raise* coverage by being uncovered. The
    IDF floor at 2.0 makes that unreachable, and this is the assertion that says so.
    """
    vocabulary = CorpusVocabulary.from_chunks([{"text": "widget"}, {"text": "widget again"}])
    assert vocabulary.idf("widget") == pytest.approx(math.log(2 / 3))
    assert vocabulary.idf("widget") < 0
    anchors, _ = select_anchors("widget", vocabulary)
    assert anchors == []


def test_document_frequency_counts_documents_not_occurrences() -> None:
    """Matching ``reporting.probes.term_occurrences``, so the two count one universe.

    If a probe's ``basis_evidence`` counted occurrences and the vocabulary counted
    documents, a ``term_absent_from_corpus`` label and an ``oov_anchors`` entry would be
    claims about different things while looking like the same claim.
    """
    vocabulary = CorpusVocabulary.from_chunks(
        [{"text": "widget widget widget"}, {"text": "widget"}, {"text": "gadget"}]
    )
    assert vocabulary.n_documents == 3
    assert vocabulary.document_frequency("widget") == 2
    assert vocabulary.document_frequency("gadget") == 1
    assert vocabulary.document_frequency("nothing") == 0
    assert vocabulary.contains("widget") is True
    assert vocabulary.contains("nothing") is False


def test_an_empty_document_still_counts_toward_n() -> None:
    """Otherwise a corpus with blank chunks would inflate every IDF in it."""
    vocabulary = CorpusVocabulary.from_chunks(
        [{"text": ""}, {"nottext": "ignored"}, {"text": None}, {"text": "widget"}]
    )
    assert (vocabulary.n_documents, vocabulary.n_terms) == (4, 1)


@pytest.mark.parametrize(
    "build",
    [
        lambda: CorpusVocabulary.from_chunks([]),
        lambda: CorpusVocabulary(df={}, n_documents=0, spec=AnchorSpec()),
        lambda: CorpusVocabulary(df={}, n_documents=-1, spec=AnchorSpec()),
    ],
    ids=["no_chunks", "zero_documents", "negative_documents"],
)
def test_a_vocabulary_with_no_documents_refuses_to_exist(build) -> None:
    """Fail closed. ``log(0/1)`` is a domain error and every coverage would be garbage.

    Same reasoning as ``metrics`` refusing a Wilson interval below n=30 and
    ``contains_normalised`` refusing an empty needle: a degenerate input must raise, not
    return a number that reads as a measurement.
    """
    with pytest.raises(ValueError, match="at least one document"):
        build()


def test_the_spec_travels_with_the_vocabulary_and_is_the_default_for_selection() -> None:
    """So a coverage number cannot be computed under a spec the vocabulary was not built for.

    Not a guarantee - ``select_anchors`` still accepts an override, which the calibration
    script does not use but a caller could. The default is the safe one.
    """
    unfolded = AnchorSpec(fold_suffixes=False)
    vocabulary = CorpusVocabulary.from_chunks([{"text": "widgets"}] * 5, unfolded)
    assert vocabulary.spec is unfolded
    assert vocabulary.document_frequency("widgets") == 5
    assert vocabulary.document_frequency("widget") == 0, "no folding on either side"


# --------------------------------------------------------------------------------------
# Anchor selection
# --------------------------------------------------------------------------------------


def test_anchors_are_deduplicated_in_first_seen_order() -> None:
    """A repeated term is one anchor, so a wordy question cannot dilute its own score.

    Order is first-seen rather than sorted because ``uncovered_anchors`` is read alongside
    the question when diagnosing a low score, and a stable order makes two runs of the same
    question comparable line by line.
    """
    anchors, n_tokens = select_anchors("widget threshold widget lessee threshold", FLAT_VOCABULARY)
    assert anchors == ["widget", "threshold", "lessee"]
    assert n_tokens == 5, "the raw token count is not deduplicated"


def test_the_query_token_count_is_raw_including_everything_dropped() -> None:
    """It is a size record, not an anchor count, and the two are reported separately.

    Keeping it means a coverage of 1.000 over two anchors can be told apart from a coverage
    of 1.000 over eighteen without re-reading the question - which matters, because the
    first is far weaker evidence than the second.
    """
    query = "Please summarise what the widget threshold is for a lessee under XX17 clause 2/4/2."
    anchors, n_tokens = select_anchors(query, FLAT_VOCABULARY)
    assert n_tokens == len(anchor_tokens(query)) == 16
    assert anchors == ["widget", "threshold", "lessee", "xx17", "clause"]
    assert "2/4/2" not in anchors, "split into digits, then dropped as bare numbers"
    assert "please" not in anchors and "summarise" not in anchors, "frame terms"


def test_the_idf_floor_and_the_shape_rules_compose() -> None:
    """A term must clear both. Asserted by moving one filter at a time.

    ``widget`` is shape-valid and, in a corpus where it is common, still not an anchor.
    That is the behaviour the module docstring describes as removing domain-generic
    vocabulary "by the corpus IDF floor instead" of by a stop list.
    """
    common = CorpusVocabulary.from_chunks(
        [{"text": "the widget"}] * 90 + [{"text": "the gadget"}] * 10
    )
    assert common.idf("widget") < AnchorSpec().min_idf <= common.idf("gadget")
    assert select_anchors("widget gadget", common)[0] == ["gadget"]
    # Same query, same shapes, a spec that accepts a lower floor: both come back.
    lower = AnchorSpec(min_idf=0.0)
    assert select_anchors("widget gadget", common, lower)[0] == ["widget", "gadget"]


# --------------------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("query", ["What is it?", "", "the a an of 5 10", "aaoifi summarise"])
def test_a_query_with_no_anchors_is_undefined_rather_than_uncovered(query) -> None:
    """Every derived quantity is ``None``, and ``is_evaluable`` says why.

    This is limit 3 in the module docstring, and it is a routing decision as much as an
    arithmetic one: absence of evidence is not evidence of absence, so the gate must
    abstain from judging rather than fire. A ``0.0`` here would fire every threshold in the
    sweep and turn an unmeasurable question into a confident abstention.
    """
    signal = compute_anchoring_signal(query, CONTEXT, FLAT_VOCABULARY)
    assert signal.anchors == ()
    assert signal.n_anchors == 0
    assert signal.is_evaluable is False
    assert signal.context_anchor_coverage is None
    assert signal.idf_weighted_coverage is None
    assert signal.oov_ratio is None
    assert signal.max_uncovered_idf is None


def test_idf_weighting_is_what_stops_a_covered_question_frame_masking_an_absent_topic() -> None:
    """The docstring's worked example, measured: 0.750 unweighted against 0.499 weighted.

    Four anchors, one uncovered - but the uncovered one is the *subject*, out of vocabulary
    at ``log(1000) = 6.91`` against ``log(1000/101) = 2.29`` for the three frame words the
    context does carry. Unweighted coverage reports a comfortable three-quarters. Weighted
    coverage reports half, because the missing term is worth three of the present ones.
    Both are recorded, and the gate reads the weighted one.
    """
    vocabulary = _weighted_vocabulary()
    signal = compute_anchoring_signal("zakat record disclose amount", CONTEXT, vocabulary)
    assert signal.anchors == ("zakat", "record", "disclose", "amount")
    assert signal.uncovered_anchors == ("zakat",)
    assert signal.oov_anchors == ("zakat",)
    assert signal.context_anchor_coverage == pytest.approx(0.75)
    assert signal.idf_weighted_coverage == pytest.approx(0.4989, abs=5e-5)
    # The weights are carried on the signal, so the ratio is re-derivable from it alone.
    weights = dict(signal.anchor_idf)
    assert signal.idf_weighted_coverage == pytest.approx(
        1 - weights["zakat"] / sum(weights.values())
    )


def test_max_uncovered_idf_names_the_single_most_diagnostic_gap() -> None:
    """A diagnostic for reading a low score: one absent topic term, or many absent verbs.

    The two look identical in a coverage ratio and need different responses, so the worst
    single gap is recorded next to the ratio.
    """
    vocabulary = _weighted_vocabulary()
    signal = compute_anchoring_signal("zakat record disclose amount", CONTEXT, vocabulary)
    assert signal.max_uncovered_idf == pytest.approx(math.log(1000))
    covered = compute_anchoring_signal("record disclose amount", CONTEXT, vocabulary)
    assert covered.uncovered_anchors == ()
    assert covered.idf_weighted_coverage == pytest.approx(1.0)
    assert covered.max_uncovered_idf is None, "nothing uncovered, so no worst gap"


def test_coverage_is_a_property_of_the_context_not_of_the_corpus() -> None:
    """An in-corpus term the retriever did not surface is uncovered but not OOV.

    Separating the two is what makes ``oov_ratio`` a diagnostic and coverage the signal:
    a high OOV ratio says the question is about something the corpus lacks, a high
    uncovered-but-in-vocabulary count says the retriever missed something it had.
    """
    vocabulary = _weighted_vocabulary()
    signal = compute_anchoring_signal("record disclose amount", [{"text": "record"}], vocabulary)
    assert signal.uncovered_anchors == ("disclose", "amount")
    assert signal.oov_anchors == (), "all three are in the corpus"
    assert signal.oov_ratio == 0.0
    assert signal.context_anchor_coverage == pytest.approx(1 / 3)


def test_an_empty_context_leaves_every_anchor_uncovered() -> None:
    """The zero-retrieval case, and the one place coverage is legitimately 0.0."""
    vocabulary = _weighted_vocabulary()
    signal = compute_anchoring_signal("zakat record disclose", [], vocabulary)
    assert signal.uncovered_anchors == signal.anchors
    assert signal.context_anchor_coverage == 0.0
    assert signal.idf_weighted_coverage == 0.0
    assert signal.is_evaluable is True, "measurable, and the measurement is zero"


# --------------------------------------------------------------------------------------
# The context surface, and the measurement error that produced it
# --------------------------------------------------------------------------------------


def test_the_context_surface_includes_the_provenance_ids_the_prompt_renders() -> None:
    """The fix for the first calibration run's measurement error.

    ``build_context_text`` writes a header above every excerpt - ``XX §1/1 occ=0 page=7
    id=clause:XX:1/1:occurrence:1`` - so the identifiers *are* on the surface the model was
    shown. Measuring coverage against ``text`` alone counted them uncovered for every item
    in both classes: a uniform penalty, and simply the wrong measurement.

    Note what the identifiers actually contribute once tokenised: the standard id, and the
    literal words ``clause`` and ``occurrence`` out of the chunk id. The clause number is
    *not* among them - ``1/1`` splits into digits, which never survive as anchors. So the
    fix restored standard-number anchoring, not clause-number anchoring.
    """
    spec = AnchorSpec()
    with_metadata = context_surface_terms(CONTEXT, spec)
    text_only = context_surface_terms(CONTEXT, AnchorSpec(context_metadata_keys=()))
    assert sorted(with_metadata - text_only) == ["1", "clause", "occurrence", "xx"]
    assert "1/1" not in with_metadata


def test_a_question_naming_a_standard_is_anchored_by_the_header_not_the_clause_text() -> None:
    """Which is exactly the case the metadata keys were added for."""
    vocabulary = CorpusVocabulary.from_chunks([{"text": "filler prose about widgets"}] * 50)
    query = "What does XX17 state?"  # "state" is a frame term, so xx17 is the only anchor
    anchored = compute_anchoring_signal(
        query, [{"text": "unrelated clause prose", "standard_id": "XX17"}], vocabulary
    )
    assert anchored.anchors == ("xx17",)
    assert anchored.uncovered_anchors == ()
    assert anchored.context_anchor_coverage == 1.0
    text_only = compute_anchoring_signal(
        query,
        [{"text": "unrelated clause prose", "standard_id": "XX17"}],
        vocabulary,
        AnchorSpec(context_metadata_keys=()),
    )
    assert text_only.uncovered_anchors == ("xx17",)
    assert text_only.context_anchor_coverage == 0.0


def test_a_missing_or_null_metadata_field_is_skipped_not_stringified() -> None:
    """``sub_clause_id`` is ``None`` on most records, and ``"none"`` must never be a term."""
    surface = context_surface_terms(CONTEXT, AnchorSpec())
    assert "none" not in surface
    assert context_surface_terms([{"text": "widget"}], AnchorSpec()) == {"widget"}


def test_the_surface_is_folded_the_same_way_the_query_is() -> None:
    """Otherwise ``widgets`` in a question would not match ``widget`` in a clause."""
    assert context_surface_terms([{"text": "The Widgets"}], AnchorSpec()) == {"the", "widget"}
    unfolded = AnchorSpec(fold_suffixes=False)
    assert context_surface_terms([{"text": "The Widgets"}], unfolded) == {"the", "widgets"}


# --------------------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------------------

#: Every key ``as_dict`` emits without ``include_terms``. Pinned as a set so an added field
#: that carries question text has to be added here too, in a test whose name says why.
PUBLIC_KEYS = {
    "n_query_tokens",
    "n_anchors",
    "n_oov_anchors",
    "n_uncovered_anchors",
    "context_anchor_coverage",
    "idf_weighted_coverage",
    "max_uncovered_idf",
    "oov_ratio",
    "is_evaluable",
    "corpus_documents",
    "spec",
    "measurement_is_lexical_only",
    "oov_ratio_status",
}


def test_the_serialised_signal_withholds_the_anchor_terms() -> None:
    """Anchors are question-derived, and ``data/private/hard_set.jsonl`` is Git-ignored.

    Counts and ratios are publishable; the terms behind them are not, which is why
    ``reports/anchoring_gate_calibration.json`` carries
    ``"terms_withheld": "hard-set question text is Git-ignored"`` for the seven answerable
    items and full anchor lists for the 25 probes, whose questions are tracked.
    """
    signal = compute_anchoring_signal("zakat record disclose", CONTEXT, _weighted_vocabulary())
    payload = signal.as_dict()
    assert set(payload) == PUBLIC_KEYS
    assert "zakat" not in json.dumps(payload)
    assert payload["n_anchors"] == 3
    assert payload["n_uncovered_anchors"] == 1


def test_the_terms_are_available_when_they_are_explicitly_asked_for() -> None:
    """Opt-in, because the calibration script needs them for the probe half of the table."""
    signal = compute_anchoring_signal("zakat record disclose", CONTEXT, _weighted_vocabulary())
    payload = signal.as_dict(include_terms=True)
    assert set(payload) - PUBLIC_KEYS == {"anchors", "oov_anchors", "uncovered_anchors"}
    assert payload["anchors"] == ["zakat", "record", "disclose"]
    assert payload["uncovered_anchors"] == ["zakat"]


def test_the_serialised_signal_carries_its_own_two_caveats() -> None:
    """A number without them invites the two misreadings the module was built to avoid.

    ``measurement_is_lexical_only`` blocks reading coverage as topical presence.
    ``oov_ratio_status`` blocks quoting the OOV rate on the ``term_absent_from_corpus``
    probes as a true-positive rate, which is circular by construction. Both travel in the
    payload rather than living only in the report, because the payload is what a future
    reader will have.
    """
    payload = compute_anchoring_signal("widget", CONTEXT, FLAT_VOCABULARY).as_dict()
    assert payload["measurement_is_lexical_only"] is True
    assert "circular" in payload["oov_ratio_status"]
    assert "0 corpus occurrences" in payload["oov_ratio_status"]
    assert payload["spec"] == AnchorSpec().as_dict()
    assert payload["corpus_documents"] == 100


def test_the_signal_is_frozen() -> None:
    signal = compute_anchoring_signal("widget", CONTEXT, FLAT_VOCABULARY)
    with pytest.raises(Exception):
        signal.anchors = ()  # type: ignore[misc]


def test_a_signal_can_be_reconstructed_from_its_recorded_weights() -> None:
    """``anchor_idf`` is stored so a coverage number stays checkable without the vocabulary.

    A year from now the corpus may have been rebuilt and the IDFs will have moved. The
    weights that produced a stored number have to travel with it or the number is not
    auditable.
    """
    stored = AnchoringSignal(
        n_query_tokens=6,
        anchors=("alpha", "beta"),
        oov_anchors=("alpha",),
        uncovered_anchors=("alpha",),
        spec=AnchorSpec(),
        corpus_documents=362,
        anchor_idf=(("alpha", 6.0), ("beta", 2.0)),
    )
    assert stored.context_anchor_coverage == 0.5
    assert stored.idf_weighted_coverage == pytest.approx(2.0 / 8.0)
    assert stored.max_uncovered_idf == 6.0
    assert stored.oov_ratio == 0.5


def test_weights_are_needed_for_the_weighted_ratio_and_absent_means_none() -> None:
    """Rather than silently falling back to unweighted, which would misreport the gate."""
    without = AnchoringSignal(
        n_query_tokens=2,
        anchors=("alpha",),
        oov_anchors=(),
        uncovered_anchors=(),
        spec=AnchorSpec(),
        corpus_documents=362,
    )
    assert without.context_anchor_coverage == 1.0
    assert without.idf_weighted_coverage is None
    assert without.max_uncovered_idf is None


# --------------------------------------------------------------------------------------
# The published report's arithmetic, checked without the corpus
# --------------------------------------------------------------------------------------
#
# `reports/anchoring_gate_calibration.json` is tracked and carries a per-item coverage for
# all 32 measured queries. Every aggregate in the Markdown report is a function of those
# numbers, so the report's tables can be re-derived on a clone with no private data at all.
# That is a narrower claim than the corpus-gated section below - it checks the arithmetic,
# not the measurement - and it is the half that keeps working when the corpus is absent.


@pytest.fixture(scope="session")
def calibration(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "reports" / "anchoring_gate_calibration.json"
    assert path.exists(), "reports/anchoring_gate_calibration.json is tracked"
    return json.loads(path.read_text(encoding="utf-8"))


def _grouped(variant: dict[str, Any], key: str, bases: Any) -> list[float]:
    """The chosen score for every row whose group is in ``bases``, ``None`` dropped."""
    names = {bases} if isinstance(bases, str) else set(bases)
    return [
        row[key]
        for row in variant["bm25"]
        if row["group"] in names and row[key] is not None
    ]


def _fired(scores: list[float], threshold: float) -> tuple[int, int]:
    """``(n_fired, n_evaluable)``. The gate fires *below* the threshold, strictly."""
    return sum(1 for score in scores if score < threshold), len(scores)


def _auc(answerable: list[float], probes: list[float]) -> float:
    """Mann-Whitney ``P(probe < answerable)``, ties counting a half.

    Reimplemented here rather than imported from ``scripts/calibrate_anchoring_gate.py``:
    that script imports the BM25 retriever at module scope, and ``conftest`` rule 1 forbids
    a test depending on ``rank_bm25``. An independent implementation is also the stronger
    check - a shared bug would not cancel out.
    """
    wins = sum(
        1.0 if probe < item else 0.5 if probe == item else 0.0
        for item in answerable
        for probe in probes
    )
    return wins / (len(answerable) * len(probes))


#: ``reports/anchoring_gate_calibration.md`` §3, verbatim: threshold -> (answerable n=7,
#: valid probes n=11, circular term probes n=8, cross-standard n=6), each as n_fired.
PUBLISHED_SWEEP = {
    0.10: (0, 0, 0, 0),
    0.20: (0, 0, 1, 0),
    0.25: (0, 0, 2, 0),
    0.30: (1, 0, 3, 1),
    0.34: (1, 1, 3, 2),
    0.40: (1, 1, 5, 4),
    0.50: (2, 4, 6, 4),
    0.60: (2, 4, 7, 5),
    0.67: (3, 4, 8, 5),
    0.75: (3, 7, 8, 5),
}


def test_the_published_threshold_sweep_is_reproducible(calibration) -> None:
    """§3 of the report, re-derived from the per-item coverages it was computed from."""
    variant = calibration["variants"][0]
    assert variant["spec"] == AnchorSpec().as_dict(), "row 1 is the default spec"
    valid, circular = calibration["non_circular_bases"], calibration["circular_bases"]
    assert list(PUBLISHED_SWEEP) == calibration["thresholds"]
    for threshold, expected in PUBLISHED_SWEEP.items():
        observed = tuple(
            _fired(_grouped(variant, "coverage", bases), threshold)[0]
            for bases in ("answerable_n7", valid, circular, "cross_standard_comparison")
        )
        assert observed == expected, f"threshold {threshold}"


def test_the_group_sizes_the_report_quotes_are_the_group_sizes_in_the_data(calibration) -> None:
    """7 / 11 / 8 / 6, and the 11 are the two non-circular bases combined.

    The split is the whole basis for the report's §2 warning: an 82% detection rate over the
    8 term probes is a tautology about how those probes were selected, and only the 11 are
    evidence. If the group names drifted, every rate in the report would silently change
    meaning.
    """
    variant = calibration["variants"][0]
    assert set(calibration["non_circular_bases"]) == {
        "standard_absent_from_corpus",
        "clause_absent_from_standard",
    }
    assert set(calibration["circular_bases"]) == {"term_absent_from_corpus"}
    sizes = {
        name: len(_grouped(variant, "coverage", name))
        for name in {row["group"] for row in variant["bm25"]}
    }
    assert sizes == {
        "answerable_n7": 7,
        "standard_absent_from_corpus": 6,
        "clause_absent_from_standard": 5,
        "term_absent_from_corpus": 8,
        "cross_standard_comparison": 6,
    }
    assert sizes["standard_absent_from_corpus"] + sizes["clause_absent_from_standard"] == 11


def test_at_the_zero_false_positive_threshold_the_gate_detects_nothing(calibration) -> None:
    """The bolded row, isolated, because it is the finding.

    0.25 is the largest threshold in the sweep with no false positive on the seven
    answerable items, and at 0.25 the true-positive rate on the 11 valid probes is also
    zero: the gate is inert exactly where it is safe. Every threshold that detects anything
    has already abstained on an answerable item. That is not a tuning problem, and this
    assertion is what stops it being quietly re-tuned.
    """
    variant = calibration["variants"][0]
    answerable = _grouped(variant, "coverage", "answerable_n7")
    valid = _grouped(variant, "coverage", calibration["non_circular_bases"])
    assert _fired(answerable, 0.25) == (0, 7)
    assert _fired(valid, 0.25) == (0, 11)
    # The next threshold up costs an answerable item and still detects nothing.
    assert _fired(answerable, 0.30) == (1, 7)
    assert _fired(valid, 0.30) == (0, 11)
    # And the first threshold that detects anything has already paid for it.
    first_detecting = min(t for t in calibration["thresholds"] if _fired(valid, t)[0] > 0)
    assert first_detecting == 0.34
    assert _fired(answerable, first_detecting)[0] >= 1


#: ``reports/anchoring_gate_calibration.md`` §4, verbatim. Keyed by
#: ``(variant index, score field)``; values are ``(vs valid n=11, vs cross-standard n=6,
#: vs circular term probes)``. The third column is printed in the report in italics and is
#: not evidence; it is asserted so it cannot be quietly promoted into the first.
PUBLISHED_AUC = {
    (0, "coverage"): (0.513, 0.738, 0.821),
    (0, "unweighted_coverage"): (0.513, 0.702, 0.812),
    (0, "-oov_ratio"): (0.643, 0.643, 0.920),
    (1, "coverage"): (0.597, 0.738, 0.821),
    (1, "-oov_ratio"): (0.792, 0.607, 0.830),
    (2, "coverage"): (0.513, 0.810, 0.804),
    (2, "-oov_ratio"): (0.610, 0.619, 0.768),
}
#: The ``(2, "coverage")`` circular cell was published as 0.762 and is 0.804. 0.762 is this
#: variant's *unweighted*-vs-cross-standard AUC, a cell the report does not print, so it was
#: transcribed from the wrong computed line. Corrected in
#: ``reports/anchoring_gate_calibration.md`` §4 with a dated note; it sits in the column the
#: report itself marks "not evidence", so no conclusion moves. Every other cell in the table
#: reproduces exactly, which is how the outlier was found.


@pytest.mark.parametrize(("key", "expected"), sorted(PUBLISHED_AUC.items()))
def test_every_published_auc_is_reproducible(key, expected, calibration) -> None:
    """§4, all seven rows. Two of them are the ones that matter.

    ``(0, "coverage")`` at 0.513 is the number that disabled the gate: the signal the gate
    actually reads, against the only non-circular negative class, at chance. ``(1,
    "-oov_ratio")`` at 0.792 is the number the report explicitly rejects as an artefact of
    one author writing six probe questions from one template - retained in the variant list
    so the rejection stays auditable rather than becoming a claim.

    The OOV rows are negated because a probe is expected to have a *higher* OOV ratio than
    an answerable item, so the comparison has to run in the other direction to stay
    "probe scores worse".
    """
    variant_index, field = key
    variant = calibration["variants"][variant_index]
    negate = field.startswith("-")
    name = field.lstrip("-")
    sign = -1.0 if negate else 1.0

    def scores(bases: Any) -> list[float]:
        return [sign * value for value in _grouped(variant, name, bases)]

    observed = (
        _auc(scores("answerable_n7"), scores(calibration["non_circular_bases"])),
        _auc(scores("answerable_n7"), scores("cross_standard_comparison")),
        _auc(scores("answerable_n7"), scores(calibration["circular_bases"])),
    )
    assert observed == pytest.approx(expected, abs=5e-4)


def test_the_variant_list_is_the_three_specs_the_report_tabulates(calibration) -> None:
    """Order matters: the tests above index into it, and so does the report's row order."""
    specs = [variant["spec"] for variant in calibration["variants"]]
    assert specs == [
        AnchorSpec().as_dict(),
        AnchorSpec(drop_frame_terms=False).as_dict(),
        AnchorSpec(fold_suffixes=False).as_dict(),
    ]
    assert specs[0]["min_idf"] == 2.0 and specs[0]["min_term_length"] == 4
    # Folding is the only spec change that alters the vocabulary, so it is the only one
    # whose term count moves.
    counts = [variant["vocabulary"]["n_terms"] for variant in calibration["variants"]]
    assert counts == [1491, 1491, 1782]
    assert all(v["vocabulary"]["n_documents"] == 362 for v in calibration["variants"])


def test_the_signal_is_at_chance_and_is_not_rescued_by_a_better_retriever(calibration) -> None:
    """§8: the false-positive curve is the same shape on the context Jais-2 actually saw.

    The probes can only be retrieved locally by BM25, so a reader might reasonably suspect
    the whole finding is an artefact of retrieving the negative class with a weaker
    retriever. It is not. Under the stored hybrid+rerank top-5 the zero-false-positive
    threshold is still 0.25 and the curve above it still climbs, because the uncovered
    anchors on the worst items are words that appear nowhere in the corpus - no ranking over
    that corpus can cover them.
    """
    variant = calibration["variants"][0]
    stored = [row["coverage"] for row in variant["stored_hybrid_rerank"]]
    assert len(stored) == 7
    assert {threshold: _fired(stored, threshold)[0] for threshold in (0.25, 0.30, 0.50, 0.67, 0.75)} == {
        0.25: 0,
        0.30: 1,
        0.50: 1,
        0.67: 3,
        0.75: 3,
    }
    bm25 = _grouped(variant, "coverage", "answerable_n7")
    assert _fired(bm25, 0.25)[0] == _fired(stored, 0.25)[0] == 0
    assert _fired(bm25, 0.67)[0] == _fired(stored, 0.67)[0] == 3


def test_the_answerable_rows_carry_no_question_vocabulary(calibration) -> None:
    """``reports/`` is tracked; the hard set is not. Enforced on the artefact, not the code.

    ``tests/test_probes.py`` makes the same check on the probe file from the other side -
    there the terms *are* published, because the probe questions are. The two together are
    the publication boundary for this report.
    """
    for variant in calibration["variants"]:
        for row in variant["bm25"] + variant["stored_hybrid_rerank"]:
            if row["group"] != "answerable_n7":
                continue
            assert row["terms_withheld"] == "hard-set question text is Git-ignored"
            assert not {"anchors", "uncovered", "oov"} & set(row)


# --------------------------------------------------------------------------------------
# The published measurement, recomputed from live code against the real corpus
# --------------------------------------------------------------------------------------
#
# The section above checks that the report's *tables* follow from the tracked per-item
# numbers. This one checks that the tracked per-item numbers follow from the *code*, by
# recomputing all 21 `stored_hybrid_rerank` rows (3 specs x 7 items) from
# `data/private/extracted/clause_chunks.jsonl`, `data/private/hard_set.jsonl` and the
# stored Colab run. Together the two sections close the loop: code -> JSON -> Markdown.
#
# Only the stored hybrid+rerank contexts are recomputed here. The BM25 rows would need
# `rank_bm25` and the persisted index, which conftest rule 1 forbids a test from
# requiring; the stored contexts are just chunk ids in a tracked-shaped run artefact
# rejoined to clause text, so they need no retriever at all.


@pytest.fixture(scope="session")
def stored_contexts(
    clause_chunks: list[dict[str, Any]], stored_run: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """The top-5 Jais-2 actually saw, rank-ordered and rejoined to clause text.

    Mirrors ``scripts/calibrate_anchoring_gate.stored_contexts``. Duplicated rather than
    imported for the same reason ``_auc`` is: that module imports BM25 at import time.
    """
    by_id = {record["chunk_id"]: record for record in clause_chunks}
    contexts: dict[str, list[dict[str, Any]]] = {}
    for item in stored_run["items"]:
        ordered = sorted(item["top5"], key=lambda entry: entry["rank"])
        contexts[str(item["item_id"])] = [by_id[entry["chunk_id"]] for entry in ordered]
    return contexts


#: The three specs ``scripts/calibrate_anchoring_gate.main`` measures, in the order the
#: JSON stores them. Positional: variant *i* of the JSON must be spec *i* here.
VARIANT_SPECS = (
    AnchorSpec(),
    AnchorSpec(drop_frame_terms=False),
    AnchorSpec(fold_suffixes=False),
)


def test_the_corpus_is_the_362_chunks_every_number_was_measured_against(
    clause_chunks: list[dict[str, Any]], calibration
) -> None:
    """A different corpus size silently rescales every IDF, so it is pinned first."""
    assert len(clause_chunks) == 362
    for variant in calibration["variants"]:
        assert variant["vocabulary"]["n_documents"] == 362


@pytest.mark.requires_private_data
@pytest.mark.parametrize("index", range(len(VARIANT_SPECS)))
def test_the_published_vocabulary_size_is_what_the_corpus_yields(
    index: int, clause_chunks: list[dict[str, Any]], calibration
) -> None:
    """1,491 folded terms at the default spec, 1,782 unfolded. Folding merges 291."""
    vocabulary = CorpusVocabulary.from_chunks(clause_chunks, VARIANT_SPECS[index])
    published = calibration["variants"][index]["vocabulary"]
    assert vocabulary.n_documents == published["n_documents"]
    assert vocabulary.n_terms == published["n_terms"]


@pytest.mark.requires_private_data
@pytest.mark.parametrize("index", range(len(VARIANT_SPECS)))
def test_every_published_stored_context_row_is_recomputed_field_for_field(
    index: int,
    clause_chunks: list[dict[str, Any]],
    hard_set: list[dict[str, Any]],
    stored_contexts: dict[str, list[dict[str, Any]]],
    calibration,
) -> None:
    """All 7 rows of one variant, recomputed from the corpus and compared exactly.

    Exact equality, not ``approx``: both sides are the same float arithmetic on the same
    inputs, so any difference at all means the code no longer produces the published
    measurement. The comparison covers every scalar the JSON carries, including the two
    counts (``n_uncovered``, ``n_oov``) that ``redact_for_publication`` substitutes for
    the withheld term lists - so a change in *which* anchors are covered fails here even
    when the ratio happens to survive.
    """
    spec = VARIANT_SPECS[index]
    variant = calibration["variants"][index]
    assert variant["spec"] == spec.as_dict(), "variant order must match VARIANT_SPECS"

    vocabulary = CorpusVocabulary.from_chunks(clause_chunks, spec)
    published = {row["id"]: row for row in variant["stored_hybrid_rerank"]}
    assert len(published) == 7

    for item in hard_set:
        item_id = str(item["item_id"])
        signal = compute_anchoring_signal(
            str(item["question_text"]), stored_contexts[item_id], vocabulary, spec
        )
        row = published[item_id]
        assert signal.idf_weighted_coverage == row["coverage"], item_id
        assert signal.context_anchor_coverage == row["unweighted_coverage"], item_id
        assert signal.oov_ratio == row["oov_ratio"], item_id
        assert signal.max_uncovered_idf == row["max_uncovered_idf"], item_id
        assert signal.n_anchors == row["n_anchors"], item_id
        assert len(signal.uncovered_anchors) == row["n_uncovered"], item_id
        assert len(signal.oov_anchors) == row["n_oov"], item_id


@pytest.mark.requires_private_data
def test_an_out_of_vocabulary_anchor_takes_the_log_of_the_corpus_size(
    clause_chunks: list[dict[str, Any]], calibration
) -> None:
    """``log(362) = 5.8916``, the ceiling ``max_uncovered_idf`` saturates at.

    Five of the seven items saturate it at the default spec: their single most
    diagnostic uncovered anchor appears nowhere in the corpus, which is the specific
    failure §7 of the report diagnoses as synonymy rather than retrieval.

    Under ``drop_frame_terms=False`` **all seven** saturate, and that is the mechanical
    footprint of the artefact §5 rejects: ``aaoifi`` is out-of-vocabulary by
    construction and uncovered on every single item, so with frame terms retained the
    most diagnostic uncovered anchor is the same constant for all seven and the
    statistic stops distinguishing anything.
    """
    vocabulary = CorpusVocabulary.from_chunks(clause_chunks, AnchorSpec())
    ceiling = vocabulary.idf("qqqzzz-not-a-word")
    assert ceiling == pytest.approx(math.log(362))
    assert ceiling == pytest.approx(5.8916442118, abs=1e-9)

    def saturated(index: int) -> list[str]:
        return sorted(
            row["id"]
            for row in calibration["variants"][index]["stored_hybrid_rerank"]
            if row["max_uncovered_idf"] == ceiling
        )

    assert saturated(0) == ["H01", "H02", "H03", "H05", "H07"]
    assert saturated(2) == saturated(0), "folding does not change which anchors are OOV"
    assert len(saturated(1)) == 7, "frame terms retained: the ceiling stops separating"


@pytest.mark.requires_private_data
def test_the_provenance_header_fix_moves_exactly_the_items_that_name_a_standard(
    clause_chunks: list[dict[str, Any]],
    hard_set: list[dict[str, Any]],
    stored_contexts: dict[str, list[dict[str, Any]]],
) -> None:
    """§13 of the report quotes these three deltas; this is where they come from.

    Dropping ``context_metadata_keys`` reverts the measurement error §7.4 describes.
    Three of seven items move and four do not, which is the evidence for the report's
    claim that the fix was load-bearing but *narrow*: it restores standard-id anchoring,
    not clause-number anchoring.
    """
    default = AnchorSpec()
    text_only = AnchorSpec(context_metadata_keys=())
    vocabulary = CorpusVocabulary.from_chunks(clause_chunks, default)

    moved: dict[str, tuple[float, float]] = {}
    for item in hard_set:
        context = stored_contexts[str(item["item_id"])]
        with_header = compute_anchoring_signal(
            str(item["question_text"]), context, vocabulary, default
        ).idf_weighted_coverage
        without = compute_anchoring_signal(
            str(item["question_text"]), context, vocabulary, text_only
        ).idf_weighted_coverage
        if with_header != without:
            moved[str(item["item_id"])] = (with_header, without)

    assert sorted(moved) == ["H03", "H04", "H05"]
    assert moved["H03"] == pytest.approx((0.800998, 0.601997), abs=5e-7)
    assert moved["H04"] == pytest.approx((1.0, 0.835682), abs=5e-7)
    assert moved["H05"] == pytest.approx((0.614935, 0.504650), abs=5e-7)


@pytest.mark.requires_private_data
def test_no_answerable_item_is_touched_by_the_unreachable_slash_branch(
    clause_chunks: list[dict[str, Any]],
    hard_set: list[dict[str, Any]],
    stored_contexts: dict[str, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The measurement behind §12's "unmeasured, not zero".

    Swap the two branches of the alternation so ``2/4/2`` survives as one token, and the
    seven answerable items do not move **at all** - not one coverage, not one anchor
    count. That is the whole reason a clause-reference signal looks free here: none of
    the seven questions exercises it, so its false-positive rate is *unmeasured* rather
    than zero, and quoting the AUC it buys on the probes (0.513 -> 0.656) as an
    improvement would be a true-positive rate with no false-positive rate beside it.

    The corpus vocabulary *does* move (1,491 -> 1,511 folded terms), because clause
    cross-references in the prose now survive as single terms. The point is that none of
    those new terms is an anchor for any of these seven questions.
    """
    spec = AnchorSpec()

    def coverages() -> dict[str, tuple[float | None, int]]:
        vocabulary = CorpusVocabulary.from_chunks(clause_chunks, spec)
        return {
            str(item["item_id"]): (
                signal.idf_weighted_coverage,
                signal.n_anchors,
            )
            for item in hard_set
            for signal in [
                compute_anchoring_signal(
                    str(item["question_text"]),
                    stored_contexts[str(item["item_id"])],
                    vocabulary,
                    spec,
                )
            ]
        }

    shipped = coverages()
    shipped_vocabulary = CorpusVocabulary.from_chunks(clause_chunks, spec).n_terms

    monkeypatch.setattr(
        anchoring_module,
        "ANCHOR_TOKEN_RE",
        re.compile(r"\d+(?:/\d+)*|[\w]+(?:['’][\w]+)?"),
    )
    assert anchoring_module.anchor_tokens("clause 2/4/2 applies") == [
        "clause",
        "2/4/2",
        "applies",
    ], "the patch must actually reach the tokeniser"

    patched = coverages()
    patched_vocabulary = CorpusVocabulary.from_chunks(clause_chunks, spec).n_terms

    assert patched == shipped, "no answerable item may move; see reports §12"
    assert (shipped_vocabulary, patched_vocabulary) == (1491, 1511)


def _string_leaves(node: Any) -> set[str]:
    """Every string *value* in a nested structure. Keys are not values."""
    if isinstance(node, dict):
        return set().union(*(_string_leaves(v) for v in node.values())) if node else set()
    if isinstance(node, list):
        return set().union(*(_string_leaves(v) for v in node)) if node else set()
    return {node} if isinstance(node, str) else set()


def _without_probe_rows(calibration: dict[str, Any]) -> dict[str, Any]:
    """The artefact with every probe row dropped, keeping only answerable rows.

    Probe questions are authored, tracked and published in ``data/probes/``, and §6 of
    the report prints their anchors on purpose, so their vocabulary is not a leak and
    including it would only measure coincidental domain overlap.
    """
    return {
        **calibration,
        "variants": [
            {
                **variant,
                "bm25": [
                    row for row in variant["bm25"] if row["group"] == "answerable_n7"
                ],
            }
            for variant in calibration["variants"]
        ],
    }


#: Every string value the calibration artefact may contain once probe rows are removed:
#: item ids, group and basis names, spec field names, the script path, the withholding
#: sentinel. Deliberately an allow-list rather than a pattern - a *new* string value is
#: exactly the event this test exists to catch, and the reviewer should have to look at
#: it. Substring sweeping the serialised JSON instead does not work: ``document`` is a
#: hard-set question token and also a substring of the key ``n_documents``.
ARTEFACT_STRING_VALUES = frozenset(
    {
        "H01",
        "H02",
        "H03",
        "H04",
        "H05",
        "H06",
        "H07",
        "anchor_spec_v1",
        "answerable_n7",
        "chunk_id",
        "clause_absent_from_standard",
        "clause_id",
        "hard-set question text is Git-ignored",
        "scripts/calibrate_anchoring_gate.py",
        "standard_absent_from_corpus",
        "standard_id",
        "sub_clause_id",
        "term_absent_from_corpus",
    }
)


def test_the_artefact_carries_no_free_text_outside_the_probe_rows(calibration) -> None:
    """A guard on this file's own blast radius, not on the module.

    These tests hold real question text in memory, and
    ``reports/anchoring_gate_calibration.json`` is the artefact a computed anchor list
    would actually land in. Once the published probe rows are removed, what remains is
    structural: ids, group names, spec fields, one sentinel sentence. Pinning that set
    exactly is stronger than any redaction pattern, and it needs no private data, so it
    keeps working on a clone.
    """
    assert _string_leaves(_without_probe_rows(calibration)) == ARTEFACT_STRING_VALUES


@pytest.mark.requires_private_data
def test_no_hard_set_vocabulary_appears_among_those_values(
    hard_set: list[dict[str, Any]], calibration
) -> None:
    """The other half: cross-check the allow-list against the real questions.

    The test above says the artefact contains only these 18 strings. This one says none
    of them is question vocabulary - the property that actually matters, and one that can
    only be checked where the hard set exists.

    Matching is on **whole tokens**, not substrings, and the fixed sentinel sentence is
    excluded. Both concessions are forced by the artefact's own field names rather than by
    convenience: ``standard`` is a hard-set question token and a substring of the spec
    field ``standard_id``, and ``document`` is a substring of the key ``n_documents``, so
    a substring sweep reports structural collisions as leaks. Underscores are word
    characters to :data:`ANCHOR_TOKEN_RE`, so ``standard_id`` tokenises to one token and
    cannot collide with ``standard`` under whole-token comparison. The sentinel is an
    authored English sentence and is pinned as a literal instead.
    """
    values = _string_leaves(_without_probe_rows(calibration))
    sentinel = "hard-set question text is Git-ignored"
    assert sentinel in values, "the withholding sentinel must be present"

    artefact_tokens = {
        token
        for value in values - {sentinel}
        for token in anchor_tokens(value)
    }
    question_tokens = {
        token
        for item in hard_set
        for token in anchor_tokens(str(item["question_text"]))
        if len(token) >= 5
    }
    assert not artefact_tokens & question_tokens, (
        "hard-set question vocabulary has reached "
        "reports/anchoring_gate_calibration.json"
    )


def test_the_markdown_report_says_in_writing_that_it_withholds_the_terms(
    repo_root: Path,
) -> None:
    """§6's withholding statement is the Markdown half of the publication boundary.

    The Markdown is deliberately not swept for question vocabulary. It is editorial
    prose, and its own narrative collides with ordinary domain words - "routing agreement
    1/7" in §9 contains "agreement" - so a substring sweep over it reports collisions,
    not leaks. Its real risk is a per-item table carrying anchor terms, and §6 states in
    writing that it withholds them.
    """
    text = (repo_root / "reports" / "anchoring_gate_calibration.md").read_text(
        encoding="utf-8"
    )
    assert "Anchor terms are withheld here" in text
    assert "hard-set question text is Git-ignored" in text
    assert "probe questions are published" in text

