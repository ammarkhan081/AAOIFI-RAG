"""Mechanical text checks: normalisation, script detection, and the echo ratio.

These four functions are the measurement floor of the whole reliability layer - the echo
gate, the citation audit and the anchoring signal all resolve to them - so their edge
behaviour is pinned rather than left to be inferred from the callers.

Three properties get particular attention because a plausible-looking alternative
implementation would break a published finding:

* **Normalisation folds typography but does not strip punctuation, and does not stem.**
  Both are load-bearing. Folding U+2019 is what lets H05's corrupted transcription be
  recognised as corpus text rather than as invention; *not* stripping punctuation is what
  makes ``term_occurrences`` mean what the probe labels say it means.
* **The repetition detector fires at four, not three.** Doubled letters are ordinary
  English; ``aaaa`` is not. The boundary is asserted on both sides.
* **The echo ratio is an asymmetric, shingle-counted quantity with a documented
  degenerate case below eight words.** A short answer scoring 1.0 means "too short to
  distinguish", and the test says so, because reading it as evidence of transcription is
  the obvious misuse.

No AAOIFI clause prose appears here. Every string is invented.
"""

from __future__ import annotations

import pytest

from aaoifi_rag.reliability.text_checks import (
    REPEATED_CHAR_RE,
    SHINGLE_WORDS,
    UNEXPECTED_SCRIPT_RE,
    contains_normalised,
    normalise_for_match,
    repeated_char_runs,
    strip_unexpected_script,
    unexpected_script_chars,
    verbatim_overlap_ratio,
    words,
)

#: Eight invented words, which is exactly one shingle at the default width.
EIGHT = "the institution shall record the widget at cost"


# --------------------------------------------------------------------------------------
# normalise_for_match
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The Institution", "the institution"),
        ("a\t b\n\nc", "a b c"),
        ("  padded  ", "padded"),
        # NFKC: a compatibility ligature and a full-width letter decompose.
        ("ﬁnancial", "financial"),
        ("ＳＳ８", "ss8"),
        # Typographic folding, one per class in the fold table.
        ("the institution’s", "the institution's"),
        ("“quoted”", '"quoted"'),
        ("«quoted»", '"quoted"'),
        ("2019—2020", "2019-2020"),
        ("a…b", "a...b"),
        ("a b", "a b"),
        (None, ""),
        ("", ""),
    ],
)
def test_normalisation_folds_case_form_typography_and_whitespace(raw, expected) -> None:
    assert normalise_for_match(raw) == expected


def test_normalisation_does_not_strip_punctuation() -> None:
    """A deliberate non-property, and the one ``term_occurrences`` depends on.

    ``probes.term_occurrences`` documents that a probe term must be spelled the way the
    corpus spells it. If normalisation deleted punctuation, ``clause 2/4/2`` and
    ``clause 242`` would collide and a ``term_absent_from_corpus`` label would claim more
    than it had measured.
    """
    assert normalise_for_match("clause 2/4/2.") == "clause 2/4/2."
    assert normalise_for_match("(a), (b);") == "(a), (b);"


def test_normalisation_does_not_stem() -> None:
    """The suffix folding in :mod:`aaoifi_rag.reliability.anchoring` is separate."""
    assert normalise_for_match("assets") != normalise_for_match("asset")


def test_normalisation_is_idempotent() -> None:
    once = normalise_for_match("The  Institution’s ﬁnancial “notes”…")
    assert normalise_for_match(once) == once


def test_the_apostrophe_h05_replaced_folds_rather_than_matching() -> None:
    """U+2019 is the character H05 emitted Arabic in place of.

    It has to fold to ASCII, because otherwise a transcription differing from the corpus
    only in quote style would score as a non-match and be reported as an unsupported
    claim. The right reading of H05 is a corrupted quotation of real text.
    """
    assert normalise_for_match("institution’s") == normalise_for_match("institution's")
    assert unexpected_script_chars("institution’s") == ()


# --------------------------------------------------------------------------------------
# Script detection
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("在", ("在",)),  # the stray character observed in reports/sac_ablation_report.md
        ("و", ("و",)),  # H05's substitution for U+2019
        ("مرابحة", ("م", "ر", "ا", "ب", "ح", "ة")),
        ("ひらがな", ("ひ", "ら", "が", "な")),
        ("한글", ("한", "글")),
        ("Привет", ("П", "р", "и", "в", "е", "т")),
        ("aaa在bbb在ccc", ("在",)),  # distinct, deduplicated
        ("在ロ在", ("在", "ロ")),  # first-seen order, not sorted
    ],
)
def test_out_of_script_characters_are_detected_deduplicated_and_ordered(
    text, expected
) -> None:
    assert unexpected_script_chars(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "The institution shall record the asset at cost.",
        "Café naïve Zürich Åland",  # accented Latin is expected in English prose
        "2/4/2 §8 [1] 10.10 — “notes”",
        "",
        None,
    ],
)
def test_expected_text_produces_no_script_findings(text) -> None:
    assert unexpected_script_chars(text) == ()
    assert not UNEXPECTED_SCRIPT_RE.search(text or "")


def test_stripping_script_collapses_the_gap_it_leaves() -> None:
    """Otherwise the stripped text would not match the corpus on whitespace."""
    assert strip_unexpected_script("the institutionو s books") == "the institution s books"
    assert strip_unexpected_script("在 在 在") == ""
    assert strip_unexpected_script(None) == ""


def test_stripping_is_what_separates_a_corrupted_quote_from_an_invented_one() -> None:
    """H05's defect, reproduced on invented text.

    Two substitutions in a 24-token quotation break most of its windows and drag the
    ratio to 0.33; stripping them restores 1.0. That difference is what
    ``citations.Attribution.CORRUPTED`` reports, and why the audit does not call such a
    segment unsupported.
    """
    source = (
        "the institution's books shall record the widget at the agreed value and "
        "disclose the trustee's fee in the notes for the period"
    )
    garbled = source.replace("'", "و")
    assert verbatim_overlap_ratio(garbled, source) == pytest.approx(1 / 3)
    assert verbatim_overlap_ratio(strip_unexpected_script(garbled), source) == 1.0


def test_a_single_substitution_stays_above_the_citation_threshold() -> None:
    """Which is why the audit checks ``SUPPORTED`` before ``CORRUPTED``.

    One stray character breaks only the windows containing it - here 2 of 11, leaving
    0.818 - so a lightly garbled transcription is still attributed to its cited rank and
    is *not* reported as corrupted. ``CORRUPTED`` is reached only when the corruption is
    dense enough to push the cited-rank overlap under 0.5. Worth knowing before reading
    the attribution counts: a low ``corrupted`` count does not mean clean script.
    """
    source = (
        "the institution's books shall record the widget at the agreed value and "
        "disclose it in the notes"
    )
    garbled = source.replace("institution's", "institutionو s")
    assert verbatim_overlap_ratio(garbled, source) == pytest.approx(9 / 11)
    assert unexpected_script_chars(garbled) == ("و",)


# --------------------------------------------------------------------------------------
# Degenerate decoding
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("aaaa", ("aaaa",)),
        ("aaaaaa", ("aaaaaa",)),  # greedy: one run, not overlapping matches
        ("the ---- rule", ("----",)),
        ("a....b", ("....",)),
        ("xxxx and yyyy", ("xxxx", "yyyy")),
        ("xxxx and xxxx", ("xxxx",)),  # deduplicated
    ],
)
def test_repetition_runs_are_reported_distinct_and_in_order(text, expected) -> None:
    assert repeated_char_runs(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "aaa",  # three is below the threshold
        "The committee will reassess the balloon lessee's fees.",
        "...",
        "",
        None,
    ],
)
def test_three_repetitions_are_ordinary_text(text) -> None:
    assert repeated_char_runs(text) == ()


def test_the_repetition_boundary_is_exactly_four() -> None:
    """Asserted on both sides so a change to ``{3,}`` cannot pass unnoticed.

    English has plenty of doubled letters and the occasional tripled one in informal
    text; four identical characters in a row in generated prose is decoding failure.
    """
    assert REPEATED_CHAR_RE.pattern == r"(.)\1{3,}"
    assert repeated_char_runs("a" * 3) == ()
    assert repeated_char_runs("a" * 4) == ("aaaa",)


# --------------------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------------------


def test_words_splits_on_punctuation_including_the_clause_separator() -> None:
    """``2/4/2`` becomes three tokens, and so does it in the anchoring tokeniser.

    The two paths differ in *mechanism* and agree in *result*. ``words`` splits on
    ``\\W+``, so the slashes are separators. ``anchoring.ANCHOR_TOKEN_RE`` has a
    ``\\d+(?:/\\d+)*`` branch that looks like it would keep the reference whole, but it
    sits second in an ordered alternation behind ``[\\w]+`` and is unreachable - see
    ``tests/test_anchoring.py`` and §12 of ``reports/anchoring_gate_calibration.md``.
    Either way it costs overlap measurement nothing, because overlap counts eight-word
    windows and three cheap tokens inside one window change no shingle boundary.
    """
    assert words("clause 2/4/2 applies") == ["clause", "2", "4", "2", "applies"]
    assert words("a_b") == ["a", "b"], "underscore is a separator, not a word character"
    assert words("The Institution’s") == ["the", "institution", "s"]
    assert words("") == []


# --------------------------------------------------------------------------------------
# verbatim_overlap_ratio
# --------------------------------------------------------------------------------------


def test_shingle_width_is_eight() -> None:
    assert SHINGLE_WORDS == 8


def test_a_full_transcription_scores_one_and_disjoint_prose_scores_zero() -> None:
    context = f"{EIGHT} and disclose the value in the notes to the statements"
    assert verbatim_overlap_ratio(context, context) == 1.0
    assert verbatim_overlap_ratio(context, "wholly unrelated prose about nothing at all") == 0.0


def test_partial_overlap_is_the_fraction_of_matching_windows() -> None:
    """Nine words, one word changed at the end: one of two shingles matches.

    An exact arithmetic claim rather than an approximate one, because the whole threshold
    sweep in ``sensitivity.py`` treats these values as exact transition points.
    """
    context = f"{EIGHT} today"
    answer = f"{EIGHT} tomorrow"
    assert len(words(answer)) == 9
    assert verbatim_overlap_ratio(answer, context) == 0.5


def test_the_ratio_is_asymmetric_because_the_answer_is_the_denominator() -> None:
    """It measures how much of the *answer* is borrowed, not how much of the context.

    A one-sentence quotation from a long context scores 1.0; the same pair swapped scores
    much lower. Reading it as a similarity would invert the finding on H03.
    """
    long_context = f"{EIGHT} and disclose the value in the notes to the statements for the period"
    assert verbatim_overlap_ratio(EIGHT, long_context) == 1.0
    assert verbatim_overlap_ratio(long_context, EIGHT) < 1.0


def test_a_short_answer_is_compared_at_its_own_width() -> None:
    """A sub-window verbatim quote scores 1.0, and used not to.

    Before this rule the answer's single short shingle was compared against the
    context's eight-word windows and could never match on length alone, so a seven-word
    transcription measured 0.0 and passed the echo gate while the same quote plus two
    words measured 1.0. The cliff at exactly eight words was a bypass, not a design.
    """
    context = "the institution shall record the widget at the agreed value and disclose it"
    assert verbatim_overlap_ratio("shall record the widget", context) == 1.0
    assert verbatim_overlap_ratio("shall record the widget at the agreed", context) == 1.0
    assert verbatim_overlap_ratio("shall record the trustee", context) == 0.0


def test_a_short_answer_scoring_high_is_uninformative_not_evidence() -> None:
    """One word will match almost any context. The module says to read it that way.

    Kept as a separate test from the one above so the two claims stay distinct: the
    measurement is now well defined at short lengths, and it is still not usable as
    evidence there. A caller wanting to act on the echo ratio needs a length floor of
    its own; nothing in the gate battery currently imposes one.
    """
    assert verbatim_overlap_ratio("widget", "the widget") == 1.0
    assert verbatim_overlap_ratio("the", "the institution shall record") == 1.0


def test_answers_at_or_above_the_window_are_unaffected_by_the_short_rule() -> None:
    """The n=7 observed values depend on this: all three answer attempts exceed 8 words."""
    context = f"{EIGHT} today"
    assert verbatim_overlap_ratio(f"{EIGHT} tomorrow", context) == 0.5
    assert verbatim_overlap_ratio(EIGHT, context) == 1.0


def test_exactly_eight_words_is_one_shingle() -> None:
    assert verbatim_overlap_ratio(EIGHT, EIGHT) == 1.0
    assert verbatim_overlap_ratio(EIGHT, f"prefix {EIGHT} suffix") == 1.0


@pytest.mark.parametrize(
    ("answer", "context"),
    [("", "anything at all"), (None, "anything at all"), ("   ", "anything")],
)
def test_an_empty_answer_scores_zero_rather_than_dividing_by_zero(answer, context) -> None:
    assert verbatim_overlap_ratio(answer, context) == 0.0


def test_an_empty_context_scores_zero_for_a_real_answer() -> None:
    """The zero-retrieval case. Nothing can be echoed from nothing."""
    assert verbatim_overlap_ratio(f"{EIGHT} today", "") == 0.0
    assert verbatim_overlap_ratio(f"{EIGHT} today", None) == 0.0


def test_the_ratio_ignores_case_and_typography() -> None:
    """Both sides go through normalisation, so a restyled quote still counts as echo."""
    context = "the institution’s books shall record the widget at the agreed value"
    answer = "The Institution's  BOOKS shall record the widget at the agreed value"
    assert verbatim_overlap_ratio(answer, context) == 1.0


def test_a_narrower_window_is_available_for_callers_that_want_one() -> None:
    """``width`` is a parameter; the default is the calibrated value, not the only one."""
    context = f"{EIGHT} today"
    answer = f"{EIGHT} tomorrow"
    assert verbatim_overlap_ratio(answer, context, width=4) == pytest.approx(5 / 6)
    assert verbatim_overlap_ratio(answer, context, width=8) == 0.5


# --------------------------------------------------------------------------------------
# contains_normalised
# --------------------------------------------------------------------------------------


def test_containment_normalises_both_sides() -> None:
    assert contains_normalised("The Institution’s Books", "institution's books")
    assert contains_normalised("a b", "a b")
    assert not contains_normalised("the institution", "the trustee")


def test_an_empty_needle_is_false_not_vacuously_true() -> None:
    """A blank probe term must not silently match every document.

    The same reasoning as ``assert_no_gold_answer_leak`` treating a blank gold answer as
    nothing to leak: a degenerate input must fail closed, not open.
    """
    assert contains_normalised("anything", "") is False
    assert contains_normalised("anything", "   ") is False
    assert contains_normalised("", "something") is False
