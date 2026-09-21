"""Surface-form response classification, and the rule change that moved a headline number.

The stored n=7 run labelled a response ``abstained`` iff the stripped string equalled the
abstention line, and ``answered`` otherwise. That two-state rule is wrong in both
directions, and the tests here pin the correction rather than describing it:

* **A response can contain the abstention line and still assert things.** H05 quotes two
  clauses and then appends the line; whole-string equality called that an answer. Under
  :func:`classify_response` it is ``mixed``, and the corpus-gated test at the end shows
  H05 is the *only* item on the stored run where the two rules disagree - so the stored
  ``n_answered = 3`` is 2 genuine answers plus 1 self-contradiction, not 3 answers.
* **A response can be nothing but the abstention line and still fail equality.** A
  trailing period, surrounding quotes, a different case or a line break all defeat the
  legacy rule. Each is asserted to classify as ``abstained``.
* **An empty response was counted as an answer.** Legacy has no ``empty`` state, so a
  blank generation - or one consisting only of digits and punctuation - scored as an
  answer. ``empty`` means *no letter-bearing residue*, which is not the same as no
  characters, and the test says so.

One further distinction that is easy to misread: ``abstention_line_is_whole_response``
records the *normalised* legacy condition, not the literal one. It is a strict superset -
every legacy ``abstained`` sets it, but so does ``I CANNOT ANSWER FROM THE GIVEN
CONTEXT``, which legacy called an answer.

No AAOIFI clause prose appears here. Every string is invented.
"""

from __future__ import annotations

import dataclasses

import pytest

from aaoifi_rag.generation.prompt import ABSTENTION_LINE
from aaoifi_rag.reliability.response_class import (
    CURRENT_RULE,
    LEGACY_RULE,
    MIN_RESIDUE_WORDS,
    ResponseClass,
    ResponseClassification,
    classify_response,
    classify_response_legacy,
)
from aaoifi_rag.reliability.signals import compute_response_signals

#: An invented normative-looking sentence, long enough to be substantive residue.
QUOTE = (
    "The institution shall record the widget at the value agreed between the parties."
)

#: H05's shape: substantive content followed by the abstention line.
MIXED_RESPONSE = f"{QUOTE} {ABSTENTION_LINE}"


# --------------------------------------------------------------------------------------
# The four states
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (ABSTENTION_LINE, ResponseClass.ABSTAINED),
        (QUOTE, ResponseClass.ANSWERED),
        (MIXED_RESPONSE, ResponseClass.MIXED),
        (f"{ABSTENTION_LINE} {QUOTE}", ResponseClass.MIXED),
        ("", ResponseClass.EMPTY),
        (None, ResponseClass.EMPTY),
    ],
)
def test_the_four_states_cover_the_shapes_observed_at_n7(text, expected) -> None:
    assert classify_response(text).response_class is expected


def test_none_is_empty_rather_than_an_exception() -> None:
    """Generation backends can return ``None``; that is a defect to route, not a crash."""
    classification = classify_response(None)
    assert classification.response_class is ResponseClass.EMPTY
    assert classification.residue_word_count == 0
    assert classification.abstention_line_present is False


@pytest.mark.parametrize(
    ("response_class", "expected"),
    [
        (ResponseClass.EMPTY, False),
        (ResponseClass.ABSTAINED, False),
        (ResponseClass.MIXED, True),
        (ResponseClass.ANSWERED, True),
    ],
)
def test_an_answer_attempt_is_answered_or_mixed(response_class, expected) -> None:
    """``mixed`` counts as an attempt: the citation audit must still run on it.

    This is the property that makes H05's misattributed ``[2]`` visible. Treating a
    response containing the abstention line as an abstention would skip the audit and the
    misattribution would never be measured.
    """
    classification = ResponseClassification(
        response_class=response_class,
        abstention_line_present=response_class in (ResponseClass.ABSTAINED, ResponseClass.MIXED),
        abstention_line_is_whole_response=False,
        residue_word_count=0,
        residue_text="",
    )
    assert classification.is_answer_attempt is expected


# --------------------------------------------------------------------------------------
# Abstention is detected by removal, not by equality
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        ABSTENTION_LINE,
        f"{ABSTENTION_LINE}.",
        f'"{ABSTENTION_LINE}"',
        f"**{ABSTENTION_LINE}**",
        f"  {ABSTENTION_LINE}  ",
        ABSTENTION_LINE.upper(),
        "I cannot answer\nfrom the given   context",
        f"{ABSTENTION_LINE} {ABSTENTION_LINE}",
    ],
)
def test_a_response_that_is_only_the_line_abstains_however_it_is_punctuated(text) -> None:
    """Seven of these eight are labelled ``answered`` by the stored run's rule.

    Only the bare line and the padded line survive ``strip() == line``. Every other form
    here - a trailing period, quotes, emphasis, a different case, a line break, the line
    twice - was counted as an answer. On the stored n=7 responses none of these forms
    happens to occur, so the correction changes no stored label by itself; it removes a
    fragility that would have changed the count on any further generation.
    """
    classification = classify_response(text)
    assert classification.response_class is ResponseClass.ABSTAINED
    assert classification.abstention_line_present is True
    assert classification.is_answer_attempt is False


def test_removing_the_line_removes_every_occurrence_of_it() -> None:
    """Otherwise a doubled line would leave the second copy as substantive residue."""
    classification = classify_response(f"{ABSTENTION_LINE} {ABSTENTION_LINE}")
    assert classification.residue_word_count == 0
    assert classification.residue_text == ""


def test_the_whole_response_flag_is_the_normalised_legacy_condition() -> None:
    """A superset of the legacy condition, which is why it is recorded separately.

    Reading the flag as "the stored run would have called this abstained" is wrong in the
    case-difference case, so the two are asserted against each other here rather than
    assumed equal anywhere downstream.
    """
    assert classify_response(ABSTENTION_LINE).abstention_line_is_whole_response is True
    assert classify_response(f"  {ABSTENTION_LINE}  ").abstention_line_is_whole_response is True
    # Normalisation folds case and whitespace, so these set the flag but legacy did not
    # call them abstentions.
    assert classify_response(ABSTENTION_LINE.upper()).abstention_line_is_whole_response is True
    assert classify_response_legacy(ABSTENTION_LINE.upper()) == "answered"
    # Punctuation is not folded, so a trailing period clears the flag while the class
    # still resolves to abstained via the residue count.
    trailing = classify_response(f"{ABSTENTION_LINE}.")
    assert trailing.abstention_line_is_whole_response is False
    assert trailing.response_class is ResponseClass.ABSTAINED


@pytest.mark.parametrize(
    "text",
    [ABSTENTION_LINE, f"  {ABSTENTION_LINE}  ", ABSTENTION_LINE.upper(), "i cannot answer from the given context"],
)
def test_the_flag_implies_abstention(text) -> None:
    """The one direction that must hold: nothing but the line cannot be an answer."""
    classification = classify_response(text)
    assert classification.abstention_line_is_whole_response is True
    assert classification.response_class is ResponseClass.ABSTAINED


# --------------------------------------------------------------------------------------
# The residue threshold
# --------------------------------------------------------------------------------------


def test_the_residue_boundary_is_exactly_two_letter_bearing_words() -> None:
    """Asserted on both sides, because the threshold is what separates the two classes.

    One trailing word is an apology or a hedge; two is content. The threshold is
    deliberately at the bottom of the range - it exists to distinguish "the line plus
    stray text" from "the line plus assertions", not to score answer length.
    """
    assert MIN_RESIDUE_WORDS == 2
    one = classify_response(f"{ABSTENTION_LINE} Sorry.")
    assert (one.residue_word_count, one.response_class) == (1, ResponseClass.ABSTAINED)
    two = classify_response(f"{ABSTENTION_LINE} Sorry, truly.")
    assert (two.residue_word_count, two.response_class) == (2, ResponseClass.MIXED)


@pytest.mark.parametrize(
    ("residue", "expected_words"),
    [
        ("a b c d", 0),  # single letters are not words
        ("2/4/2 8 10", 0),  # bare clause references and page numbers are not words
        ("-- ... ***", 0),
        ("x9 y7", 0),  # a letter and a digit is not two letters
        ("co-op", 2),  # the hyphen is a boundary, as in text_checks.words
    ],
)
def test_only_multi_letter_words_count_as_residue(residue, expected_words) -> None:
    """``[^\\W\\d_]{2,}``: at least two letters, digits excluded.

    Digits are excluded on purpose. A response of ``[2] 2/4/2`` is a citation marker and a
    clause reference with nothing asserted around them, and counting those as content
    would promote it to ``mixed`` and send it to the citation audit as an answer attempt.
    """
    classification = classify_response(f"{ABSTENTION_LINE} {residue}")
    assert classification.residue_word_count == expected_words


def test_the_residue_is_what_remains_after_the_line_is_cut_out() -> None:
    """Recorded so a ``mixed`` label can be inspected without re-deriving it."""
    classification = classify_response(MIXED_RESPONSE)
    assert classification.response_class is ResponseClass.MIXED
    assert classification.residue_text == QUOTE.rstrip(".")
    assert classification.residue_word_count == 13


def test_the_line_can_appear_first_and_the_class_is_unchanged() -> None:
    """Order carries no information: both orders are self-contradiction."""
    before = classify_response(f"{ABSTENTION_LINE} {QUOTE}")
    after = classify_response(f"{QUOTE} {ABSTENTION_LINE}")
    assert before.response_class is after.response_class is ResponseClass.MIXED
    assert before.residue_word_count == after.residue_word_count


# --------------------------------------------------------------------------------------
# Empty
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", None, "   \n\t ", "... --- ***", "2/4/2 8", "[1] [2]"])
def test_empty_means_no_letter_bearing_residue_not_no_characters(text) -> None:
    """A response of only markers, numbers or punctuation asserts nothing.

    Legacy had no such state, so all six of these scored as ``answered``: a generation
    failure and a substantive answer landed in the same bucket. ``empty`` is separated from
    ``abstained`` because the two need different handling - an abstention is the model
    doing what it was told, an empty response is a broken generation.
    """
    classification = classify_response(text)
    assert classification.response_class is ResponseClass.EMPTY
    assert classification.residue_word_count == 0
    assert classification.abstention_line_present is False
    assert classification.is_answer_attempt is False
    assert classify_response_legacy(text) == "answered"


def test_empty_retains_the_residue_it_rejected() -> None:
    """The digits are still visible in ``residue_text``, so the label can be checked."""
    assert classify_response("2/4/2 8").residue_text == "2/4/2 8"


# --------------------------------------------------------------------------------------
# The legacy rule, kept for comparison only
# --------------------------------------------------------------------------------------


def test_the_legacy_rule_has_only_two_outcomes() -> None:
    """Both of its failure directions in one place.

    Anything that is not byte-identical to the line is an answer, so a self-contradicting
    response, a differently-punctuated abstention and an empty string all read
    ``answered``. That is the rule ``reports/e2e_batch_smoke_results.json`` was written
    with, which is why its ``n_answered`` is an upper bound rather than a count.
    """
    assert classify_response_legacy(ABSTENTION_LINE) == "abstained"
    assert classify_response_legacy(f"  {ABSTENTION_LINE}  ") == "abstained"
    assert classify_response_legacy(f"{ABSTENTION_LINE}.") == "answered"
    assert classify_response_legacy(MIXED_RESPONSE) == "answered"
    assert classify_response_legacy("") == "answered"
    assert classify_response_legacy(None) == "answered"


def test_both_rules_are_recorded_as_strings_so_a_stored_label_stays_readable() -> None:
    """A label without its rule is not interpretable a year later."""
    assert ABSTENTION_LINE in LEGACY_RULE
    assert str(MIN_RESIDUE_WORDS) in CURRENT_RULE
    assert "not a quality, entailment" in CURRENT_RULE
    assert classify_response(QUOTE).rule == CURRENT_RULE


def test_the_classification_is_frozen() -> None:
    """Signals are evidence; nothing downstream may edit them after the fact."""
    classification = classify_response(QUOTE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        classification.response_class = ResponseClass.ABSTAINED  # type: ignore[misc]


def test_the_residue_text_never_reaches_a_signal_dictionary() -> None:
    """``residue_text`` is a fragment of the response, so it can carry clause text.

    It is kept on the classification because the gates read it, and dropped from every
    serialised view. Asserted here, next to the field that carries the risk, as well as in
    the trace tests: the word count travels, the words do not.
    """
    signals = compute_response_signals(MIXED_RESPONSE, [])
    payload = signals.as_dict()
    assert "residue_text" not in payload
    assert payload["residue_word_count"] == 13
    assert QUOTE not in repr(payload)


# --------------------------------------------------------------------------------------
# Against the stored run
# --------------------------------------------------------------------------------------


@pytest.mark.requires_private_data
def test_the_stored_labels_are_exactly_what_the_legacy_rule_produces(stored_run) -> None:
    """Establishes that the divergence below is the rule changing, not the data.

    If a stored label disagreed with ``classify_response_legacy`` on its own stored
    response, the comparison that follows would be measuring two different things.
    """
    for item in stored_run["items"]:
        assert item["response_class"] == classify_response_legacy(item["model_response"]), (
            f"{item['item_id']} stored label is not reproducible from its stored response"
        )


@pytest.mark.requires_private_data
def test_h05_is_the_only_item_where_the_two_rules_disagree(stored_run) -> None:
    """The whole basis for saying the stored ``n_answered = 3`` overstates.

    Six items classify identically under both rules. H05 moves ``answered`` -> ``mixed``:
    it quotes clause text, cites ``[2]`` for a passage that transcribes rank 4, and then
    appends the abstention line. Its 88-word residue is far past the 2-word threshold, so
    this is not a boundary case.
    """
    diverging = {
        item["item_id"]: (
            item["response_class"],
            classify_response(item["model_response"]).response_class.value,
        )
        for item in stored_run["items"]
        if item["response_class"] != classify_response(item["model_response"]).response_class
    }
    assert diverging == {"H05": ("answered", "mixed")}
    h05 = next(item for item in stored_run["items"] if item["item_id"] == "H05")
    classification = classify_response(h05["model_response"])
    assert classification.abstention_line_present is True
    assert classification.abstention_line_is_whole_response is False
    assert classification.residue_word_count == 88
    assert classification.is_answer_attempt is True


@pytest.mark.requires_private_data
def test_the_stored_answered_count_decomposes_into_two_answers_and_one_mixed(
    stored_run,
) -> None:
    """``n_answered = 3`` is 2 + 1, and the four abstentions are unambiguous.

    All four abstaining items emitted the bare line, so no reclassification is available
    to argue the abstention rate up or down. The only movement is on the answering side.
    """
    classes = {
        item["item_id"]: classify_response(item["model_response"]).response_class.value
        for item in stored_run["items"]
    }
    assert classes == {
        "H01": "answered",
        "H02": "abstained",
        "H03": "answered",
        "H04": "abstained",
        "H05": "mixed",
        "H06": "abstained",
        "H07": "abstained",
    }
    assert stored_run["aggregates"]["n_answered"] == 3
    assert sum(value == "answered" for value in classes.values()) == 2
    assert all(
        classify_response(item["model_response"]).abstention_line_is_whole_response
        for item in stored_run["items"]
        if classes[item["item_id"]] == "abstained"
    )
