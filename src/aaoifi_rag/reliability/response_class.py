"""Response classification (plan Layer 4).

Why this module exists
----------------------
The n=7 batch run classified responses with whole-string equality::

    abstained iff model_response.strip() == "I cannot answer from the given context"

That rule is wrong in one direction that materially changed the headline
number. H05 quotes two clauses and *then appends the abstention line*; because
the stripped response is not equal to the line, the rule labelled it
``answered``. So ``reports/e2e_batch_smoke_results.json`` reports
``n_answered = 3`` where only two items (H01 and H03) assert anything of their
own, and H03's assertion is 0.90 verbatim overlap with its context - a judgement
that belongs to the echo ratio in :mod:`~aaoifi_rag.reliability.text_checks`,
not to this module.

:func:`classify_response` therefore looks for the abstention line *anywhere*
in the response and reports four states instead of two, so a self-contradicting
response can be routed rather than silently counted as an answer.
:func:`classify_response_legacy` is kept so the divergence from the stored run
can be asserted in tests rather than asserted in prose.

Scope limit: these are surface-form labels. Nothing here judges whether an
answer is correct, entailed by the context, or Shari'ah-compliant.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
import unicodedata

from ..generation.prompt import ABSTENTION_LINE

#: Minimum number of letter-bearing words that must survive after the
#: abstention line is removed before the residue counts as an answer attempt.
#: 2 is deliberately low: the point is to separate "the line plus stray
#: punctuation" from "the line plus actual content", not to score length.
MIN_RESIDUE_WORDS = 2

LEGACY_RULE = (
    "abstained iff model_response.strip() == the fixed line "
    f"'{ABSTENTION_LINE}'; else answered"
)

CURRENT_RULE = (
    "empty iff no letter-bearing residue; abstained iff the fixed abstention "
    "line is present and fewer than "
    f"{MIN_RESIDUE_WORDS} letter-bearing words remain once every occurrence of "
    "the line is removed; mixed iff the line is present and the residue is "
    "substantive; else answered. Surface form only - not a quality, entailment "
    "or Shari'ah label."
)

_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
# Sentence-final punctuation and quoting marks that models append to the line.
_TRIM_CHARS = " \t\r\n.,;:!?\"'`*_-()[]{}‘’“”«»"


class ResponseClass(StrEnum):
    """Surface form of a generated response."""

    EMPTY = "empty"
    ABSTAINED = "abstained"
    #: Abstention line present *and* substantive other content: H05's shape.
    MIXED = "mixed"
    ANSWERED = "answered"


def _normalise(text: str) -> str:
    """Casefold, NFKC-normalise and collapse whitespace for matching."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _WHITESPACE_RE.sub(" ", folded).strip()


def _abstention_pattern() -> re.Pattern[str]:
    """Match the abstention line tolerantly across whitespace and case."""
    parts = [re.escape(token) for token in ABSTENTION_LINE.split()]
    return re.compile(r"\s+".join(parts), re.IGNORECASE)


_ABSTENTION_RE = _abstention_pattern()


@dataclass(frozen=True)
class ResponseClassification:
    """Result of surface-form classification, with its evidence."""

    response_class: ResponseClass
    abstention_line_present: bool
    #: True when the response *normalised* (NFKC, casefolded, whitespace-collapsed) is
    #: nothing but the abstention line. A strict superset of the condition the stored n=7
    #: run tested for: that rule was byte equality after ``strip()``, so it missed a
    #: case or whitespace difference. Recorded rather than derived so the stored labels
    #: stay comparable (``tests/test_response_class.py`` asserts both directions).
    abstention_line_is_whole_response: bool
    residue_word_count: int
    #: What the response looks like once the abstention line is removed.
    #: Retained for gate input; it may contain licensed clause text, so it is
    #: dropped by the public trace view.
    residue_text: str
    rule: str = CURRENT_RULE

    @property
    def is_answer_attempt(self) -> bool:
        """True when the model produced content beyond the abstention line."""
        return self.response_class in (ResponseClass.ANSWERED, ResponseClass.MIXED)


def classify_response(text: str) -> ResponseClassification:
    """Classify a raw model response by surface form."""
    raw = text or ""
    residue = _ABSTENTION_RE.sub(" ", raw)
    residue = residue.strip(_TRIM_CHARS).strip()
    residue_words = len(_WORD_RE.findall(residue))
    present = bool(_ABSTENTION_RE.search(raw))
    whole = _normalise(raw) == _normalise(ABSTENTION_LINE)

    if present:
        response_class = (
            ResponseClass.MIXED
            if residue_words >= MIN_RESIDUE_WORDS
            else ResponseClass.ABSTAINED
        )
    elif residue_words == 0:
        response_class = ResponseClass.EMPTY
    else:
        response_class = ResponseClass.ANSWERED

    return ResponseClassification(
        response_class=response_class,
        abstention_line_present=present,
        abstention_line_is_whole_response=whole,
        residue_word_count=residue_words,
        residue_text=residue,
    )


def classify_response_legacy(text: str) -> str:
    """Reproduce the stored n=7 rule exactly, for comparison only.

    Do not use this to route. It is here so tests can demonstrate which stored
    ``response_class`` values are artefacts of the rule: on the stored run
    exactly one item diverges (H05, ``answered`` -> ``mixed``), and the rule's
    other failure directions - a punctuated abstention, an empty generation -
    are asserted on invented text in ``tests/test_response_class.py``.
    """
    if (text or "").strip() == ABSTENTION_LINE:
        return "abstained"
    return "answered"
