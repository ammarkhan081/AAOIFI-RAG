"""Mechanical text checks shared by the reliability gates (plan Layer 4).

Every function here is deterministic, dependency-free and locally testable. No
function judges meaning: they measure surface properties (script, repetition,
verbatim overlap with the supplied context) that were each observed as a
concrete defect in the stored n=7 run.

Provenance of the script ranges: ported from ``validate_summary_text`` in
``scripts/colab_sac_build.py``, which uses them as a corruption detector for
generated English summaries of an English-only corpus.

This module contains no AAOIFI clause prose.
"""

from __future__ import annotations

import re
import unicodedata

#: CJK / Hiragana-Katakana / Hangul / Cyrillic / Arabic. None are expected in
#: generated English answers over this corpus. Observed twice: the stray ``在``
#: in ``reports/sac_ablation_report.md`` and the Arabic substitutions in H05's
#: answer, where the model emitted Arabic tokens in place of U+2019 (``’``).
UNEXPECTED_SCRIPT_RE = re.compile(
    r"[一-鿿぀-ヿ가-힯Ѐ-ӿ؀-ۿ]"
)

#: Four or more of the same character in a row: degenerate decoding.
REPEATED_CHAR_RE = re.compile(r"(.)\1{3,}")

_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

#: Characters folded to an ASCII equivalent before matching, so that a quote
#: differing from the corpus only by typography still matches. U+2019 is the
#: character H05 replaced with Arabic text, so it must fold rather than match.
_PUNCTUATION_FOLD = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "«": '"', "»": '"', "‹": "'", "›": "'",
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ",
    "…": "...",
}
_FOLD_TABLE = str.maketrans(_PUNCTUATION_FOLD)

#: Word-shingle width used by :func:`verbatim_overlap_ratio`. Eight consecutive
#: words matching the context is well past coincidence for prose of this kind,
#: and short enough to still register partial echo.
SHINGLE_WORDS = 8


def normalise_for_match(text: str) -> str:
    """NFKC-normalise, fold typography, casefold and collapse whitespace.

    Applied to both sides of every string comparison in this package so that a
    match means "same words", not "same bytes".
    """
    folded = unicodedata.normalize("NFKC", text or "").translate(_FOLD_TABLE)
    return _WHITESPACE_RE.sub(" ", folded.casefold()).strip()


def unexpected_script_chars(text: str) -> tuple[str, ...]:
    """Return the distinct out-of-script characters found, in first-seen order."""
    seen: dict[str, None] = {}
    for match in UNEXPECTED_SCRIPT_RE.finditer(text or ""):
        seen.setdefault(match.group(0), None)
    return tuple(seen)


def strip_unexpected_script(text: str) -> str:
    """Remove out-of-script characters and collapse the resulting whitespace.

    Used to tell a *corrupted transcription* from an *unsupported claim*: if a
    quotation only matches the context after stripping, the model garbled a
    real excerpt rather than inventing one. That is H05's actual defect.
    """
    return _WHITESPACE_RE.sub(" ", UNEXPECTED_SCRIPT_RE.sub(" ", text or "")).strip()


def repeated_char_runs(text: str) -> tuple[str, ...]:
    """Return distinct 4+ character repetitions, in first-seen order."""
    seen: dict[str, None] = {}
    for match in REPEATED_CHAR_RE.finditer(text or ""):
        seen.setdefault(match.group(0), None)
    return tuple(seen)


def words(text: str) -> list[str]:
    """Tokenise for overlap measurement, after :func:`normalise_for_match`."""
    return _WORD_RE.findall(normalise_for_match(text))


def _shingles(tokens: list[str], width: int) -> set[tuple[str, ...]]:
    if len(tokens) < width:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + width]) for i in range(len(tokens) - width + 1)}


def _effective_width(answer_tokens: list[str], width: int) -> int:
    """Narrow the window to the answer's own length when the answer is shorter.

    Without this, a sub-``width`` answer produced one short shingle that was compared
    against the context's ``width``-length shingles and could never match on length
    alone: a seven-word verbatim quote of a long excerpt scored 0.0 and passed the echo
    gate, while the same quote plus two words scored 1.0. Comparing at the answer's own
    width removes that cliff and leaves every answer of ``width`` words or more scored
    exactly as before, so the stored n=7 values are unaffected
    (``tests/test_sensitivity.py`` re-derives them).
    """
    return min(width, len(answer_tokens)) if answer_tokens else width


def verbatim_overlap_ratio(
    answer: str,
    context: str,
    width: int = SHINGLE_WORDS,
) -> float:
    """Fraction of the answer's word shingles that also occur in the context.

    1.0 means every window of ``width`` consecutive words in the answer appears
    verbatim in the supplied excerpts: the answer is a transcription, not a
    response. In the stored n=7 run the three answer attempts measure H01
    0.443038, H05 0.717391 and H03 0.900000 - H03's response is almost entirely a
    quoted clause, which is why that run's ``n_answered = 3`` overstates how many
    items produced an actual answer. (Those three values are also the only
    transition points the ``max_echo_ratio`` threshold has; see
    :mod:`~aaoifi_rag.reliability.sensitivity`.)

    An answer shorter than ``width`` words is compared at its own length instead:
    the whole answer is looked for among the context's windows of that size, so a
    six-word verbatim quote scores 1.0 rather than falling through a length
    mismatch. It is still **too short to distinguish** - a two-word answer will
    often match by coincidence - so read a high ratio on a short answer as
    uninformative, not as evidence of transcription. Answers of ``width`` words or
    more are unaffected by this rule.
    """
    answer_tokens = words(answer)
    if not answer_tokens:
        return 0.0
    effective_width = _effective_width(answer_tokens, width)
    answer_shingles = _shingles(answer_tokens, effective_width)
    if not answer_shingles:
        return 0.0
    context_shingles = _shingles(words(context), effective_width)
    matched = sum(1 for shingle in answer_shingles if shingle in context_shingles)
    return matched / len(answer_shingles)


def contains_normalised(haystack: str, needle: str) -> bool:
    """Substring test under :func:`normalise_for_match` on both sides."""
    normalised_needle = normalise_for_match(needle)
    if not normalised_needle:
        return False
    return normalised_needle in normalise_for_match(haystack)
