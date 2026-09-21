"""Mechanical citation auditing (plan Layer 4).

What this checks, and why it is shaped this way
-----------------------------------------------
The context block gives the model 1-indexed ranks ``[1]``..``[k]`` and those are
the only citation targets it has (see :mod:`aaoifi_rag.generation.prompt`). So
citation *indices* can be checked mechanically without any judgement.

The parsing follows the shape the model actually produced in the stored n=7 run
rather than an assumed one. Inspecting ``reports/e2e_batch_smoke_results.json``:
six of seven responses contain no quotation marks at all, and the three answer
attempts instead emit ``[n]`` followed by unquoted transcribed clause text. A
quotation-mark-based extractor would therefore have missed H05 entirely. So a
response is split into *segments*: each marker owns the text up to the next
marker, and that text is the content attributed to that rank.

The one hard defect this catches is H05's: its ``[2]`` segment transcribes
content that was really at rank 4. ``reports/citation_entailment_check_n7.md``
describes that as invented content; it is not. The text is genuine corpus
(``clause:SS26:3/1``), cited under the wrong index.

Deliberate limitation
---------------------
A segment that is *not* substantially verbatim is classified
:attr:`Attribution.UNVERIFIED`, never "unsupported". Separating a legitimate
paraphrase from a fabricated claim needs entailment checking, which this layer
does not attempt and must not pretend to. Only out-of-range ranks,
misattribution and script corruption are treated as failures, because only
those are mechanically certain. ``UNVERIFIED`` is recorded and does not gate.

How to read the attribution counts
----------------------------------
The tests are checked in a fixed order, and two consequences follow that a
counts table does not show on its own:

* ``supported`` is tested before ``corrupted``, so a *lightly* garbled
  transcription of the right excerpt is reported supported with its stray
  characters in ``unexpected_script``. On the stored n=7 run H05's first segment
  carries ten distinct Arabic characters and still matches its cited rank at
  0.64, so the whole run reports ``corrupted: 0``. A zero there means "no
  segment needed stripping to match", not "no script corruption".
* ``misattributed`` is tested before ``corrupted``, so a segment that is *both*
  misattributed and densely garbled is reported ``corrupted``. Both gate, so no
  decision is affected, but the ``misattributed`` count is a lower bound
  whenever any segment carries out-of-script characters.

Both are asserted in ``tests/test_citations.py``.

This module contains no AAOIFI clause prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Any, Mapping, Sequence

from ..generation.prompt import clause_label
from .text_checks import (
    strip_unexpected_script,
    unexpected_script_chars,
    verbatim_overlap_ratio,
    words,
)

#: ``[12]``-style markers. Two digits is ample for a top-k context block.
CITATION_MARKER_RE = re.compile(r"\[(\d{1,2})\]")

#: Shingle-overlap fraction at which a segment counts as a transcription of a
#: particular excerpt. Chosen so that a lightly-edited transcription still
#: matches while genuine prose does not; validated against the three n=7 answer
#: attempts in ``tests/test_citations.py``, which re-derives every published
#: segment score from the stored contexts.
MIN_VERBATIM_OVERLAP = 0.5

#: Segments shorter than this are not attributable either way.
MIN_SEGMENT_WORDS = 6


class Attribution(StrEnum):
    """Outcome of attributing one citation segment to its cited rank."""

    #: Cited rank does not exist in the context block. Always a failure.
    UNKNOWN_RANK = "unknown_rank"
    #: Segment is verbatim in the rank it cites.
    SUPPORTED = "supported"
    #: Segment is verbatim in a *different* rank. H05's ``[2]``. Always a failure.
    MISATTRIBUTED = "misattributed"
    #: Only matches once out-of-script characters are stripped: the model garbled
    #: a real excerpt. Treated as a failure because the served text is corrupt.
    CORRUPTED = "corrupted"
    #: Not substantially verbatim anywhere. Paraphrase or fabrication - this
    #: layer cannot tell which, and does not gate on it.
    UNVERIFIED = "unverified"
    #: Too few words to attribute.
    TOO_SHORT = "too_short"


#: The subset of outcomes that are mechanically certain defects.
FAILING_ATTRIBUTIONS = frozenset(
    {Attribution.UNKNOWN_RANK, Attribution.MISATTRIBUTED, Attribution.CORRUPTED}
)


@dataclass(frozen=True)
class CitationSegment:
    """One ``[n]`` marker and the text it claims."""

    rank: int
    marker_start: int
    word_count: int
    attribution: Attribution
    overlap_with_cited_rank: float
    best_matching_rank: int | None
    best_overlap: float
    unexpected_script: tuple[str, ...] = ()

    @property
    def failed(self) -> bool:
        return self.attribution in FAILING_ATTRIBUTIONS

    def as_dict(self) -> dict[str, Any]:
        """Identifiers and scores only; no segment text is retained."""
        return {
            "rank": self.rank,
            "marker_start": self.marker_start,
            "word_count": self.word_count,
            "attribution": self.attribution.value,
            "overlap_with_cited_rank": round(self.overlap_with_cited_rank, 4),
            "best_matching_rank": self.best_matching_rank,
            "best_overlap": round(self.best_overlap, 4),
            "unexpected_script": list(self.unexpected_script),
        }


@dataclass(frozen=True)
class CitationAudit:
    """Every citation claim in one response, plus the derived failure counts."""

    context_size: int
    segments: tuple[CitationSegment, ...]
    #: True when the response made an answer attempt but cited nothing at all.
    #: Untested at n=7 (all three answer attempts cited at least one rank), so
    #: it is enforced on principle - an uncitable answer is unverifiable - not
    #: on evidence.
    uncited_answer_attempt: bool
    min_verbatim_overlap: float = MIN_VERBATIM_OVERLAP

    @property
    def cited_ranks(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(segment.rank for segment in self.segments))

    @property
    def out_of_range_ranks(self) -> tuple[int, ...]:
        return tuple(
            dict.fromkeys(
                segment.rank
                for segment in self.segments
                if segment.attribution is Attribution.UNKNOWN_RANK
            )
        )

    @property
    def failing_segments(self) -> tuple[CitationSegment, ...]:
        return tuple(segment for segment in self.segments if segment.failed)

    @property
    def passed(self) -> bool:
        """No mechanically certain citation defect, and something was cited."""
        return not self.failing_segments and not self.uncited_answer_attempt

    def counts(self) -> dict[str, int]:
        tally = {attribution.value: 0 for attribution in Attribution}
        for segment in self.segments:
            tally[segment.attribution.value] += 1
        return tally

    def as_dict(self) -> dict[str, Any]:
        return {
            "context_size": self.context_size,
            "n_segments": len(self.segments),
            "cited_ranks": list(self.cited_ranks),
            "out_of_range_ranks": list(self.out_of_range_ranks),
            "uncited_answer_attempt": self.uncited_answer_attempt,
            "passed": self.passed,
            "min_verbatim_overlap": self.min_verbatim_overlap,
            "attribution_counts": self.counts(),
            "segments": [segment.as_dict() for segment in self.segments],
        }


def rendered_rank_texts(
    context_records: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Per-rank text exactly as the model saw it: provenance label plus clause.

    The label is included because the stored H01 response transcribes it
    (``[1] SS8 §2/4/4 occ=0 page=10 id=...``), so matching against
    ``record["text"]`` alone would under-count that segment's overlap.
    """
    return [
        f"{clause_label(record)}\n{record.get('text', '')}"
        for record in context_records
    ]


def split_citation_segments(text: str) -> list[tuple[int, int, str]]:
    """Split a response into ``(rank, marker_start, owned_text)`` triples.

    Each marker owns the text up to the next marker. Text before the first
    marker is not owned by any citation and is ignored here; the whole-response
    echo ratio in :mod:`aaoifi_rag.reliability.signals` covers it.
    """
    matches = list(CITATION_MARKER_RE.finditer(text or ""))
    segments: list[tuple[int, int, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segments.append((int(match.group(1)), match.start(), text[match.end() : end]))
    return segments


def audit_citations(
    response_text: str,
    context_records: Sequence[Mapping[str, Any]],
    *,
    is_answer_attempt: bool = True,
    min_verbatim_overlap: float = MIN_VERBATIM_OVERLAP,
    min_segment_words: int = MIN_SEGMENT_WORDS,
) -> CitationAudit:
    """Audit every ``[n]`` claim in ``response_text`` against the context block."""
    rank_texts = rendered_rank_texts(context_records)
    raw_segments = split_citation_segments(response_text or "")
    segments: list[CitationSegment] = []

    for rank, marker_start, owned in raw_segments:
        word_count = len(words(owned))
        script = unexpected_script_chars(owned)
        in_range = 1 <= rank <= len(rank_texts)

        overlaps = [
            verbatim_overlap_ratio(owned, rank_text) for rank_text in rank_texts
        ]
        cited_overlap = overlaps[rank - 1] if in_range else 0.0
        best_overlap = max(overlaps, default=0.0)
        best_rank = overlaps.index(best_overlap) + 1 if overlaps else None

        if not in_range:
            attribution = Attribution.UNKNOWN_RANK
        elif word_count < min_segment_words:
            attribution = Attribution.TOO_SHORT
        elif cited_overlap >= min_verbatim_overlap:
            attribution = Attribution.SUPPORTED
        elif best_overlap >= min_verbatim_overlap and best_rank != rank:
            attribution = Attribution.MISATTRIBUTED
        elif script and _matches_after_stripping(
            owned, rank_texts, rank, min_verbatim_overlap
        ):
            attribution = Attribution.CORRUPTED
        else:
            attribution = Attribution.UNVERIFIED

        segments.append(
            CitationSegment(
                rank=rank,
                marker_start=marker_start,
                word_count=word_count,
                attribution=attribution,
                overlap_with_cited_rank=cited_overlap,
                best_matching_rank=best_rank if best_overlap > 0.0 else None,
                best_overlap=best_overlap,
                unexpected_script=script,
            )
        )

    return CitationAudit(
        context_size=len(rank_texts),
        segments=tuple(segments),
        uncited_answer_attempt=is_answer_attempt and not segments,
        min_verbatim_overlap=min_verbatim_overlap,
    )


def _matches_after_stripping(
    owned: str,
    rank_texts: Sequence[str],
    rank: int,
    threshold: float,
) -> bool:
    """True when removing out-of-script characters makes the segment verbatim."""
    cleaned = strip_unexpected_script(owned)
    if 1 <= rank <= len(rank_texts):
        if verbatim_overlap_ratio(cleaned, rank_texts[rank - 1]) >= threshold:
            return True
    return any(
        verbatim_overlap_ratio(cleaned, rank_text) >= threshold
        for rank_text in rank_texts
    )
