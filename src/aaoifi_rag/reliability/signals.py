"""Reliability signals for one answer attempt (plan Layer 4).

Signals are separated from the policy that consumes them
(:mod:`aaoifi_rag.reliability.policy`) so that every signal is recorded on every
item regardless of whether it currently gates anything. That separation is the
point: the n=7 analysis rejected the obvious retrieval-confidence signal, and
the honest response is to keep measuring it rather than to delete it or to
pretend it works.

Evidentiary status of each signal, from ``reports/reliability_signal_analysis_n7.md``
and ``reports/e2e_batch_smoke_results.json``:

* ``reranker_top_1`` - **known not predictive at n=7.** H07 had the lowest score
  (3.78) and was the only item with complete gold retrieval; H01 had the highest
  (10.10) with 2/3 golds. Recorded, never gates.
* ``reranker_margin`` - top-1 minus top-2. Never analysed against outcomes; the
  n=7 report lists it only as a future direction. Recorded, never gates by
  default.
* ``echo_ratio`` - directly diagnostic. H03 scores 0.90 and H05 0.72, and both
  were counted as answers by the stored run's whole-string rule.
* ``unexpected_script`` - fired on H05's Arabic substitutions for U+2019.
* ``citation_audit`` - fired on H05's ``[2]``, which transcribes rank 4.
* ``clause_grounding`` - clause *paths* named in the answer, resolved against the
  clause identities actually retrieved. Mechanically certain, so unlike
  ``anchoring`` there is nothing to calibrate. Fired on **none** of the seven
  stored responses: only H01 names any clause path, and both of its distinct
  references are grounded. The extractor's precision is what was measured
  instead - 36/36 on the corpus (``reports/clause_grounding_gate.md``).
* ``heading_like_top_1`` - fired on H02 and H07, whose rank 1 is the bare
  section heading ``SS9 §8`` (12 words, no sentence punctuation). Recorded as a
  retrieval-quality diagnostic; it does not gate, because the remaining ranks
  can still carry the answer.
* ``anchoring`` - **calibrated and rejected.** IDF-weighted query-to-context
  anchor coverage separates the seven answerable items from the 11 non-circular
  unanswerable probes at AUC 0.513, i.e. chance
  (``reports/anchoring_gate_calibration.md``). Recorded when a caller supplies a
  corpus vocabulary; the gate that reads it ships disabled.

Nothing here is calibrated uncertainty. No coverage or risk guarantee follows
from any of these numbers. This module contains no AAOIFI clause prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Sequence

from ..generation.prompt import build_context_text
from ..retrieval.bge_reranker import RerankerSignal
from .anchoring import AnchoringSignal
from .citations import CitationAudit, audit_citations
from .grounding import ClauseGroundingAudit, CorpusClauseIndex, audit_clause_grounding
from .response_class import ResponseClass, ResponseClassification, classify_response
from .text_checks import repeated_char_runs, unexpected_script_chars, verbatim_overlap_ratio, words

#: A record at or below this length with no sentence-final punctuation is
#: treated as a heading rather than normative text. Calibrated against the n=7
#: contexts: ``clause:SS9:8`` (12 words) and ``clause:SS17:5/1`` (4 words) are
#: headings; ``clause:SS17:5/2/2`` (58 words, no terminal period) is not.
HEADING_MAX_WORDS = 12
_TERMINAL_PUNCTUATION = (".", ";", ":", "?", "!")
_SENTENCE_TAIL_RE = re.compile(r"[.;:?!]\s*$")


def is_heading_like(record: Mapping[str, Any]) -> bool:
    """True when a record looks like a section heading, not normative text."""
    text = (record.get("text") or "").strip()
    if not text:
        return True
    if _SENTENCE_TAIL_RE.search(text):
        return False
    return len(words(text)) <= HEADING_MAX_WORDS


@dataclass(frozen=True)
class RetrievalSignals:
    """What retrieval alone says about the evidence, before generation."""

    retrieved_count: int
    distinct_standards: tuple[str, ...]
    heading_like_ranks: tuple[int, ...]
    #: Uncalibrated. See the module docstring: not predictive at n=7.
    reranker_top_1: float | None = None
    reranker_top_2: float | None = None
    reranker_margin: float | None = None
    #: Present only when the caller supplied a :class:`CorpusVocabulary`. ``None``
    #: means "not measured", which the anchoring gate treats as not-evaluable
    #: rather than as a failure.
    anchoring: AnchoringSignal | None = None

    @property
    def is_empty(self) -> bool:
        return self.retrieved_count == 0

    @property
    def heading_like_top_1(self) -> bool:
        return 1 in self.heading_like_ranks

    @property
    def all_heading_like(self) -> bool:
        return self.retrieved_count > 0 and len(self.heading_like_ranks) == self.retrieved_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "retrieved_count": self.retrieved_count,
            "distinct_standards": list(self.distinct_standards),
            "heading_like_ranks": list(self.heading_like_ranks),
            "heading_like_top_1": self.heading_like_top_1,
            "all_heading_like": self.all_heading_like,
            "reranker_top_1": self.reranker_top_1,
            "reranker_top_2": self.reranker_top_2,
            "reranker_margin": self.reranker_margin,
            "reranker_signal_status": (
                "recorded_not_gated: not predictive of retrieval completeness or "
                "response class at n=7 (reports/reliability_signal_analysis_n7.md)"
            ),
            "anchoring": self.anchoring.as_dict() if self.anchoring else None,
            "anchoring_status": (
                "recorded_not_gated: AUC 0.513 against the non-circular "
                "unanswerable probes, i.e. chance "
                "(reports/anchoring_gate_calibration.md)"
            ),
        }


def compute_retrieval_signals(
    context_records: Sequence[Mapping[str, Any]],
    reranker_signal: RerankerSignal | None = None,
    *,
    anchoring: AnchoringSignal | None = None,
) -> RetrievalSignals:
    """Derive retrieval-side signals from the ranked context block.

    ``anchoring`` is supplied by the caller rather than computed here, because it
    needs a :class:`~aaoifi_rag.reliability.anchoring.CorpusVocabulary` built over
    the whole clause corpus. Callers without one pass nothing and the field stays
    ``None``, which the anchoring gate reads as not-evaluable, not as a failure.
    """
    return RetrievalSignals(
        retrieved_count=len(context_records),
        distinct_standards=tuple(
            dict.fromkeys(
                str(record.get("standard_id"))
                for record in context_records
                if record.get("standard_id")
            )
        ),
        heading_like_ranks=tuple(
            rank
            for rank, record in enumerate(context_records, start=1)
            if is_heading_like(record)
        ),
        reranker_top_1=getattr(reranker_signal, "top_1_score", None),
        reranker_top_2=getattr(reranker_signal, "top_2_score", None),
        reranker_margin=getattr(reranker_signal, "top_1_top_2_margin", None),
        anchoring=anchoring,
    )


@dataclass(frozen=True)
class ResponseSignals:
    """Surface properties of the generated text, measured against its context."""

    classification: ResponseClassification
    citation_audit: CitationAudit
    #: Fraction of the response's 8-word windows that occur verbatim in the
    #: context block. 1.0 means transcription rather than answer.
    echo_ratio: float
    unexpected_script: tuple[str, ...]
    repeated_runs: tuple[str, ...]
    answer_word_count: int
    #: Clause references in the response, resolved against the context. Always
    #: computed - unlike :attr:`RetrievalSignals.anchoring` this needs no
    #: caller-supplied vocabulary, only the response and context already in hand.
    clause_grounding: ClauseGroundingAudit | None = None

    @property
    def response_class(self) -> ResponseClass:
        return self.classification.response_class

    @property
    def is_answer_attempt(self) -> bool:
        return self.classification.is_answer_attempt

    def as_dict(self) -> dict[str, Any]:
        """Scores and labels only. No response or clause text is retained."""
        return {
            "response_class": self.response_class.value,
            "response_class_rule": self.classification.rule,
            "abstention_line_present": self.classification.abstention_line_present,
            "abstention_line_is_whole_response": (
                self.classification.abstention_line_is_whole_response
            ),
            "residue_word_count": self.classification.residue_word_count,
            "answer_word_count": self.answer_word_count,
            "echo_ratio": round(self.echo_ratio, 4),
            "unexpected_script": list(self.unexpected_script),
            "repeated_char_runs": list(self.repeated_runs),
            "citation_audit": self.citation_audit.as_dict(),
            "clause_grounding": (
                self.clause_grounding.as_dict() if self.clause_grounding else None
            ),
        }


def compute_response_signals(
    response_text: str,
    context_records: Sequence[Mapping[str, Any]],
    *,
    classification: ResponseClassification | None = None,
    corpus_index: CorpusClauseIndex | None = None,
) -> ResponseSignals:
    """Derive response-side signals. Pure text measurement, no model calls.

    ``corpus_index`` refines the clause-grounding diagnosis and never changes its
    verdict, so omitting it is safe - see
    :mod:`aaoifi_rag.reliability.grounding`.
    """
    resolved = classification or classify_response(response_text)
    audit = audit_citations(
        response_text,
        context_records,
        is_answer_attempt=resolved.is_answer_attempt,
    )
    return ResponseSignals(
        classification=resolved,
        citation_audit=audit,
        echo_ratio=verbatim_overlap_ratio(
            response_text, build_context_text(context_records)
        ),
        unexpected_script=unexpected_script_chars(response_text),
        repeated_runs=repeated_char_runs(response_text),
        answer_word_count=len(words(response_text)),
        clause_grounding=audit_clause_grounding(
            response_text, context_records, corpus_index=corpus_index
        ),
    )


@dataclass(frozen=True)
class ReliabilitySignals:
    """Everything the policy is allowed to look at for one item."""

    retrieval: RetrievalSignals
    response: ResponseSignals | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "retrieval": self.retrieval.as_dict(),
            "response": self.response.as_dict() if self.response else None,
            "extra": dict(self.extra),
        }


def compute_signals(
    context_records: Sequence[Mapping[str, Any]],
    response_text: str | None = None,
    *,
    reranker_signal: RerankerSignal | None = None,
    anchoring: AnchoringSignal | None = None,
    corpus_index: CorpusClauseIndex | None = None,
) -> ReliabilitySignals:
    """Compute retrieval signals, plus response signals when text is present."""
    retrieval = compute_retrieval_signals(
        context_records, reranker_signal, anchoring=anchoring
    )
    response = (
        compute_response_signals(
            response_text, context_records, corpus_index=corpus_index
        )
        if response_text is not None
        else None
    )
    return ReliabilitySignals(retrieval=retrieval, response=response)
