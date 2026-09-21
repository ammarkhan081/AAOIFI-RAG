"""Lexical anchoring: does the retrieved context actually mention what was asked?

Why this gate had to be invented
--------------------------------
The existing retrieval-stage gate ``retrieval_non_empty`` cannot fire on a
question the corpus does not cover. BM25 and a dense index both return *k*
records for **any** query - a question about deferred tax returns the five
least-bad clauses about something else, with a low score but a full context
block. So the negative class in ``data/probes/unanswerable_probes.jsonl`` would
be routed on the strength of the model's own hedging alone, and the retrieval
layer would contribute nothing to abstention. This module gives the retrieval
layer a signal it can contribute.

What it measures
----------------
An *anchor* is a query term the corpus treats as discriminative: high IDF against
the 362-chunk corpus, not a function word, long enough to be a term rather than
an artefact. Two ratios follow:

``context_anchor_coverage``
    Fraction of the query's anchors that appear anywhere in the retrieved
    context. Available at run time for any query, needs no corpus vocabulary at
    decision time, and is the quantity the gate actually reads.
``oov_ratio``
    Fraction of anchors absent from the whole corpus vocabulary. A diagnostic
    only - see the circularity warning below.

Honest limits, stated before any number is quoted
-------------------------------------------------
1. **This is lexical, not semantic.** A question that is genuinely outside the
   corpus but phrased entirely in in-corpus vocabulary ("what is the maximum
   permitted profit rate?") will score high coverage and pass. The gate raises
   the floor; it does not close the hole. Semantic absence detection needs the
   dense index and is a Stage 2 question.
2. **``oov_ratio`` on the ``term_absent_from_corpus`` probes is circular.** Those
   probes were *selected* by having a term with 0 corpus occurrences, so a
   detector keyed on 0-occurrence terms is guaranteed to fire. That number is not
   evidence and must never be reported as the gate's true-positive rate. The
   non-circular measurements are (a) the false-positive rate on the seven
   answerable hard-set items, which were authored independently against real
   clauses, and (b) the true-positive rate on the ``standard_absent_from_corpus``
   and ``clause_absent_from_standard`` probes, which were selected by membership
   criteria unrelated to term frequency.
3. **Coverage is undefined, not zero, for a query with no anchors.** Absence of
   evidence is not evidence of absence, so :data:`AnchoringSignal.context_anchor_coverage`
   is ``None`` in that case and the gate abstains from judging rather than firing.

This module contains no AAOIFI clause prose. It reads clause text to compute
document frequencies and never stores it; :meth:`AnchoringSignal.as_dict` emits
counts and ratios only, because the anchor terms come from the question and
``data/private/hard_set.jsonl`` is not public.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from .text_checks import normalise_for_match

#: Mirrors ``aaoifi_rag.retrieval.bm25_baseline.BM25_TOKEN_RE`` so anchoring sees
#: the same tokens BM25 scores on. Duplicated rather than imported on purpose: the
#: reliability layer must stay free of the ``rank_bm25`` dependency so it imports
#: without a GPU or an index. Behavioural sync is asserted in
#: ``tests/test_anchoring.py`` (the two pattern *strings* differ only in the order of
#: the two apostrophe characters inside a character class).
#:
#: Note what this does **not** do, because an earlier version of this comment claimed
#: the opposite: a slash numeral is *not* kept whole. Python's ``|`` is ordered and
#: ``[\w]+`` comes first, so at the ``2`` of ``2/4/2`` the first branch matches and
#: consumes a single digit; the ``\d+(?:/\d+)*`` branch is unreachable, since every
#: position where it could match is a position where ``[\w]+`` matches too. Clause
#: references therefore reach :func:`_is_candidate` as bare digits and are dropped.
#: BM25 inherits the same behaviour, so the two really are in sync - but neither keeps
#: ``9/2`` as one token. Left as-is deliberately: this is the tokeniser the shipped
#: BM25 index was built with and every published coverage number was measured under.
ANCHOR_TOKEN_RE = re.compile(r"[\w]+(?:['’][\w]+)?|\d+(?:/\d+)*")

#: English function words plus interrogatives and the verbs of a question frame.
#: Deliberately short. Domain-generic vocabulary ("contract", "institution") is
#: *not* listed here - it is removed by the corpus IDF floor instead, so the stop
#: list carries no judgement about which Islamic-finance terms matter.
STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after against all also am an and any are as at be because been
    before being below between both but by can cannot could did do does doing
    down during each few for from further had has have having he her here hers
    him his how i if in into is it its itself just me more most must my no nor
    not of off on once only or other others ought our out over own same shall she
    should so some such than that the their them then there these they this those
    through to too under until up very was we were what when where which while
    who whom why will with within would you your
    """.split()
)

#: Extended once, after the first calibration run showed which anchors were
#: systematically uncovered. Every term here is either fixed prompt boilerplate or
#: a question-frame directive, and each was uncovered in **both** classes - on
#: answerable items and probes alike - so removing them cannot pull the measurement
#: toward either label. ``aaoifi`` is the clearest case: it appears in
#: :data:`aaoifi_rag.generation.prompt.SYSTEM_PROMPT` for every single query and
#: essentially never in clause prose, so it contributed a constant penalty and zero
#: information. The directive verbs describe how to answer, not what the corpus
#: must contain.
FRAME_TERMS: frozenset[str] = frozenset(
    """
    aaoifi summarise summarize quote compare contrast describe explain outline
    state reconcile discuss identify please kindly
    """.split()
)



def anchor_tokens(text: str) -> list[str]:
    """Normalise then tokenise. Same folding as every other comparison here."""
    return ANCHOR_TOKEN_RE.findall(normalise_for_match(text))


#: Suffixes stripped by :func:`fold_suffix`, longest first. Crude by design: a
#: real stemmer would be another dependency and another thing to justify, and the
#: only question that matters is whether folding lowers the false-positive rate
#: on the seven answerable items. That is measured, not assumed - see
#: ``reports/anchoring_gate_calibration.md``.
_SUFFIXES: tuple[str, ...] = ("ies", "ing", "ed", "es", "s")


def fold_suffix(term: str, *, min_stem: int = 4) -> str:
    """Strip one plural/participle suffix when a long enough stem remains.

    ``assets -> asset``, ``leasing -> leas``, ``cities -> citi``, and ``fees -> fees``
    because stripping would leave a three-character stem. The ``ies`` branch
    re-appends a ``y``, so ``policies -> policy``, not ``polic``. The stem need not be
    a word: both sides of every comparison go through this function, so an ugly stem
    still matches consistently.
    """
    for suffix in _SUFFIXES:
        if term.endswith(suffix) and len(term) - len(suffix) >= min_stem:
            stem = term[: -len(suffix)]
            return stem + "y" if suffix == "ies" else stem
    return term


@dataclass(frozen=True)
class AnchorSpec:
    """How an anchor is chosen. Every threshold here is a knob, so it is recorded.

    ``min_idf`` is expressed in nats against the corpus size. At N=362 a term in
    36 chunks (10%) has idf ~2.29 and a term in 4 chunks ~4.29, so the default
    2.0 keeps anything rarer than roughly one chunk in eight.
    """

    min_idf: float = 2.0
    min_term_length: int = 4
    fold_suffixes: bool = True
    #: Numeric-only tokens are dropped; a bare ``5`` is noise. A token containing
    #: ``/`` is kept instead, and skips :attr:`min_term_length`, on the theory that a
    #: clause reference is among the most diagnostic tokens a question can carry.
    #:
    #: **This switch is unreachable through the public path and changes nothing.**
    #: :data:`ANCHOR_TOKEN_RE` never emits a token containing ``/`` (see the comment
    #: there), so :func:`_is_candidate` never reaches this branch when called from
    #: :func:`select_anchors`. It is exercised only by calling ``_is_candidate``
    #: directly, which ``tests/test_anchoring.py`` does in order to pin that fact.
    #: Retained rather than deleted because it documents the intent a future
    #: tokeniser change would implement, and because flipping it cannot silently
    #: alter any published number.
    keep_slash_numerals: bool = True
    #: Drop :data:`FRAME_TERMS`. Exposed as a switch so the calibration report can
    #: show the measurement with and without it rather than asserting it helps.
    drop_frame_terms: bool = True
    #: Record fields whose values form part of the context surface, because
    #: :func:`aaoifi_rag.generation.prompt.build_context_text` renders a provenance
    #: header above every excerpt: ``SS8 §2/4/2 occ=0 page=7 id=clause:SS8:...``.
    #: A question naming ``SS17`` is therefore anchored by what the model was actually
    #: shown, and measuring coverage against ``text`` alone under-counts it. (A clause
    #: *number* is not recovered by this: ``2/4/2`` is tokenised into bare digits on
    #: both sides and dropped - see :data:`ANCHOR_TOKEN_RE`. What the metadata surface
    #: restores is the standard id plus the literal words ``clause`` and ``occurrence``
    #: from ``chunk_id``.) Named here rather than importing the prompt builder, so the
    #: reliability layer keeps no dependency on the generation layer.
    context_metadata_keys: tuple[str, ...] = (
        "standard_id",
        "clause_id",
        "sub_clause_id",
        "chunk_id",
    )
    spec_version: str = "anchor_spec_v1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_idf": self.min_idf,
            "min_term_length": self.min_term_length,
            "fold_suffixes": self.fold_suffixes,
            "keep_slash_numerals": self.keep_slash_numerals,
            "drop_frame_terms": self.drop_frame_terms,
            "context_metadata_keys": list(self.context_metadata_keys),
            "spec_version": self.spec_version,
        }



class CorpusVocabulary:
    """Document frequencies over the clause corpus, built once and reused.

    "Document" is a clause chunk, matching :func:`aaoifi_rag.reporting.probes.term_occurrences`
    so an anchoring measurement and a probe's ``basis_evidence`` count the same
    universe. Holds only folded term -> count; no clause text is retained.
    """

    __slots__ = ("_df", "_n_documents", "_spec")

    def __init__(self, df: Mapping[str, int], n_documents: int, spec: AnchorSpec):
        if n_documents <= 0:
            raise ValueError("corpus vocabulary needs at least one document")
        self._df = dict(df)
        self._n_documents = n_documents
        self._spec = spec

    @classmethod
    def from_chunks(
        cls,
        chunks: Iterable[Mapping[str, Any]],
        spec: AnchorSpec | None = None,
        *,
        text_key: str = "text",
    ) -> "CorpusVocabulary":
        spec = spec or AnchorSpec()
        df: dict[str, int] = {}
        n_documents = 0
        for chunk in chunks:
            n_documents += 1
            terms = {
                _fold(token, spec)
                for token in anchor_tokens(str(chunk.get(text_key) or ""))
            }
            for term in terms:
                df[term] = df.get(term, 0) + 1
        return cls(df, n_documents, spec)

    @property
    def n_documents(self) -> int:
        return self._n_documents

    @property
    def n_terms(self) -> int:
        return len(self._df)

    @property
    def spec(self) -> AnchorSpec:
        return self._spec

    def document_frequency(self, term: str) -> int:
        return self._df.get(_fold(term, self._spec), 0)

    def idf(self, term: str) -> float:
        """``log(N / (1 + df))``. Out-of-vocabulary terms get the maximum value.

        The ``1 +`` keeps an unseen term finite so it can be compared against a
        threshold rather than special-cased, and makes the score monotone in df.
        """
        return math.log(self._n_documents / (1 + self.document_frequency(term)))

    def contains(self, term: str) -> bool:
        return self.document_frequency(term) > 0


def _fold(token: str, spec: AnchorSpec) -> str:
    return fold_suffix(token) if spec.fold_suffixes else token


def _is_candidate(token: str, spec: AnchorSpec) -> bool:
    """Shape test only. Nothing here consults the corpus."""
    if token in STOPWORDS:
        return False
    if spec.drop_frame_terms and token in FRAME_TERMS:
        return False
    if "/" in token:
        return spec.keep_slash_numerals
    if token.isdigit():
        return False
    return len(token) >= spec.min_term_length


@dataclass(frozen=True)
class AnchoringSignal:
    """Query-to-context lexical overlap on the terms that carry information.

    ``anchors``, ``oov_anchors`` and ``uncovered_anchors`` hold question-derived
    terms. :meth:`as_dict` omits them unless explicitly asked, because the public
    trace view is built for a repository that does not ship the hard set.
    """

    n_query_tokens: int
    anchors: tuple[str, ...]
    oov_anchors: tuple[str, ...]
    uncovered_anchors: tuple[str, ...]
    spec: AnchorSpec
    corpus_documents: int
    #: ``(anchor, idf)`` pairs, aligned in meaning with :attr:`anchors`. Carried so
    #: :attr:`idf_weighted_coverage` needs no second pass over the vocabulary and
    #: so a stored signal stays interpretable without it.
    anchor_idf: tuple[tuple[str, float], ...] = ()

    @property
    def n_anchors(self) -> int:
        return len(self.anchors)

    @property
    def context_anchor_coverage(self) -> float | None:
        """``None`` when the query has no anchors - undefined, not zero."""
        if not self.anchors:
            return None
        covered = self.n_anchors - len(self.uncovered_anchors)
        return covered / self.n_anchors

    @property
    def idf_weighted_coverage(self) -> float | None:
        """Coverage weighted by how diagnostic each anchor is. The gate reads this.

        Unweighted coverage treats "zakat" and "calculated" as equally important,
        so a question whose *topic* is absent but whose verbs are present scores a
        comfortable 0.5. Weighting by IDF is the standard IR correction: an
        out-of-vocabulary topic term carries the maximum weight ``log(N/1)`` and
        dominates, which is the behaviour a corpus-absence detector needs.
        """
        if not self.anchor_idf:
            return None
        total = sum(weight for _, weight in self.anchor_idf)
        if total <= 0:
            return None
        uncovered = set(self.uncovered_anchors)
        lost = sum(
            weight for term, weight in self.anchor_idf if term in uncovered
        )
        return (total - lost) / total

    @property
    def max_uncovered_idf(self) -> float | None:
        """IDF of the single most diagnostic uncovered anchor, or ``None``."""
        uncovered = set(self.uncovered_anchors)
        weights = [w for term, w in self.anchor_idf if term in uncovered]
        return max(weights) if weights else None


    @property
    def oov_ratio(self) -> float | None:
        if not self.anchors:
            return None
        return len(self.oov_anchors) / self.n_anchors

    @property
    def is_evaluable(self) -> bool:
        return bool(self.anchors)

    def as_dict(self, *, include_terms: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "n_query_tokens": self.n_query_tokens,
            "n_anchors": self.n_anchors,
            "n_oov_anchors": len(self.oov_anchors),
            "n_uncovered_anchors": len(self.uncovered_anchors),
            "context_anchor_coverage": self.context_anchor_coverage,
            "idf_weighted_coverage": self.idf_weighted_coverage,
            "max_uncovered_idf": self.max_uncovered_idf,
            "oov_ratio": self.oov_ratio,
            "is_evaluable": self.is_evaluable,
            "corpus_documents": self.corpus_documents,
            "spec": self.spec.as_dict(),
            "measurement_is_lexical_only": True,
            "oov_ratio_status": (
                "diagnostic_only: circular on term_absent_from_corpus probes, "
                "which were selected by having 0 corpus occurrences"
            ),
        }
        if include_terms:
            payload["anchors"] = list(self.anchors)
            payload["oov_anchors"] = list(self.oov_anchors)
            payload["uncovered_anchors"] = list(self.uncovered_anchors)
        return payload


def select_anchors(
    query_text: str,
    vocabulary: CorpusVocabulary,
    spec: AnchorSpec | None = None,
) -> tuple[list[str], int]:
    """Return ``(folded anchors in first-seen order, n_query_tokens)``.

    Deduplicated: a term repeated in the question is one anchor, so a wordy
    question cannot dilute its own coverage score.
    """
    spec = spec or vocabulary.spec
    tokens = anchor_tokens(query_text)
    anchors: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if not _is_candidate(token, spec):
            continue
        folded = _fold(token, spec)
        if folded in seen:
            continue
        if vocabulary.idf(folded) >= spec.min_idf:
            seen.add(folded)
            anchors.append(folded)
    return anchors, len(tokens)


def context_surface_terms(
    context_records: Sequence[Mapping[str, Any]],
    spec: AnchorSpec,
    *,
    text_key: str = "text",
) -> set[str]:
    """Every folded term on the context surface: clause text plus provenance ids.

    This is the fix for a measurement error the first calibration run exposed.
    Coverage was computed against ``text`` alone, so a standard id such as ``SS17``
    counted as uncovered for **every** item - answerable and probe alike - even though
    the prompt puts it in the header line above each excerpt. That depressed coverage
    uniformly and destroyed the discrimination the gate depends on. The identifiers
    are part of what the retrieval layer supplied, so they belong in the surface.

    It does *not* recover clause numbers, and never did: ``2/4/2`` is split into bare
    digits by :data:`ANCHOR_TOKEN_RE` and dropped before it can be an anchor, on the
    query side as well as here.
    """
    terms: set[str] = set()
    for record in context_records:
        surfaces = [str(record.get(text_key) or "")]
        surfaces.extend(
            str(record.get(key) or "") for key in spec.context_metadata_keys
        )
        for surface in surfaces:
            for token in anchor_tokens(surface):
                terms.add(_fold(token, spec))
    return terms


def compute_anchoring_signal(
    query_text: str,
    context_records: Sequence[Mapping[str, Any]],
    vocabulary: CorpusVocabulary,
    spec: AnchorSpec | None = None,
    *,
    text_key: str = "text",
) -> AnchoringSignal:
    """Measure how much of the question's distinctive vocabulary the context has.

    ``context_records`` is the ranked block as passed to the generator, so this is
    a property of what the model was actually shown - not of the whole corpus.
    """
    spec = spec or vocabulary.spec
    anchors, n_tokens = select_anchors(query_text, vocabulary, spec)
    context_terms = context_surface_terms(context_records, spec, text_key=text_key)
    return AnchoringSignal(
        n_query_tokens=n_tokens,
        anchors=tuple(anchors),
        oov_anchors=tuple(term for term in anchors if not vocabulary.contains(term)),
        uncovered_anchors=tuple(
            term for term in anchors if term not in context_terms
        ),
        spec=spec,
        corpus_documents=vocabulary.n_documents,
        anchor_idf=tuple((term, vocabulary.idf(term)) for term in anchors),
    )






