"""Interfaces the orchestrator depends on (plan Layer 5).

Both real backends need resources the local sandbox does not have - a GPU for
the reranker and the generator, model downloads for both - so the pipeline talks
to protocols and never constructs a backend itself. Colab injects the real
implementations; :mod:`tests` injects scripted ones replayed from
``reports/e2e_batch_smoke_results.json``.

Nothing in this module imports torch or transformers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from ..retrieval.bge_reranker import RerankerSignal


@dataclass(frozen=True)
class RetrievalResult:
    """Ranked context for one query, plus how it was produced.

    ``records`` is in final rank order: index 0 is the excerpt the model will see
    as ``[1]``. Each record carries the clause-chunk fields
    (``chunk_id``, ``standard_id``, ``clause_id``, ``sub_clause_id``,
    ``occurrence_index``, ``source_page``, ``text``).
    """

    records: tuple[Mapping[str, Any], ...]
    reranker_signal: RerankerSignal | None = None
    #: Free-form description of the configuration used, recorded verbatim in the
    #: trace: candidate counts, fusion rule, whether reranking ran, index names.
    retrieval_policy: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.records, tuple):
            object.__setattr__(self, "records", tuple(self.records))

    @property
    def size(self) -> int:
        return len(self.records)

    def chunk_ids(self) -> list[str]:
        return [str(record.get("chunk_id")) for record in self.records]


@runtime_checkable
class Retriever(Protocol):
    """Anything that can produce ranked context for a query."""

    def retrieve(self, query_text: str, k: int) -> RetrievalResult:  # pragma: no cover
        ...


def as_retrieval_result(
    records: Sequence[Mapping[str, Any]],
    reranker_signal: RerankerSignal | None = None,
    **policy: Any,
) -> RetrievalResult:
    """Convenience wrapper for callers holding a plain list of records."""
    return RetrievalResult(
        records=tuple(records),
        reranker_signal=reranker_signal,
        retrieval_policy=dict(policy),
    )
