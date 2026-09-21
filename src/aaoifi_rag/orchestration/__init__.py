"""Routing orchestration (plan Layer 5).

A plain explicit Python state machine wiring retrieval, generation and the
reliability policy into one callable pipeline. Backends are injected through the
protocols in :mod:`~aaoifi_rag.orchestration.protocols`, so the whole layer is
importable and testable without a GPU.

``RouteDecision`` is re-exported here for convenience but is defined with the
policy that produces it, in :mod:`aaoifi_rag.reliability.policy`.
"""

from ..reliability.policy import RouteDecision
from .pipeline import (
    CONTEXT_AFTER_RERANK_K,
    PIPELINE_VERSION,
    AnswerPipeline,
    PipelineResult,
    PipelineStage,
    StageRecord,
)
from .protocols import RetrievalResult, Retriever, as_retrieval_result

__all__ = [
    "CONTEXT_AFTER_RERANK_K",
    "PIPELINE_VERSION",
    "AnswerPipeline",
    "PipelineResult",
    "PipelineStage",
    "RetrievalResult",
    "Retriever",
    "RouteDecision",
    "StageRecord",
    "as_retrieval_result",
]
