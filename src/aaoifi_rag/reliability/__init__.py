"""Reliability and selective prediction (plan Layer 4).

Import-safe without a GPU: no module here imports torch or transformers.

Layout:

* :mod:`~aaoifi_rag.reliability.text_checks` - deterministic string measures.
* :mod:`~aaoifi_rag.reliability.response_class` - surface form of a response.
* :mod:`~aaoifi_rag.reliability.citations` - citation index and attribution audit.
* :mod:`~aaoifi_rag.reliability.grounding` - clause-reference grounding: did the
  answer cite a clause it was actually shown? Mechanically certain, so unlike
  ``anchoring`` there is no threshold to calibrate.
* :mod:`~aaoifi_rag.reliability.anchoring` - query-to-context lexical anchoring.
  Calibrated against the 25-probe negative class and **rejected** (AUC 0.513);
  shipped implemented and disabled.
* :mod:`~aaoifi_rag.reliability.signals` - all signals, recorded whether or not
  they gate.
* :mod:`~aaoifi_rag.reliability.policy` - the gate set and the routing decision.

The policy deliberately contains no calibrated confidence score. See
:mod:`~aaoifi_rag.reliability.policy` for the evidentiary basis of each gate and
for the n=7 finding that ruled out reranker score as a selective-prediction
signal.
"""

from .anchoring import (
    ANCHOR_TOKEN_RE,
    FRAME_TERMS,
    STOPWORDS,
    AnchoringSignal,
    AnchorSpec,
    CorpusVocabulary,
    anchor_tokens,
    compute_anchoring_signal,
    context_surface_terms,
    fold_suffix,
    select_anchors,
)
from .citations import (
    Attribution,
    CitationAudit,
    CitationSegment,
    audit_citations,
    split_citation_segments,
)
from .grounding import (
    CLAUSE_REFERENCE_RE,
    FAILING_STATUSES,
    MIN_PATH_DEPTH,
    ClauseGroundingAudit,
    ClauseReference,
    CorpusClauseIndex,
    ReferenceStatus,
    audit_clause_grounding,
    clause_ancestors,
    extract_clause_references,
    normalise_standard,
)
from .policy import (
    GATES,
    EvidentiaryBasis,
    Gate,
    GateResult,
    GateStage,
    PolicyConfig,
    RouteDecision,
    RoutingOutcome,
    SelectivePredictionPolicy,
    describe_gates,
)
from .response_class import (
    ResponseClass,
    ResponseClassification,
    classify_response,
    classify_response_legacy,
)
from .sensitivity import (
    MAX_LATTICE_GATES,
    THRESHOLD_SIGNALS,
    AblationLattice,
    DecisionFlip,
    GateContribution,
    SensitivityItem,
    ThresholdSensitivity,
    ThresholdSweepPoint,
    compute_ablation_lattice,
    compute_gate_contributions,
    sweep_threshold,
)
from .signals import (
    ReliabilitySignals,
    ResponseSignals,
    RetrievalSignals,
    compute_response_signals,
    compute_retrieval_signals,
    compute_signals,
    is_heading_like,
)
from .text_checks import (
    normalise_for_match,
    strip_unexpected_script,
    unexpected_script_chars,
    verbatim_overlap_ratio,
)

__all__ = [
    "ANCHOR_TOKEN_RE",
    "AblationLattice",
    "AnchorSpec",
    "AnchoringSignal",
    "Attribution",
    "CLAUSE_REFERENCE_RE",
    "CitationAudit",
    "CitationSegment",
    "ClauseGroundingAudit",
    "ClauseReference",
    "CorpusClauseIndex",
    "CorpusVocabulary",
    "DecisionFlip",
    "EvidentiaryBasis",
    "FAILING_STATUSES",
    "FRAME_TERMS",
    "GATES",
    "Gate",
    "GateContribution",
    "GateResult",
    "GateStage",
    "MAX_LATTICE_GATES",
    "MIN_PATH_DEPTH",
    "PolicyConfig",
    "ReferenceStatus",
    "ReliabilitySignals",
    "ResponseClass",
    "ResponseClassification",
    "ResponseSignals",
    "RetrievalSignals",
    "RouteDecision",
    "RoutingOutcome",
    "STOPWORDS",
    "SelectivePredictionPolicy",
    "SensitivityItem",
    "THRESHOLD_SIGNALS",
    "ThresholdSensitivity",
    "ThresholdSweepPoint",
    "anchor_tokens",
    "audit_citations",
    "audit_clause_grounding",
    "classify_response",
    "classify_response_legacy",
    "clause_ancestors",
    "compute_ablation_lattice",
    "compute_anchoring_signal",
    "compute_gate_contributions",
    "compute_response_signals",
    "compute_retrieval_signals",
    "compute_signals",
    "context_surface_terms",
    "describe_gates",
    "extract_clause_references",
    "fold_suffix",
    "is_heading_like",
    "normalise_for_match",
    "normalise_standard",
    "select_anchors",
    "split_citation_segments",
    "strip_unexpected_script",
    "sweep_threshold",
    "unexpected_script_chars",
    "verbatim_overlap_ratio",
]
