"""Trace logging, evaluation labels and descriptive metrics (plan Layer 6).

Three concerns, deliberately separated:

* :mod:`~aaoifi_rag.reporting.trace` - the per-answer record, in a full view and
  a publishable view. The licensing boundary between them is enforced by
  :func:`~aaoifi_rag.reporting.trace.assert_private_destination`, not by
  convention.
* :mod:`~aaoifi_rag.reporting.evaluation` - clause identity, gold-clause recall
  and the reference labels a decision is compared against.
* :mod:`~aaoifi_rag.reporting.probes` - the mechanically labelled negative class,
  re-verified against the live corpus rather than trusted as a stored label.
* :mod:`~aaoifi_rag.reporting.metrics` - descriptive aggregates only. No Wilson
  interval is emitted below :data:`~aaoifi_rag.reporting.metrics.MIN_N_FOR_INTERVALS`;
  :func:`~aaoifi_rag.reporting.metrics.clopper_pearson_interval` computes at any n
  and reports its own uninformativeness instead. Metrics this evidence cannot
  support are reported as ``None`` with a reason.

Nothing here imports torch or transformers, and no module contains AAOIFI clause
prose.
"""

from .evaluation import (
    EXPECTED_BEHAVIOURS,
    VERIFICATION_BASES,
    ClauseKey,
    EvaluationLabel,
    GoldClauseHit,
    RetrievalScore,
    build_evaluation_label,
    clause_key,
    clause_keys,
    normalise_sub_clause_id,
    score_retrieval,
)
from .metrics import (
    ESCALATION_KEYS,
    EXPECTED_BEHAVIORS,
    MAX_INFORMATIVE_WIDTH,
    MIN_N_FOR_INTERVALS,
    NOT_COMPUTABLE,
    ExactInterval,
    InsufficientSampleError,
    RoutingMetrics,
    SelectiveRiskMetrics,
    clopper_pearson_interval,
    compute_routing_metrics,
    compute_selective_risk,
    metrics_from_traces,
    wilson_interval,
)
from .probes import (
    ESCALATION_STIPULATION,
    EXPECTED_BEHAVIOR_BY_BASIS,
    PROBE_SCHEMA_VERSION,
    VERIFICATION_BASIS_BY_BASIS,
    Probe,
    ProbeVerificationError,
    UnanswerableBasis,
    corpus_clause_ids,
    corpus_standards,
    load_probes,
    term_occurrences,
    verify_probe,
    verify_probes,
    write_probes,
)
from .trace import (
    DIGEST_CHARS,
    PRIVATE_ROOTS,
    TRACE_SCHEMA_VERSION,
    AnswerTrace,
    PublicationBoundaryError,
    TraceWriter,
    assert_no_clause_text,
    assert_private_destination,
    read_traces,
    text_digest,
)

__all__ = [
    "DIGEST_CHARS",
    "ESCALATION_KEYS",
    "ESCALATION_STIPULATION",
    "EXPECTED_BEHAVIORS",
    "EXPECTED_BEHAVIOURS",
    "EXPECTED_BEHAVIOR_BY_BASIS",
    "MAX_INFORMATIVE_WIDTH",
    "MIN_N_FOR_INTERVALS",
    "NOT_COMPUTABLE",
    "PRIVATE_ROOTS",
    "PROBE_SCHEMA_VERSION",
    "TRACE_SCHEMA_VERSION",
    "VERIFICATION_BASES",
    "VERIFICATION_BASIS_BY_BASIS",
    "AnswerTrace",
    "ClauseKey",
    "EvaluationLabel",
    "ExactInterval",
    "GoldClauseHit",
    "InsufficientSampleError",
    "Probe",
    "ProbeVerificationError",
    "PublicationBoundaryError",
    "RetrievalScore",
    "RoutingMetrics",
    "SelectiveRiskMetrics",
    "TraceWriter",
    "UnanswerableBasis",
    "assert_no_clause_text",
    "assert_private_destination",
    "build_evaluation_label",
    "clause_key",
    "clause_keys",
    "clopper_pearson_interval",
    "compute_routing_metrics",
    "compute_selective_risk",
    "corpus_clause_ids",
    "corpus_standards",
    "load_probes",
    "metrics_from_traces",
    "normalise_sub_clause_id",
    "read_traces",
    "score_retrieval",
    "term_occurrences",
    "text_digest",
    "verify_probe",
    "verify_probes",
    "wilson_interval",
    "write_probes",
]
