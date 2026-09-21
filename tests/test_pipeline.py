"""The single-pass state machine and the answer-release rule (plan Layer 5).

The pipeline owns two things that nothing else in the project can own for it, and both
are the kind that fail silently:

* **Answer text is released only on ``ANSWER``.** ``served_answer`` returns ``None`` for
  ``ABSTAIN`` and ``ESCALATE`` even though ``generated_text`` is still populated for the
  reviewer. This one property is the difference between a system that is
  abstention-*aware* and one that is merely abstention-*shaped*: a caller that renders
  ``generated_text`` instead would show every escalated answer to the user while the
  metrics happily reported it as withheld. Asserted on all three decisions.
* **Generation is never attempted when retrieval is unusable.** The short-circuit is
  checked with a generator that raises if called, so "we skipped the model" is proved
  rather than inferred from a stage flag that the same code path writes.

The corpus-gated half then reproduces all seven published routing decisions - their
triggering gates and their full rationale strings - by driving ``AnswerPipeline`` itself
over the real corpus and the stored model responses. ``scripts/replay_n7_router.py``
produced the tracked artefact; this shows the artefact is a property of the *pipeline*
and not of the script's own wiring. It also records a real discrepancy it found while
doing so: see
``test_the_published_artefact_predates_the_anchoring_gate_but_no_decision_moved``.

One honest gap is pinned rather than papered over:
``test_the_pipeline_never_computes_the_anchoring_signal_so_that_gate_is_unreachable``.

No AAOIFI clause prose appears here. Clause identifiers and stored *decisions* do; the
corpus-gated tests read real prose at run time and never assert it as a literal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pytest

from aaoifi_rag.generation.prompt import (
    ABSTENTION_LINE,
    PROMPT_VERSION,
    GoldAnswerLeak,
)
from aaoifi_rag.generation.types import GeneratedAnswer, GenerationConfig
from aaoifi_rag.orchestration.pipeline import (
    CONTEXT_AFTER_RERANK_K,
    PIPELINE_VERSION,
    AnswerPipeline,
    PipelineResult,
    PipelineStage,
    StageRecord,
)
from aaoifi_rag.orchestration.protocols import RetrievalResult
from aaoifi_rag.reliability.policy import (
    GATES,
    PolicyConfig,
    RouteDecision,
    SelectivePredictionPolicy,
    describe_gates,
)

#: Invented prose. Long enough that a transcription of it exceeds the shingle window.
INVENTED_CLAUSE = (
    "The institution shall record the widget at the value agreed between the parties "
    "and shall disclose that value in the notes to the financial statements."
)

#: A paraphrase: cites rank 1, echoes no eight-word window, so every gate passes.
PARAPHRASE = "The institution records the widget at the agreed value [1]."

PUBLIC_TRACES = "reports/replay_n7_public_traces.jsonl"


# --------------------------------------------------------------------------------------
# Scripted backends
# --------------------------------------------------------------------------------------


class ScriptedRetriever:
    def __init__(self, records: Sequence[Mapping[str, Any]], **policy: Any) -> None:
        self._records = tuple(records)
        self.calls: list[tuple[str, int]] = []
        self._policy = policy or {"scripted": True}

    def retrieve(self, query_text: str, k: int) -> RetrievalResult:
        self.calls.append((query_text, k))
        return RetrievalResult(
            records=self._records[:k],
            reranker_signal=None,
            retrieval_policy={**self._policy, "requested_k": k},
        )


class ScriptedGenerator:
    def __init__(self, text: str) -> None:
        self._text = text
        self.prompts: list[Any] = []

    def generate(self, prompt: Any) -> GeneratedAnswer:
        self.prompts.append(prompt)
        return GeneratedAnswer(
            text=self._text,
            config=GenerationConfig(),
            input_token_count=len(prompt.user_prompt.split()),
            new_tokens=len(self._text.split()),
            generation_seconds=0.0,
            truncated=False,
            extra={"source": "scripted_test_generator"},
        )


class ExplodingGenerator:
    """Proves a code path did not reach generation, rather than trusting a flag."""

    def generate(self, prompt: Any) -> GeneratedAnswer:  # pragma: no cover
        raise AssertionError("the generator must not be called on this path")


@dataclass(frozen=True)
class StoredRerankerSignal:
    """Duck-typed stand-in. ``compute_retrieval_signals`` reads it via ``getattr``."""

    top_1_score: float | None
    top_2_score: float | None
    top_1_top_2_margin: float | None


def records(count: int = 2) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": f"clause:XX:1/{index}:occurrence:0",
            "standard_id": "XX",
            "clause_id": f"1/{index}",
            "sub_clause_id": None,
            "occurrence_index": 0,
            "source_page": 10 + index,
            "reranker_score": 5.5 - index,
            "retrieval_sources": ["bm25", "dense"],
            "text": f"{INVENTED_CLAUSE} Record {index}.",
        }
        for index in range(1, count + 1)
    ]


def run_with(response: str, context: Sequence[Mapping[str, Any]] | None = None,
             **kwargs: Any) -> PipelineResult:
    context = records() if context is None else context
    pipeline = AnswerPipeline(
        retriever=ScriptedRetriever(context), generator=ScriptedGenerator(response)
    )
    return pipeline.run(item_id="T01", query_text="How is a widget recorded?", **kwargs)


@pytest.fixture
def answered() -> PipelineResult:
    return run_with(PARAPHRASE)


@pytest.fixture
def abstained() -> PipelineResult:
    return run_with(ABSTENTION_LINE)


@pytest.fixture
def escalated() -> PipelineResult:
    """A transcription of rank 1: echo 0.95, above the 0.85 threshold."""
    return run_with(f"[1] {records()[0]['text']}")


# --------------------------------------------------------------------------------------
# The release rule
# --------------------------------------------------------------------------------------


def test_an_answer_is_served_only_when_every_gate_passed(answered) -> None:
    assert answered.decision is RouteDecision.ANSWER
    assert answered.routing.triggering_gate is None
    assert answered.served_answer == PARAPHRASE
    assert answered.generated_text == PARAPHRASE


def test_an_abstention_serves_nothing_while_the_text_stays_in_the_result(
    abstained,
) -> None:
    assert abstained.decision is RouteDecision.ABSTAIN
    assert abstained.routing.triggering_gate == "model_did_not_abstain"
    assert abstained.served_answer is None
    assert abstained.generated_text == ABSTENTION_LINE, "retained for the trace"


def test_an_escalation_serves_nothing_while_the_text_stays_for_the_reviewer(
    escalated,
) -> None:
    """The load-bearing case. There *is* an answer here, and it must not be shown.

    ``ESCALATE`` means an answer exists and failed a gate - which is precisely when a
    naive implementation leaks it, because ``generated_text`` is not empty and the
    decision is not ``ABSTAIN``.
    """
    assert escalated.decision is RouteDecision.ESCALATE
    assert escalated.routing.triggering_gate == "substantive_answer"
    assert escalated.served_answer is None
    assert escalated.generated_text is not None
    assert len(escalated.generated_text) > 0


def test_the_release_rule_holds_across_every_decision_in_one_place(
    answered, abstained, escalated
) -> None:
    """Written as an exhaustive sweep so a fourth decision cannot be added untested."""
    served = {
        result.decision: result.served_answer is not None
        for result in (answered, abstained, escalated)
    }
    assert served == {
        RouteDecision.ANSWER: True,
        RouteDecision.ABSTAIN: False,
        RouteDecision.ESCALATE: False,
    }
    assert set(served) == set(RouteDecision), "every decision is covered above"


def test_a_result_with_no_generation_serves_and_reports_nothing(abstained) -> None:
    """``served_answer`` guards on ``generated is None`` as well as on the decision."""
    from dataclasses import replace

    forced = replace(abstained, generated=None)
    assert forced.generated_text is None
    assert forced.served_answer is None


# --------------------------------------------------------------------------------------
# The short-circuit
# --------------------------------------------------------------------------------------


def test_empty_retrieval_terminates_before_generation_is_attempted() -> None:
    pipeline = AnswerPipeline(
        retriever=ScriptedRetriever([]), generator=ExplodingGenerator()
    )
    result = pipeline.run(item_id="T02", query_text="anything")

    assert result.short_circuited is True
    assert result.decision is RouteDecision.ABSTAIN
    assert result.routing.triggering_gate == "retrieval_non_empty"
    assert result.routing.stage_scope == "retrieval"
    assert result.prompt is None and result.generated is None
    assert result.served_answer is None and result.generated_text is None


def test_the_skipped_stages_are_recorded_rather_than_omitted() -> None:
    """All six stages appear on every run, so a trace is comparable across paths.

    Omitting the skipped ones would make ``stage_seconds()`` a different shape depending
    on the outcome, and any aggregate over it would be silently ragged.
    """
    pipeline = AnswerPipeline(
        retriever=ScriptedRetriever([]), generator=ExplodingGenerator()
    )
    result = pipeline.run(item_id="T02", query_text="anything")

    assert [record.stage for record in result.stages] == list(PipelineStage)
    entered = {record.stage: record.entered for record in result.stages}
    assert entered[PipelineStage.RETRIEVE] is True
    assert entered[PipelineStage.GATE_RETRIEVAL] is True
    for stage in (
        PipelineStage.BUILD_PROMPT,
        PipelineStage.GENERATE,
        PipelineStage.SIGNALS,
        PipelineStage.ROUTE,
    ):
        assert entered[stage] is False
    skipped = [record for record in result.stages if not record.entered]
    assert all(record.detail == "skipped: retrieval gate terminated" for record in skipped)
    assert all(record.seconds == 0.0 for record in skipped)


def test_a_short_circuited_run_has_retrieval_signals_but_no_response_signals() -> None:
    """Which is what makes ``response_present`` distinguishable from "not attempted"."""
    pipeline = AnswerPipeline(
        retriever=ScriptedRetriever([]), generator=ExplodingGenerator()
    )
    result = pipeline.run(item_id="T02", query_text="anything")
    assert result.signals.retrieval.retrieved_count == 0
    assert result.signals.retrieval.is_empty is True
    assert result.signals.response is None
    assert result.cited_chunk_ids == []


def test_an_all_heading_context_also_short_circuits() -> None:
    """The second retrieval gate. Both n=7 items with a heading at rank 1 had prose
    below it, so this has never fired on real data - it is a prompt-contract gate."""
    headings = [
        {
            "chunk_id": f"clause:XX:{index}:occurrence:0",
            "standard_id": "XX",
            "clause_id": str(index),
            "occurrence_index": 0,
            "text": "Murabahah to the Purchase Orderer",
        }
        for index in (8, 9)
    ]
    pipeline = AnswerPipeline(
        retriever=ScriptedRetriever(headings), generator=ExplodingGenerator()
    )
    result = pipeline.run(item_id="T03", query_text="anything")
    assert result.short_circuited is True
    assert result.routing.triggering_gate == "retrieval_normative"
    assert result.decision is RouteDecision.ABSTAIN


def test_disabling_the_heading_gate_lets_a_heading_only_context_reach_the_model() -> None:
    """The switch is real, and its effect is the whole downstream path opening up."""
    headings = [
        {"chunk_id": "clause:XX:8:occurrence:0", "standard_id": "XX", "clause_id": "8",
         "occurrence_index": 0, "text": "Murabahah to the Purchase Orderer"}
    ]
    pipeline = AnswerPipeline(
        retriever=ScriptedRetriever(headings),
        generator=ScriptedGenerator(PARAPHRASE),
        policy=SelectivePredictionPolicy(PolicyConfig(enable_heading_only_gate=False)),
    )
    result = pipeline.run(item_id="T03", query_text="anything")
    assert result.short_circuited is False
    assert result.generated is not None


def test_a_full_run_enters_every_stage(answered) -> None:
    assert answered.short_circuited is False
    assert all(record.entered for record in answered.stages)
    assert [record.stage for record in answered.stages] == list(PipelineStage)
    assert answered.routing.stage_scope == "all"


# --------------------------------------------------------------------------------------
# cited_chunk_ids
# --------------------------------------------------------------------------------------


def test_citations_resolve_to_chunk_ids_in_citation_order() -> None:
    result = run_with("Second rank first [2], then the first [1].")
    assert result.cited_chunk_ids == [
        "clause:XX:1/2:occurrence:0",
        "clause:XX:1/1:occurrence:0",
    ]


def test_an_out_of_range_citation_is_dropped_rather_than_wrapped_or_raised() -> None:
    """A model citing ``[9]`` over five records must not resolve to *some* clause.

    Negative indexing would make ``[0]`` the last record, and an unguarded lookup would
    raise mid-trace. Both would be worse than reporting no chunk id: the audit already
    records the out-of-range rank as a citation *defect*, which is where that finding
    belongs.
    """
    result = run_with("As stated [9] and also [1].")
    assert result.signals.response is not None
    audit = result.signals.response.citation_audit
    assert 9 in audit.cited_ranks
    assert audit.out_of_range_ranks == (9,)
    assert result.cited_chunk_ids == ["clause:XX:1/1:occurrence:0"]
    assert result.decision is RouteDecision.ESCALATE
    assert result.routing.triggering_gate == "citation_integrity"


def test_a_zero_citation_is_dropped_rather_than_indexing_from_the_end() -> None:
    result = run_with("As stated [0].")
    assert result.cited_chunk_ids == []


def test_a_repeated_citation_yields_one_chunk_id() -> None:
    """``cited_ranks`` deduplicates, so the ids are distinct clauses, not mentions."""
    result = run_with("First [1], again [1], and once more [1].")
    assert result.cited_chunk_ids == ["clause:XX:1/1:occurrence:0"]


def test_an_uncited_answer_attempt_escalates_and_cites_nothing() -> None:
    """Untested at n=7 - all three answer attempts cited at least one rank - so this is
    enforced on the principle that an uncitable answer is unverifiable."""
    result = run_with("The institution records the widget at the agreed value.")
    assert result.cited_chunk_ids == []
    assert result.signals.response is not None
    assert result.signals.response.citation_audit.uncited_answer_attempt is True
    assert result.decision is RouteDecision.ESCALATE
    assert result.routing.triggering_gate == "citation_integrity"


# --------------------------------------------------------------------------------------
# The gold-answer leak guard
# --------------------------------------------------------------------------------------


def test_a_gold_answer_in_the_context_stops_the_run_before_the_model_is_called() -> None:
    """Refusing to generate is the only safe response: the measurement is already void.

    Reached inside ``BUILD_PROMPT``, which is why an ``ExplodingGenerator`` is used - the
    guard must fire before the model call, not after.
    """
    leaked = records()
    leaked[0]["text"] = "SECRET REFERENCE ANSWER: the widget is recorded at cost."
    pipeline = AnswerPipeline(
        retriever=ScriptedRetriever(leaked), generator=ExplodingGenerator()
    )
    with pytest.raises(GoldAnswerLeak, match="T04"):
        pipeline.run(
            item_id="T04",
            query_text="anything",
            gold_answer="SECRET REFERENCE ANSWER: the widget is recorded at cost.",
        )


def test_a_gold_answer_absent_from_the_prompt_passes_and_never_reaches_the_model() -> None:
    pipeline = AnswerPipeline(
        retriever=ScriptedRetriever(records()), generator=ScriptedGenerator(PARAPHRASE)
    )
    result = pipeline.run(
        item_id="T05",
        query_text="How is a widget recorded?",
        gold_answer="A wholly different reference answer about leasing.",
    )
    assert result.prompt is not None
    assert "wholly different reference answer" not in result.prompt.user_prompt
    assert "wholly different reference answer" not in result.prompt.system_prompt
    assert result.decision is RouteDecision.ANSWER


def test_the_build_prompt_stage_records_that_the_guard_ran(answered) -> None:
    """A trace reader must be able to see the guard passed, not assume it."""
    detail = next(
        record.detail
        for record in answered.stages
        if record.stage is PipelineStage.BUILD_PROMPT
    )
    assert "leak_guard=passed" in detail
    assert f"version={PROMPT_VERSION}" in detail
    assert "context_size=2" in detail


# --------------------------------------------------------------------------------------
# k, context_k and policy injection
# --------------------------------------------------------------------------------------


def test_the_default_context_width_is_the_stored_runs_five() -> None:
    assert CONTEXT_AFTER_RERANK_K == 5
    retriever = ScriptedRetriever(records(8))
    AnswerPipeline(retriever=retriever, generator=ScriptedGenerator(PARAPHRASE)).run(
        item_id="T06", query_text="q"
    )
    assert retriever.calls == [("q", 5)]


def test_a_per_call_k_overrides_the_configured_width() -> None:
    retriever = ScriptedRetriever(records(8))
    result = AnswerPipeline(
        retriever=retriever, generator=ScriptedGenerator(PARAPHRASE), context_k=5
    ).run(item_id="T06", query_text="q", k=3)
    assert retriever.calls == [("q", 3)]
    assert result.retrieval.size == 3
    assert result.prompt is not None and result.prompt.context_size == 3


def test_a_k_of_zero_is_honoured_rather_than_falling_back_to_the_default() -> None:
    """``k=0`` is a real request, and ``if k is None`` is what keeps it from becoming 5.

    Written because ``k or self.context_k`` is the tempting spelling and would silently
    turn a zero-retrieval experiment into a five-record one.
    """
    retriever = ScriptedRetriever(records(8))
    result = AnswerPipeline(
        retriever=retriever, generator=ExplodingGenerator()
    ).run(item_id="T06", query_text="q", k=0)
    assert retriever.calls == [("q", 0)]
    assert result.short_circuited is True


def test_the_pipeline_reports_the_policy_config_it_is_running(answered) -> None:
    pipeline = AnswerPipeline(
        retriever=ScriptedRetriever(records()), generator=ScriptedGenerator(PARAPHRASE)
    )
    assert pipeline.policy_config.policy_version == "gates_v1"
    assert pipeline.policy_config.max_echo_ratio == 0.85
    assert answered.routing.policy.policy_version == "gates_v1"


def test_an_injected_threshold_changes_the_decision_and_is_recorded_with_it(
    escalated,
) -> None:
    """Config travels inside the outcome, so a decision is never orphaned from its policy.

    The same response escalates at 0.85 and answers at 0.99, and both traces carry the
    threshold that produced them.
    """
    lenient = AnswerPipeline(
        retriever=ScriptedRetriever(records()),
        generator=ScriptedGenerator(f"[1] {records()[0]['text']}"),
        policy=SelectivePredictionPolicy(PolicyConfig(max_echo_ratio=0.99)),
    ).run(item_id="T01", query_text="How is a widget recorded?")

    assert escalated.decision is RouteDecision.ESCALATE
    assert escalated.routing.policy.max_echo_ratio == 0.85
    assert lenient.decision is RouteDecision.ANSWER
    assert lenient.routing.policy.max_echo_ratio == 0.99
    assert lenient.served_answer is not None


def test_disabling_the_echo_gate_records_it_as_disabled_not_as_passed(escalated) -> None:
    """A gate that never ran must not be indistinguishable from one that passed."""
    off = AnswerPipeline(
        retriever=ScriptedRetriever(records()),
        generator=ScriptedGenerator(f"[1] {records()[0]['text']}"),
        policy=SelectivePredictionPolicy(PolicyConfig(enable_echo_gate=False)),
    ).run(item_id="T01", query_text="How is a widget recorded?")
    gate = next(
        result for result in off.routing.gate_results if result.name == "substantive_answer"
    )
    assert gate.enabled is False
    assert gate.passed is True, "vacuous pass"
    assert gate.detail == "disabled by configuration"
    assert off.decision is RouteDecision.ANSWER


def test_the_requested_generation_config_is_the_pipelines_not_the_backends(answered) -> None:
    """Two configs exist deliberately: what was asked for, and what the backend reports."""
    pipeline = AnswerPipeline(
        retriever=ScriptedRetriever(records()),
        generator=ScriptedGenerator(PARAPHRASE),
        generation_config=GenerationConfig(seed=7, max_new_tokens=64),
    )
    assert pipeline.generation_config.seed == 7
    assert answered.generated is not None
    assert answered.generated.config.seed == 42, "the backend reports its own"


# --------------------------------------------------------------------------------------
# Stage records
# --------------------------------------------------------------------------------------


def test_stage_seconds_covers_every_stage_and_is_non_negative(answered) -> None:
    timings = answered.stage_seconds()
    assert list(timings) == [stage.value for stage in PipelineStage]
    assert all(value >= 0.0 for value in timings.values())


def test_the_stage_record_rounds_to_microseconds_for_a_stable_artefact() -> None:
    record = StageRecord(PipelineStage.RETRIEVE, True, 0.123456789, "detail")
    assert record.as_dict() == {
        "stage": "retrieve",
        "entered": True,
        "seconds": 0.123457,
        "detail": "detail",
    }


def test_the_signals_stage_detail_names_the_three_measurements_it_took(answered) -> None:
    detail = next(
        record.detail
        for record in answered.stages
        if record.stage is PipelineStage.SIGNALS
    )
    assert "class=answered" in detail
    assert "echo=0.0000" in detail
    assert "citations_passed=True" in detail


def test_the_version_constants_are_the_published_ones(answered) -> None:
    assert PIPELINE_VERSION == "single_pass_v1"
    assert answered.pipeline_version == "single_pass_v1"


def test_the_result_is_frozen(answered) -> None:
    with pytest.raises(Exception):
        answered.short_circuited = True  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# run_batch
# --------------------------------------------------------------------------------------


def test_run_batch_uses_the_hard_set_field_names_and_preserves_order() -> None:
    items = [
        {"item_id": "A", "question_text": "first?", "gold_answer": "ref A"},
        {"item_id": "B", "question_text": "second?", "gold_answer": "ref B"},
    ]
    results = AnswerPipeline(
        retriever=ScriptedRetriever(records()), generator=ScriptedGenerator(PARAPHRASE)
    ).run_batch(items)
    assert [result.item_id for result in results] == ["A", "B"]
    assert [result.query_text for result in results] == ["first?", "second?"]


def test_run_batch_tolerates_a_missing_gold_answer_but_not_a_missing_question() -> None:
    """The asymmetry is deliberate: the guard has nothing to check without a reference,
    but a missing question would silently generate against an empty query."""
    pipeline = AnswerPipeline(
        retriever=ScriptedRetriever(records()), generator=ScriptedGenerator(PARAPHRASE)
    )
    assert len(pipeline.run_batch([{"item_id": "A", "question_text": "q?"}])) == 1
    with pytest.raises(KeyError):
        pipeline.run_batch([{"item_id": "A"}])
    with pytest.raises(KeyError):
        pipeline.run_batch([{"question_text": "q?"}])


def test_run_batch_field_names_are_overridable_for_the_probe_file() -> None:
    """``data/probes/unanswerable_probes.jsonl`` keys items by ``probe_id``."""
    items = [{"probe_id": "P01", "question_text": "what does SS99 say?"}]
    results = AnswerPipeline(
        retriever=ScriptedRetriever(records()), generator=ScriptedGenerator(ABSTENTION_LINE)
    ).run_batch(items, id_field="probe_id")
    assert results[0].item_id == "P01"
    assert results[0].decision is RouteDecision.ABSTAIN


def test_run_batch_on_an_empty_list_returns_an_empty_list() -> None:
    pipeline = AnswerPipeline(
        retriever=ScriptedRetriever(records()), generator=ExplodingGenerator()
    )
    assert pipeline.run_batch([]) == []


# --------------------------------------------------------------------------------------
# A known gap, recorded rather than hidden
# --------------------------------------------------------------------------------------


def test_the_pipeline_never_computes_the_anchoring_signal_so_that_gate_is_unreachable() -> None:
    """``AnswerPipeline`` cannot evaluate ``lexical_anchoring``, even if it is enabled.

    ``run`` calls ``compute_retrieval_signals(records, reranker_signal)`` with no
    ``anchoring=`` argument, because computing it needs a ``CorpusVocabulary`` built over
    the whole clause corpus and the pipeline is not given one. So the signal is always
    ``None`` here, and the gate reads ``None`` as *not evaluable* - correctly, since
    abstaining on an absent measurement would be inventing evidence.

    The consequence is that turning ``enable_anchoring_gate`` on through this path buys
    nothing: the gate reports ``enabled=True, passed=True`` with detail "not evaluable"
    on every item. That is currently harmless - ``reports/anchoring_gate_calibration.md``
    measured AUC 0.513 against the non-circular probes and the gate ships disabled - but
    it is a wiring gap, not a design decision, and it must not be mistaken for evidence
    that anchoring was tried in the pipeline and did nothing. Closing it means injecting
    a vocabulary into ``AnswerPipeline``. ``tests/test_policy.py`` covers the gate's own
    logic, which is reachable through ``SelectivePredictionPolicy`` directly.
    """
    enabled = AnswerPipeline(
        retriever=ScriptedRetriever(records()),
        generator=ScriptedGenerator(PARAPHRASE),
        policy=SelectivePredictionPolicy(
            PolicyConfig(enable_anchoring_gate=True, min_anchor_coverage=0.25)
        ),
    ).run(item_id="T07", query_text="How is a widget recorded?")

    assert enabled.signals.retrieval.anchoring is None
    gate = next(
        result for result in enabled.routing.gate_results if result.name == "lexical_anchoring"
    )
    assert gate.enabled is True
    assert gate.passed is True
    assert gate.detail == "anchoring signal not computed; not evaluable"
    assert enabled.decision is RouteDecision.ANSWER


# --------------------------------------------------------------------------------------
# Corpus-gated: the seven published decisions, through the pipeline object
# --------------------------------------------------------------------------------------

#: decision, triggering gate. From ``reports/replay_n7_public_traces.jsonl``.
PUBLISHED_ROUTING: dict[str, tuple[str, str | None]] = {
    "H01": ("answer", None),
    "H02": ("abstain", "model_did_not_abstain"),
    "H03": ("escalate", "substantive_answer"),
    "H04": ("abstain", "model_did_not_abstain"),
    "H05": ("escalate", "no_self_contradiction"),
    "H06": ("abstain", "model_did_not_abstain"),
    "H07": ("abstain", "model_did_not_abstain"),
}


class ReplayRetriever:
    """Return the stored context for a named item. Mirrors the replay script's shape."""

    def __init__(self, contexts: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        self._contexts = contexts
        self._item_id: str | None = None

    def for_item(self, item_id: str) -> "ReplayRetriever":
        if item_id not in self._contexts:
            raise KeyError(item_id)
        self._item_id = item_id
        return self

    def retrieve(self, query_text: str, k: int) -> RetrievalResult:
        assert self._item_id is not None
        stored = self._contexts[self._item_id]
        scores = [record.get("reranker_score") for record in stored]
        return RetrievalResult(
            records=tuple(stored[:k]),
            reranker_signal=StoredRerankerSignal(
                top_1_score=scores[0] if scores else None,
                top_2_score=scores[1] if len(scores) > 1 else None,
                top_1_top_2_margin=(
                    scores[0] - scores[1] if len(scores) > 1 else None
                ),
            ),
            retrieval_policy={"source": "stored_colab_run"},
        )


class StoredResponseGenerator:
    def __init__(self, responses: Mapping[str, str]) -> None:
        self._responses = responses
        self._item_id: str | None = None

    def for_item(self, item_id: str) -> "StoredResponseGenerator":
        if item_id not in self._responses:
            raise KeyError(item_id)
        self._item_id = item_id
        return self

    def generate(self, prompt: Any) -> GeneratedAnswer:
        assert self._item_id is not None
        return GeneratedAnswer(
            text=self._responses[self._item_id],
            config=GenerationConfig(),
            extra={"source": "replayed_from_stored_run"},
        )


@pytest.fixture(scope="session")
def replayed(hard_set, stored_run, clause_chunks) -> dict[str, PipelineResult]:
    """Drive ``AnswerPipeline`` over the real corpus and the stored model responses.

    Context is rejoined from ``chunk_id`` to the private corpus rather than taken from
    the run artefact, so the clause *text* the pipeline measures against is the licensed
    source, not a copy. The merge order matches ``rejoin_context`` in
    ``scripts/replay_n7_router.py`` exactly - corpus record first, then the run's score
    and sources - because ``source_page`` reaches the prompt through ``clause_label``,
    and a different provenance header would change every echo ratio. Gold answers are
    passed so the leak guard runs on real data.
    """
    by_id = {row["item_id"]: row for row in hard_set}
    chunks = {chunk["chunk_id"]: chunk for chunk in clause_chunks}
    contexts = {
        row["item_id"]: [
            {
                **chunks[entry["chunk_id"]],
                "reranker_score": entry.get("reranker_score"),
                "retrieval_sources": entry.get("retrieval_sources"),
                "stored_rank": entry["rank"],
            }
            for entry in sorted(row["top5"], key=lambda item: item["rank"])
        ]
        for row in stored_run["items"]
    }
    responses = {row["item_id"]: row["model_response"] for row in stored_run["items"]}

    retriever = ReplayRetriever(contexts)
    generator = StoredResponseGenerator(responses)
    results: dict[str, PipelineResult] = {}
    for item_id in sorted(contexts):
        pipeline = AnswerPipeline(
            retriever=retriever.for_item(item_id),
            generator=generator.for_item(item_id),
        )
        item = by_id[item_id]
        results[item_id] = pipeline.run(
            item_id=item_id,
            query_text=item["question_text"],
            gold_answer=item.get("gold_answer"),
        )
    return results


@pytest.mark.requires_private_data
def test_the_pipeline_reproduces_every_published_decision_and_triggering_gate(
    replayed,
) -> None:
    observed = {
        item_id: (result.decision.value, result.routing.triggering_gate)
        for item_id, result in replayed.items()
    }
    assert observed == PUBLISHED_ROUTING


@pytest.mark.requires_private_data
def test_the_pipeline_agrees_with_the_tracked_artefact_field_for_field(
    repo_root, replayed
) -> None:
    """Decision, gate, rationale, stage scope and the response-side signals.

    The rationale string is included on purpose: it embeds the measured values that
    produced the decision (``echo_ratio=0.9000 threshold=0.85``), so matching it pins the
    arithmetic and not just the verdict.
    """
    published = {
        json.loads(line)["item_id"]: json.loads(line)
        for line in (repo_root / PUBLIC_TRACES).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert set(published) == set(replayed)
    for item_id, result in replayed.items():
        expected = published[item_id]["routing"]
        assert result.decision.value == expected["decision"], item_id
        assert result.routing.triggering_gate == expected["triggering_gate"], item_id
        assert result.routing.rationale == expected["rationale"], item_id
        assert result.routing.stage_scope == expected["stage_scope"], item_id

        signals = published[item_id]["signals"]["response"]
        assert result.signals.response is not None
        live = result.signals.response.as_dict()
        for key in (
            "response_class",
            "echo_ratio",
            "answer_word_count",
            "unexpected_script",
            "abstention_line_present",
            "residue_word_count",
        ):
            assert live[key] == signals[key], f"{item_id}.{key}"


@pytest.mark.requires_private_data
def test_only_h01_releases_an_answer_and_it_is_the_stored_response(
    replayed, stored_run
) -> None:
    """One served answer out of seven. The four abstentions and two escalations serve
    nothing, and their text is still present for a reviewer."""
    stored = {row["item_id"]: row["model_response"] for row in stored_run["items"]}
    served = {item_id for item_id, result in replayed.items() if result.served_answer}
    assert served == {"H01"}
    assert replayed["H01"].served_answer == stored["H01"]
    for item_id in ("H03", "H05"):
        assert replayed[item_id].decision is RouteDecision.ESCALATE
        assert replayed[item_id].served_answer is None
        assert replayed[item_id].generated_text == stored[item_id]


@pytest.mark.requires_private_data
def test_no_hard_set_item_short_circuits_so_all_seven_reached_the_model(replayed) -> None:
    """Both retrieval gates are prompt-contract gates: neither has fired on real data."""
    assert all(result.short_circuited is False for result in replayed.values())
    assert all(result.retrieval.size == 5 for result in replayed.values())
    assert all(
        result.prompt is not None and result.prompt.context_size == 5
        for result in replayed.values()
    )
    assert all(result.signals.response is not None for result in replayed.values())


@pytest.mark.requires_private_data
def test_no_gold_answer_leaks_into_any_of_the_seven_prompts(replayed, hard_set) -> None:
    """The guard ran on all seven with the real references - reaching a result proves it
    did not raise - and this checks the substring directly as well."""
    gold = {row["item_id"]: row["gold_answer"] for row in hard_set}
    for item_id, result in replayed.items():
        assert result.prompt is not None
        assert gold[item_id] and gold[item_id].strip(), f"{item_id} has a reference answer"
        assert gold[item_id] not in result.prompt.user_prompt, item_id
        assert gold[item_id] not in result.prompt.system_prompt, item_id


@pytest.mark.requires_private_data
def test_the_stored_reranker_scores_reach_the_signals_without_gating_anything(
    replayed, stored_run
) -> None:
    """Recorded, never gated - the n=7 analysis rejected top-1 as a predictor.

    H07 had the lowest top-1 and the only complete gold retrieval; H01 the highest with
    2/3. Keeping the number in the trace while refusing to gate on it is the honest
    position, and this asserts both halves.
    """
    top1 = {
        row["item_id"]: sorted(row["top5"], key=lambda entry: entry["rank"])[0][
            "reranker_score"
        ]
        for row in stored_run["items"]
    }
    for item_id, result in replayed.items():
        assert result.signals.retrieval.reranker_top_1 == top1[item_id]
        for name in ("reranker_top_1", "reranker_margin"):
            gate = next(g for g in result.routing.gate_results if g.name == name)
            assert gate.enabled is False
            assert gate.detail == "disabled by configuration"
    assert min(top1.values()) == top1["H07"], "lowest score, best retrieval"
    assert max(top1.values()) == top1["H01"]


@pytest.mark.requires_private_data
def test_the_published_artefact_matches_the_live_gate_set_exactly(
    repo_root, replayed
) -> None:
    """The tracked artefact and the live gate set agree gate-for-gate, no drift.

    ``reports/replay_n7_public_traces.jsonl`` was regenerated when the
    ``clause_grounding`` gate was enabled, so it now records the full live set: 13
    gates per item, 10 enabled. Pinning both counts here means that if anyone adds a
    further gate, enables ``lexical_anchoring``, or otherwise changes the set, this test
    fails and forces a deliberate regeneration rather than letting artefact and code
    drift apart.
    """
    published = [
        json.loads(line)
        for line in (repo_root / PUBLIC_TRACES).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in published:
        assert len(row["routing"]["gates"]) == 13
        assert sum(1 for gate in row["routing"]["gates"] if gate["enabled"]) == 10

    assert len(GATES) == 13
    assert sum(1 for gate in describe_gates() if gate["enabled"]) == 10
    live_names = {gate["name"] for gate in describe_gates()}
    published_names = {gate["name"] for gate in published[0]["routing"]["gates"]}
    assert live_names == published_names

    for item_id, result in replayed.items():
        assert len(result.routing.gate_results) == 13, item_id
        assert sum(1 for g in result.routing.gate_results if g.enabled) == 10, item_id
