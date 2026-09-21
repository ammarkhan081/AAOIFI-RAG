"""Single-pass answer-or-abstain-or-escalate pipeline (plan Layer 5).

A plain explicit Python state machine, as ``docs/open_questions.md`` requires -
no graph framework. Stages run in a fixed order and every transition is recorded:

1. ``RETRIEVE`` - ask the injected retriever for ranked context.
2. ``GATE_RETRIEVAL`` - run the retrieval-stage gates. If they object, the
   pipeline terminates here and **generation is never attempted**, which is both
   the correct behaviour with no usable evidence and a saved GPU call.
3. ``BUILD_PROMPT`` - build the context-only prompt and run the gold-answer leak
   guard.
4. ``GENERATE`` - one call to the injected generator.
5. ``SIGNALS`` - measure the response against the context it was given.
6. ``ROUTE`` - full gate set decides ``ANSWER`` / ``ABSTAIN`` / ``ESCALATE``.

Single pass, deliberately. The plan's bounded-retry configuration (D+) is
optional and out of Stage 1 scope, so there is no retry loop here. Adding one
would change what the recorded ``ABSTAIN`` rate means, so it must be an explicit
future configuration rather than a quiet default.

Answer text is released only on ``ANSWER``. On ``ESCALATE`` the generated text is
still retained inside the result for the reviewer, but
:attr:`PipelineResult.served_answer` is ``None``, so a caller that renders only
``served_answer`` cannot show ungated text by accident.

Nothing in this module imports torch or transformers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import time
from typing import Any, Mapping

from ..generation.prompt import AnswerPrompt, build_answer_prompt
from ..generation.prompt_config import PromptConfig
from ..generation.types import GeneratedAnswer, GenerationConfig, Generator
from ..reliability.policy import (
    PolicyConfig,
    RouteDecision,
    RoutingOutcome,
    SelectivePredictionPolicy,
)
from ..reliability.signals import (
    ReliabilitySignals,
    compute_response_signals,
    compute_retrieval_signals,
)
from .protocols import RetrievalResult, Retriever

#: Context size after reranking, matching the stored n=7 run
#: (``scripts/colab_e2e_batch_test.py``: ``CONTEXT_AFTER_RERANK_K = 5``).
#: Candidate pool sizes belong to the retriever, not the pipeline.
CONTEXT_AFTER_RERANK_K = 5

#: Bump when the stage sequence or the release rule changes.
PIPELINE_VERSION = "single_pass_v1"


class PipelineStage(StrEnum):
    """The fixed stage sequence. Recorded in order on every run."""

    RETRIEVE = "retrieve"
    GATE_RETRIEVAL = "gate_retrieval"
    BUILD_PROMPT = "build_prompt"
    GENERATE = "generate"
    SIGNALS = "signals"
    ROUTE = "route"


@dataclass(frozen=True)
class StageRecord:
    """One stage's outcome and wall-clock cost."""

    stage: PipelineStage
    entered: bool
    seconds: float
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "entered": self.entered,
            "seconds": round(self.seconds, 6),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PipelineResult:
    """Everything one item produced, decision included."""

    item_id: str
    query_text: str
    retrieval: RetrievalResult
    signals: ReliabilitySignals
    routing: RoutingOutcome
    stages: tuple[StageRecord, ...]
    prompt: AnswerPrompt | None = None
    generated: GeneratedAnswer | None = None
    pipeline_version: str = PIPELINE_VERSION
    #: Set when the pipeline stopped before generation.
    short_circuited: bool = False

    @property
    def decision(self) -> RouteDecision:
        return self.routing.decision

    @property
    def served_answer(self) -> str | None:
        """The answer text, and only when every gate passed."""
        if self.decision is not RouteDecision.ANSWER or self.generated is None:
            return None
        return self.generated.text

    @property
    def generated_text(self) -> str | None:
        """Raw text regardless of decision. For reviewers and traces only."""
        return self.generated.text if self.generated else None

    @property
    def cited_chunk_ids(self) -> list[str]:
        """Chunk ids the response actually cited, in citation order."""
        response = self.signals.response
        if response is None:
            return []
        records = self.retrieval.records
        ids: list[str] = []
        for rank in response.citation_audit.cited_ranks:
            if 1 <= rank <= len(records):
                ids.append(str(records[rank - 1].get("chunk_id")))
        return ids

    def stage_seconds(self) -> dict[str, float]:
        return {record.stage.value: round(record.seconds, 6) for record in self.stages}


class AnswerPipeline:
    """Wire retrieval, generation and the reliability policy into one call.

    The pipeline owns sequencing and the answer-release rule. It does not own the
    decision: :class:`~aaoifi_rag.reliability.policy.SelectivePredictionPolicy`
    does. Nor does it own retrieval or generation: both are injected.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        policy: SelectivePredictionPolicy | None = None,
        *,
        context_k: int = CONTEXT_AFTER_RERANK_K,
        generation_config: GenerationConfig | None = None,
        prompt_config: PromptConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.policy = policy or SelectivePredictionPolicy()
        self.context_k = context_k
        #: Recorded in traces. The generator is free to report its own config on
        #: each :class:`GeneratedAnswer`; this is the requested configuration.
        self.generation_config = generation_config or GenerationConfig()
        #: Optional configurable prompt. When ``None``, the pipeline uses the
        #: legacy :func:`build_answer_prompt` (v1) for full backward compat.
        self.prompt_config = prompt_config

    @property
    def policy_config(self) -> PolicyConfig:
        return self.policy.config

    def run(
        self,
        item_id: str,
        query_text: str,
        *,
        gold_answer: str | None = None,
        k: int | None = None,
    ) -> PipelineResult:
        """Run one item end to end.

        ``gold_answer`` is passed only to the leak guard: it is checked against
        the built prompt and never sent to the model. See
        :func:`aaoifi_rag.generation.prompt.assert_no_gold_answer_leak`.
        """
        stages: list[StageRecord] = []
        top_k = self.context_k if k is None else k

        started = time.perf_counter()
        retrieval = self.retriever.retrieve(query_text, top_k)
        stages.append(
            StageRecord(
                PipelineStage.RETRIEVE,
                True,
                time.perf_counter() - started,
                f"records={retrieval.size} k={top_k}",
            )
        )

        retrieval_signals = compute_retrieval_signals(
            retrieval.records, retrieval.reranker_signal
        )
        pre_signals = ReliabilitySignals(retrieval=retrieval_signals)

        started = time.perf_counter()
        pre_routing = self.policy.decide_pre_generation(pre_signals)
        stages.append(
            StageRecord(
                PipelineStage.GATE_RETRIEVAL,
                True,
                time.perf_counter() - started,
                f"decision={pre_routing.decision.value} gate={pre_routing.triggering_gate}",
            )
        )

        if pre_routing.decision is not RouteDecision.ANSWER:
            for stage in (
                PipelineStage.BUILD_PROMPT,
                PipelineStage.GENERATE,
                PipelineStage.SIGNALS,
                PipelineStage.ROUTE,
            ):
                stages.append(
                    StageRecord(stage, False, 0.0, "skipped: retrieval gate terminated")
                )
            return PipelineResult(
                item_id=item_id,
                query_text=query_text,
                retrieval=retrieval,
                signals=pre_signals,
                routing=pre_routing,
                stages=tuple(stages),
                short_circuited=True,
            )
        return self._generate_and_route(
            item_id, query_text, retrieval, retrieval_signals, gold_answer, stages
        )

    def _generate_and_route(
        self,
        item_id: str,
        query_text: str,
        retrieval: RetrievalResult,
        retrieval_signals: Any,
        gold_answer: str | None,
        stages: list[StageRecord],
    ) -> PipelineResult:
        """Stages 3-6. Split out so the short-circuit path stays readable."""
        started = time.perf_counter()
        if self.prompt_config is not None:
            prompt = self.prompt_config.build_prompt(
                item_id=item_id,
                query_text=query_text,
                context_records=retrieval.records,
                gold_answer=gold_answer,
            )
        else:
            prompt = build_answer_prompt(
                item_id=item_id,
                query_text=query_text,
                context_records=retrieval.records,
                gold_answer=gold_answer,
            )
        stages.append(
            StageRecord(
                PipelineStage.BUILD_PROMPT,
                True,
                time.perf_counter() - started,
                f"version={prompt.prompt_version} context_size={prompt.context_size} "
                f"leak_guard=passed",
            )
        )

        started = time.perf_counter()
        generated = self.generator.generate(prompt)
        stages.append(
            StageRecord(
                PipelineStage.GENERATE,
                True,
                time.perf_counter() - started,
                f"new_tokens={generated.new_tokens} truncated={generated.truncated}",
            )
        )

        started = time.perf_counter()
        response_signals = compute_response_signals(generated.text, retrieval.records)
        signals = ReliabilitySignals(
            retrieval=retrieval_signals, response=response_signals
        )
        stages.append(
            StageRecord(
                PipelineStage.SIGNALS,
                True,
                time.perf_counter() - started,
                f"class={response_signals.response_class.value} "
                f"echo={response_signals.echo_ratio:.4f} "
                f"citations_passed={response_signals.citation_audit.passed}",
            )
        )

        started = time.perf_counter()
        routing = self.policy.decide(signals)
        stages.append(
            StageRecord(
                PipelineStage.ROUTE,
                True,
                time.perf_counter() - started,
                f"decision={routing.decision.value} gate={routing.triggering_gate}",
            )
        )

        return PipelineResult(
            item_id=item_id,
            query_text=query_text,
            retrieval=retrieval,
            signals=signals,
            routing=routing,
            stages=tuple(stages),
            prompt=prompt,
            generated=generated,
        )

    def run_batch(
        self,
        items: list[Mapping[str, Any]],
        *,
        query_field: str = "question_text",
        id_field: str = "item_id",
        gold_field: str = "gold_answer",
    ) -> list[PipelineResult]:
        """Run each item in order.

        Field defaults are the hard-set field names from
        ``data/manifests/hard_set_schema.json``.
        """
        return [
            self.run(
                item_id=str(item[id_field]),
                query_text=str(item[query_field]),
                gold_answer=item.get(gold_field),
            )
            for item in items
        ]
