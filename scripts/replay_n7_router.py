"""Replay the stored n=7 run through the real pipeline. GPU-free.

Why this exists
---------------
``scripts/colab_e2e_batch_test.py`` produced ``reports/e2e_batch_smoke_results.json``
on a GPU: retrieval, reranking and one Jais-2-8B generation per item. Those
generations are fixed and reproducible (greedy, seed 42), so the *decision* layer
can be re-derived from them locally without a GPU. This script does exactly that:

1. Rejoin each stored ``top5`` chunk_id against ``data/private/extracted/clause_chunks.jsonl``
   to recover the context block the model actually saw. The stored JSON carries no
   ``text`` field, by design.
2. Feed that context and the stored ``model_response`` through the real
   :class:`~aaoifi_rag.orchestration.pipeline.AnswerPipeline` via a replay
   retriever and a scripted generator, both satisfying the injected protocols.
3. Emit both trace views and the descriptive metrics.

So the numbers this prints are produced by the shipped modules over the real run,
not by a bespoke analysis path. What it does **not** do is re-run the model: it
cannot tell you what Jais-2 would say under a different prompt, and it is not a
substitute for the Colab batch test.

Requires ``data/private/``. Exits 2 with an explanation if the licensed corpus is
absent, so a clone without the data fails loudly rather than silently producing
empty metrics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aaoifi_rag.generation.types import GeneratedAnswer, GenerationConfig  # noqa: E402
from aaoifi_rag.generation.prompt import AnswerPrompt  # noqa: E402
from aaoifi_rag.orchestration import AnswerPipeline, RetrievalResult  # noqa: E402
from aaoifi_rag.reliability.policy import (  # noqa: E402
    PolicyConfig,
    SelectivePredictionPolicy,
)
from aaoifi_rag.reporting import (  # noqa: E402
    AnswerTrace,
    TraceWriter,
    build_evaluation_label,
    compute_routing_metrics,
    metrics_from_traces,
)

STORED_RESULTS = REPO_ROOT / "reports" / "e2e_batch_smoke_results.json"
CLAUSE_CHUNKS = REPO_ROOT / "data" / "private" / "extracted" / "clause_chunks.jsonl"
HARD_SET = REPO_ROOT / "data" / "private" / "hard_set.jsonl"
GATES_CONFIG = REPO_ROOT / "configs" / "reliability" / "gates_v1.json"

DEFAULT_FULL_OUT = REPO_ROOT / "runs" / "replay_n7" / "replay_n7_traces_full.jsonl"
DEFAULT_PUBLIC_OUT = REPO_ROOT / "reports" / "replay_n7_public_traces.jsonl"
DEFAULT_METRICS_OUT = REPO_ROOT / "reports" / "replay_n7_metrics.json"


class MissingLicensedData(SystemExit):
    """Raised as exit code 2 when ``data/private/`` is not present."""

    def __init__(self, path: Path) -> None:
        super().__init__(2)
        self.path = path


def _require(path: Path) -> Path:
    if not path.exists():
        print(
            f"missing required input: {path.relative_to(REPO_ROOT).as_posix()}\n"
            "This replay needs the licensed corpus and the stored Colab run. Both "
            "are Git-ignored; see data/README.md. Refusing to emit metrics from "
            "partial inputs.",
            file=sys.stderr,
        )
        raise MissingLicensedData(path)
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class ReplayRetriever:
    """Return the stored context block for a query. Satisfies ``Retriever``.

    Keyed on ``item_id`` rather than on query text, because the point is to
    reproduce the exact ranked block the model saw - not to re-run retrieval.
    Raises on an unknown item instead of returning an empty block, since an empty
    block would be silently routed ``ABSTAIN`` and look like a finding.
    """

    def __init__(
        self,
        contexts: Mapping[str, Sequence[Mapping[str, Any]]],
        retrieval_policy: Mapping[str, Any],
    ) -> None:
        self._contexts = contexts
        self._policy = dict(retrieval_policy)
        self._current_item: str | None = None

    def for_item(self, item_id: str) -> "ReplayRetriever":
        self._current_item = item_id
        return self

    def retrieve(self, query_text: str, k: int) -> RetrievalResult:
        if self._current_item is None:
            raise RuntimeError("call for_item() before retrieve()")
        records = self._contexts.get(self._current_item)
        if records is None:
            raise KeyError(f"no stored context for item {self._current_item!r}")
        return RetrievalResult(
            records=tuple(records[:k]),
            reranker_signal=_StoredRerankerSignal.from_records(records),
            retrieval_policy={
                **self._policy,
                "replayed_from": STORED_RESULTS.name,
                "requested_k": k,
            },
        )


class _StoredRerankerSignal:
    """The reranker scores the stored run recorded, in the shape signals expect."""

    __slots__ = ("top_1_score", "top_2_score", "top_1_top_2_margin")

    def __init__(self, top_1: float | None, top_2: float | None) -> None:
        self.top_1_score = top_1
        self.top_2_score = top_2
        self.top_1_top_2_margin = (
            None if top_1 is None or top_2 is None else top_1 - top_2
        )

    @classmethod
    def from_records(
        cls, records: Sequence[Mapping[str, Any]]
    ) -> "_StoredRerankerSignal":
        scores = [record.get("reranker_score") for record in records]
        return cls(
            scores[0] if len(scores) > 0 else None,
            scores[1] if len(scores) > 1 else None,
        )


class StoredResponseGenerator:
    """Replay the stored ``model_response``. Satisfies ``Generator``.

    ``generation_seconds`` and token counts are the stored GPU measurements, not
    this process's, and are labelled ``replayed`` in the trace so nobody reads
    them as local timings.
    """

    def __init__(
        self,
        responses: Mapping[str, str],
        telemetry: Mapping[str, Mapping[str, Any]],
        config: GenerationConfig,
    ) -> None:
        self._responses = responses
        self._telemetry = telemetry
        self._config = config
        self._current_item: str | None = None

    def for_item(self, item_id: str) -> "StoredResponseGenerator":
        self._current_item = item_id
        return self

    def generate(self, prompt: AnswerPrompt) -> GeneratedAnswer:
        item_id = self._current_item or prompt.item_id
        if item_id not in self._responses:
            raise KeyError(f"no stored response for item {item_id!r}")
        stored = self._telemetry.get(item_id, {})
        return GeneratedAnswer(
            text=self._responses[item_id],
            config=self._config,
            input_token_count=stored.get("input_token_count"),
            new_tokens=stored.get("new_tokens"),
            generation_seconds=stored.get("generation_seconds"),
            truncated=bool(
                stored.get("new_tokens") == self._config.max_new_tokens
            ),
            extra={
                "source": "replayed_from_stored_run",
                "timings_are_from": "colab_gpu_run_not_this_process",
                "vram_peak_generation_gb": stored.get("vram_peak_generation_gb"),
            },
        )


def load_inputs() -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Stored run, chunk_id -> clause record, and the hard set."""
    stored = json.loads(_require(STORED_RESULTS).read_text(encoding="utf-8"))
    chunks = {
        record["chunk_id"]: record for record in read_jsonl(_require(CLAUSE_CHUNKS))
    }
    hard_set = read_jsonl(_require(HARD_SET))
    return stored, chunks, hard_set


def rejoin_context(
    top5: Sequence[Mapping[str, Any]], chunks: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Recover the ranked context block, text included, in the stored rank order.

    Raises on an unresolvable ``chunk_id``: a silently dropped record would
    shorten the context and change both the echo ratio and the citation ranks.
    """
    records: list[dict[str, Any]] = []
    for entry in sorted(top5, key=lambda item: item["rank"]):
        chunk_id = entry["chunk_id"]
        source = chunks.get(chunk_id)
        if source is None:
            raise KeyError(
                f"chunk_id {chunk_id!r} from the stored run is not in "
                f"{CLAUSE_CHUNKS.name}; the corpus and the run have diverged"
            )
        records.append(
            {
                **source,
                "reranker_score": entry.get("reranker_score"),
                "retrieval_sources": entry.get("retrieval_sources"),
                "stored_rank": entry["rank"],
            }
        )
    return records


def environment_block(stored: Mapping[str, Any]) -> dict[str, Any]:
    """Provenance for the trace: where each number came from."""
    return {
        "replay_script": Path(__file__).name,
        "replay_is_gpu_free": True,
        "generation_source": "stored Colab run, replayed verbatim",
        "stored_run_label": stored.get("label"),
        "package_versions_at_generation": stored.get("package_versions"),
        "retrieval_policy_at_generation": stored.get("retrieval_policy"),
        "python": sys.version.split()[0],
    }


def replay(*, run_id: str) -> dict[str, Any]:
    """Run all stored items through the real pipeline. Writes nothing."""
    stored, chunks, hard_set = load_inputs()
    items = stored["items"]
    by_id = {item["item_id"]: item for item in items}
    hard_by_id = {item["item_id"]: item for item in hard_set}

    contexts = {
        item["item_id"]: rejoin_context(item["top5"], chunks) for item in items
    }
    responses = {item["item_id"]: item.get("model_response") or "" for item in items}
    telemetry = {
        item["item_id"]: {
            key: item.get(key)
            for key in (
                "input_token_count",
                "new_tokens",
                "generation_seconds",
                "vram_peak_generation_gb",
            )
        }
        for item in items
    }

    generation = stored.get("generation", {})
    config = GenerationConfig(
        model_id=generation.get("model_id", GenerationConfig.model_id),
        hub_revision=generation.get("hub_revision"),
        do_sample=bool(generation.get("do_sample", False)),
        max_new_tokens=int(generation.get("max_new_tokens", 512)),
    )

    retriever = ReplayRetriever(contexts, {"described_as": stored.get("retrieval_policy")})
    generator = StoredResponseGenerator(responses, telemetry, config)
    policy = SelectivePredictionPolicy(
        PolicyConfig.from_json_file(GATES_CONFIG)
        if GATES_CONFIG.exists()
        else PolicyConfig()
    )
    pipeline = AnswerPipeline(retriever, generator, policy, generation_config=config)

    environment = environment_block(stored)
    traces: list[AnswerTrace] = []
    labels = []
    gates: list[str | None] = []

    for item_id in sorted(by_id):
        hard_item = hard_by_id.get(item_id)
        if hard_item is None:
            raise KeyError(f"item {item_id!r} is in the stored run but not the hard set")
        retriever.for_item(item_id)
        generator.for_item(item_id)
        result = pipeline.run(
            item_id=item_id,
            query_text=str(hard_item["question_text"]),
            gold_answer=hard_item.get("gold_answer"),
        )
        label = build_evaluation_label(
            hard_item, result.decision.value, list(result.retrieval.records)
        )
        labels.append(label)
        gates.append(result.routing.triggering_gate)
        traces.append(
            AnswerTrace.from_pipeline_result(
                result, run_id=run_id, evaluation=label, environment=environment
            )
        )

    metrics = compute_routing_metrics(labels, triggering_gates=gates)
    return {
        "run_id": run_id,
        "traces": traces,
        "metrics": metrics,
        "labels": labels,
        "gates": gates,
    }


def print_summary(outcome: Mapping[str, Any]) -> None:
    """Per-item lines then the aggregate. Prints no clause or response text."""
    metrics = outcome["metrics"]
    print(f"run_id: {outcome['run_id']}")
    print(f"{'item':<5} {'decision':<9} {'expected':<9} {'gate':<24} recall")
    print("-" * 62)
    for label, gate in zip(outcome["labels"], outcome["gates"]):
        recall = label.retrieval.recall if label.retrieval else float("nan")
        hits = (
            f"{label.retrieval.n_gold_in_context}/{label.retrieval.n_gold}"
            if label.retrieval
            else "-"
        )
        print(
            f"{label.item_id:<5} {label.decision:<9} {label.expected_behavior:<9} "
            f"{(gate or '-'):<24} {recall:.3f} ({hits})"
        )
    print("-" * 62)
    payload = metrics.as_dict()
    print(
        f"answer {metrics.n_answer} / abstain {metrics.n_abstain} / "
        f"escalate {metrics.n_escalate}   coverage {metrics.coverage:.3f}"
    )
    print(
        f"routing agreement vs expected_behavior: "
        f"{metrics.n_decision_matches_expected}/{metrics.n_items} "
        f"({metrics.routing_agreement:.3f})"
    )
    print(
        f"clause recall@5  macro {payload['clause_recall']['macro']}  "
        f"micro {payload['clause_recall']['micro']} "
        f"({metrics.n_gold_in_context_total}/{metrics.n_gold_total})"
    )
    print(f"over-abstention (expected answer, did not answer): {metrics.n_over_abstention}")
    print(f"intervals: {payload['intervals']}")
    for name, reason in payload["not_computable"].items():
        print(f"not computable - {name}: {reason}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-id",
        default=None,
        help="Defaults to replay_n7_<UTC timestamp>.",
    )
    parser.add_argument(
        "--full-out",
        type=Path,
        default=DEFAULT_FULL_OUT,
        help=(
            "Full traces, containing verbatim clause text. Must be under a "
            "Git-ignored root; TraceWriter refuses otherwise."
        ),
    )
    parser.add_argument(
        "--public-out",
        type=Path,
        default=DEFAULT_PUBLIC_OUT,
        help="Redacted traces: identifiers, hashes, scores, decisions.",
    )
    parser.add_argument(
        "--metrics-out", type=Path, default=DEFAULT_METRICS_OUT
    )
    parser.add_argument(
        "--include-query-text",
        action="store_true",
        help=(
            "Include the authored question text in the public traces. Off by "
            "default because data/private/hard_set.jsonl is Git-ignored in full; "
            "the questions are ours, not AAOIFI's, so this is safe to enable once "
            "the hard set itself is released."
        ),
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the summary only. Writes nothing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = args.run_id or (
        "replay_n7_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    outcome = replay(run_id=run_id)
    print_summary(outcome)

    if args.no_write:
        print("\n--no-write: nothing written.")
        return 0

    writer = TraceWriter(
        full_path=args.full_out,
        public_path=args.public_out,
        include_query_text=args.include_query_text,
    )
    counts = writer.write(outcome["traces"])
    print(f"\nwrote {counts['full']} full traces -> {args.full_out}")
    print(f"wrote {counts['public']} public traces -> {args.public_out}")

    # Re-derive the metrics from the public traces alone. If the redacted view has
    # lost something the aggregates depend on, this disagrees and the run fails
    # here rather than in a paper.
    public_payloads = [
        trace.as_public_dict(include_query_text=args.include_query_text)
        for trace in outcome["traces"]
    ]
    from_public = metrics_from_traces(public_payloads)
    if from_public.as_dict() != outcome["metrics"].as_dict():
        print(
            "FAIL: metrics re-derived from the public traces differ from the "
            "metrics computed during the run.",
            file=sys.stderr,
        )
        return 1
    print("public traces reproduce every metric: confirmed")

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "source": "scripts/replay_n7_router.py",
                "replayed_from": STORED_RESULTS.name,
                "policy": outcome["traces"][0].routing["policy"],
                "environment": outcome["traces"][0].environment,
                **outcome["metrics"].as_dict(),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote metrics -> {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

