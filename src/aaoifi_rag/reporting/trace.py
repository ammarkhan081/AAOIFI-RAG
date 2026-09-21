"""Per-answer trace logging (plan Layer 6).

The plan requires a complete trace per answer: retrieved clauses, the exact
prompt, model version, seed, reliability scores, the routing decision and the
evaluation label. All of that is here.

The licensing boundary is enforced in code, not by convention
--------------------------------------------------------------
A complete trace necessarily contains verbatim AAOIFI clause text - inside the
retrieved records, inside the built prompt, and often inside the model's own
answer, since H03's entire response is a quoted clause. Per
``docs/governance/`` and ``data/README.md`` that text may not reach a public
repository. So every trace has two views:

* :meth:`AnswerTrace.as_full_dict` - includes clause text, prompt text and
  response text. Writable only under a Git-ignored root; :func:`assert_private_destination`
  refuses anything else.
* :meth:`AnswerTrace.as_public_dict` - identifiers, ranks, hashes, scores,
  decisions and labels. No clause text, no prompt text, no response text. This
  view is safe to publish, and the SHA-256 digests let anyone holding the
  licensed corpus verify it against the full trace.

Nothing in this module imports torch or transformers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..orchestration.pipeline import PipelineResult
from .evaluation import EvaluationLabel

#: Bump when the trace field set changes.
TRACE_SCHEMA_VERSION = "answer_trace_v1"

#: Repository-relative roots that ``.gitignore`` covers and that may therefore
#: hold verbatim clause text. This tuple is a *claim about another file*, so
#: ``tests/test_trace.py`` reads the real ``.gitignore`` and checks every entry
#: against it - a root dropped from ``.gitignore`` would otherwise leave the
#: guard below approving a tracked directory.
PRIVATE_ROOTS = ("runs", "artifacts", "logs", "data/private", "data/derived", "data/raw")

#: Length of the truncated digests used in the public view.
DIGEST_CHARS = 16


def text_digest(text: str | None) -> str | None:
    """Truncated SHA-256 of ``text``, or ``None``.

    Truncation is for readability. It is a verification aid for someone who
    already holds the licensed corpus, not a security control.
    """
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:DIGEST_CHARS]


class PublicationBoundaryError(RuntimeError):
    """Raised when a full trace would be written outside a Git-ignored root."""


def assert_private_destination(path: Path, repo_root: Path | None = None) -> Path:
    """Refuse to write clause-bearing output outside a Git-ignored root.

    ``repo_root`` defaults to the package's own repository root. A path outside
    the repository entirely is also refused: it cannot be checked against
    ``.gitignore``, and silently trusting it is how licensed text escapes.
    """
    root = (repo_root or _repo_root()).resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise PublicationBoundaryError(
            f"{resolved} is outside the repository ({root}); refusing to write a "
            "trace containing licensed clause text to an unverifiable location"
        ) from error
    parts = relative.as_posix()
    if not any(parts == root_name or parts.startswith(f"{root_name}/") for root_name in PRIVATE_ROOTS):
        raise PublicationBoundaryError(
            f"{relative.as_posix()} is not under a Git-ignored root "
            f"({', '.join(PRIVATE_ROOTS)}); use as_public_dict() to write there"
        )
    return resolved


def _repo_root() -> Path:
    """``src/aaoifi_rag/reporting/trace.py`` -> repository root."""
    return Path(__file__).resolve().parents[3]


def _public_record(rank: int, record: Mapping[str, Any]) -> dict[str, Any]:
    """Clause identity and provenance, with the text replaced by a digest."""
    text = record.get("text")
    return {
        "rank": rank,
        "chunk_id": record.get("chunk_id"),
        "standard_id": record.get("standard_id"),
        "clause_id": record.get("clause_id"),
        "sub_clause_id": record.get("sub_clause_id"),
        "occurrence_index": record.get("occurrence_index", 0),
        "source_page": record.get("source_page"),
        "reranker_score": record.get("reranker_score"),
        "retrieval_sources": record.get("retrieval_sources"),
        "text_sha256_16": text_digest(text),
        "text_char_count": len(text) if isinstance(text, str) else None,
    }


@dataclass(frozen=True)
class AnswerTrace:
    """One item's complete record, in two views.

    Built from a :class:`~aaoifi_rag.orchestration.pipeline.PipelineResult` by
    :meth:`from_pipeline_result`; the fields are kept plain so a trace can also
    be reconstructed from a stored run for replay.
    """

    run_id: str
    item_id: str
    created_at: str
    query_text: str
    context_records: tuple[Mapping[str, Any], ...]
    retrieval_policy: dict[str, Any]
    reranker_signal: dict[str, Any] | None
    system_prompt: str | None
    user_prompt: str | None
    prompt_version: str | None
    generation: dict[str, Any]
    response_text: str | None
    signals: dict[str, Any]
    routing: dict[str, Any]
    stages: tuple[dict[str, Any], ...]
    pipeline_version: str
    evaluation: dict[str, Any] | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    schema_version: str = TRACE_SCHEMA_VERSION
    #: Repeated on every trace so a stray file is still self-describing.
    verification_note: str = (
        "Hard-set labels are corpus_cross_reference only, NOT qualified Shari'ah "
        "scholar review. No field in this trace is a fatwa, Shari'ah ruling or "
        "compliance approval."
    )

    @classmethod
    def from_pipeline_result(
        cls,
        result: PipelineResult,
        *,
        run_id: str,
        evaluation: EvaluationLabel | None = None,
        environment: Mapping[str, Any] | None = None,
        created_at: str | None = None,
    ) -> "AnswerTrace":
        signal = result.retrieval.reranker_signal
        generation: dict[str, Any] = {}
        if result.generated is not None:
            generation = dict(result.generated.config.as_dict())
            generation.update(
                {
                    "input_token_count": result.generated.input_token_count,
                    "new_tokens": result.generated.new_tokens,
                    "generation_seconds": result.generated.generation_seconds,
                    "truncated": result.generated.truncated,
                    **dict(result.generated.extra),
                }
            )
        return cls(
            run_id=run_id,
            item_id=result.item_id,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
            query_text=result.query_text,
            context_records=tuple(result.retrieval.records),
            retrieval_policy=dict(result.retrieval.retrieval_policy),
            reranker_signal=(
                {
                    "top_1_score": signal.top_1_score,
                    "top_2_score": signal.top_2_score,
                    "top_1_top_2_margin": signal.top_1_top_2_margin,
                }
                if signal is not None
                else None
            ),
            system_prompt=result.prompt.system_prompt if result.prompt else None,
            user_prompt=result.prompt.user_prompt if result.prompt else None,
            prompt_version=result.prompt.prompt_version if result.prompt else None,
            generation=generation,
            response_text=result.generated_text,
            signals=result.signals.as_dict(),
            routing=result.routing.as_dict(),
            stages=tuple(record.as_dict() for record in result.stages),
            pipeline_version=result.pipeline_version,
            evaluation=evaluation.as_dict() if evaluation else None,
            environment=dict(environment or {}),
        )

    def as_public_dict(self, include_query_text: bool = True) -> dict[str, Any]:
        """Publishable view: identifiers, ranks, hashes, scores, decisions.

        Contains no clause text, no prompt text and no response text. The
        ``*_sha256_16`` digests let a holder of the licensed corpus verify this
        view against the full trace.

        ``question_text`` is authored by this project rather than taken from
        AAOIFI, so it is included by default. ``data/private/hard_set.jsonl`` is
        nonetheless Git-ignored in full, so set ``include_query_text=False`` when
        publishing traces before the hard set itself is released.
        """
        return {
            "schema_version": self.schema_version,
            "view": "public_no_clause_text",
            "run_id": self.run_id,
            "item_id": self.item_id,
            "created_at": self.created_at,
            "pipeline_version": self.pipeline_version,
            "query_text": self.query_text if include_query_text else None,
            "retrieval": {
                "policy": self.retrieval_policy,
                "reranker_signal": self.reranker_signal,
                "context_size": len(self.context_records),
                "records": [
                    _public_record(rank, record)
                    for rank, record in enumerate(self.context_records, start=1)
                ],
            },
            "prompt": {
                "prompt_version": self.prompt_version,
                "system_prompt_sha256_16": text_digest(self.system_prompt),
                "user_prompt_sha256_16": text_digest(self.user_prompt),
                "user_prompt_char_count": (
                    len(self.user_prompt) if self.user_prompt is not None else None
                ),
            },
            "generation": self.generation,
            "response": {
                "sha256_16": text_digest(self.response_text),
                "char_count": (
                    len(self.response_text) if self.response_text is not None else None
                ),
            },
            "signals": self.signals,
            "routing": self.routing,
            "evaluation": self.evaluation,
            "stages": list(self.stages),
            "environment": self.environment,
            "verification_note": self.verification_note,
        }

    def as_full_dict(self) -> dict[str, Any]:
        """Complete view, including licensed clause text. Private roots only."""
        payload = self.as_public_dict()
        payload["view"] = "full_contains_licensed_clause_text"
        payload["retrieval"]["records"] = [
            {**_public_record(rank, record), "text": record.get("text")}
            for rank, record in enumerate(self.context_records, start=1)
        ]
        payload["prompt"]["system_prompt"] = self.system_prompt
        payload["prompt"]["user_prompt"] = self.user_prompt
        payload["response"]["text"] = self.response_text
        return payload

    def query_text_only(self) -> str:
        """The authored question. Ours, not AAOIFI's, so it is safe to publish."""
        return self.query_text


class TraceWriter:
    """Write traces as JSONL, with the publication boundary enforced on open.

    Two destinations, deliberately asymmetric:

    * ``full_path`` must be under a Git-ignored root. :func:`assert_private_destination`
      raises :class:`PublicationBoundaryError` otherwise, so a mistyped path
      fails at construction rather than after licensed text has been written.
    * ``public_path`` may be anywhere; only the redacted view is written to it.
    """

    def __init__(
        self,
        full_path: Path | str | None = None,
        public_path: Path | str | None = None,
        *,
        repo_root: Path | None = None,
        include_query_text: bool = True,
    ) -> None:
        if full_path is None and public_path is None:
            raise ValueError("TraceWriter needs at least one destination")
        self.full_path = (
            assert_private_destination(Path(full_path), repo_root)
            if full_path is not None
            else None
        )
        self.public_path = Path(public_path).resolve() if public_path else None
        self.include_query_text = include_query_text
        self._counts = {"full": 0, "public": 0}

    def write(self, traces: Iterable[AnswerTrace]) -> dict[str, int]:
        """Write every trace to whichever destinations are configured."""
        materialised = list(traces)
        if self.full_path is not None:
            self._write_jsonl(
                self.full_path, (trace.as_full_dict() for trace in materialised)
            )
            self._counts["full"] += len(materialised)
        if self.public_path is not None:
            self._write_jsonl(
                self.public_path,
                (
                    trace.as_public_dict(include_query_text=self.include_query_text)
                    for trace in materialised
                ),
            )
            self._counts["public"] += len(materialised)
        return dict(self._counts)

    @staticmethod
    def _write_jsonl(path: Path, payloads: Iterable[Mapping[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for payload in payloads:
                stream.write(json.dumps(payload, ensure_ascii=False))
                stream.write("\n")


def read_traces(path: Path | str) -> list[dict[str, Any]]:
    """Read a trace JSONL file written by :class:`TraceWriter`."""
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def assert_no_clause_text(
    payloads: Sequence[Mapping[str, Any]],
    forbidden_snippets: Sequence[str],
) -> None:
    """Assert that no forbidden snippet appears anywhere in ``payloads``.

    Used by the tests to check the public view against real clause text drawn
    from the private corpus at test time, so the guarantee is verified rather
    than asserted. The snippets are supplied by the caller; none are stored here.
    """
    blob = json.dumps(list(payloads), ensure_ascii=False)
    leaked = [snippet for snippet in forbidden_snippets if snippet and snippet in blob]
    if leaked:
        raise PublicationBoundaryError(
            f"{len(leaked)} clause snippet(s) leaked into a public view"
        )
