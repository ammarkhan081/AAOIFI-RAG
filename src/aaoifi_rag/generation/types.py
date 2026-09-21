"""Generator interface and result type (plan Layer 3).

The concrete Jais-2 generator needs a GPU, 4-bit bitsandbytes quantisation and
a Hugging Face token, none of which exist in the local sandbox. So the
generator is expressed here as a :class:`Generator` protocol plus a plain
result dataclass, following the deferred-import pattern already used by
``aaoifi_rag.retrieval.bge_reranker``: nothing in this module imports torch or
transformers, so the orchestration and reliability layers stay importable and
unit-testable without a GPU.

Colab supplies a real implementation; tests supply a scripted one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .prompt import AnswerPrompt


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding settings recorded verbatim in every trace.

    Defaults reproduce the n=7 batch run
    (``reports/e2e_batch_smoke_results.json`` -> ``generation``).
    """

    model_id: str = "inception42/Jais-2-8B-Chat"
    hub_revision: str | None = "da0e1639cd92b508b24120f7f77f5270a8465dc4"
    do_sample: bool = False
    max_new_tokens: int = 512
    seed: int = 42
    #: Set when a determinism-forcing env/flag combination was applied by the
    #: caller (``CUBLAS_WORKSPACE_CONFIG=:4096:8`` plus
    #: ``torch.use_deterministic_algorithms``). Recorded, not enforced here.
    deterministic_algorithms: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "hub_revision": self.hub_revision,
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "deterministic_algorithms": self.deterministic_algorithms,
        }


@dataclass(frozen=True)
class GeneratedAnswer:
    """Raw model output plus whatever cost metadata the backend exposed.

    ``text`` is stored exactly as returned. No stripping, normalisation or
    repair happens here; every judgement about the text belongs to
    :mod:`aaoifi_rag.reliability`.
    """

    text: str
    config: GenerationConfig
    input_token_count: int | None = None
    new_tokens: int | None = None
    generation_seconds: float | None = None
    truncated: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Generator(Protocol):
    """Anything that can turn a built prompt into text."""

    def generate(self, prompt: AnswerPrompt) -> GeneratedAnswer:  # pragma: no cover
        ...
