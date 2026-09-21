"""Canonical context-only answer prompt (plan Layer 3).

This module is the single source of truth for the prompt that was used to
produce ``reports/e2e_batch_smoke_results.json``. The strings below are
reproduced verbatim from ``scripts/colab_e2e_batch_test.py`` so that the
stored n=7 results stay interpretable against this package; changing any of
them invalidates comparison with that run and requires a new prompt version.

Design constraints carried over from the plan and the governance docs:

* Context-only. The system prompt forbids outside knowledge, so an answer
  that is not supported by the supplied excerpts is a defect, not a
  variation.
* Ranks in the context are 1-indexed and stable. ``[1]`` .. ``[k]`` are the
  only citation targets the model is given, which is what makes mechanical
  citation-index checking possible downstream
  (:mod:`aaoifi_rag.reliability.citations`).
* No gold answer may reach the model. :func:`assert_no_gold_answer_leak`
  reproduces the guard that ran on every n=7 generation.

This module contains no AAOIFI clause prose. Clause text is supplied by the
caller at run time from the Git-ignored private corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

#: Prompt version recorded in traces. Bump on any change to the strings here.
PROMPT_VERSION = "answer_context_only_v1"

#: The exact abstention string the system prompt demands.
ABSTENTION_LINE = "I cannot answer from the given context"

SYSTEM_PROMPT = (
    "You are answering a question using only the retrieved AAOIFI clause "
    "excerpts provided in the user message. Do not use any other knowledge. "
    "If the provided excerpts are insufficient to answer, reply exactly: "
    f"{ABSTENTION_LINE}"
)


class GoldAnswerLeak(RuntimeError):
    """Raised when a reference answer would have been visible to the model."""


def clause_label(record: Mapping[str, Any]) -> str:
    """Render the one-line provenance header shown above each excerpt.

    Verbatim port of ``clause_label`` in ``scripts/colab_e2e_batch_test.py``.
    """
    sub = record.get("sub_clause_id")
    sub_bit = f"({sub})" if sub else ""
    return (
        f"{record['standard_id']} §{record['clause_id']}{sub_bit} "
        f"occ={record.get('occurrence_index', 0)} "
        f"page={record.get('source_page', '?')} "
        f"id={record.get('chunk_id', '?')}"
    )


def build_context_text(context_records: Sequence[Mapping[str, Any]]) -> str:
    """Join excerpts into the 1-indexed context block the model receives."""
    blocks = [
        f"[{rank}] {clause_label(record)}\n{record['text']}"
        for rank, record in enumerate(context_records, start=1)
    ]
    return "\n\n".join(blocks)


def build_user_prompt(
    query_text: str,
    context_records: Sequence[Mapping[str, Any]],
) -> str:
    """Build the user turn. Empty context yields an empty excerpt block.

    An empty context is deliberately *not* an error here: the orchestrator
    decides what to do about zero retrieval before generation is attempted
    (see :mod:`aaoifi_rag.orchestration.pipeline`).
    """
    return (
        "Retrieved clause excerpts (this is the only source you may use):\n\n"
        f"{build_context_text(context_records)}\n\n"
        f"Question:\n{query_text}\n\n"
        "Answer only from the excerpts above. If they are insufficient, reply exactly: "
        f"{ABSTENTION_LINE}"
    )


def assert_no_gold_answer_leak(
    item_id: str,
    prompts: Iterable[str],
    gold_answer: str | None,
) -> None:
    """Refuse to generate if the reference answer appears in any prompt.

    Mirrors the guard in ``scripts/colab_e2e_batch_test.py``. A blank or
    missing ``gold_answer`` is treated as "nothing to leak" rather than as a
    substring that trivially matches every prompt.
    """
    if not gold_answer or not gold_answer.strip():
        return
    for prompt in prompts:
        if gold_answer in prompt:
            raise GoldAnswerLeak(
                f"Refusing to generate for {item_id}: gold_answer leaked into the prompt."
            )


@dataclass(frozen=True)
class AnswerPrompt:
    """A fully-built, leak-checked prompt pair plus its version tag."""

    item_id: str
    system_prompt: str
    user_prompt: str
    prompt_version: str
    context_size: int

    def as_messages(self) -> list[dict[str, str]]:
        """Chat-template input, in the order the n=7 run used."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]


def build_answer_prompt(
    item_id: str,
    query_text: str,
    context_records: Sequence[Mapping[str, Any]],
    gold_answer: str | None = None,
) -> AnswerPrompt:
    """Build and leak-check the prompt for one item."""
    user_prompt = build_user_prompt(query_text, context_records)
    assert_no_gold_answer_leak(item_id, (user_prompt, SYSTEM_PROMPT), gold_answer)
    return AnswerPrompt(
        item_id=item_id,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        prompt_version=PROMPT_VERSION,
        context_size=len(context_records),
    )
