"""Versioned prompt configurations for the prompt ablation study.

Each :class:`PromptConfig` encodes a specific hypothesis about what causes
Jais-2 over-abstention.  The configurations are:

v1 (baseline)
    Exact reproduction of the n=7 prompt that produced 57% abstention.
    Double abstention instruction (system + user).

v2 (single_abstention)
    Remove the user-turn abstention instruction, leaving only the
    system prompt.  **Hypothesis**: the duplicated instruction amplifies
    the model's refusal tendency beyond what a single instruction would.

v3 (synthesis_guidance)
    Replace the blanket abstention instruction in the user turn with
    explicit permission to synthesise an answer from partial evidence.
    **Hypothesis**: Jais-2 interprets "insufficient" too broadly and needs
    explicit guidance that partial but relevant evidence is sufficient.

v4 (specific_trigger)
    Replace the vague "insufficient" abstention trigger with a specific
    condition: abstain only if the excerpts contain no information at all
    about the topic.  **Hypothesis**: the abstention threshold in v1 is
    too vague, and a narrow trigger condition reduces false abstentions.

Design constraints
------------------
* ``ABSTENTION_LINE`` is constant across all variants.  The response
  classifier (:mod:`aaoifi_rag.reliability.response_class`) detects
  abstention via that exact string; changing it would break classification.
* ``build_context_text`` and ``clause_label`` are shared — only the system
  and user *framing* varies, not the context block format.
* Each config is frozen and serialisable so it can be recorded in traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .prompt import (
    ABSTENTION_LINE,
    AnswerPrompt,
    GoldAnswerLeak,
    assert_no_gold_answer_leak,
    build_context_text,
    clause_label,  # noqa: F401 — re-export for convenience
)

# ---------------------------------------------------------------------------
# Prompt variant definitions
# ---------------------------------------------------------------------------

# -- v1: exact reproduction of the n=7 baseline prompt ---------------------

_V1_VERSION = "answer_context_only_v1"

_V1_SYSTEM = (
    "You are answering a question using only the retrieved AAOIFI clause "
    "excerpts provided in the user message. Do not use any other knowledge. "
    "If the provided excerpts are insufficient to answer, reply exactly: "
    f"{ABSTENTION_LINE}"
)

_V1_USER_TEMPLATE = (
    "Retrieved clause excerpts (this is the only source you may use):\n\n"
    "{context}\n\n"
    "Question:\n{query}\n\n"
    "Answer only from the excerpts above. If they are insufficient, "
    f"reply exactly: {ABSTENTION_LINE}"
)

# -- v2: single abstention (remove user-turn duplication) ------------------

_V2_VERSION = "single_abstention_v2"

_V2_SYSTEM = _V1_SYSTEM  # keep system instruction unchanged

_V2_USER_TEMPLATE = (
    "Retrieved clause excerpts (this is the only source you may use):\n\n"
    "{context}\n\n"
    "Question:\n{query}\n\n"
    "Answer the question using only the excerpts above."
)

# -- v3: synthesis guidance ------------------------------------------------

_V3_VERSION = "synthesis_guidance_v3"

_V3_SYSTEM = (
    "You are an AAOIFI compliance assistant. Answer questions using only the "
    "retrieved clause excerpts provided in the user message. You may "
    "synthesise information across multiple excerpts to form a complete "
    "answer. Cite excerpts by their [n] reference numbers. "
    "If the excerpts contain no relevant information at all, reply exactly: "
    f"{ABSTENTION_LINE}"
)

_V3_USER_TEMPLATE = (
    "Retrieved clause excerpts (this is the only source you may use):\n\n"
    "{context}\n\n"
    "Question:\n{query}\n\n"
    "Use the excerpts above to answer. Even if only part of the answer is "
    "covered, provide what the excerpts support and cite the relevant "
    "excerpt numbers [n]. If the excerpts contain nothing relevant, reply "
    f"exactly: {ABSTENTION_LINE}"
)

# -- v4: specific abstention trigger ---------------------------------------

_V4_VERSION = "specific_trigger_v4"

_V4_SYSTEM = (
    "You are answering a question using only the retrieved AAOIFI clause "
    "excerpts provided in the user message. Do not use any other knowledge."
)

_V4_USER_TEMPLATE = (
    "Retrieved clause excerpts (this is the only source you may use):\n\n"
    "{context}\n\n"
    "Question:\n{query}\n\n"
    "Answer only from the excerpts above. Cite relevant excerpts by their "
    "[n] reference numbers.\n\n"
    "Abstain ONLY if NONE of the excerpts mention the topic of the question "
    "at all. In that case, reply exactly: "
    f"{ABSTENTION_LINE}"
)


# -- v5: strong guidance (synthesis + specific trigger + english only) --

_V5_VERSION = "strong_guidance_v5"

_V5_SYSTEM = (
    "You are an expert AAOIFI compliance assistant. Your task is to answer "
    "questions using ONLY the retrieved clause excerpts provided in the user "
    "message. Do not use any outside knowledge.\n\n"
    "RULES:\n"
    "1. You MUST answer entirely in English. Do not use Arabic script.\n"
    "2. You may synthesise information across multiple excerpts to form a complete answer.\n"
    "3. Cite excerpts by their [n] reference numbers.\n"
    "4. Abstain ONLY if the excerpts contain absolutely no relevant information "
    "about the topic of the question. If so, reply exactly: "
    f"{ABSTENTION_LINE}"
)

_V5_USER_TEMPLATE = (
    "Retrieved clause excerpts (this is the only source you may use):\n\n"
    "{context}\n\n"
    "Question:\n{query}\n\n"
    "Answer the question in English using only the excerpts above. "
    "Cite relevant excerpts by their [n] reference numbers.\n\n"
    "If the excerpts are completely irrelevant to the question, reply exactly: "
    f"{ABSTENTION_LINE}"
)


# ---------------------------------------------------------------------------
# PromptConfig dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptConfig:
    """A versioned, serialisable prompt configuration.

    Instances are constructed via the class methods :meth:`v1` through
    :meth:`v5`, not directly.  The version tag is recorded in every
    :class:`~aaoifi_rag.reporting.trace.AnswerTrace` so that results can be
    attributed to the exact prompt that produced them.
    """

    version: str
    system_prompt: str
    user_template: str

    # -- factories ---------------------------------------------------------

    @classmethod
    def v1(cls) -> "PromptConfig":
        """Exact n=7 baseline prompt (double abstention)."""
        return cls(_V1_VERSION, _V1_SYSTEM, _V1_USER_TEMPLATE)

    @classmethod
    def v2(cls) -> "PromptConfig":
        """Single abstention instruction (user-turn removed)."""
        return cls(_V2_VERSION, _V2_SYSTEM, _V2_USER_TEMPLATE)

    @classmethod
    def v3(cls) -> "PromptConfig":
        """Synthesis guidance with partial-evidence permission."""
        return cls(_V3_VERSION, _V3_SYSTEM, _V3_USER_TEMPLATE)

    @classmethod
    def v4(cls) -> "PromptConfig":
        """Specific abstention trigger (topic-absence only)."""
        return cls(_V4_VERSION, _V4_SYSTEM, _V4_USER_TEMPLATE)

    @classmethod
    def v5(cls) -> "PromptConfig":
        """Synthesis + specific trigger + English-only rule."""
        return cls(_V5_VERSION, _V5_SYSTEM, _V5_USER_TEMPLATE)

    @classmethod
    def by_name(cls, name: str) -> "PromptConfig":
        """Look up a prompt config by version name or short alias.

        Accepted names: ``v1``, ``v2``, ``v3``, ``v4``, ``v5`` or the full version
        string (e.g. ``answer_context_only_v1``).
        """
        _registry: dict[str, PromptConfig] = {
            "v1": cls.v1(),
            _V1_VERSION: cls.v1(),
            "v2": cls.v2(),
            _V2_VERSION: cls.v2(),
            "v3": cls.v3(),
            _V3_VERSION: cls.v3(),
            "v4": cls.v4(),
            _V4_VERSION: cls.v4(),
            "v5": cls.v5(),
            _V5_VERSION: cls.v5(),
        }
        config = _registry.get(name)
        if config is None:
            known = sorted({_V1_VERSION, _V2_VERSION, _V3_VERSION, _V4_VERSION, _V5_VERSION})
            raise ValueError(
                f"Unknown prompt config {name!r}. "
                f"Known: v1..v5 or {known}"
            )
        return config

    @classmethod
    def all_configs(cls) -> list["PromptConfig"]:
        """Return all registered configs in version order."""
        return [cls.v1(), cls.v2(), cls.v3(), cls.v4(), cls.v5()]

    # -- prompt building ---------------------------------------------------

    def build_user_prompt(
        self,
        query_text: str,
        context_records: Sequence[Mapping[str, Any]],
    ) -> str:
        """Render the user turn from the template."""
        return self.user_template.format(
            context=build_context_text(context_records),
            query=query_text,
        )

    def build_prompt(
        self,
        item_id: str,
        query_text: str,
        context_records: Sequence[Mapping[str, Any]],
        gold_answer: str | None = None,
    ) -> AnswerPrompt:
        """Build and leak-check the prompt for one item."""
        user_prompt = self.build_user_prompt(query_text, context_records)
        assert_no_gold_answer_leak(
            item_id, (user_prompt, self.system_prompt), gold_answer
        )
        return AnswerPrompt(
            item_id=item_id,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            prompt_version=self.version,
            context_size=len(context_records),
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialise for trace recording."""
        return {
            "version": self.version,
            "system_prompt": self.system_prompt,
            "user_template": self.user_template,
            "abstention_line": ABSTENTION_LINE,
        }
