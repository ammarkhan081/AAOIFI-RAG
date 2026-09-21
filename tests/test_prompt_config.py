"""Tests for prompt configuration system (prompt ablation support).

Verifies:
1. All four prompt variants are constructible and serialisable.
2. Each variant preserves the exact ABSTENTION_LINE.
3. v1 produces prompts byte-identical to the legacy build_answer_prompt.
4. All variants pass the gold-answer leak guard.
5. by_name lookup works for short and full names.
6. Pipeline integration with PromptConfig produces correct prompt versions.
"""

from __future__ import annotations

import pytest

from aaoifi_rag.generation.prompt import (
    ABSTENTION_LINE,
    build_answer_prompt,
)
from aaoifi_rag.generation.prompt_config import PromptConfig


# -- Synthetic context (identical to conftest.py pattern) ------------------

def _make_context(n: int = 3) -> list[dict]:
    return [
        {
            "chunk_id": f"clause:XX{i}:1/{i}:occurrence:1",
            "standard_id": f"XX{i}",
            "clause_id": f"1/{i}",
            "sub_clause_id": None,
            "occurrence_index": 1,
            "page_number": 10 + i,
            "text": (
                f"The institution shall record the widget number {i} in its "
                f"books at the value agreed between the parties."
            ),
        }
        for i in range(1, n + 1)
    ]


QUERY = "What must the institution record in its books?"
ITEM_ID = "TEST_01"
CONTEXT = _make_context()


# =========================================================================
# Section 1: All variants constructible and serialisable
# =========================================================================

class TestPromptConfigConstruction:
    """Each factory method returns a frozen, serialisable config."""

    @pytest.mark.parametrize("factory", ["v1", "v2", "v3", "v4", "v5"])
    def test_factory_returns_frozen_config(self, factory: str) -> None:
        config = getattr(PromptConfig, factory)()
        assert isinstance(config, PromptConfig)
        assert config.version  # non-empty
        assert config.system_prompt  # non-empty
        assert config.user_template  # non-empty

    @pytest.mark.parametrize("factory", ["v1", "v2", "v3", "v4", "v5"])
    def test_as_dict_round_trip(self, factory: str) -> None:
        config = getattr(PromptConfig, factory)()
        d = config.as_dict()
        assert d["version"] == config.version
        assert d["abstention_line"] == ABSTENTION_LINE

    def test_all_configs_returns_five(self) -> None:
        configs = PromptConfig.all_configs()
        assert len(configs) == 5
        versions = [c.version for c in configs]
        assert len(set(versions)) == 5  # all distinct

    def test_versions_are_unique(self) -> None:
        versions = {c.version for c in PromptConfig.all_configs()}
        assert len(versions) == 5


# =========================================================================
# Section 2: Abstention line preservation
# =========================================================================

class TestAbstentionLinePreservation:
    """Every prompt variant must produce the exact ABSTENTION_LINE."""

    @pytest.mark.parametrize("factory", ["v1", "v2", "v3", "v4", "v5"])
    def test_abstention_line_in_system_or_user(self, factory: str) -> None:
        config = getattr(PromptConfig, factory)()
        prompt = config.build_prompt(ITEM_ID, QUERY, CONTEXT)
        # At least one of system/user must contain the exact abstention line
        assert (
            ABSTENTION_LINE in prompt.system_prompt
            or ABSTENTION_LINE in prompt.user_prompt
        ), f"{factory}: ABSTENTION_LINE not found in either prompt"


# =========================================================================
# Section 3: v1 backward compatibility
# =========================================================================

class TestV1BackwardCompatibility:
    """v1 must produce prompts identical to legacy build_answer_prompt."""

    def test_system_prompt_matches_legacy(self) -> None:
        v1 = PromptConfig.v1()
        legacy = build_answer_prompt(ITEM_ID, QUERY, CONTEXT)
        assert v1.system_prompt == legacy.system_prompt

    def test_user_prompt_matches_legacy(self) -> None:
        v1 = PromptConfig.v1()
        v1_prompt = v1.build_prompt(ITEM_ID, QUERY, CONTEXT)
        legacy = build_answer_prompt(ITEM_ID, QUERY, CONTEXT)
        assert v1_prompt.user_prompt == legacy.user_prompt

    def test_prompt_version_matches_legacy(self) -> None:
        v1 = PromptConfig.v1()
        v1_prompt = v1.build_prompt(ITEM_ID, QUERY, CONTEXT)
        legacy = build_answer_prompt(ITEM_ID, QUERY, CONTEXT)
        assert v1_prompt.prompt_version == legacy.prompt_version

    def test_context_size_matches_legacy(self) -> None:
        v1 = PromptConfig.v1()
        v1_prompt = v1.build_prompt(ITEM_ID, QUERY, CONTEXT)
        legacy = build_answer_prompt(ITEM_ID, QUERY, CONTEXT)
        assert v1_prompt.context_size == legacy.context_size


# =========================================================================
# Section 4: Prompt variant differentiation
# =========================================================================

class TestPromptDifferentiation:
    """Variants v2-v5 must actually differ from v1."""

    def test_v2_removes_user_abstention(self) -> None:
        v1 = PromptConfig.v1().build_prompt(ITEM_ID, QUERY, CONTEXT)
        v2 = PromptConfig.v2().build_prompt(ITEM_ID, QUERY, CONTEXT)
        # v1 user prompt has abstention, v2 should not
        assert ABSTENTION_LINE in v1.user_prompt
        assert ABSTENTION_LINE not in v2.user_prompt
        # But v2 system still has it
        assert ABSTENTION_LINE in v2.system_prompt

    def test_v3_has_synthesis_language(self) -> None:
        v3 = PromptConfig.v3().build_prompt(ITEM_ID, QUERY, CONTEXT)
        assert "synthesi" in v3.system_prompt.lower() or "synthesi" in v3.user_prompt.lower()

    def test_v4_has_specific_trigger(self) -> None:
        v4 = PromptConfig.v4().build_prompt(ITEM_ID, QUERY, CONTEXT)
        assert "none of the excerpts" in v4.user_prompt.lower()

    def test_v5_has_english_only_rule(self) -> None:
        v5 = PromptConfig.v5().build_prompt(ITEM_ID, QUERY, CONTEXT)
        assert "english" in v5.system_prompt.lower() or "english" in v5.user_prompt.lower()

    def test_all_variants_different_user_prompts(self) -> None:
        prompts = {
            f: getattr(PromptConfig, f)().build_prompt(ITEM_ID, QUERY, CONTEXT).user_prompt
            for f in ("v1", "v2", "v3", "v4", "v5")
        }
        # All user prompts must be distinct
        assert len(set(prompts.values())) == 5


# =========================================================================
# Section 5: Leak guard integration
# =========================================================================

class TestLeakGuard:
    """All variants must enforce the gold-answer leak guard."""

    @pytest.mark.parametrize("factory", ["v1", "v2", "v3", "v4", "v5"])
    def test_leak_guard_fires(self, factory: str) -> None:
        config = getattr(PromptConfig, factory)()
        # Use text that appears in a synthetic context record
        leaking_answer = "The institution shall record the widget number 1"
        from aaoifi_rag.generation.prompt import GoldAnswerLeak
        with pytest.raises(GoldAnswerLeak):
            config.build_prompt(ITEM_ID, QUERY, CONTEXT, gold_answer=leaking_answer)

    @pytest.mark.parametrize("factory", ["v1", "v2", "v3", "v4", "v5"])
    def test_no_leak_on_safe_answer(self, factory: str) -> None:
        config = getattr(PromptConfig, factory)()
        # Answer that does NOT appear in context
        safe_answer = "The answer is forty-two."
        prompt = config.build_prompt(ITEM_ID, QUERY, CONTEXT, gold_answer=safe_answer)
        assert prompt is not None


# =========================================================================
# Section 6: by_name lookup
# =========================================================================

class TestByNameLookup:
    """Prompt configs are discoverable by short and full name."""

    @pytest.mark.parametrize(
        "name,expected_version",
        [
            ("v1", "answer_context_only_v1"),
            ("v2", "single_abstention_v2"),
            ("v3", "synthesis_guidance_v3"),
            ("v4", "specific_trigger_v4"),
            ("v5", "strong_guidance_v5"),
            ("answer_context_only_v1", "answer_context_only_v1"),
            ("single_abstention_v2", "single_abstention_v2"),
            ("synthesis_guidance_v3", "synthesis_guidance_v3"),
            ("specific_trigger_v4", "specific_trigger_v4"),
            ("strong_guidance_v5", "strong_guidance_v5"),
        ],
    )
    def test_by_name_resolves(self, name: str, expected_version: str) -> None:
        config = PromptConfig.by_name(name)
        assert config.version == expected_version

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown prompt config"):
            PromptConfig.by_name("nonexistent")


# =========================================================================
# Section 7: Pipeline integration (PromptConfig wiring)
# =========================================================================

class TestPipelinePromptConfigIntegration:
    """AnswerPipeline uses PromptConfig when provided."""

    def test_pipeline_default_no_prompt_config(self) -> None:
        """Without prompt_config, pipeline.prompt_config is None."""
        from aaoifi_rag.orchestration.pipeline import AnswerPipeline

        class StubRetriever:
            def retrieve(self, query_text, k):
                from aaoifi_rag.orchestration.protocols import RetrievalResult
                return RetrievalResult(records=tuple(_make_context()))

        class StubGenerator:
            def generate(self, prompt):
                from aaoifi_rag.generation.types import GeneratedAnswer, GenerationConfig
                return GeneratedAnswer(text="stub", config=GenerationConfig())

        pipeline = AnswerPipeline(StubRetriever(), StubGenerator())
        assert pipeline.prompt_config is None

    def test_pipeline_with_prompt_config(self) -> None:
        """Pipeline stores and uses the provided PromptConfig."""
        from aaoifi_rag.orchestration.pipeline import AnswerPipeline

        class StubRetriever:
            def retrieve(self, query_text, k):
                from aaoifi_rag.orchestration.protocols import RetrievalResult
                return RetrievalResult(records=tuple(_make_context()))

        class StubGenerator:
            def generate(self, prompt):
                from aaoifi_rag.generation.types import GeneratedAnswer, GenerationConfig
                return GeneratedAnswer(text="stub answer", config=GenerationConfig())

        v3 = PromptConfig.v3()
        pipeline = AnswerPipeline(StubRetriever(), StubGenerator(), prompt_config=v3)
        assert pipeline.prompt_config is v3

        result = pipeline.run("T01", QUERY)
        assert result.prompt is not None
        assert result.prompt.prompt_version == "synthesis_guidance_v3"

    def test_pipeline_v1_config_matches_default(self) -> None:
        """Pipeline with PromptConfig.v1() produces same prompt as default."""
        from aaoifi_rag.orchestration.pipeline import AnswerPipeline

        class StubRetriever:
            def retrieve(self, query_text, k):
                from aaoifi_rag.orchestration.protocols import RetrievalResult
                return RetrievalResult(records=tuple(_make_context()))

        class StubGenerator:
            def generate(self, prompt):
                self.last_prompt = prompt
                from aaoifi_rag.generation.types import GeneratedAnswer, GenerationConfig
                return GeneratedAnswer(text="stub", config=GenerationConfig())

        gen_default = StubGenerator()
        gen_v1 = StubGenerator()

        pipeline_default = AnswerPipeline(StubRetriever(), gen_default)
        pipeline_v1 = AnswerPipeline(StubRetriever(), gen_v1, prompt_config=PromptConfig.v1())

        pipeline_default.run("T01", QUERY)
        pipeline_v1.run("T01", QUERY)

        assert gen_default.last_prompt.system_prompt == gen_v1.last_prompt.system_prompt
        assert gen_default.last_prompt.user_prompt == gen_v1.last_prompt.user_prompt
