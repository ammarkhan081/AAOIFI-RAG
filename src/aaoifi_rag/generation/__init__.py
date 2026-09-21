"""Context-only answer generation (plan Layer 3).

Prompt construction and the generator interface live here. No module in this
package imports torch or transformers at import time.
"""

from .prompt import (
    ABSTENTION_LINE,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    AnswerPrompt,
    GoldAnswerLeak,
    assert_no_gold_answer_leak,
    build_answer_prompt,
    build_context_text,
    build_user_prompt,
    clause_label,
)
from .prompt_config import PromptConfig
from .types import GeneratedAnswer, GenerationConfig, Generator

__all__ = [
    "ABSTENTION_LINE",
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "AnswerPrompt",
    "GeneratedAnswer",
    "GenerationConfig",
    "Generator",
    "GoldAnswerLeak",
    "PromptConfig",
    "assert_no_gold_answer_leak",
    "build_answer_prompt",
    "build_context_text",
    "build_user_prompt",
    "clause_label",
]
