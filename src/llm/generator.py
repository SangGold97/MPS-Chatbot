"""Answer Generator — builds final response with streaming."""
from collections.abc import AsyncGenerator
from functools import lru_cache
from typing import Any

from loguru import logger

from src.llm.client import get_llm_client

# System prompt for answer generation
ANSWER_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions about figures and documents.
Use the provided context (figures, knowledge base results, conversation history) \
to give accurate, detailed answers in the same language as the user's query.

RULES:
- If a figure image is provided, describe and analyze its visual content.
- If knowledge base context is provided, incorporate it into your answer.
- If both figure and context are provided, combine insights.
- If no additional context is available, answer based on conversation history.
- Be concise but thorough. Use bullet points when listing multiple items.
- Answer in the same language the user used (Vietnamese or English).
"""


class AnswerGenerator:
    """Generates final answers from aggregated context."""

    def __init__(self) -> None:
        """Initialize with LLM client singleton."""
        self._client = get_llm_client()
        logger.info("AnswerGenerator initialized")

    async def generate(
        self,
        aggregated_context: list[dict[str, Any]],
        max_tokens: int = 1024,
    ) -> str:
        """Generate a complete answer (non-streaming)."""
        return await self._client.chat(
            messages=aggregated_context, max_tokens=max_tokens,
        )

    async def stream_generate(
        self,
        aggregated_context: list[dict[str, Any]],
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """Stream answer tokens."""
        async for token in self._client.stream_chat(
            messages=aggregated_context, max_tokens=max_tokens,
        ):
            yield token


@lru_cache
def get_answer_generator() -> AnswerGenerator:
    """Get cached answer generator singleton."""
    return AnswerGenerator()
