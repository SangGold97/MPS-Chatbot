"""vLLM OpenAI-compatible LLM client."""
from collections.abc import AsyncGenerator
from functools import lru_cache

from loguru import logger
from openai import AsyncOpenAI

from src.config.settings import get_settings


class LLMClient:
    """Async client for vLLM via OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize OpenAI async client for vLLM."""
        settings = get_settings()
        self.model = model or settings.vllm_model_name

        self._client = AsyncOpenAI(
            base_url=base_url or settings.vllm_base_url,
            api_key="EMPTY",
            timeout=180.0,
        )
        logger.info(f"LLMClient initialized: {base_url or settings.vllm_base_url}")

    async def chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Send a non-streaming chat completion request."""
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        return response.choices[0].message.content or ""

    async def stream_chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion tokens."""
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


@lru_cache
def get_llm_client() -> LLMClient:
    """Get cached LLM client singleton."""
    return LLMClient()
