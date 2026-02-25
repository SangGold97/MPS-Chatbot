"""Router Agent — analyzes queries and decides tool actions."""
from functools import lru_cache

from langchain_openai import ChatOpenAI
from loguru import logger

from src.config.settings import get_settings
from src.models.schemas import RouterOutput


# --- System prompt with few-shot examples ---

ROUTER_SYSTEM_PROMPT = """\
You are a routing agent that analyzes user queries about figures and documents.

Given a query and conversation history, decide what actions are needed:
1. "get_figure" — When the query asks about visual content about figure.
2. "semantic_search" — When the query needs additional context about biology from knowledge base.
3. Both — When the query needs the figure AND additional biology context.
4. None — When you can answer directly from conversation history.

RULES:
- If the user explicitly mentions a figure (e.g., "figure scatter1"), use get_figure.
- If the user asks a general knowledge question, use semantic_search.
- If the user asks about a figure AND wants an explanation, use BOTH.
- If the user is just greeting or asking follow-up that you can answer from history, \
set ready_to_answer=true and skip actions.
- For get_figure: figure_id is the filename stem (e.g., "scatter1", "bar2").
- For semantic_search: rewrite the query to be suitable for vector search.

EXAMPLES:
Query: "Mô tả figure scatter1"
→ actions: [{{"type": "get_figure", "figure_id": "scatter1"}}], ready_to_answer: false

Query: "BCA là gì?"
→ actions: [{{"type": "semantic_search", "query": "BCA definition"}}], ready_to_answer: false

Query: "Giải thích figure bar2 và cho biết BCA liên quan"
→ actions: [
    {{"type": "get_figure", "figure_id": "bar2"}},
    {{"type": "semantic_search", "query": "BCA related concepts"}}
  ], ready_to_answer: false

Query: "Cảm ơn bạn"
→ actions: [], ready_to_answer: true

Current figure_id in context: {figure_id}
"""


class RouterAgent:
    """Decides which tools to invoke based on the user query."""

    def __init__(self) -> None:
        """Initialize LangChain ChatOpenAI with structured output."""
        settings = get_settings()

        # LangChain wrapper pointing to vLLM
        llm = ChatOpenAI(
            base_url=settings.vllm_base_url,
            api_key="EMPTY",
            model=settings.vllm_model_name,
            temperature=0.1,
            max_tokens=512,
        )

        # Bind Pydantic model for structured output
        self._chain = llm.with_structured_output(
            RouterOutput, method="json_mode"
        )
        logger.info("RouterAgent initialized")

    async def route(
        self,
        query: str,
        conversation_history: list[dict[str, str]],
        figure_id: str | None = None,
    ) -> RouterOutput:
        """Analyze query and return routing decision."""
        # Build messages
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT.format(
                figure_id=figure_id or "None"
            )},
        ]
        messages.extend(conversation_history[-5:])
        messages.append({"role": "user", "content": query})

        try:
            result = await self._chain.ainvoke(messages)
            logger.info(
                f"Router: {len(result.actions)} actions, "
                f"ready={result.ready_to_answer}"
            )
            return result

        except Exception as e:
            logger.warning(f"Router parse failed, fallback: {e}")
            return RouterOutput(
                reasoning=f"Router fallback: {e}",
                actions=[],
                ready_to_answer=True,
            )


@lru_cache
def get_router_agent() -> RouterAgent:
    """Get cached router agent singleton."""
    return RouterAgent()
