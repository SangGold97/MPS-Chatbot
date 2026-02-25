"""LangGraph workflow node implementations."""
from typing import Any

from loguru import logger

from src.database.postgres_manager import PostgresManager
from src.llm.generator import ANSWER_SYSTEM_PROMPT, get_answer_generator
from src.llm.router import get_router_agent
from src.models.state import AgentState
from src.tools.tool_registry import get_tool_registry


async def load_conversation_history(state: AgentState) -> dict[str, Any]:
    """Load last 5 turns from PostgreSQL as chat messages."""
    conv_id = state["conversation_id"]
    logger.info(f"[load_history] conversation: {conv_id}")

    db = PostgresManager()
    try:
        await db.connect()
        turns = await db.get_conversation(conv_id, limit=5)

        # Format as chat messages
        history: list[dict[str, str]] = []
        for turn in turns:
            history.append({"role": "user", "content": turn.query})
            history.append({"role": "assistant", "content": turn.answer})

        logger.info(f"[load_history] {len(turns)} turns loaded")
        return {"conversation_history": history}
    except Exception as e:
        logger.warning(f"[load_history] failed: {e}")
        return {"conversation_history": []}
    finally:
        await db.close()


async def router_agent(state: AgentState) -> dict[str, Any]:
    """Invoke router LLM to decide actions."""
    router = get_router_agent()
    result = await router.route(
        query=state["query"],
        conversation_history=state.get("conversation_history", []),
        figure_id=state.get("figure_id"),
    )
    logger.info(f"[router] {result.reasoning}")
    return {"router_output": result.model_dump()}


async def get_figure(state: AgentState) -> dict[str, Any]:
    """Retrieve figure image as base64 via FigureTool."""
    # Extract figure_id from router actions or state
    figure_id = state.get("figure_id")
    for action in state.get("router_output", {}).get("actions", []):
        if action.get("type") == "get_figure" and action.get("figure_id"):
            figure_id = action["figure_id"]
            break

    if not figure_id:
        logger.warning("[get_figure] no figure_id found")
        return {"figure_data": None}

    result = await get_tool_registry().execute(
        "get_figure", figure_id=figure_id
    )
    if result.status.value == "success" and result.data:
        logger.info(f"[get_figure] retrieved: {figure_id}")
        return {"figure_data": result.data.base64_image}

    logger.warning(f"[get_figure] not found: {figure_id}")
    return {"figure_data": None}


async def semantic_search(state: AgentState) -> dict[str, Any]:
    """Search knowledge base for relevant context chunks."""
    # Extract search query from router actions
    search_query = state["query"]
    for action in state.get("router_output", {}).get("actions", []):
        if action.get("type") == "semantic_search" and action.get("query"):
            search_query = action["query"]
            break

    result = await get_tool_registry().execute(
        "semantic_search", query=search_query, top_k=1
    )
    if result.status.value == "success" and result.data:
        rag_results = [
            {"id": r.id, "content": r.content,
             "metadata": r.metadata, "distance": r.distance}
            for r in result.data
        ]
        logger.info(f"[semantic_search] {len(rag_results)} chunks")
        return {"rag_results": rag_results}

    logger.warning(f"[semantic_search] failed: {result.message}")
    return {"rag_results": []}


async def aggregate_context(state: AgentState) -> dict[str, Any]:
    """Merge figure, RAG results, and history into LLM messages."""
    messages: list[dict[str, Any]] = []

    # System message with optional RAG context
    system_parts = [ANSWER_SYSTEM_PROMPT]
    rag_results = state.get("rag_results") or []
    if rag_results:
        context_text = "\n\n".join(
            f"[{r['id']}] {r['content']}" for r in rag_results
        )
        system_parts.append(
            f"\n\nRelevant knowledge base context:\n{context_text}"
        )
    messages.append({"role": "system", "content": "\n".join(system_parts)})

    # Conversation history
    messages.extend(state.get("conversation_history", []))

    # User query — multimodal if figure exists
    figure_data = state.get("figure_data")
    if figure_data:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": state["query"]},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{figure_data}",
                }},
            ],
        })
    else:
        messages.append({"role": "user", "content": state["query"]})

    logger.info(f"[aggregate] {len(messages)} messages")
    return {"aggregated_context": messages}


async def generate_answer(state: AgentState) -> dict[str, Any]:
    """Generate final answer from aggregated context."""
    try:
        answer = await get_answer_generator().generate(
            aggregated_context=state["aggregated_context"],
        )
        logger.info(f"[generate] {len(answer)} chars")
        return {"final_answer": answer}
    except Exception as e:
        logger.error(f"[generate] failed: {e}")
        return {
            "final_answer": "Xin lỗi, đã xảy ra lỗi khi tạo câu trả lời.",
            "error": str(e),
        }


async def save_to_memory(state: AgentState) -> dict[str, Any]:
    """Persist Q&A pair to PostgreSQL."""
    conv_id = state["conversation_id"]
    db = PostgresManager()
    try:
        await db.connect()
        existing = await db.get_conversation(conv_id, limit=1)
        if existing:
            await db.insert_turn(conv_id, state["query"], state.get("final_answer", ""))
        else:
            await db.create_conversation(conv_id, state["query"], state.get("final_answer", ""))
        logger.info(f"[save_memory] conversation: {conv_id}")
    except Exception as e:
        logger.error(f"[save_memory] failed: {e}")
    finally:
        await db.close()
    return {}
