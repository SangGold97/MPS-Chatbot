"""LangGraph workflow definition — the main agentic graph."""
from typing import Any

from langgraph.graph import END, START, StateGraph
from loguru import logger

from src.models.state import AgentState
from src.workflow.edges import route_decision
from src.workflow.nodes import (
    aggregate_context,
    generate_answer,
    get_figure,
    load_conversation_history,
    router_agent,
    save_to_memory,
    semantic_search,
)


def build_graph() -> StateGraph:
    """Construct the LangGraph workflow.

    Flow:
        START → load_conversation_history → router_agent
            → [get_figure | semantic_search | aggregate_context]  (fan-out)
            → aggregate_context  (fan-in)
            → generate_answer → save_to_memory → END
    """
    graph = StateGraph(AgentState)

    # --- Add nodes ---
    graph.add_node("load_conversation_history", load_conversation_history)
    graph.add_node("router_agent", router_agent)
    graph.add_node("get_figure", get_figure)
    graph.add_node("semantic_search", semantic_search)
    graph.add_node("aggregate_context", aggregate_context)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("save_to_memory", save_to_memory)

    # --- Linear edges ---
    graph.add_edge(START, "load_conversation_history")
    graph.add_edge("load_conversation_history", "router_agent")

    # --- Conditional fan-out from router ---
    graph.add_conditional_edges(
        "router_agent",
        route_decision,
        {
            "get_figure": "get_figure",
            "semantic_search": "semantic_search",
            "aggregate_context": "aggregate_context",
        },
    )

    # --- Fan-in: tools → aggregate_context ---
    graph.add_edge("get_figure", "aggregate_context")
    graph.add_edge("semantic_search", "aggregate_context")

    # --- Post-aggregation linear flow ---
    graph.add_edge("aggregate_context", "generate_answer")
    graph.add_edge("generate_answer", "save_to_memory")
    graph.add_edge("save_to_memory", END)

    return graph


# Compiled graph singleton
_compiled_graph = None


def get_workflow():
    """Get the compiled LangGraph workflow."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph().compile()
        logger.info("LangGraph workflow compiled")
    return _compiled_graph


async def run_workflow(
    conversation_id: str,
    query: str,
    figure_id: str | None = None,
) -> dict[str, Any]:
    """Execute the full workflow and return final state.

    Args:
        conversation_id: Conversation identifier.
        query: User query text.
        figure_id: Optional figure ID in context.

    Returns:
        Final AgentState dict with final_answer populated.
    """
    workflow = get_workflow()

    # Build initial state
    initial_state: AgentState = {
        "conversation_id": conversation_id,
        "query": query,
        "figure_id": figure_id,
        "conversation_history": [],
        "router_output": None,
        "figure_data": None,
        "rag_results": None,
        "aggregated_context": [],
        "final_answer": "",
        "error": None,
    }

    logger.info(
        f"Running workflow: conv={conversation_id}, "
        f"query='{query[:50]}...', figure={figure_id}"
    )

    # Execute graph
    result = await workflow.ainvoke(initial_state)
    logger.info("Workflow completed")
    return result
