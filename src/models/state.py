"""LangGraph AgentState definition."""
from typing import Any, Optional, TypedDict


class AgentState(TypedDict):
    """State schema for the LangGraph workflow.

    Each field flows through the graph nodes, accumulating
    context until the final answer is generated.
    """

    # --- Input fields ---
    conversation_id: str
    query: str
    figure_id: Optional[str]

    # --- Memory ---
    conversation_history: list[dict[str, str]]

    # --- Router ---
    router_output: Optional[dict[str, Any]]

    # --- Tool results ---
    figure_data: Optional[str]
    rag_results: Optional[list[dict[str, Any]]]

    # --- Context aggregation ---
    aggregated_context: list[dict[str, Any]]

    # --- Output ---
    final_answer: str
    error: Optional[str]
