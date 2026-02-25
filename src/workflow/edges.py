"""Conditional edge functions for the LangGraph workflow."""
from loguru import logger

from src.models.state import AgentState


def route_decision(state: AgentState) -> list[str]:
    """Determine which nodes to execute after the router agent.

    Returns a list of node names. LangGraph will fan-out and
    execute them in parallel, then fan-in to aggregate_context.

    Possible returns:
        ["aggregate_context"]              — direct answer
        ["get_figure"]                     — figure only
        ["semantic_search"]                — RAG only
        ["get_figure", "semantic_search"]  — both (parallel)
    """
    router_output = state.get("router_output", {})

    # If ready to answer directly, skip tools
    if router_output.get("ready_to_answer", False):
        logger.info("Route: direct answer (skip tools)")
        return ["aggregate_context"]

    # Collect requested tool nodes
    next_nodes: list[str] = []
    for action in router_output.get("actions", []):
        action_type = action.get("type")
        if action_type == "get_figure" and "get_figure" not in next_nodes:
            next_nodes.append("get_figure")
        elif (
            action_type == "semantic_search"
            and "semantic_search" not in next_nodes
        ):
            next_nodes.append("semantic_search")

    # Fallback to direct answer if no valid actions
    if not next_nodes:
        logger.info("Route: no actions, fallback to direct answer")
        return ["aggregate_context"]

    logger.info(f"Route: {next_nodes}")
    return next_nodes
