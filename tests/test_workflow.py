"""Integration test for the agentic workflow — step-by-step with streaming.

Runs each workflow node sequentially to verify the full pipeline:
  load_history → router → tools → aggregate → stream answer

Usage:
    python tests/test_workflow.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.llm.generator import get_answer_generator
from src.models.state import AgentState
from src.workflow.edges import route_decision
from src.workflow.nodes import (
    aggregate_context,
    get_figure,
    load_conversation_history,
    router_agent,
    semantic_search,
)


def _build_initial_state(query: str, figure_id: str | None = None) -> AgentState:
    """Build a clean initial AgentState for testing."""
    return {
        "conversation_id": "test_workflow",
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


async def run_step_by_step(query: str, figure_id: str | None = None) -> None:
    """Execute workflow nodes one-by-one with logging and streaming."""
    state = _build_initial_state(query, figure_id)
    logger.info(f"Query: '{query}' | figure_id: {figure_id}")
    print("=" * 60)

    # Step 1: Load conversation history
    logger.info("Step 1: load_conversation_history")
    result = await load_conversation_history(state)
    state.update(result)
    logger.info(f"  → history: {len(state['conversation_history'])} messages")
    print("-" * 60)

    # Step 2: Router agent
    logger.info("Step 2: router_agent")
    result = await router_agent(state)
    state.update(result)
    router_out = state["router_output"]
    logger.info(f"  → reasoning: {router_out['reasoning']}")
    logger.info(f"  → actions: {router_out['actions']}")
    logger.info(f"  → ready_to_answer: {router_out['ready_to_answer']}")
    print("-" * 60)

    # Step 3: Conditional routing
    next_nodes = route_decision(state)
    logger.info(f"Step 3: route_decision → {next_nodes}")
    print("-" * 60)

    # Step 4: Execute tool nodes
    if "get_figure" in next_nodes:
        logger.info("Step 4a: get_figure")
        result = await get_figure(state)
        state.update(result)
        has_fig = state["figure_data"] is not None
        logger.info(f"  → figure loaded: {has_fig}")

    if "semantic_search" in next_nodes:
        logger.info("Step 4b: semantic_search")
        result = await semantic_search(state)
        state.update(result)
        n_chunks = len(state.get("rag_results") or [])
        logger.info(f"  → RAG chunks: {n_chunks}")
        logger.info(f"  Context chunks: {state.get('rag_results')[0].get('content')}")
    print("-" * 60)

    # Step 5: Aggregate context
    logger.info("Step 5: aggregate_context")
    result = await aggregate_context(state)
    state.update(result)
    logger.info(f"  → {len(state['aggregated_context'])} messages ready")
    print("-" * 60)

    # Step 6: Stream answer
    logger.info("Step 6: generate_answer (streaming)")
    print()
    generator = get_answer_generator()
    full_answer = ""
    async for token in generator.stream_generate(
        aggregated_context=state["aggregated_context"],
    ):
        print(token, end="", flush=True)
        full_answer += token
    print("\n")
    print("=" * 60)
    logger.info(f"Answer length: {len(full_answer)} chars")
    logger.info("Workflow completed successfully!")


async def main() -> None:
    """Run integration test with figure query."""
    logger.info("=== Workflow Integration Test ===")

    # Test: figure query (should trigger get_figure)
    await run_step_by_step(
        query="giải thích và mô tả hình ảnh này",
        figure_id="bar3.png",
    )


if __name__ == "__main__":
    asyncio.run(main())
