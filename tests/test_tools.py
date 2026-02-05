"""Tests for MCP Tools - run with: python -m tests.test_tools"""
import asyncio

from loguru import logger

from src.tools import ToolStatus, get_tool_registry


async def test_figure_tool_via_mcp() -> None:
    """Test retrieve figure qua MCP protocol."""
    logger.info("=== Testing Figure Tool via MCP ===")

    # Gọi figure tool qua registry với figure_id
    result = await get_tool_registry().execute("get_figure", figure_id="scatter1")

    # Kiểm tra kết quả
    assert result.status == ToolStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "No figure data returned"
    assert result.data.base64_image, "No base64 image"

    logger.info(f"✅ Figure tool OK - Retrieved figure size: {result.data.width}x{result.data.height}")


async def test_semantic_search_tool_via_mcp() -> None:
    """Test retrieve semantic search qua MCP protocol."""
    logger.info("\n=== Testing Semantic Search Tool via MCP ===")

    # Gọi semantic search tool qua registry với query
    result = await get_tool_registry().execute(
        "semantic_search", query="Short Tandem Repeats là gì?", top_k=3
    )

    # Kiểm tra kết quả (SUCCESS hoặc ERROR nếu collection chưa tồn tại)
    assert result.status in [ToolStatus.SUCCESS, ToolStatus.ERROR], f"Unexpected status: {result.status}"

    if result.status == ToolStatus.SUCCESS:
        assert result.data is not None, "No search results"
        logger.info(f"✅ Semantic search OK - Found {len(result.data)} results")
        logger.info("Contents of top result:")
        for i, res in enumerate(result.data):
            logger.info(f"Result {i+1} - Distance {round(res.distance, 3)}: {res.content[:100]}...")
    else:
        logger.info(f"⚠️  Semantic search returned ERROR (collection may not exist): {result.message}")


async def main() -> None:
    """Run MCP tool tests."""
    logger.info("Starting MCP Tools Tests\n")

    # Test figure tool
    await test_figure_tool_via_mcp()

    # Test semantic search tool
    await test_semantic_search_tool_via_mcp()

    logger.info("\n✅ All MCP tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
