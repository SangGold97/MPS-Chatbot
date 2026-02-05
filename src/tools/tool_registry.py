"""MCP Tool Registry for centralized tool management."""
from typing import Any, Optional

from loguru import logger

from src.tools.figure_tool import get_figure_tool
from src.tools.mcp_protocol import MCPTool, MCPToolOutput, ToolStatus
from src.tools.semantic_search_tool import get_semantic_search_tool


class ToolRegistry:
    """Registry for managing MCP tools."""

    def __init__(self) -> None:
        """Initialize registry with empty tool map."""
        self._tools: dict[str, MCPTool] = {}

    def register(self, tool: MCPTool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Optional[MCPTool]:
        """Get tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List registered tool names."""
        return list(self._tools.keys())

    def get_all_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI-compatible schemas for all tools."""
        return [t.get_schema() for t in self._tools.values()]

    async def execute(self, tool_name: str, **kwargs: Any) -> MCPToolOutput:
        """Execute a tool by name."""
        tool = self.get(tool_name)
        if not tool:
            return MCPToolOutput(
                status=ToolStatus.ERROR,
                message=f"Tool '{tool_name}' not found",
            )

        try:
            return await tool(**kwargs)
        except Exception as e:
            logger.error(f"Tool error ({tool_name}): {e}")
            return MCPToolOutput(status=ToolStatus.ERROR, message=str(e))


# Global registry singleton
_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get global tool registry with default tools."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        _registry.register(get_figure_tool())
        _registry.register(get_semantic_search_tool())
    return _registry
