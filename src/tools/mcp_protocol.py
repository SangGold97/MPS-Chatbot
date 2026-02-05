"""MCP Protocol definitions for standardized tool calling."""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolStatus(str, Enum):
    """Tool execution status."""

    SUCCESS = "success"
    ERROR = "error"
    NOT_FOUND = "not_found"


class MCPToolOutput(BaseModel):
    """Base output schema for MCP tools."""

    status: ToolStatus = Field(description="Execution status")
    message: str = Field(default="")
    data: Any = Field(default=None)


class MCPTool(ABC):
    """Abstract base class for MCP tools."""

    name: str
    description: str

    @abstractmethod
    async def execute(self, **kwargs: Any) -> MCPToolOutput:
        """Execute the tool with provided arguments."""
        pass

    async def __call__(self, **kwargs: Any) -> MCPToolOutput:
        """Allow calling tool instance directly."""
        return await self.execute(**kwargs)

    def get_schema(self) -> dict[str, Any]:
        """Get OpenAI-compatible tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": getattr(self, "input_schema", {}).model_json_schema()
                if hasattr(self, "input_schema")
                else {"type": "object"},
            },
        }
