"""Pydantic models for request/response and router output."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class RouterAction(BaseModel):
    """Single action decided by the router agent."""

    type: Literal["get_figure", "semantic_search"] = Field(
        description="Action type to execute"
    )
    figure_id: Optional[str] = Field(
        default=None, description="Figure ID for get_figure action"
    )
    query: Optional[str] = Field(
        default=None,
        description="Search query for semantic_search action",
    )


class RouterOutput(BaseModel):
    """Structured output from the router agent."""

    reasoning: str = Field(
        default="",
        description="Brief explanation of the routing decision",
    )
    actions: list[RouterAction] = Field(
        default_factory=list,
        description="List of actions to execute",
    )
    ready_to_answer: bool = Field(
        default=False,
        description="Whether to skip tools and answer directly",
    )


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    conversation_id: str = Field(description="Conversation identifier")
    query: str = Field(description="User query text")
    figure_id: Optional[str] = Field(
        default=None, description="Optional figure ID in context"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    conversation_id: str
    answer: str
    turn: int
