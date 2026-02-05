"""Semantic Search Tool for RAG pipeline via MCP protocol."""
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.rag.embedder import get_embedding_client
from src.tools.mcp_protocol import MCPTool, MCPToolOutput, ToolStatus


class SemanticSearchInput(BaseModel):
    """Input schema for semantic search."""

    query: str = Field(description="Search query text")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    collection_name: Optional[str] = Field(
        default=None, description="ChromaDB collection name (uses default if None)"
    )


class SearchResult(BaseModel):
    """Single search result."""

    id: str
    content: str
    metadata: dict[str, Any]
    distance: float


class SemanticSearchOutput(MCPToolOutput):
    """Output schema for semantic search."""

    data: Optional[list[SearchResult]] = None


class SemanticSearchTool(MCPTool):
    """MCP Tool for semantic search in vector database."""

    name = "semantic_search"
    description = "Search knowledge base using semantic similarity to find relevant context chunks"
    input_schema = SemanticSearchInput

    def __init__(self, persist_dir: Optional[str] = None) -> None:
        """Initialize embedder and ChromaDB client."""
        settings = get_settings()

        # Initialize embedding client
        self._embedder = get_embedding_client()

        # Initialize ChromaDB client
        self._persist_dir = persist_dir or settings.chroma_persist_dir
        self._default_collection = settings.chroma_collection_name
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info(f"SemanticSearchTool initialized: {self._persist_dir}")

    def list_collections(self) -> list[str]:
        """List available collections."""
        return [c.name for c in self._client.list_collections()]

    async def execute(self, **kwargs: Any) -> SemanticSearchOutput:
        """Execute semantic search: embed query → search → format results."""
        # Validate input
        inp = SemanticSearchInput(**kwargs)
        collection_name = inp.collection_name or self._default_collection
        logger.info(f"Searching '{collection_name}': {inp.query[:50]}...")

        try:
            # Get collection
            collection = self._client.get_collection(name=collection_name)

            # Encode query to embedding
            query_emb = self._embedder.encode(inp.query)

            # Search vector DB
            results = collection.query(
                query_embeddings=query_emb.tolist(),
                n_results=inp.top_k,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            search_results = self._format_results(results)
            logger.info(f"Found {len(search_results)} results")

            return SemanticSearchOutput(
                status=ToolStatus.SUCCESS,
                message=f"Found {len(search_results)} relevant chunks",
                data=search_results,
            )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return SemanticSearchOutput(
                status=ToolStatus.ERROR,
                message=f"Search error: {str(e)}",
            )

    def _format_results(self, raw_results: dict[str, Any]) -> list[SearchResult]:
        """Format ChromaDB results to SearchResult list."""
        results = []
        ids = raw_results.get("ids", [[]])[0]
        docs = raw_results.get("documents", [[]])[0]
        metas = raw_results.get("metadatas", [[]])[0]
        dists = raw_results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            results.append(
                SearchResult(
                    id=doc_id,
                    content=docs[i] if i < len(docs) else "",
                    metadata=metas[i] if i < len(metas) else {},
                    distance=dists[i] if i < len(dists) else 0.0,
                )
            )
        return results


# Singleton instance
_semantic_search_tool: Optional[SemanticSearchTool] = None


def get_semantic_search_tool() -> SemanticSearchTool:
    """Get global semantic search tool instance."""
    global _semantic_search_tool
    if _semantic_search_tool is None:
        _semantic_search_tool = SemanticSearchTool()
    return _semantic_search_tool
