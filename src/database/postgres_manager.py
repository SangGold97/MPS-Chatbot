"""PostgreSQL manager for long-term memory conversations."""
from datetime import datetime
from typing import Optional

import asyncpg
from loguru import logger
from pydantic import BaseModel

from src.config.settings import get_settings


class ConversationTurn(BaseModel):
    """Model for a conversation turn."""
    conversation_id: str
    turn: int
    query: str
    answer: str
    created_at: Optional[datetime] = None


class PostgresManager:
    """Manages PostgreSQL operations for conversation memory."""

    CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id VARCHAR(255) NOT NULL,
            turn INT NOT NULL,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (conversation_id, turn)
        );
    """

    def __init__(self, database_url: Optional[str] = None) -> None:
        """Initialize with database URL."""
        # Convert SQLAlchemy URL to asyncpg format
        url = database_url or get_settings().database_url
        self.database_url = url.replace("postgresql+asyncpg://", "postgresql://")
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Create connection pool and initialize tables."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.database_url, min_size=2, max_size=10)
            async with self._pool.acquire() as conn:
                await conn.execute(self.CREATE_TABLE_SQL)
            logger.info("PostgreSQL connection pool created and tables initialized")

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL connection pool closed")

    # --- CRUD Operations ---

    async def create_conversation(self, conv_id: str, query: str, answer: str) -> ConversationTurn:
        """Create a new conversation with turn=1."""
        sql = """
            INSERT INTO conversations (conversation_id, turn, query, answer)
            VALUES ($1, 1, $2, $3) RETURNING created_at
        """
        async with self._pool.acquire() as conn:
            created_at = await conn.fetchval(sql, conv_id, query, answer)

        logger.debug(f"Created conversation {conv_id}")
        return ConversationTurn(
            conversation_id=conv_id, turn=1, query=query, answer=answer, created_at=created_at
        )

    async def insert_turn(self, conv_id: str, query: str, answer: str) -> ConversationTurn:
        """Insert a new turn at the end of a conversation."""
        turn = await self.get_next_turn(conv_id)
        sql = """
            INSERT INTO conversations (conversation_id, turn, query, answer)
            VALUES ($1, $2, $3, $4) RETURNING created_at
        """
        async with self._pool.acquire() as conn:
            created_at = await conn.fetchval(sql, conv_id, turn, query, answer)

        logger.debug(f"Inserted turn {turn} for conversation {conv_id}")
        return ConversationTurn(
            conversation_id=conv_id, turn=turn, query=query, answer=answer, created_at=created_at
        )

    async def get_turn(self, conv_id: str, turn: int) -> Optional[ConversationTurn]:
        """Get a specific conversation turn."""
        sql = "SELECT * FROM conversations WHERE conversation_id = $1 AND turn = $2"
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, conv_id, turn)

        return ConversationTurn(**dict(row)) if row else None

    async def get_conversation(self, conv_id: str, limit: int = 10) -> list[ConversationTurn]:
        """Get all turns for a conversation, ordered by turn number."""
        sql = """
            SELECT * FROM conversations WHERE conversation_id = $1
            ORDER BY turn DESC LIMIT $2
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, conv_id, limit)

        # Return in ascending order (oldest first)
        return [ConversationTurn(**dict(r)) for r in reversed(rows)]

    async def delete_conversation(self, conv_id: str) -> int:
        """Delete all turns for a conversation. Returns number of deleted rows."""
        sql = "DELETE FROM conversations WHERE conversation_id = $1"
        async with self._pool.acquire() as conn:
            result = await conn.execute(sql, conv_id)

        # Parse "DELETE X" to get count
        count = int(result.split()[-1])
        logger.debug(f"Deleted {count} turns for conversation {conv_id}")
        return count

    async def get_next_turn(self, conv_id: str) -> int:
        """Get the next turn number for a conversation."""
        sql = "SELECT COALESCE(MAX(turn), 0) + 1 FROM conversations WHERE conversation_id = $1"
        async with self._pool.acquire() as conn:
            return await conn.fetchval(sql, conv_id)

