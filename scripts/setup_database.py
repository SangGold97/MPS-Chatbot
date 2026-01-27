"""Database setup script for PostgreSQL."""
import asyncio
from loguru import logger
import asyncpg


DATABASE_URL = "postgresql://postgres:postgres@localhost:5433/chatbot"

CREATE_TABLES_SQL = """
-- Create conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create messages table
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    figure_id VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_messages_conversation 
ON messages(conversation_id, created_at DESC);
"""


async def create_tables() -> bool:
    """Create database tables."""
    try:
        # Connect to database
        conn = await asyncpg.connect(DATABASE_URL)
        logger.info("Connected to PostgreSQL database")

        # Execute SQL
        await conn.execute(CREATE_TABLES_SQL)
        logger.info("Tables created successfully")

        # Close connection
        await conn.close()
        return True

    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        return False


async def verify_tables() -> bool:
    """Verify tables exist."""
    try:
        conn = await asyncpg.connect(DATABASE_URL)

        # Check tables exist
        tables = await conn.fetch("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        table_names = [t["table_name"] for t in tables]

        # Verify required tables
        required = ["conversations", "messages"]
        missing = [t for t in required if t not in table_names]

        if missing:
            logger.error(f"Missing tables: {missing}")
            await conn.close()
            return False

        logger.info(f"Tables verified: {table_names}")
        await conn.close()
        return True

    except Exception as e:
        logger.error(f"Failed to verify tables: {e}")
        return False


async def main():
    """Run database setup and tests."""
    logger.info("Starting database setup...")

    # Test 1: Create tables
    result = await create_tables()
    assert result, "Failed to create tables"
    logger.info("✓ Test 1: Create tables passed")

    # Test 2: Verify tables
    result = await verify_tables()
    assert result, "Failed to verify tables"
    logger.info("✓ Test 2: Verify tables passed")

    # Test 3: Insert test data
    try:
        conn = await asyncpg.connect(DATABASE_URL)

        # Insert conversation
        conv_id = await conn.fetchval("""
            INSERT INTO conversations DEFAULT VALUES RETURNING id
        """)
        logger.info(f"Created conversation: {conv_id}")

        # Insert message
        msg_id = await conn.fetchval("""
            INSERT INTO messages (conversation_id, role, content) 
            VALUES ($1, $2, $3) RETURNING id
        """, conv_id, "user", "Test message")
        logger.info(f"Created message: {msg_id}")

        # Cleanup test data
        await conn.execute("DELETE FROM conversations WHERE id = $1", conv_id)
        logger.info("Cleaned up test data")

        await conn.close()
        logger.info("✓ Test 3: Insert/delete operations passed")

    except Exception as e:
        logger.error(f"Test 3 failed: {e}")
        raise

    logger.info("All database tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
