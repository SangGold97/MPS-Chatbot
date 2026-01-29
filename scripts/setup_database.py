"""Database setup script for PostgreSQL."""
import asyncio
from loguru import logger
import asyncpg


DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/chatbot"

CREATE_TABLES_SQL = """
-- Create conversations table
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id VARCHAR(255) NOT NULL,
    turn INT NOT NULL,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (conversation_id, turn)
);
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
    """Verify tables exist and have correct structure."""
    try:
        conn = await asyncpg.connect(DATABASE_URL)

        # Check conversations table columns
        columns = await conn.fetch("""
            SELECT column_name, data_type FROM information_schema.columns 
            WHERE table_name = 'conversations' AND table_schema = 'public'
            ORDER BY ordinal_position
        """)
        
        if not columns:
            logger.error("conversations table does not exist")
            await conn.close()
            return False

        col_dict = {c["column_name"]: c["data_type"] for c in columns}
        
        # Verify required columns
        required_cols = {
            "conversation_id": "character varying",
            "turn": "integer",
            "query": "text",
            "answer": "text",
            "created_at": "timestamp without time zone"
        }
        
        missing = [c for c in required_cols if c not in col_dict]
        if missing:
            logger.error(f"Missing columns: {missing}")
            await conn.close()
            return False

        logger.info(f"Table 'conversations' verified with columns: {list(col_dict.keys())}")
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
        test_conv_id = "test_conv_001"
        test_turn = 1
        test_query = "Test query"
        test_answer = "Test answer"
        
        await conn.execute("""
            INSERT INTO conversations (conversation_id, turn, query, answer)
            VALUES ($1, $2, $3, $4)
        """, test_conv_id, test_turn, test_query, test_answer)
        logger.info(f"Created conversation: {test_conv_id}")

        # Verify insertion
        result = await conn.fetchrow("""
            SELECT * FROM conversations WHERE conversation_id = $1 AND turn = $2
        """, test_conv_id, test_turn)
        assert result is not None, "Failed to retrieve inserted data"
        logger.info(f"Verified data: {dict(result)}")

        # Cleanup test data
        await conn.execute("""
            DELETE FROM conversations WHERE conversation_id = $1
        """, test_conv_id)
        logger.info("Cleaned up test data")

        await conn.close()
        logger.info("✓ Test 3: Insert/delete operations passed")

    except Exception as e:
        logger.error(f"Test 3 failed: {e}")
        raise

    logger.info("All database tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
