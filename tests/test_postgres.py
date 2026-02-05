"""Tests for PostgresManager CRUD operations."""
import asyncio
import uuid

from loguru import logger

from src.database.postgres_manager import PostgresManager


async def run_tests() -> None:
    """Run all CRUD tests for PostgresManager."""
    db = PostgresManager()
    await db.connect()
    conv_id = f"test_{uuid.uuid4().hex[:8]}"

    # Test 1: Create conversation
    turn = await db.create_conversation(conv_id, "Hello", "Hi there!")
    assert turn.conversation_id == conv_id and turn.turn == 1 and turn.query == "Hello"
    logger.info("✓ Test 1: Create conversation passed")

    # Test 2: Get turn
    result = await db.get_turn(conv_id, 1)
    assert result and result.query == "Hello" and result.answer == "Hi there!"
    logger.info("✓ Test 2: Get turn passed")

    # Test 3: Insert turn (appends to conversation)
    turn2 = await db.insert_turn(conv_id, "Q2", "A2")
    assert turn2.turn == 2
    turn3 = await db.insert_turn(conv_id, "Q3", "A3")
    assert turn3.turn == 3
    logger.info("✓ Test 3: Insert turn passed")

    # Test 4: Get conversation
    turns = await db.get_conversation(conv_id)
    assert len(turns) == 3 and turns[0].turn == 1 and turns[2].turn == 3
    logger.info("✓ Test 4: Get conversation passed")

    # Test 5: Get next turn
    assert await db.get_next_turn(conv_id) == 4
    logger.info("✓ Test 5: Get next turn passed")

    # Test 6: Delete conversation
    count = await db.delete_conversation(conv_id)
    assert count == 3 and await db.get_conversation(conv_id) == []
    logger.info("✓ Test 6: Delete conversation passed")

    await db.close()
    logger.info("All PostgresManager tests passed!")


if __name__ == "__main__":
    asyncio.run(run_tests())
