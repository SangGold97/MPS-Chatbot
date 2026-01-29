"""Test case for document indexing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.config import Settings
from loguru import logger

from src.config.settings import get_settings
from src.rag.embedder import EmbeddingClient


def test_indexing():
    """Test that documents are indexed correctly in ChromaDB."""
    settings = get_settings()
    base_dir = Path(__file__).resolve().parent.parent
    persist_dir = str(base_dir / settings.chroma_persist_dir.lstrip("./"))

    # Load ChromaDB
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(settings.chroma_collection_name)

    # Check document count
    count = collection.count()
    assert count == 22, f"Expected 22 documents, got {count}"
    logger.info(f"✓ Document count: {count}")

    # Test semantic search
    embedder = EmbeddingClient()
    query_emb = embedder.encode("ADN ty thể là gì")
    results = collection.query(query_embeddings=query_emb.tolist(), n_results=3)

    # Verify results
    assert len(results["ids"][0]) == 3, "Expected 3 results"
    logger.info(f"✓ Search results: {results['ids'][0]}")
    logger.info(f"✓ Top result: {results['documents'][0][0][:100]}...")

    logger.info("All indexing tests passed!")


if __name__ == "__main__":
    test_indexing()
