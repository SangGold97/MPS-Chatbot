"""Document indexing script for ChromaDB."""
import os
import json
from pathlib import Path
from loguru import logger
import chromadb
from chromadb.config import Settings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.embedder import EmbeddingServer


class DocumentIndexer:
    """Index documents into ChromaDB."""

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        collection_name: str = "documents",
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "cuda",
    ):
        """Initialize document indexer.

        Args:
            persist_dir: Directory to persist ChromaDB
            collection_name: Name of the collection
            embedding_model: Path to embedding model
            device: Device for embedding model
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # Initialize embedding server
        self.embedder = EmbeddingServer(embedding_model, device)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = None

    def setup(self) -> bool:
        """Setup indexer and load model."""
        try:
            # Load embedding model
            if not self.embedder.load_model():
                return False

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(f"Collection '{self.collection_name}' ready")
            return True

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False

    def index_documents(
        self,
        documents: list[dict],
        batch_size: int = 32,
    ) -> int:
        """Index documents into ChromaDB.

        Args:
            documents: List of dicts with 'id', 'content', 'metadata'
            batch_size: Batch size for embedding

        Returns:
            Number of documents indexed
        """
        if self.collection is None:
            raise RuntimeError("Indexer not setup. Call setup() first.")

        indexed = 0

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            # Extract data
            ids = [doc["id"] for doc in batch]
            contents = [doc["content"] for doc in batch]
            metadatas = [doc.get("metadata", {}) for doc in batch]

            # Generate embeddings
            embeddings = self.embedder.encode(contents)

            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=contents,
                metadatas=metadatas,
            )

            indexed += len(batch)
            logger.info(f"Indexed {indexed}/{len(documents)} documents")

        return indexed

    def index_from_directory(
        self,
        directory: str,
        extensions: list[str] = [".txt", ".md"],
    ) -> int:
        """Index all documents from a directory.

        Args:
            directory: Path to documents directory
            extensions: File extensions to include

        Returns:
            Number of documents indexed
        """
        documents = []
        dir_path = Path(directory)

        # Find all matching files
        for ext in extensions:
            for file_path in dir_path.rglob(f"*{ext}"):
                try:
                    content = file_path.read_text(encoding="utf-8")
                    doc_id = str(file_path.relative_to(dir_path))

                    documents.append({
                        "id": doc_id,
                        "content": content,
                        "metadata": {
                            "source": str(file_path),
                            "filename": file_path.name,
                        },
                    })
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")

        if not documents:
            logger.warning(f"No documents found in {directory}")
            return 0

        logger.info(f"Found {len(documents)} documents to index")
        return self.index_documents(documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Search for similar documents.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results with content and metadata
        """
        if self.collection is None:
            raise RuntimeError("Indexer not setup")

        # Generate query embedding
        query_embedding = self.embedder.encode(query)

        # Search
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
        )

        # Format results
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results["distances"] else None,
            })

        return formatted

    def get_collection_info(self) -> dict:
        """Get collection statistics."""
        if self.collection is None:
            return {}

        return {
            "name": self.collection.name,
            "count": self.collection.count(),
        }


def main():
    """Run indexer tests."""
    import torch

    logger.info("Starting document indexer tests...")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test directory
    test_dir = Path("./data/documents")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create test documents
    test_docs = [
        ("doc1.txt", "Machine learning is a subset of artificial intelligence."),
        ("doc2.txt", "Deep learning uses neural networks with many layers."),
        ("doc3.txt", "Python is a popular programming language for data science."),
    ]

    for filename, content in test_docs:
        (test_dir / filename).write_text(content)

    # Test 1: Initialize indexer
    indexer = DocumentIndexer(
        persist_dir="./data/chroma_db_test",
        collection_name="test_collection",
        device=device,
    )
    logger.info("✓ Test 1: Indexer initialized")

    # Test 2: Setup indexer
    result = indexer.setup()
    assert result, "Failed to setup indexer"
    logger.info("✓ Test 2: Indexer setup complete")

    # Test 3: Index documents from directory
    count = indexer.index_from_directory(str(test_dir))
    assert count == 3, f"Expected 3 documents, got {count}"
    logger.info(f"✓ Test 3: Indexed {count} documents")

    # Test 4: Get collection info
    info = indexer.get_collection_info()
    assert info["count"] == 3, "Collection count mismatch"
    logger.info(f"✓ Test 4: Collection info = {info}")

    # Test 5: Search documents
    results = indexer.search("neural networks", top_k=2)
    assert len(results) == 2, "Expected 2 search results"
    logger.info(f"✓ Test 5: Search returned {len(results)} results")

    # Test 6: Verify search relevance
    top_result = results[0]
    assert "neural" in top_result["content"].lower() or "deep" in top_result["content"].lower()
    logger.info(f"✓ Test 6: Top result is relevant: {top_result['content'][:50]}...")

    # Cleanup test files
    for filename, _ in test_docs:
        (test_dir / filename).unlink(missing_ok=True)

    # Cleanup test database
    import shutil
    shutil.rmtree("./data/chroma_db_test", ignore_errors=True)

    logger.info("All document indexer tests passed!")


if __name__ == "__main__":
    main()
