"""Document indexing script for ChromaDB using vLLM embedding API."""
import sys
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.config import Settings
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rag.embedder import EmbeddingClient


def load_xlsx_as_chunks(file_path: str) -> list[dict]:
    """Load xlsx file and create chunks from rows.

    Args:
        file_path: Path to xlsx file

    Returns:
        List of document chunks with id, content, metadata
    """
    df = pd.read_excel(file_path)
    chunks = []

    # Process each row as a chunk
    for idx, row in df.iterrows():
        term = str(row["Term (Thuật ngữ)"]).strip()
        description = str(row["Description (Mô tả/Định nghĩa)"]).strip()

        # Combine term and description
        content = f"Thuật ngữ: {term}\nĐịnh nghĩa: {description}"

        chunks.append({
            "id": f"term_{idx}",
            "content": content,
            "metadata": {"term": term, "source": Path(file_path).name},
        })

    logger.info(f"Loaded {len(chunks)} chunks from {file_path}")
    return chunks


def index_to_chromadb(
    chunks: list[dict],
    embedder: EmbeddingClient,
    persist_dir: str,
    collection_name: str = "bca_terms",
) -> chromadb.Collection:
    """Index chunks into ChromaDB with FLAT index.

    Args:
        chunks: List of document chunks
        embedder: Embedding client
        persist_dir: Directory to persist ChromaDB
        collection_name: Name of the collection

    Returns:
        ChromaDB collection
    """
    # Extract data and generate embeddings first
    ids = [c["id"] for c in chunks]
    contents = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    embeddings = embedder.encode(contents)
    logger.info(f"Generated embeddings: {embeddings.shape}")

    # Initialize ChromaDB with FLAT index (exact search)
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )

    # Delete existing collection if exists
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    # Create collection with cosine distance
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Add to collection
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=contents,
        metadatas=metadatas,
    )

    logger.info(f"Indexed {len(chunks)} documents to '{collection_name}'")
    return collection


def main():
    """Index xlsx documents into ChromaDB."""
    # Paths - use absolute paths
    base_dir = Path(__file__).resolve().parent.parent
    xlsx_path = base_dir / "data/documents/[BCA]TERM_DATA.xlsx"
    persist_dir = str(base_dir / "src/database/volumes/chromadb")

    # Initialize embedding client
    embedder = EmbeddingClient()
    logger.info(f"Embedding dimension: {embedder.get_embedding_dim()}")

    # Load and chunk xlsx
    chunks = load_xlsx_as_chunks(str(xlsx_path))

    # Index to ChromaDB
    collection = index_to_chromadb(chunks, embedder, persist_dir)

    # Verify indexing
    logger.info(f"Collection count: {collection.count()}")
    logger.info("Indexing completed successfully!")


if __name__ == "__main__":
    main()
