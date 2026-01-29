"""Embedding client using vLLM OpenAI-compatible API."""
import httpx
import numpy as np
from loguru import logger

from src.config.settings import get_settings


class EmbeddingClient:
    """Client for vLLM embedding API."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        """Initialize embedding client from settings."""
        settings = get_settings()
        self.base_url = base_url or settings.embedding_base_url
        self.model = model or settings.embedding_model_name
        self.client = httpx.Client(timeout=60.0)

    def encode(self, texts: str | list[str]) -> np.ndarray:
        """Encode texts to embeddings via vLLM API.
        
        Note: Qwen3-Embedding outputs are pre-normalized (L2 norm = 1).
        """
        # Ensure list format
        if isinstance(texts, str):
            texts = [texts]

        # Call vLLM embedding API
        response = self.client.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": texts},
        )
        response.raise_for_status()

        # Extract embeddings (already normalized by model)
        data = response.json()["data"]
        return np.array([d["embedding"] for d in data])

    def get_embedding_dim(self) -> int:
        """Get embedding dimension by encoding a test text."""
        emb = self.encode("test")
        return emb.shape[1]


# Global client instance
_client: EmbeddingClient | None = None


def get_embedding_client() -> EmbeddingClient:
    """Get or create embedding client singleton."""
    global _client
    if _client is None:
        _client = EmbeddingClient()
    return _client


def main():
    """Run embedding client tests."""
    logger.info("Starting embedding client tests...")

    # Initialize client
    client = EmbeddingClient()
    logger.info("✓ Test 1: Client initialized")

    # Get embedding dimension
    dim = client.get_embedding_dim()
    assert dim > 0, "Invalid embedding dimension"
    logger.info(f"✓ Test 2: Embedding dimension = {dim}")

    # Encode single text
    embedding = client.encode("This is a test sentence.")
    assert embedding.shape == (1, dim), f"Wrong shape: {embedding.shape}"
    logger.info(f"✓ Test 3: Single text encoding shape = {embedding.shape}")

    # Encode multiple texts
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    embeddings = client.encode(texts)
    assert embeddings.shape == (3, dim), f"Wrong shape: {embeddings.shape}"
    logger.info(f"✓ Test 4: Batch encoding shape = {embeddings.shape}")

    # Semantic similarity check
    emb_similar = client.encode(["I love programming", "I enjoy coding"])
    emb_different = client.encode(["The weather is sunny"])

    sim_score = np.dot(emb_similar[0], emb_similar[1])
    diff_score = np.dot(emb_similar[0], emb_different[0])

    logger.info(f"Similar texts similarity: {sim_score:.4f}")
    logger.info(f"Different texts similarity: {diff_score:.4f}")
    assert sim_score > diff_score, "Semantic similarity check failed"
    logger.info("✓ Test 5: Semantic similarity works correctly")

    logger.info("All embedding client tests passed!")


if __name__ == "__main__":
    main()
