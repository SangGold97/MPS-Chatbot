"""Embedding server using local model loading."""
import torch
import numpy as np
from typing import Union
from loguru import logger
from transformers import AutoModel, AutoTokenizer


class EmbeddingServer:
    """Local embedding server using Qwen3-Embedding model."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "cuda",
    ):
        """Initialize embedding server.

        Args:
            model_path: HuggingFace model path or local path
            device: Device to run model on (cuda/cpu)
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self) -> bool:
        """Load model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_path}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )

            # Load model
            self.model = AutoModel.from_pretrained(
                self.model_path, trust_remote_code=True
            )

            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def encode(
        self,
        texts: Union[str, list[str]],
        max_length: int = 512,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: Single text or list of texts
            max_length: Maximum token length
            normalize: Whether to L2 normalize embeddings

        Returns:
            Numpy array of embeddings
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure list format
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

            # Use CLS token or mean pooling
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                # Mean pooling
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                )
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embeddings = sum_embeddings / sum_mask

        # Convert to numpy
        embeddings = embeddings.cpu().numpy()

        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-9, None)

        return embeddings

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model.config.hidden_size


# Global server instance
_server: EmbeddingServer | None = None


def get_embedding_server(
    model_path: str = "Qwen/Qwen3-Embedding-0.6B",
    device: str = "cuda",
) -> EmbeddingServer:
    """Get or create embedding server singleton."""
    global _server
    if _server is None:
        _server = EmbeddingServer(model_path, device)
        _server.load_model()
    return _server


def main():
    """Run unit tests for embedding server."""
    logger.info("Starting embedding server tests...")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Test 1: Initialize server
    server = EmbeddingServer(
        model_path="Qwen/Qwen3-Embedding-0.6B",
        device=device,
    )
    logger.info("✓ Test 1: Server initialized")

    # Test 2: Load model
    result = server.load_model()
    assert result, "Failed to load model"
    logger.info("✓ Test 2: Model loaded")

    # Test 3: Get embedding dimension
    dim = server.get_embedding_dim()
    assert dim > 0, "Invalid embedding dimension"
    logger.info(f"✓ Test 3: Embedding dimension = {dim}")

    # Test 4: Encode single text
    text = "This is a test sentence."
    embedding = server.encode(text)
    assert embedding.shape == (1, dim), f"Wrong shape: {embedding.shape}"
    logger.info(f"✓ Test 4: Single text encoding shape = {embedding.shape}")

    # Test 5: Encode multiple texts
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    embeddings = server.encode(texts)
    assert embeddings.shape == (3, dim), f"Wrong shape: {embeddings.shape}"
    logger.info(f"✓ Test 5: Batch encoding shape = {embeddings.shape}")

    # Test 6: Verify normalization
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Embeddings not normalized"
    logger.info("✓ Test 6: Embeddings are normalized")

    # Test 7: Semantic similarity check
    similar_texts = ["I love programming", "I enjoy coding"]
    different_text = ["The weather is sunny"]

    emb_similar = server.encode(similar_texts)
    emb_different = server.encode(different_text)

    # Cosine similarity (already normalized, so just dot product)
    sim_score = np.dot(emb_similar[0], emb_similar[1])
    diff_score = np.dot(emb_similar[0], emb_different[0])

    logger.info(f"Similar texts similarity: {sim_score:.4f}")
    logger.info(f"Different texts similarity: {diff_score:.4f}")
    assert sim_score > diff_score, "Semantic similarity check failed"
    logger.info("✓ Test 7: Semantic similarity works correctly")

    logger.info("All embedding server tests passed!")


if __name__ == "__main__":
    main()
