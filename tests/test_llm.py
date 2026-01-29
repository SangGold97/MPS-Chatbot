"""Test script for Qwen3-VL-4B-Instruct-FP8 LLM with streaming output."""
import json

import httpx
from loguru import logger


def stream_chat(
    prompt: str,
    base_url: str = "http://localhost:8000/v1",
    model: str = "Qwen/Qwen3-VL-4B-Instruct-FP8",
    max_tokens: int = 1024,
) -> str:
    """Stream chat completion from vLLM API.

    Args:
        prompt: User message to send.
        base_url: vLLM API base URL.
        model: Model name.
        max_tokens: Maximum tokens to generate.

    Returns:
        Full response text.
    """
    # Build request payload
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    full_response = ""

    # Stream response with httpx
    with httpx.Client(timeout=180.0) as client:
        with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()

            # Process SSE stream
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue

                # Parse data
                data = line[6:]  # Remove "data: " prefix
                if data == "[DONE]":
                    break

                # Extract content delta
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"]

                if "content" not in delta or not delta["content"]:
                    continue

                content = delta["content"]
                print(content, end="", flush=True)
                full_response += content

    print()  # Newline after stream
    return full_response


def main() -> None:
    """Run LLM streaming tests."""
    logger.info("Starting LLM streaming test...")
    logger.info("Model: Qwen/Qwen3-VL-4B-Instruct-FP8")
    logger.info("API: http://localhost:8000/v1")
    print("-" * 50)

    # Test prompt
    prompt = "Confusion matrix là gì?"
    logger.info(f"Prompt: {prompt}")
    print("-" * 50)
    print("Response (streaming):")
    print()

    # Stream response
    response = stream_chat(prompt)

    print("-" * 50)
    logger.info(f"Response length: {len(response)} characters")
    logger.info("✓ LLM streaming test completed!")


if __name__ == "__main__":
    main()
