"""Test script for LLM with streaming output."""
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from loguru import logger

from src.config.settings import get_settings


def stream_chat(
    prompt: str,
    base_url: str | None = None,
    model: str | None = None,
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
    # Load from settings if not provided
    settings = get_settings()
    base_url = base_url or settings.vllm_base_url
    model = model or settings.vllm_model_name

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
    settings = get_settings()
    logger.info("Starting LLM streaming test...")
    logger.info(f"Model: {settings.vllm_model_name}")
    logger.info(f"API: {settings.vllm_base_url}")
    print("-" * 50)

    # Test with image: venn1.png
    image_path = Path(__file__).parent.parent / "data" / "figures" / "scatter1.png"
    
    # Encode image to base64
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    
    # Prompt for image description
    prompt = "Mô tả ngắn gọn figure này"
    logger.info(f"Image: {image_path.name}")
    logger.info(f"Prompt: {prompt}")
    print("-" * 50)
    print("Response (streaming):")
    print()

    # Stream response with image
    response = stream_chat_with_image(prompt, image_data)

    print("-" * 50)
    logger.info(f"Response length: {len(response)} characters")
    logger.info("✓ LLM streaming test completed!")


def stream_chat_with_image(
    prompt: str,
    image_base64: str,
    base_url: str | None = None,
    model: str | None = None,
    max_tokens: int = 1024,
) -> str:
    """Stream chat completion with image from vLLM API.

    Args:
        prompt: User message to send.
        image_base64: Base64 encoded image.
        base_url: vLLM API base URL.
        model: Model name.
        max_tokens: Maximum tokens to generate.

    Returns:
        Full response text.
    """
    # Load from settings if not provided
    settings = get_settings()
    base_url = base_url or settings.vllm_base_url
    model = model or settings.vllm_model_name

    # Build request payload with image
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ],
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
                data = line[6:]
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

    print()
    return full_response


if __name__ == "__main__":
    main()
