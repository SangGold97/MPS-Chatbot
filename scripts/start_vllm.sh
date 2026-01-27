#!/bin/bash
# Start vLLM server for Qwen3-VL-4B-Thinking-FP8

set -e

MODEL_NAME="Qwen/Qwen3-VL-4B-Thinking-FP8"
PORT=8000
MAX_MODEL_LEN=8192
GPU_MEMORY_UTIL=0.85

echo "Starting vLLM server..."
echo "Model: $MODEL_NAME"
echo "Port: $PORT"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --port $PORT \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
