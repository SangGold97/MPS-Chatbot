#!/bin/bash
# Start vLLM servers for LLM and Embedding models

set -e

# Load environment variables from .env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

# LLM Configuration (from .env or defaults)
LLM_MODEL="${VLLM_MODEL_NAME:-Qwen/Qwen3-VL-4B-Instruct-FP8}"
LLM_PORT="${VLLM_PORT:-8000}"
LLM_MAX_LEN="${VLLM_MAX_LEN:-4096}"
LLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.70}"

# Embedding Configuration (from .env or defaults)
EMB_MODEL="${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-0.6B}"
EMB_PORT="${EMBEDDING_PORT:-8001}"
EMB_MAX_LEN="${EMBEDDING_MAX_LEN:-512}"
EMB_GPU_UTIL="${EMBEDDING_GPU_UTIL:-0.10}"

# Cleanup function
cleanup() {
    echo "Stopping servers..."
    kill $LLM_PID $EMB_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "=========================================="
echo "Starting vLLM servers..."
echo "=========================================="

# Start LLM server (background)
echo "[1/2] Starting LLM server on port $LLM_PORT..."
python -m vllm.entrypoints.openai.api_server \
    --model "$LLM_MODEL" \
    --port $LLM_PORT \
    --max-model-len $LLM_MAX_LEN \
    --gpu-memory-utilization $LLM_GPU_UTIL \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --enforce-eager &
LLM_PID=$!

# Wait for LLM server to start
sleep 5

# Start Embedding server (background)
echo "[2/2] Starting Embedding server on port $EMB_PORT..."
python -m vllm.entrypoints.openai.api_server \
    --model "$EMB_MODEL" \
    --port $EMB_PORT \
    --max-model-len $EMB_MAX_LEN \
    --gpu-memory-utilization $EMB_GPU_UTIL \
    --trust-remote-code \
    --enforce-eager &
EMB_PID=$!

echo "=========================================="
echo "Servers started:"
echo "  - LLM:       http://localhost:$LLM_PORT"
echo "  - Embedding: http://localhost:$EMB_PORT"
echo "=========================================="

# Wait for both processes
wait $LLM_PID $EMB_PID
