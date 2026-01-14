#!/bin/bash
#
# vLLM Server Launch Script
# This script starts a vLLM server for self-hosted LLM inference.
#
# Usage:
#   ./scripts/launch_vllm_server.sh [MODEL_SIZE]
#   MODEL_SIZE: 405B (default), 70B, or 8B
#
# Requirements:
#   - vLLM installed: pip install vllm
#   - Sufficient GPU memory (see comments below)
#   - Model weights downloaded or accessible via HuggingFace

set -e

# Parse arguments
MODEL_SIZE=${1:-405B}

# Configuration based on model size
case $MODEL_SIZE in
  405B)
    MODEL_NAME="meta-llama/Meta-Llama-3.1-405B"
    TENSOR_PARALLEL_SIZE=8  # 8× A100 80GB or H100
    MAX_MODEL_LEN=15000
    MAX_NUM_SEQS=256
    echo "Launching Llama-3.1-405B (requires 8× A100 80GB or H100 GPUs)"
    ;;
  70B)
    MODEL_NAME="meta-llama/Llama-3.1-70B"
    TENSOR_PARALLEL_SIZE=4  # 4× A40 or 2× A100 80GB
    MAX_MODEL_LEN=15000
    MAX_NUM_SEQS=256
    echo "Launching Llama-3.1-70B (requires 4× A40 or 2× A100 80GB GPUs)"
    ;;
  8B)
    MODEL_NAME="meta-llama/Llama-3.1-8B"
    TENSOR_PARALLEL_SIZE=1  # 1× A100 40GB or A6000
    MAX_MODEL_LEN=15000
    MAX_NUM_SEQS=256
    echo "Launching Llama-3.1-8B (requires 1× A100 40GB or A6000 GPU)"
    ;;
  *)
    echo "Error: Invalid model size '$MODEL_SIZE'"
    echo "Usage: $0 [405B|70B|8B]"
    exit 1
    ;;
esac

# Server configuration
HOST="0.0.0.0"
PORT=8000
GPU_MEMORY_UTILIZATION=0.9

echo "========================================="
echo "vLLM Server Configuration"
echo "========================================="
echo "Model: $MODEL_NAME"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Max Sequences: $MAX_NUM_SEQS"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "========================================="
echo ""

# Check if vLLM is installed
if ! python -c "import vllm" &> /dev/null; then
    echo "Error: vLLM is not installed. Install with: pip install vllm"
    exit 1
fi

# Launch vLLM server
echo "Starting vLLM server..."
echo "The server will be accessible at http://$HOST:$PORT"
echo "Press Ctrl+C to stop the server"
echo ""

# Use Python module instead of vllm CLI command (works even if vllm not in PATH)
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --enable-prefix-caching \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --disable-log-requests

# Note: Prefix caching is critical for ICM performance
# It caches the KV cache for repeated prompt prefixes, which speeds up
# inference by 5-10x for the leave-one-out demonstration pattern
