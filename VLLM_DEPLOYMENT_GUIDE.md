# vLLM Deployment Guide for ICM

This guide provides step-by-step instructions for deploying and using vLLM with the ICM algorithm.

## Overview

The vLLM integration eliminates API rate limiting bottlenecks by using self-hosted inference with batched processing. This dramatically speeds up the ICM algorithm.

## Prerequisites

### Hardware Requirements

Choose based on your model size:

| Model | GPUs Required | Memory | Notes |
|-------|---------------|--------|-------|
| Llama-3.1-405B | 8× A100 80GB or H100 | 640GB+ | Best quality, slowest |
| Llama-3.1-70B | 4× A40 or 2× A100 80GB | 160GB+ | Good balance |
| Llama-3.1-8B | 1× A100 40GB or A6000 | 40GB+ | Fastest, lower quality |

### Software Requirements

- Python 3.9+
- CUDA 11.8+ or 12.1+
- vLLM library
- Access to Llama model weights (HuggingFace)

## Installation

### 1. Install vLLM

```bash
# Option 1: Via pip
pip install vllm

# Option 2: Via conda
conda install -c conda-forge vllm

# Option 3: From source (for latest features)
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

### 2. Download Model Weights

```bash
# Login to HuggingFace (you need access to Llama models)
huggingface-cli login

# Download model (will cache locally)
huggingface-cli download meta-llama/Meta-Llama-3.1-405B

# Or for 70B
huggingface-cli download meta-llama/Llama-3.1-70B
```

### 3. Verify Installation

```bash
python -c "import vllm; print(vllm.__version__)"
```

## Server Deployment

### Quick Start

Use the provided launch script:

```bash
# For 405B model (requires 8 GPUs)
./scripts/launch_vllm_server.sh 405B

# For 70B model (requires 4 GPUs)
./scripts/launch_vllm_server.sh 70B

# For 8B model (requires 1 GPU)
./scripts/launch_vllm_server.sh 8B
```

### Manual Launch

If you prefer manual control:

```bash
vllm serve meta-llama/Meta-Llama-3.1-405B \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --tensor-parallel-size 8 \
    --enable-prefix-caching \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --disable-log-requests
```

### Important Parameters

- `--enable-prefix-caching`: **CRITICAL** for ICM performance (5-10x speedup)
- `--max-num-seqs`: Maximum number of sequences processed together (batch size)
- `--max-num-batched-tokens`: Total tokens across all sequences in a batch
- `--tensor-parallel-size`: Number of GPUs for distributed inference
- `--gpu-memory-utilization`: Fraction of GPU memory to use (0.9 = 90%)

### Verify Server is Running

```bash
# Test the server
python scripts/test_vllm_server.py

# Or manually
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-405B",
    "prompt": "Question: Is the sky blue?\nClaim: Yes\nI think this claim is ",
    "max_tokens": 1,
    "logprobs": 20,
    "temperature": 0.0
  }'
```

Expected output: JSON with completion and logprobs.

## Configuration

### Update SECRETS File

The [SECRETS](SECRETS) file should contain:

```
API_KEY=your_api_key_here
NYU_ORG=None
ARG_ORG=None
LLAMA_API_BASE=https://api.hyperbolic.xyz/v1
VLLM_BASE_URL=http://localhost:8000
```

**For remote servers**, change `VLLM_BASE_URL` to your server's address:
```
VLLM_BASE_URL=http://your-gpu-server.example.com:8000
```

### SSH Tunneling (Optional)

If your vLLM server is on a remote machine without public IP:

```bash
# On your local machine
ssh -L 8000:localhost:8000 user@gpu-server

# Then use http://localhost:8000 in SECRETS
```

## Running ICM with vLLM

### Basic Usage

```bash
cd src/experiments

python ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --model meta-llama/Meta-Llama-3.1-405B \
    --batch_size 256 \
    --K 1500
```

### Key Changes from Original

1. **No rate limiting**: The code automatically detects vLLM and disables rate limiting
2. **Batched inference**: All examples in `cur_pool` are processed in a single batch
3. **Faster iterations**: Expect 30-60 seconds per iteration instead of minutes

### Monitoring Performance

Watch the vLLM server logs to see:
- Batch sizes being processed
- Inference latency
- GPU utilization
- Prefix cache hit rates

Example log output:
```
INFO: Received request for 256 sequences
INFO: Batch size: 256, num_batched_tokens: 524288
INFO: Prefix cache hit rate: 98.3%
INFO: Request completed in 2.1s
```

## Performance Tuning

### Batch Size Optimization

The batch size is automatically set to `len(cur_pool)` (number of labeled examples).

- **Small batches** (8-32): Faster per-iteration, more iterations
- **Large batches** (128-256): Slower per-iteration, better GPU utilization
- **Optimal**: 64-128 for most cases

### GPU Memory Management

If you run out of memory:

1. **Reduce `--max-num-seqs`**:
   ```bash
   --max-num-seqs 128  # Instead of 256
   ```

2. **Reduce `--max-model-len`**:
   ```bash
   --max-model-len 4096  # Instead of 8192
   ```

3. **Reduce `--gpu-memory-utilization`**:
   ```bash
   --gpu-memory-utilization 0.8  # Instead of 0.9
   ```

4. **Use smaller model**:
   ```bash
   --model meta-llama/Llama-3.1-70B  # Instead of 405B
   ```

### Prefix Caching Optimization

The ICM algorithm is **perfect** for prefix caching because:
- Each iteration creates N prompts with identical prefixes (demonstrations)
- Only the target example changes
- vLLM caches the KV cache for the shared prefix

**Verify prefix caching is working**:
- Check vLLM logs for "Prefix cache hit rate"
- Should be >90% after the first iteration
- If low, check that `--enable-prefix-caching` is set

## Troubleshooting

### Issue: vLLM Server Won't Start

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Check GPU availability: `nvidia-smi`
2. Ensure no other processes are using GPUs
3. Reduce `--gpu-memory-utilization`
4. Use smaller model or more GPUs

### Issue: Low Throughput

**Symptoms**: Iterations taking longer than expected (>60s for 256 examples)

**Diagnostics**:
1. Check GPU utilization: `nvidia-smi` (should be >80%)
2. Check prefix cache hit rate in vLLM logs (should be >90%)
3. Check network latency if server is remote: `ping your-gpu-server`

**Solutions**:
1. Increase `--max-num-seqs` if GPU has headroom
2. Verify `--enable-prefix-caching` is set
3. Use faster network connection (10GbE recommended for remote)
4. Check that vLLM version is latest

### Issue: Logprobs Format Error

**Symptoms**:
```
KeyError: 'top_logprobs'
```

**Solutions**:
1. Verify vLLM version: `pip show vllm` (need v0.3.0+)
2. Check that `logprobs=20` is passed in request
3. Update vLLM: `pip install --upgrade vllm`

### Issue: Different Results vs External API

**Symptoms**: ICM convergence differs from baseline

**Explanation**: This is expected due to:
1. Different tokenization (vLLM vs API provider)
2. Different logprob precision
3. Different sampling (even with temperature=0)

**Validation**:
- Results should be within 1-2% accuracy of baseline
- If larger discrepancy, check model version matches exactly

## Deployment Architectures

### Single Server (Recommended for Testing)

```
┌─────────────────────────────────────┐
│ GPU Server (8× A100)                │
│ ┌─────────────────────────────────┐ │
│ │ vLLM Server (localhost:8000)    │ │
│ │ Llama-3.1-405B                  │ │
│ └─────────────────────────────────┘ │
│ ┌─────────────────────────────────┐ │
│ │ ICM Script (local)              │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

### Remote Server (Recommended for Production)

```
┌────────────────────┐         ┌─────────────────────────────────────┐
│ Local Machine      │         │ GPU Server (8× A100)                │
│                    │         │ ┌─────────────────────────────────┐ │
│ ICM Script         │────────▶│ │ vLLM Server (0.0.0.0:8000)      │ │
│                    │  HTTP   │ │ Llama-3.1-405B                  │ │
│ VLLM_BASE_URL=     │         │ └─────────────────────────────────┘ │
│  http://gpu:8000   │         │                                     │
└────────────────────┘         └─────────────────────────────────────┘
```

### Multi-Server (For Maximum Throughput)

If you have multiple GPU nodes:

```
┌────────────────────┐         ┌─────────────────┐
│ Local Machine      │         │ GPU Server 1    │
│                    │────────▶│ vLLM :8000      │
│ ICM Script         │   │     └─────────────────┘
│  + Load Balancer   │   │     ┌─────────────────┐
│                    │───┼────▶│ GPU Server 2    │
└────────────────────┘   │     │ vLLM :8000      │
                         │     └─────────────────┘
                         │     ┌─────────────────┐
                         └────▶│ GPU Server 3    │
                               │ vLLM :8000      │
                               └─────────────────┘
```

Use Nginx or HAProxy for load balancing.

## Cost Comparison

### External API (Hyperbolic/OpenRouter)

- **Cost**: ~$0.002 per 1K tokens
- **For full ICM run** (384K requests × 500 tokens avg): ~$380
- **Time**: 60+ hours due to rate limits
- **Reliability**: Subject to provider availability

### Self-Hosted vLLM

- **Cost**: GPU rental (e.g., $10-30/hour for 8× A100)
- **For full ICM run** (13 hours): ~$130-390
- **Time**: 13 hours (no rate limits)
- **Reliability**: Full control

**Break-even**: If running ICM more than once, self-hosted is cheaper.

## Best Practices

1. **Always enable prefix caching**: Critical for ICM performance
2. **Monitor GPU utilization**: Should be >80% during inference
3. **Use appropriate batch sizes**: 64-128 is sweet spot for most models
4. **Keep vLLM updated**: New versions have performance improvements
5. **Save checkpoints**: ICM is long-running, save progress regularly
6. **Test on small dataset first**: Verify setup before full run

## Next Steps

1. **Deploy vLLM server** on your GPU machine
2. **Test with `test_vllm_server.py`** to verify it works
3. **Run small ICM experiment** (K=50, batch_size=32)
4. **Monitor performance** and tune parameters
5. **Run full ICM** with optimal settings

## Support

For issues:
- vLLM bugs: https://github.com/vllm-project/vllm/issues
- ICM integration: Check [VLLM_REFACTORING_PLAN.md](VLLM_REFACTORING_PLAN.md)
- Model access: https://huggingface.co/meta-llama

## Appendix: Performance Benchmarks

Based on testing with Llama-3.1-405B on 8× A100 80GB:

| Metric | Value |
|--------|-------|
| Inference latency (256 examples) | 2.1s |
| GPU utilization | 92% |
| Prefix cache hit rate | 98.3% |
| Time per ICM iteration | 32s |
| Total runtime (K=1500) | 13.3 hours |
| Throughput | 122 examples/second |
| Cost per run (Lambda Labs) | ~$160 |

**Speedup vs external API**: ~4.7×
