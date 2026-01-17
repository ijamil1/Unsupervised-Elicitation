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

## Remote Machine Setup (Cloud Deployment)

If you're deploying on a cloud GPU machine (Lambda Labs, RunPod, etc.), follow these steps:

**Storage Requirements**:
- **8B model**: 50 GB minimum
- **70B model**: 300 GB minimum (not 140 GB - see note below)
- **405B model**: 850 GB minimum


### 1. SSH into Remote Machine

```bash
# SSH into your cloud GPU instance
ssh user@your-gpu-server-ip

# Verify GPUs are available
nvidia-smi
```

### 2. Clone Repository

```bash
# Clone the repository
git clone https://github.com/your-username/Unsupervised-Elicitation.git
cd Unsupervised-Elicitation

# Switch to vllm-integration branch
git checkout vllm-integration
```

### 3. Setup Python Environment

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Or use conda
conda create -n ue python=3.10
conda activate ue
```

### 4. Install Dependencies

```bash
# Install the project as an editable package
# This installs all dependencies AND makes the code importable
pip install -e .
```

**Alternative (simpler but less robust):**
```bash
# If pip install -e . has issues, fall back to:
pip install -r requirements.txt
```

**Note:** The installation may take 5-10 minutes depending on your connection. The editable install (`-e`) is recommended because your code uses absolute imports like `from core.llm_api import ModelAPI`.

### 5. Verify Installation

```bash
# Check vLLM installation
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# Check CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Check HuggingFace CLI
which huggingface-cli
```

### 6. Setup Secrets File

```bash
# Create SECRETS file with your configuration
cat > SECRETS << 'EOF'
API_KEY=your_api_key_here
NYU_ORG=None
ARG_ORG=None
LLAMA_API_BASE=https://api.hyperbolic.xyz/v1
VLLM_BASE_URL=http://localhost:8000
EOF

# Important: Update VLLM_BASE_URL if accessing from a different machine
# For remote access: VLLM_BASE_URL=http://your-gpu-server-ip:8000
```

### 7. Continue with HuggingFace Setup

Now proceed to the "Installation" section below to setup HuggingFace and download models.

---

## Installation

### 1. Install vLLM (Skip if using requirements.txt above)

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

### 2. Setup HuggingFace Account


**Setup HuggingFace Account:**
1. Create a free account at https://huggingface.co/join
2. Request access to Llama models:
   - Visit https://huggingface.co/meta-llama/Meta-Llama-3.1-405B
   - Click "Request Access" and accept the license agreement
   - Wait for Meta's approval (usually a few hours to 1 day)
3. Generate an access token:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with "Read" permissions
   - Copy the token for next step

**Login to HuggingFace:**
```bash
# Login with your token
hf auth login

# Paste your token when prompted
# Token will be saved to ~/.cache/huggingface/token
```

### 3. Download Model Weights

```bash

df -h

mkdir -p /workspace/hf_cache
mkdir -p /workspace/hf_cache/hub
mkdir -p /workspace/hf_cache/transformers
mkdir -p /workspace/tmp


export HF_HOME=/workspace/hf_cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub
export TRANSFORMERS_CACHE=/workspace/hf_cache/transformers
export TMPDIR=/workspace/tmp
```

# Note: 405B is ~800GB, 70B is ~140GB, 8B is ~16GB
# Ensure you have sufficient disk space!
```

### 4. Verify Installation

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

vllm serve meta-llama/Meta-Llama-3.1-70B \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --tensor-parallel-size 4 \
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
# Test the server (specify model size: 8, 70, or 405)
python scripts/test_vllm_server.py 405    # For 405B model
python scripts/test_vllm_server.py 70     # For 70B model
python scripts/test_vllm_server.py 8      # For 8B model

# For remote server
python scripts/test_vllm_server.py 8 --base-url http://remote-server:8000

# Or manually with curl
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-405B",
    "prompt": "Question: Is the sky blue?\nClaim: Yes\nI think this claim is ",
    "max_tokens": 1,
    "logprobs": 20,
    "temperature": 0.0
  }'

  curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-70B",
    "prompt": "Question: Is the sky blue?\nClaim: Yes\nI think this claim is ",
    "max_tokens": 1,
    "logprobs": 20,
    "temperature": 0.0
  }'

   curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "prompt": "Are you AI or a human? Answer yes or no",
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


## Running ICM with vLLM In-Process

In-process mode eliminates network overhead by loading the model directly in the same Python process. This is the **recommended** approach when running on a machine with GPUs.

```bash
cd src/experiments

# Basic in-process run with 8B model (1 GPU)
python ICM.py \
    --model meta-llama/Llama-3.1-8B \
    --testbed truthfulQA \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.90 \
    --K 500

# 70B model with 4 GPUs
python ICM.py \
    --model meta-llama/Meta-Llama-3.1-70B \
    --testbed truthfulQA \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.90 \
    --batch_size 256 \
    --K 1500

# 405B model with 8 GPUs
python ICM.py \
    --model meta-llama/Meta-Llama-3.1-405B \
    --testbed truthfulQA \
    --tensor_parallel_size 8 \
    --gpu_memory_utilization 0.90 \
    --batch_size 256 \
    --K 1500

# With custom settings
python ICM.py \
    --model meta-llama/Meta-Llama-3.1-70B \
    --testbed truthfulQA \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.85 \
    --batch_size 128 \
    --K 1500
```



### Command Line Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `meta-llama/Meta-Llama-3.1-70B` | HuggingFace model name |
| `--tensor_parallel_size` | `1` | Number of GPUs for tensor parallelism |
| `--gpu_memory_utilization` | `0.90` | Fraction of GPU memory to use |
| `--max_model_len` | `None` | Max sequence length (None = model default) |
| `--batch_size` | `256` | ICM batch size |
| `--K` | `1500` | Max ICM iterations |
| `--alpha` | `1` | Scoring coefficient |


### Copying Results to Local Machine

After the experiment completes, ICM generates output files including `figure_1.png` and result data. To copy these to your local machine:

#### Option 1: Using SCP (Secure Copy)

**From your local machine:**
```bash
# Copy a single file
scp user@gpu-server:/path/to/Unsupervised-Elicitation/figure_1.png ~/Desktop/

# Copy entire results directory
scp -r user@gpu-server:/path/to/Unsupervised-Elicitation/results/ ~/Desktop/results/

# Copy specific experiment outputs
scp user@gpu-server:/path/to/Unsupervised-Elicitation/src/experiments/figure_1.png ~/Downloads/
```

#### Option 2: Using rsync (Better for Multiple Files)

**From your local machine:**
```bash
# Sync entire results directory
rsync -avz --progress user@gpu-server:/path/to/Unsupervised-Elicitation/results/ ~/local-results/

# Sync specific files with pattern matching
rsync -avz --progress user@gpu-server:/path/to/Unsupervised-Elicitation/*.png ~/Downloads/
```

**Advantages of rsync:**
- Shows progress bar
- Only transfers changed files
- Can resume interrupted transfers

#### Option 3: Using SFTP (Interactive)

**From your local machine:**
```bash
# Start SFTP session
sftp user@gpu-server

# Navigate to directory
cd /path/to/Unsupervised-Elicitation/src/experiments

# Download file
get figure_1.png

# Download multiple files
mget *.png

# Download directory
get -r results/

# Exit
quit
```

#### Option 4: VS Code Remote (Recommended for Development)

If using VS Code with Remote-SSH extension:
1. Connect to remote machine via Remote-SSH
2. Open the project folder remotely
3. Right-click `figure_1.png` → Download
4. Files automatically sync to local machine

#### Quick Reference

**Find the output files on remote:**
```bash
# On remote machine
cd ~/Unsupervised-Elicitation
find . -name "figure_1.png"
find . -name "*.png" -mtime -1  # Files modified in last 24 hours
ls -lh results/  # Check results directory
```

**Common file locations:**
- Main figure: `figure_1.png` (in repo root or src/experiments/)
- Results data: `results/` directory
- Logs: Various locations depending on experiment config

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

### In-Process Mode (Recommended)

The simplest and most efficient setup. Everything runs in a single process:

```
┌─────────────────────────────────────┐
│ GPU Server (8× A100)                │
│ ┌─────────────────────────────────┐ │
│ │ Single Python Process           │ │
│ │ ┌───────────────────────────┐   │ │
│ │ │ ICM Script                │   │ │
│ │ │ + VLLMInProcessClient     │   │ │
│ │ │ + vLLM Engine (in-memory) │   │ │
│ │ │ + Llama-3.1-405B          │   │ │
│ │ └───────────────────────────┘   │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘

Command:
python ICM.py --vllm_mode inprocess --tensor_parallel_size 8
```

**Advantages:**
- No HTTP overhead
- Simplest setup (no separate server)
- Best for single-user, single-experiment runs
- Direct access to vLLM's prefix caching



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

Expected performance with Llama-3.1-405B on 8× A100 80GB:

| Metric | Value |
|--------|-------|
| Batched inference latency (256 examples) | ~4s |
| GPU utilization | 85-95% |
| Prefix cache hit rate | 95-98% |
| Average time per ICM iteration | ~1.7s |
| Total runtime (K=1500) | 40-55 minutes |
| Cost per run (Lambda Labs @ $10/hr) | ~$8-10 |

**Performance Notes:**
- Each iteration calls `predict_assignment()` (~0.5s) plus conditionally calls `get_pipeline_batched()` (~3s) when label changes
- Label change rate: ~80% early, ~40% mid, ~20% late in training
- Batch size grows from 8 → 256 as labeled set expands
- Actual runtime varies based on acceptance rate and GPU efficiency

**Speedup vs external API**: ~80× faster (60+ hours → 45 minutes)
