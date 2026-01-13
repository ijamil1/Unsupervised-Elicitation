# vLLM Integration & Batching Refactoring Plan

## Executive Summary

This document outlines a comprehensive refactoring plan to:
1. **Replace API endpoint calls with self-hosted vLLM** to eliminate rate limiting issues
2. **Implement batched inference** to dramatically reduce the number of API calls in the ICM algorithm
3. **Optimize the pipeline architecture** for efficient batch processing

## Problem Analysis

### Current Architecture Issues

#### 1. Rate Limiting Bottleneck
- **Location**: [core/llm_api/openai_llm.py:230-231](core/llm_api/openai_llm.py#L230-L231)
- **Current behavior**: Uses `LLAMA_API_BASE` environment variable to call external API (Hyperbolic.xyz)
- **Problem**: External APIs have strict rate limits that throttle the algorithm
- **Rate limit handling**: [src/pipeline/pipeline.py:31-43](src/pipeline/pipeline.py#L31-L43)
  - Uses `AsyncLimiter` (100 requests per 60 seconds)
  - Semaphore limits concurrency to 2
  - Still hits rate limits frequently

#### 2. Inefficient Sequential Inference in ICM Loop
- **Location**: [src/experiments/ICM.py:370-446](src/experiments/ICM.py#L370-L446)
- **Problem**: For each of K iterations (default 1500):
  1. Line 421-428: Creates a pipeline for `tmp_pool` (contains N labeled examples)
  2. Line 428: Calls `await pipeline.run()`
  3. Pipeline creates N separate prompts in [ICM.py:105-151](src/experiments/ICM.py#L105-L151)
  4. Each prompt gets leave-one-out demonstrations (all other N-1 examples)
  5. [src/runners/query_model.py:243-271](src/runners/query_model.py#L243-L271) creates N independent API calls
  6. These N calls are executed with `await asyncio.gather()` but still hit rate limits

**Inefficiency Calculation**:
- K iterations × N examples per iteration = 1500 × 256 = **384,000 API calls** (worst case)
- Each call has overhead: API roundtrip, rate limiting delays, connection setup
- Total runtime: Hours to days depending on rate limits

#### 3. No Batching Support
- **Current**: Each example is a separate API request with unique prompt
- **Opportunity**: vLLM supports batched inference where multiple prompts are processed together
- **Benefit**: Can process 8-32+ examples per forward pass (limited by GPU memory)

## Proposed Solution Architecture

### Overview
```
┌─────────────────────────────────────────────────────────────┐
│                     ICM Algorithm Loop                       │
│  (K=1500 iterations, N=256 examples per iteration)          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Batch Inference Manager (NEW)                   │
│  • Groups N examples into batches of B (e.g., B=16)         │
│  • Creates batched prompts with padding                      │
│  • Manages batch scheduling and execution                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              vLLM Client Module (NEW)                        │
│  • Async client for vLLM OpenAI-compatible API              │
│  • Handles batch requests via /v1/completions               │
│  • Extracts logprobs from batch responses                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              vLLM Server (Self-Hosted)                       │
│  • Runs Llama-3.1-405B (or 70B) with prefix caching        │
│  • Batches inference internally (continuous batching)        │
│  • Returns logprobs for top-K tokens                         │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Minimal Code Changes**: Leverage existing abstractions where possible
2. **Backward Compatibility**: Keep external API support as fallback
3. **Progressive Refactoring**: Can be implemented in phases
4. **Performance First**: Focus on batch efficiency over feature parity

## Detailed Implementation Plan

### Phase 1: vLLM Server Deployment

#### 1.1 Set Up vLLM Server

**Goal**: Deploy a self-hosted vLLM server with the base Llama model

**Hardware Requirements**:
- **For Llama-3.1-405B**: 8× A100 80GB or H100 GPUs (distributed inference)
- **For Llama-3.1-70B**: 2× A100 80GB or 4× A40 GPUs
- **For Llama-3.1-8B**: 1× A100 40GB or A6000

**Installation**:
```bash
# Install vLLM
pip install vllm

# Or use Docker
docker pull vllm/vllm-openai:latest
```

**Server Launch Script** (create as `scripts/launch_vllm_server.sh`):
```bash
#!/bin/bash

# Configuration
MODEL_NAME="meta-llama/Meta-Llama-3.1-405B"  # or 70B, 8B
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=8192
TENSOR_PARALLEL_SIZE=8  # Number of GPUs
PORT=8000

# Enable prefix caching for ICM efficiency
vllm serve $MODEL_NAME \
    --host 0.0.0.0 \
    --port $PORT \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --enable-prefix-caching \
    --max-num-seqs 256 \
    --max-num-batched-tokens 16384 \
    --disable-log-requests
```

**Why Prefix Caching is Critical**:
- ICM creates prompts where the first N-1 demonstrations are identical
- Only the target example changes
- Prefix caching caches the KV cache for repeated prefixes
- Can speed up inference by 5-10x for ICM workload

**Server Configuration File** (create as `configs/vllm_config.yaml`):
```yaml
model: "meta-llama/Meta-Llama-3.1-405B"
host: "0.0.0.0"
port: 8000
tensor_parallel_size: 8
gpu_memory_utilization: 0.9
max_model_len: 8192
enable_prefix_caching: true
max_num_seqs: 256
max_num_batched_tokens: 16384
disable_log_requests: true
trust_remote_code: false
```

#### 1.2 Verify Server Functionality

**Test Script** (create as `scripts/test_vllm_server.py`):
```python
import requests
import json

def test_vllm_server(base_url="http://localhost:8000"):
    # Test basic completion
    response = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": "meta-llama/Meta-Llama-3.1-405B",
            "prompt": "Question: Is the sky blue?\nClaim: Yes\nI think this claim is ",
            "max_tokens": 1,
            "logprobs": 20,
            "temperature": 0.0
        }
    )

    result = response.json()
    print("Response:", json.dumps(result, indent=2))

    # Verify logprobs are present
    assert "logprobs" in result["choices"][0]
    print("✓ Server is working correctly")

    return result

if __name__ == "__main__":
    test_vllm_server()
```

### Phase 2: Create vLLM Client Module

#### 2.1 New File: `core/llm_api/vllm_llm.py`

This module will handle all vLLM-specific communication.

**Key Features**:
1. OpenAI-compatible API client (vLLM mimics OpenAI API)
2. Support for batched requests
3. Logprobs extraction
4. No rate limiting (self-hosted)
5. Prefix caching awareness

**Implementation**:
```python
# core/llm_api/vllm_llm.py
import asyncio
import logging
import time
from typing import List, Union, Optional
import aiohttp
import attrs

from core.llm_api.base_llm import LLMResponse, ModelAPIProtocol, messages_to_single_prompt

LOGGER = logging.getLogger(__name__)

@attrs.define()
class VLLMClient(ModelAPIProtocol):
    """
    Client for self-hosted vLLM server.
    No rate limiting since we control the server.
    Supports batched inference for efficiency.
    """

    base_url: str = attrs.field()
    print_prompt_and_response: bool = False
    max_batch_size: int = 16  # Number of prompts to batch together
    timeout: int = 300  # Longer timeout for batch processing

    def __attrs_post_init__(self):
        # Verify server is accessible
        LOGGER.info(f"Initializing vLLM client for {self.base_url}")

    async def __call__(
        self,
        model_ids: list[str],
        prompt: Union[str, list[str], list[dict]],
        print_prompt_and_response: bool,
        max_attempts: int,
        **kwargs,
    ) -> list[LLMResponse]:
        """
        Main entry point for vLLM inference.
        Supports both single and batched requests.
        """
        model_id = model_ids[0]  # vLLM only serves one model at a time

        # Convert chat format to text if needed
        if isinstance(prompt, list) and isinstance(prompt[0], dict):
            prompt = messages_to_single_prompt(prompt)

        # Ensure prompt is a list for batching
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        start_time = time.time()

        # Batch the prompts if there are many
        if len(prompts) > self.max_batch_size:
            return await self._batched_call(
                model_id, prompts, start_time, max_attempts, **kwargs
            )
        else:
            return await self._single_call(
                model_id, prompts, start_time, max_attempts, **kwargs
            )

    async def _single_call(
        self,
        model_id: str,
        prompts: list[str],
        start_time: float,
        max_attempts: int,
        **kwargs
    ) -> list[LLMResponse]:
        """
        Make a single batch request to vLLM.
        """
        for attempt in range(max_attempts):
            try:
                responses = await self._make_vllm_request(
                    model_id, prompts, start_time, **kwargs
                )
                return responses
            except Exception as e:
                LOGGER.warning(f"vLLM request failed (attempt {attempt+1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1.5 ** attempt)
                else:
                    raise

    async def _batched_call(
        self,
        model_id: str,
        prompts: list[str],
        start_time: float,
        max_attempts: int,
        **kwargs
    ) -> list[LLMResponse]:
        """
        Split large prompt lists into multiple batches and process in parallel.
        """
        batches = [
            prompts[i:i + self.max_batch_size]
            for i in range(0, len(prompts), self.max_batch_size)
        ]

        LOGGER.info(f"Processing {len(prompts)} prompts in {len(batches)} batches")

        # Process batches in parallel
        batch_tasks = [
            self._single_call(model_id, batch, start_time, max_attempts, **kwargs)
            for batch in batches
        ]

        batch_results = await asyncio.gather(*batch_tasks)

        # Flatten results
        all_responses = []
        for batch_result in batch_results:
            all_responses.extend(batch_result)

        return all_responses

    async def _make_vllm_request(
        self,
        model_id: str,
        prompts: list[str],
        start_time: float,
        **kwargs
    ) -> list[LLMResponse]:
        """
        Make the actual HTTP request to vLLM server.
        vLLM supports batching natively via list of prompts.
        """
        api_start = time.time()

        # Prepare request payload
        payload = {
            "model": model_id,
            "prompt": prompts if len(prompts) > 1 else prompts[0],
            "max_tokens": kwargs.get("max_tokens", 1),
            "temperature": kwargs.get("temperature", 0.0),
            "top_p": kwargs.get("top_p", 1.0),
            "n": kwargs.get("n", 1),
            "logprobs": kwargs.get("top_logprobs", 20),  # vLLM uses 'logprobs' for top-k
            "echo": False,
        }

        # Make async HTTP request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"vLLM request failed: {error_text}")

                result = await response.json()

        api_duration = time.time() - api_start
        duration = time.time() - start_time

        # Parse response into LLMResponse objects
        responses = []
        for choice in result["choices"]:
            # Extract logprobs in the format expected by the codebase
            logprobs_list = None
            if "logprobs" in choice and choice["logprobs"] is not None:
                # vLLM returns logprobs in OpenAI format
                token_logprobs = choice["logprobs"].get("top_logprobs", [])
                if token_logprobs:
                    # Convert to expected format: list of dicts {token: logprob}
                    logprobs_list = token_logprobs

            responses.append(
                LLMResponse(
                    model_id=model_id,
                    completion=choice["text"],
                    stop_reason=choice["finish_reason"],
                    api_duration=api_duration,
                    duration=duration,
                    cost=0.0,  # Self-hosted, no per-token cost
                    logprobs=logprobs_list
                )
            )

        if self.print_prompt_and_response or kwargs.get("print_prompt_and_response", False):
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                print(f"\n{'='*80}\nPrompt {i+1}:\n{prompt}\n\nResponse:\n{response.completion}\n{'='*80}")

        return responses


def create_vllm_client(base_url: str, max_batch_size: int = 16) -> VLLMClient:
    """
    Factory function to create a vLLM client.
    """
    return VLLMClient(base_url=base_url, max_batch_size=max_batch_size)
```

#### 2.2 Update SECRETS Configuration

Add vLLM endpoint to [SECRETS](SECRETS):
```
API_KEY=sk_live_...
NYU_ORG=None
ARG_ORG=None
LLAMA_API_BASE=https://api.hyperbolic.xyz/v1
VLLM_BASE_URL=http://localhost:8000  # NEW: vLLM server endpoint
```

### Phase 3: Integrate vLLM Client into ModelAPI

#### 3.1 Modify `core/llm_api/llm.py`

**Changes**:
1. Add vLLM client initialization
2. Add model routing to use vLLM for base models
3. Remove rate limiting for vLLM calls

**Code Changes**:

```python
# core/llm_api/llm.py

# Add import
from core.llm_api.vllm_llm import VLLMClient

@attrs.define()
class ModelAPI:
    openai_fraction_rate_limit: float = attrs.field(
        default=0.99, validator=attrs.validators.lt(1)
    )
    organization: None = None
    print_prompt_and_response: bool = False
    use_vllm: bool = True  # NEW: Flag to enable vLLM
    vllm_base_url: str = None  # NEW: vLLM server URL

    _openai_base: OpenAIBaseModel = attrs.field(init=False)
    _openai_chat: OpenAIChatModel = attrs.field(init=False)
    _vllm_client: VLLMClient = attrs.field(init=False)  # NEW

    def __attrs_post_init__(self):
        secrets = load_secrets()
        if self.organization is None:
            self.organization = "NYU_ORG"

        # Initialize OpenAI clients (keep for fallback)
        self._openai_base = OpenAIBaseModel(
            frac_rate_limit=self.openai_fraction_rate_limit,
            organization=secrets[self.organization],
            print_prompt_and_response=self.print_prompt_and_response,
        )
        self._openai_chat = OpenAIChatModel(
            frac_rate_limit=self.openai_fraction_rate_limit,
            organization=secrets[self.organization],
            print_prompt_and_response=self.print_prompt_and_response,
        )

        # NEW: Initialize vLLM client
        if self.use_vllm:
            vllm_url = self.vllm_base_url or secrets.get('VLLM_BASE_URL', 'http://localhost:8000')
            self._vllm_client = VLLMClient(
                base_url=vllm_url,
                print_prompt_and_response=self.print_prompt_and_response,
                max_batch_size=16  # Configurable
            )
            LOGGER.info(f"Initialized vLLM client at {vllm_url}")

        Path("./prompt_history").mkdir(exist_ok=True)

    def model_id_to_class(self, model_id: str) -> ModelAPIProtocol:
        # NEW: Route base models to vLLM if enabled
        if self.use_vllm and model_id in BASE_MODELS:
            return self._vllm_client

        # Existing routing logic
        if model_id in ["gpt-4-base", "gpt-3.5-turbo-instruct"]:
            return self._openai_base_arg
        elif model_id in BASE_MODELS:
            return self._openai_base
        elif model_id in GPT_CHAT_MODELS or "ft:gpt-3.5-turbo" in model_id:
            return self._openai_chat
        elif model_id in ANTHROPIC_MODELS:
            return self._anthropic_chat
        raise ValueError(f"Invalid model id: {model_id}")
```

#### 3.2 Update `core/utils.py`

Add vLLM URL loading:
```python
def load_secrets():
    secrets_path = get_root_directory() / "SECRETS"
    secrets = {}
    with open(secrets_path) as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                secrets[key] = value if value != "None" else None

    os.environ['LLAMA_API_BASE'] = secrets['LLAMA_API_BASE']
    os.environ['VLLM_BASE_URL'] = secrets.get('VLLM_BASE_URL', 'http://localhost:8000')  # NEW

    # ... rest of function
    return secrets
```

### Phase 4: Implement Efficient Batching in ICM

This is the **most critical optimization** for performance.

#### 4.1 Problem: Current Pipeline Creates N Independent Calls

In [src/experiments/ICM.py:105-151](src/experiments/ICM.py#L105-L151), the `add_train_demonstrations` function creates a unique prompt for each example by doing leave-one-out:

```python
for idx, key in enumerate(keys):  # Loop over N examples
    # For each example, create demonstrations excluding itself
    demos = {...}  # All other N-1 examples
    copy_data[key]["demonstration"] = out_sorted_demos
```

Then [src/runners/query_model.py:243-271](src/runners/query_model.py#L243-L271) creates N separate API calls:
```python
model_requests = [
    model_api(...)  # One call per example
    for data_id in prompts.keys()
]
model_responses = await asyncio.gather(*model_requests)
```

#### 4.2 Solution: Batch-Aware Prompt Construction

**Key Insight**: vLLM can process multiple prompts in a single forward pass. We need to:
1. Group examples into batches
2. Construct batched prompts that share prefix (demonstrations)
3. Use vLLM's batching to process them together

**New Function**: `get_train_preds_batched` in [src/experiments/ICM.py](src/experiments/ICM.py)

```python
# src/experiments/ICM.py

async def compute_logprobs_batched(model_api, model_id, examples_dict, batch_size=16):
    """
    Compute log probabilities for all examples using batched inference.

    Args:
        model_api: The ModelAPI instance (should route to vLLM)
        model_id: Model identifier
        examples_dict: Dict of {example_id: example_data}
        batch_size: Number of examples to process per batch

    Returns:
        Dict of {example_id: score}
    """
    example_ids = list(examples_dict.keys())
    examples = list(examples_dict.values())

    # Prepare all prompts
    all_prompts = []
    for example in examples:
        prompt = get_judge_prompt_fewshot(example, pipeline=False)
        all_prompts.append(prompt)

    # Process in batches
    scores = {}
    for i in range(0, len(all_prompts), batch_size):
        batch_ids = example_ids[i:i+batch_size]
        batch_prompts = all_prompts[i:i+batch_size]

        # Make batched request to vLLM
        responses = await model_api(
            model_id,
            batch_prompts,  # List of prompts - vLLM handles batching
            logprobs=True,
            top_logprobs=20,
            max_tokens=1,
            temperature=0.0,
        )

        # Extract scores from responses
        for example_id, response in zip(batch_ids, responses):
            logprobs = response[0]["response"]["logprobs"][0]
            score = get_yes_no_diff_logprobs(logprobs)
            scores[example_id] = score

    return scores


def get_pipeline_batched(
    model,
    name=None,
    use_cache=True,
    num_problems=None,
    decision_id=None,
    iter=None,
    assignment=None,
    batch_size=16,  # NEW parameter
):
    """
    Modified pipeline that uses batched inference.
    """
    pipeline_name = f"iterative-truth-assign-iter-{iter}"
    if decision_id is not None:
        pipeline_name += f"-{decision_id}"
    if name is not None:
        pipeline_name += "-" + name

    ROOT_DIR = get_root_directory()
    DATA_DIR = ROOT_DIR / "data"

    pipeline_config = PipelineConfig(
        pipeline_name,
        openai_fraction_rate_limit=0.99,
        num_problems=num_problems,
        use_cache=use_cache,
    )
    pipeline = Pipeline(pipeline_config)

    assert assignment is not None
    initial_assign = pipeline.add_load_data_step(
        "get_assign", load_assignments, assignment
    )

    def add_train_demonstrations(train_data):
        """Same as before - prepare leave-one-out demonstrations"""
        copy_data = deepcopy(train_data)
        copy_data = {k: v for k, v in copy_data.items() if v["label"] is not None}
        keys = list(copy_data.keys())
        values = list(copy_data.values())
        # ... [same logic as before]
        return copy_data

    merged_train_data = pipeline.add_transformation_step(
        "add_train_demonstration",
        add_train_demonstrations,
        dependencies=[initial_assign],
    )

    # NEW: Batched inference step
    async def batched_inference_step(train_data, use_cache, index):
        """
        Custom step that replaces add_query_step for batched processing.
        """
        # Compute scores using batched API calls
        scores = await compute_logprobs_batched(
            pipeline.model_api,
            model,
            train_data,
            batch_size=batch_size
        )

        # Add scores to examples
        result = {}
        for example_id, example in train_data.items():
            example_copy = example.copy()
            example_copy["score"] = scores.get(example_id, 0)
            result[example_id] = example_copy

        return result

    get_train_preds = pipeline.add_transformation_step(
        "get_train_preds_batched",
        batched_inference_step,
        dependencies=[merged_train_data],
    )

    eval_preds = pipeline.add_eval_step(
        "evaluate",
        calculate_accuracy,
        dependencies=[get_train_preds],
    )
    return pipeline
```

#### 4.3 Update ICM Main Loop

Modify [src/experiments/ICM.py:370-446](src/experiments/ICM.py#L370-L446) to use batched pipeline:

```python
async def icm_main(args):
    train, fewshot_ids = load_train_data(args)
    demonstrations, unlabeled_ids, whole_ids, seed_ids = initialize(train, fewshot_ids, args)

    # ... initialization code ...

    # NEW: Use batched pipeline
    batch_size = args.vllm_batch_size if hasattr(args, 'vllm_batch_size') else 16

    for _ in tqdm(range(args.K), desc="searching"):
        cur_pool = {
            k: v for k, v in demonstrations.items() if v["label"] is not None
        }

        if iter == 0:
            pipeline = get_pipeline_batched(  # NEW: Use batched version
                args.model,
                name=name,
                num_problems=None,
                iter=iter,
                assignment=cur_pool,
                batch_size=batch_size,  # NEW
            )
            results = await pipeline.run()
            cur_metric = results["evaluate"]

        # ... [rest of loop logic] ...

        if demonstrations[example_id]["label"] != new_label:
            tmp_demonstrations = deepcopy(demonstrations)
            tmp_demonstrations[example_id]["label"] = new_label

            tmp_pool = {
                k: v
                for k, v in tmp_demonstrations.items()
                if v["label"] is not None
            }
            pipeline = get_pipeline_batched(  # NEW: Use batched version
                model=args.model,
                name=name,
                num_problems=None,
                iter=iter,
                assignment=tmp_pool,
                batch_size=batch_size,  # NEW
            )
            results = await pipeline.run()
            metric = results["evaluate"]
            # ... [rest of acceptance logic] ...
```

#### 4.4 Add Batch Size Argument

Update `get_args()` in [src/experiments/ICM.py:245-259](src/experiments/ICM.py#L245-L259):

```python
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=27565976)
    parser.add_argument("--testbed", type=str, default="truthfulQA")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-405B")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_seed", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=1)
    parser.add_argument("--K", type=int, default=1500)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--initial_T", type=float, default=10)
    parser.add_argument("--final_T", type=float, default=0.01)
    parser.add_argument("--scheduler", type=str, default="log")
    parser.add_argument("--vllm_batch_size", type=int, default=16)  # NEW
    parser.add_argument("--use_vllm", action="store_true", default=True)  # NEW
    args = parser.parse_args()
    return args
```

### Phase 5: Remove Rate Limiting for vLLM

#### 5.1 Conditional Rate Limiting in Pipeline

Modify [src/pipeline/pipeline.py:31-43](src/pipeline/pipeline.py#L31-L43):

```python
def limit_concurrency_with_retry(model_api, use_vllm=False, retries=5):
    """
    Apply rate limiting only for external APIs, not for vLLM.
    """
    if use_vllm:
        # No rate limiting for self-hosted vLLM
        async def unlimited_model_api(*args, **kwargs):
            return await model_api(*args, **kwargs)
        return unlimited_model_api
    else:
        # Apply rate limiting for external APIs
        semaphore = asyncio.Semaphore(2)
        rate_limiter = AsyncLimiter(
            max_rate=100,
            time_period=60
        )

        async def limited_model_api(*args, **kwargs):
            async with rate_limiter:
                async with semaphore:
                    return await model_api(*args, **kwargs)

        return limited_model_api
```

#### 5.2 Update Pipeline Initialization

Modify [src/pipeline/pipeline.py:98-120](src/pipeline/pipeline.py#L98-L120):

```python
class Pipeline:
    # Class variables shared across all instances
    _model_api = None
    _limited_model_api = None
    _initialized = False

    def __init__(self, config, use_vllm=True):  # NEW parameter
        self.config = config
        self.use_vllm = use_vllm  # NEW
        self.steps = []
        self.step_names = set()
        self.results = {}

        # Initialize shared model_api and limited_model_api on first access
        if not Pipeline._initialized:
            Pipeline._model_api = ModelAPI(
                self.config.openai_fraction_rate_limit,
                self.config.organization,
                self.config.print_prompt_and_response,
                use_vllm=use_vllm,  # NEW
            )
            Pipeline._limited_model_api = limit_concurrency_with_retry(
                Pipeline._model_api,
                use_vllm=use_vllm,  # NEW
            )
            Pipeline._initialized = True

        self.file_sem = asyncio.BoundedSemaphore(self.config.num_open_files)
        self.cost = {"red": 0, "blue": 0}
```

### Phase 6: Testing & Validation

#### 6.1 Unit Tests

Create `tests/test_vllm_integration.py`:

```python
import asyncio
import pytest
from core.llm_api.vllm_llm import VLLMClient, create_vllm_client

@pytest.mark.asyncio
async def test_vllm_single_request():
    """Test single prompt inference"""
    client = create_vllm_client("http://localhost:8000")

    responses = await client(
        ["meta-llama/Meta-Llama-3.1-405B"],
        "Question: Is the sky blue?\nClaim: Yes\nI think this claim is ",
        print_prompt_and_response=False,
        max_attempts=3,
        max_tokens=1,
        top_logprobs=20,
    )

    assert len(responses) == 1
    assert responses[0].logprobs is not None
    assert len(responses[0].logprobs) > 0

@pytest.mark.asyncio
async def test_vllm_batched_request():
    """Test batched prompt inference"""
    client = create_vllm_client("http://localhost:8000", max_batch_size=4)

    prompts = [
        "Question: Is the sky blue?\nClaim: Yes\nI think this claim is ",
        "Question: Is water wet?\nClaim: Yes\nI think this claim is ",
        "Question: Is fire cold?\nClaim: Yes\nI think this claim is ",
    ]

    responses = await client(
        ["meta-llama/Meta-Llama-3.1-405B"],
        prompts,
        print_prompt_and_response=False,
        max_attempts=3,
        max_tokens=1,
        top_logprobs=20,
    )

    assert len(responses) == 3
    for response in responses:
        assert response.logprobs is not None

@pytest.mark.asyncio
async def test_icm_batched_inference():
    """Test ICM with batched inference"""
    from src.experiments.ICM import compute_logprobs_batched
    from core.llm_api.llm import ModelAPI

    model_api = ModelAPI(use_vllm=True)

    examples = {
        0: {
            "prompt": "Q: Is sky blue?\nClaim: Yes\nI think this claim is ",
            "label": 1,
            "demonstration": {}
        },
        1: {
            "prompt": "Q: Is fire cold?\nClaim: Yes\nI think this claim is ",
            "label": 0,
            "demonstration": {}
        },
    }

    scores = await compute_logprobs_batched(
        model_api,
        "meta-llama/Meta-Llama-3.1-405B",
        examples,
        batch_size=2
    )

    assert len(scores) == 2
    assert 0 in scores and 1 in scores
```

#### 6.2 Integration Test

Create `scripts/test_icm_with_vllm.sh`:

```bash
#!/bin/bash

# Test ICM with vLLM on a small dataset
python src/experiments/ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --batch_size 32 \
    --num_seed 4 \
    --K 50 \
    --vllm_batch_size 8 \
    --use_vllm \
    --seed 42
```

#### 6.3 Performance Benchmarking

Create `scripts/benchmark_vllm.py`:

```python
import asyncio
import time
from core.llm_api.llm import ModelAPI

async def benchmark_throughput():
    """
    Compare throughput: external API vs vLLM
    """
    model_id = "meta-llama/Meta-Llama-3.1-405B"

    prompts = [f"Question {i}: Is this true?\nClaim: Yes\nI think " for i in range(100)]

    # Test with external API
    print("Testing external API...")
    model_api_external = ModelAPI(use_vllm=False)
    start = time.time()
    try:
        responses = await asyncio.gather(*[
            model_api_external(model_id, prompt, max_tokens=1, logprobs=True, top_logprobs=20)
            for prompt in prompts[:10]  # Limited to avoid rate limits
        ])
        external_time = time.time() - start
        print(f"External API: {10 / external_time:.2f} requests/sec")
    except Exception as e:
        print(f"External API failed: {e}")

    # Test with vLLM
    print("\nTesting vLLM...")
    model_api_vllm = ModelAPI(use_vllm=True)
    start = time.time()
    responses = await model_api_vllm(
        model_id,
        prompts,  # All 100 prompts
        max_tokens=1,
        logprobs=True,
        top_logprobs=20
    )
    vllm_time = time.time() - start
    print(f"vLLM: {len(prompts) / vllm_time:.2f} requests/sec")
    print(f"Speedup: {external_time * 10 / vllm_time:.2f}x")

if __name__ == "__main__":
    asyncio.run(benchmark_throughput())
```

## Expected Performance Improvements

### Baseline (Current Implementation)
- **API**: External (Hyperbolic.xyz or similar)
- **Rate limit**: ~100 requests/minute
- **Batching**: None (N individual requests per iteration)
- **Time per iteration**:
  - N=256 examples
  - 256 requests at 100 req/min = ~2.5 minutes per iteration
  - K=1500 iterations = **3,750 minutes = 62.5 hours**

### With vLLM + Batching
- **API**: Self-hosted vLLM
- **Rate limit**: None
- **Batching**: B=16 examples per batch
- **Time per iteration**:
  - N=256 examples / B=16 batch_size = 16 batches
  - Assuming 2 seconds per batch (with prefix caching)
  - 16 batches × 2s = 32 seconds per iteration
  - K=1500 iterations = **48,000 seconds = 13.3 hours**

### Speedup: ~4.7x

**Additional optimizations**:
- Larger batch sizes (B=32 or B=64): Could reduce to **8-10 hours**
- Better prefix caching: Could reduce to **6-8 hours**
- Multiple GPU servers: Could reduce to **3-4 hours**

## Implementation Timeline

### Week 1: Infrastructure Setup
- [ ] Day 1-2: Deploy vLLM server with Llama-3.1-405B
- [ ] Day 3: Write and test `core/llm_api/vllm_llm.py`
- [ ] Day 4: Integrate vLLM client into ModelAPI
- [ ] Day 5: Verification and testing

### Week 2: Batching Implementation
- [ ] Day 1-2: Implement `compute_logprobs_batched()` function
- [ ] Day 3: Create `get_pipeline_batched()` with batch-aware steps
- [ ] Day 4: Update ICM main loop to use batched pipeline
- [ ] Day 5: Testing and debugging

### Week 3: Optimization & Validation
- [ ] Day 1-2: Remove rate limiting for vLLM paths
- [ ] Day 3: Performance benchmarking and tuning
- [ ] Day 4: Full ICM run with vLLM + batching
- [ ] Day 5: Compare results with baseline, document findings

## Migration Strategy

### Phase 1: Parallel Implementation (Low Risk)
1. Keep existing API infrastructure intact
2. Add vLLM as optional path (flag: `--use_vllm`)
3. Run experiments in parallel to validate equivalence

### Phase 2: Gradual Migration
1. Use vLLM for development and testing
2. Keep external API as fallback
3. Monitor for any discrepancies in results

### Phase 3: Full Migration
1. Make vLLM the default
2. Remove external API dependencies (optional)
3. Update documentation

## Configuration Reference

### Environment Variables
```bash
# In SECRETS file
VLLM_BASE_URL=http://localhost:8000
LLAMA_API_BASE=https://api.hyperbolic.xyz/v1  # Fallback

# Or export directly
export VLLM_BASE_URL=http://your-vllm-server:8000
```

### Command Line Flags
```bash
# Use vLLM with batching
python src/experiments/ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --batch_size 256 \
    --vllm_batch_size 16 \
    --use_vllm \
    --model meta-llama/Meta-Llama-3.1-405B

# Fallback to external API
python src/experiments/ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --batch_size 256 \
    --no-use_vllm
```

### vLLM Server Configuration
```yaml
# configs/vllm_large.yaml (for 405B model)
model: "meta-llama/Meta-Llama-3.1-405B"
tensor_parallel_size: 8
gpu_memory_utilization: 0.9
max_model_len: 8192
enable_prefix_caching: true
max_num_seqs: 256
max_num_batched_tokens: 32768

# configs/vllm_small.yaml (for 70B model, faster)
model: "meta-llama/Llama-3.1-70B"
tensor_parallel_size: 4
gpu_memory_utilization: 0.9
max_model_len: 8192
enable_prefix_caching: true
max_num_seqs: 128
max_num_batched_tokens: 16384
```

## Troubleshooting

### Issue: vLLM Server OOM (Out of Memory)
**Solution**:
- Reduce `max_num_seqs` or `max_num_batched_tokens`
- Reduce `max_model_len`
- Reduce `gpu_memory_utilization` to 0.8
- Use smaller model (70B instead of 405B)

### Issue: Slow Inference Despite Batching
**Solution**:
- Check if prefix caching is enabled (`--enable-prefix-caching`)
- Increase `max_num_seqs` to allow larger batches
- Verify GPU utilization with `nvidia-smi`
- Check for network bottlenecks if server is remote

### Issue: Logprobs Format Mismatch
**Solution**:
- vLLM's logprobs format matches OpenAI
- Verify `extract_claim_logprobs()` handles the format correctly
- Add debug logging to inspect raw responses

### Issue: Model Weights Not Found
**Solution**:
- Ensure model is downloaded: `huggingface-cli download meta-llama/Meta-Llama-3.1-405B`
- Check HuggingFace token: `huggingface-cli login`
- Use `--trust-remote-code` if needed

## Summary of Changes

### New Files
1. `core/llm_api/vllm_llm.py` - vLLM client implementation
2. `scripts/launch_vllm_server.sh` - Server launch script
3. `scripts/test_vllm_server.py` - Server verification
4. `scripts/benchmark_vllm.py` - Performance benchmarking
5. `configs/vllm_config.yaml` - vLLM configuration
6. `tests/test_vllm_integration.py` - Integration tests

### Modified Files
1. `core/llm_api/llm.py` - Add vLLM routing and initialization
2. `core/utils.py` - Load vLLM URL from SECRETS
3. `src/experiments/ICM.py` - Implement batched pipeline and inference
4. `src/pipeline/pipeline.py` - Conditional rate limiting
5. `SECRETS` - Add VLLM_BASE_URL
6. `CLAUDE.md` - Update with vLLM documentation

### Lines of Code
- **Added**: ~800 lines
- **Modified**: ~200 lines
- **Deleted**: ~50 lines (rate limiting logic)

## Questions & Decisions Needed

1. **GPU Resources**: What GPUs are available? This determines whether to use 405B, 70B, or 8B model.

2. **Batch Size**: Needs tuning based on GPU memory. Start with B=16, can increase to B=32 or B=64.

3. **Prefix Caching**: Critical for performance. Verify it works correctly with your prompts.

4. **Multiple Servers**: If you have access to multiple GPU nodes, can deploy multiple vLLM servers and load balance.

5. **Model Selection**: 405B gives best quality but slowest. 70B is 3-5x faster with similar quality. 8B is 10x faster but lower quality.

6. **Fallback Strategy**: Keep external API as backup, or fully commit to vLLM?

## Next Steps

1. **Start with Phase 1**: Deploy vLLM server and verify it works
2. **Create vLLM client**: Implement `core/llm_api/vllm_llm.py`
3. **Test integration**: Run small-scale ICM experiment
4. **Implement batching**: Add `compute_logprobs_batched()` and batched pipeline
5. **Full-scale test**: Run complete ICM with K=1500 iterations
6. **Optimize**: Tune batch sizes and server configuration
7. **Document**: Update CLAUDE.md with deployment guide

---

**This refactoring will transform your ICM algorithm from being rate-limit bottlenecked to GPU-compute bottlenecked, which is exactly where you want to be for maximum efficiency.**
