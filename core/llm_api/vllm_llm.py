"""
vLLM Client Module for Self-Hosted LLM Inference

This module provides clients for vLLM inference:
1. VLLMClient - HTTP client for remote vLLM servers
2. VLLMInProcessClient - Direct in-process inference (no HTTP overhead)

Key features:
- No rate limiting (self-hosted)
- Batched inference support for efficiency
- Logprobs extraction for ICM algorithm
- In-process mode eliminates network overhead
"""

import asyncio
import logging
import time
from typing import List, Union, Optional
import attrs

from core.llm_api.base_llm import LLMResponse, ModelAPIProtocol, messages_to_single_prompt

LOGGER = logging.getLogger(__name__)

# Lazy import for vLLM to avoid import errors when not using in-process mode
_vllm_llm = None
_sampling_params = None

def _get_vllm_imports():
    """Lazy import vLLM modules to avoid import overhead when not needed."""
    global _vllm_llm, _sampling_params
    if _vllm_llm is None:
        from vllm import LLM, SamplingParams
        _vllm_llm = LLM
        _sampling_params = SamplingParams
    return _vllm_llm, _sampling_params



@attrs.define()
class VLLMInProcessClient(ModelAPIProtocol):
    """
    In-process vLLM client for direct inference without HTTP overhead.

    This client loads the model directly into GPU memory and performs
    inference in the same process. Ideal when the model server and
    client are on the same machine.

    Key advantages over HTTP client:
    - No network serialization overhead
    - No HTTP request/response latency
    - Direct access to vLLM's prefix caching
    - Better GPU memory utilization
    """

    model_name: str = attrs.field()
    tensor_parallel_size: int = attrs.field(default=1)
    gpu_memory_utilization: float = attrs.field(default=0.90)
    max_model_len: Optional[int] = attrs.field(default=None)
    max_num_batched_tokens: Optional[int] = attrs.field(default=None)
    enable_prefix_caching: bool = attrs.field(default=True)
    print_prompt_and_response: bool = attrs.field(default=False)
    dtype: str = attrs.field(default="auto")

    _llm: Optional[object] = attrs.field(init=False, default=None)
    _initialized: bool = attrs.field(init=False, default=False)

    def _ensure_initialized(self):
        """Lazily initialize the vLLM engine on first use."""
        if self._initialized:
            return

        LLM, _ = _get_vllm_imports()

        LOGGER.info(f"Initializing vLLM in-process engine for {self.model_name}")
        LOGGER.info(f"  tensor_parallel_size: {self.tensor_parallel_size}")
        LOGGER.info(f"  gpu_memory_utilization: {self.gpu_memory_utilization}")
        LOGGER.info(f"  enable_prefix_caching: {self.enable_prefix_caching}")

        init_kwargs = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enable_prefix_caching": self.enable_prefix_caching,
            "max_num_seqs": 256,
            "dtype": self.dtype,
            "trust_remote_code": True,
        }

        if self.max_model_len is not None:
            init_kwargs["max_model_len"] = self.max_model_len

        if self.max_num_batched_tokens is not None:
            init_kwargs["max_num_batched_tokens"] = self.max_num_batched_tokens

        self._llm = LLM(**init_kwargs)
        self._initialized = True
        LOGGER.info("vLLM engine initialized successfully")

    async def __call__(
        self,
        model_ids: list[str],
        prompt: Union[str, list[str], list[dict]],
        print_prompt_and_response: bool,
        max_attempts: int,
        **kwargs,
    ) -> list[LLMResponse]:
        """
        Main entry point for in-process vLLM inference.

        Args:
            model_ids: List of model IDs (ignored, uses configured model)
            prompt: Single prompt string, list of prompts, or chat messages
            print_prompt_and_response: Whether to print I/O
            max_attempts: Number of retry attempts (less relevant for in-process)
            **kwargs: Additional parameters (max_tokens, temperature, etc.)

        Returns:
            List of LLMResponse objects
        """
        # Ensure engine is initialized
        self._ensure_initialized()

        model_id = model_ids[0] if model_ids else self.model_name

        # Convert chat format to text if needed
        if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
            prompt = messages_to_single_prompt(prompt)

        # Ensure prompt is a list for consistent handling
        if isinstance(prompt, str):
            prompts = [prompt]
            single_prompt = True
        else:
            prompts = prompt
            single_prompt = False

        start_time = time.time()

        # Run inference (vLLM's generate is synchronous, run in executor to not block)
        loop = asyncio.get_event_loop()
        responses = await loop.run_in_executor(
            None,
            lambda: self._generate_sync(model_id, prompts, start_time, **kwargs)
        )

        # Format response
        if single_prompt:
            return [{"prompt": prompts[0], "response": responses[0].to_dict()}]
        else:
            return [{"prompt": p, "response": r.to_dict()} for p, r in zip(prompts, responses)]

    def _generate_sync(
        self,
        model_id: str,
        prompts: list[str],
        start_time: float,
        **kwargs
    ) -> list[LLMResponse]:
        """
        Synchronous generation using vLLM's LLM.generate().

        Args:
            model_id: Model identifier (for response metadata)
            prompts: List of prompt strings
            start_time: Timestamp when request started
            **kwargs: Sampling parameters

        Returns:
            List of LLMResponse objects
        """
        _, SamplingParams = _get_vllm_imports()

        # Build sampling params
        sampling_params = SamplingParams(
            max_tokens=kwargs.get("max_tokens", 1),
            temperature=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", 1.0),
            n=kwargs.get("n", 1),
            logprobs=kwargs.get("top_logprobs", 20),  # Number of top logprobs to return
        )

        LOGGER.debug(f"Generating for {len(prompts)} prompts with params: {sampling_params}")

        api_start = time.time()

        # Run vLLM generation
        outputs = self._llm.generate(prompts, sampling_params)

        api_duration = time.time() - api_start
        duration = time.time() - start_time

        LOGGER.debug(f"Generation completed in {api_duration:.2f}s for {len(prompts)} prompts")

        # Convert vLLM outputs to LLMResponse format
        responses = []
        for output in outputs:
            # vLLM returns one output per prompt, with potentially multiple completions
            # We take the first completion (n=1 typically for ICM)
            completion_output = output.outputs[0]

            # Extract logprobs - required for ICM algorithm
            if completion_output.logprobs is None or len(completion_output.logprobs)==0:
                raise RuntimeError(
                    "Logprobs are None/Empty. ICM requires logprobs for scoring. "
                    "Ensure the SamplingParams include logprobs > 0. "
                    f"Prompt: {output.prompt[:100]}..."
                )

            # vLLM returns list of logprobs dicts, one per generated token
            # Each dict maps token_id -> Logprob object
            # We need to convert to format: list of {token_str: logprob_value}
            logprobs_list = []
            token_logprobs = completion_output.logprobs[0]
            if token_logprobs is not None:
                # Convert to {token_string: logprob_value} format
                token_dict = {}
                for token_id, logprob_obj in token_logprobs.items():
                    # logprob_obj has .decoded_token and .logprob attributes
                    token_dict[logprob_obj.decoded_token] = logprob_obj.logprob
                logprobs_list.append(token_dict)

            responses.append(
                LLMResponse(
                    model_id=model_id,
                    completion=completion_output.text,
                    stop_reason=completion_output.finish_reason or "stop",
                    api_duration=api_duration,
                    duration=duration,
                    cost=0.0,  # Self-hosted, no per-token cost
                    logprobs=logprobs_list[0]
                )
            )

        if self.print_prompt_and_response or kwargs.get("print_prompt_and_response", False):
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                print(f"\n{'='*80}\nPrompt {i+1}:\n{prompt}\n\nResponse:\n{response.completion}\n{'='*80}")

        return responses
