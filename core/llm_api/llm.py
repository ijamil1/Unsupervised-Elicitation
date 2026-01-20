import asyncio
import json
import logging
import os
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import attrs

from core.llm_api.base_llm import LLMResponse, ModelAPIProtocol
from core.llm_api.openai_llm import (
    BASE_MODELS,
    GPT_CHAT_MODELS,
    OpenAIChatModel,
)
from core.llm_api.vllm_llm import VLLMInProcessClient
from core.utils import load_secrets

LOGGER = logging.getLogger(__name__)


@attrs.define()
class ModelAPI:

    openai_fraction_rate_limit: float = attrs.field(
        default=0.99, validator=attrs.validators.lt(1)
    )
    organization: None = None
    print_prompt_and_response: bool = False

    # vLLM configuration
    use_vllm: bool = True  # Flag to enable vLLM (default: True)

    # In-process vLLM configuration
    vllm_model_name: str = None  # Model name for in-process mode
    vllm_tensor_parallel_size: int = 1  # Number of GPUs for tensor parallelism
    vllm_gpu_memory_utilization: float = 0.90  # Fraction of GPU memory to use
    vllm_max_model_len: int = None  # Maximum sequence length
    vllm_max_num_batched_tokens: int = None  # Maximum number of batched tokens
    vllm_enable_prefix_caching: bool = True  # Enable prefix caching
    vllm_dtype: str = "auto"  # Data type

    _openai_chat: OpenAIChatModel = attrs.field(init=False)
    _vllm_client: VLLMInProcessClient = attrs.field(init=False)

    running_cost: float = attrs.field(init=False, default=0)
    model_timings: dict[str, list[float]] = attrs.field(init=False, default={})
    model_wait_times: dict[str, list[float]] = attrs.field(init=False, default={})

    def __attrs_post_init__(self):
        secrets = load_secrets()
        if self.organization is None:
            self.organization = "NYU_ORG"


        self._openai_chat = OpenAIChatModel(
            frac_rate_limit=self.openai_fraction_rate_limit,
            organization=secrets[self.organization],
            print_prompt_and_response=self.print_prompt_and_response,
        )

        # Initialize vLLM client based on mode
        if self.use_vllm:
            # In-process mode: load model directly into GPU memory
            model_name = self.vllm_model_name or secrets.get('VLLM_MODEL_NAME', 'meta-llama/Meta-Llama-3.1-8B')
            self._vllm_client = VLLMInProcessClient(
                model_name=model_name,
                tensor_parallel_size=self.vllm_tensor_parallel_size,
                gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                max_model_len=self.vllm_max_model_len,
                max_num_batched_tokens=self.vllm_max_num_batched_tokens,
                enable_prefix_caching=self.vllm_enable_prefix_caching,
                dtype=self.vllm_dtype,
                print_prompt_and_response=self.print_prompt_and_response,
            )
            LOGGER.info(f"Initialized vLLM in-process client for {model_name}")
           

        Path("./prompt_history").mkdir(exist_ok=True)

    @staticmethod
    def _load_from_cache(save_file):
        if not os.path.exists(save_file):
            return None
        else:
            with open(save_file) as f:
                cache = json.load(f)
            return cache

    async def call_single(
        self,
        model_ids: Union[str, list[str]],
        prompt: Union[list[dict[str, str]], str],
        max_tokens: int,
        print_prompt_and_response: bool = False,
        n: int = 1,
        max_attempts_per_api_call: int = 10,
        num_candidates_per_completion: int = 1,
        # is_valid: Callable[[str], bool] = lambda _: True,
        parse_fn=lambda _: True,
        insufficient_valids_behaviour: Literal[
            "error", "continue", "pad_invalids"
        ] = "error",
        **kwargs,
    ) -> str:
        assert n == 1, f"Expected a single response. {n} responses were requested."
        responses = await self(
            model_ids,
            prompt,
            max_tokens,
            print_prompt_and_response,
            n,
            max_attempts_per_api_call,
            num_candidates_per_completion,
            parse_fn,
            insufficient_valids_behaviour,
            **kwargs,
        )
        assert len(responses) == 1, "Expected a single response."
        return responses[0].completion

    async def __call__(
        self,
        model_ids: Union[str, list[str]],
        prompt: Union[list[dict[str, str]], str],
        print_prompt_and_response: bool = False,
        n: int = 1,
        max_attempts_per_api_call: int = 5,
        num_candidates_per_completion: int = 1,
        parse_fn=None,
        use_cache: bool = True,
        file_sem: asyncio.Semaphore = None,
        insufficient_valids_behaviour: Literal[
            "error", "continue", "pad_invalids"
        ] = "error",
        **kwargs,
    ) -> list[LLMResponse]:
        """
        Make maximally efficient API requests for the specified model(s) and prompt.

        Args:
            model_ids: The model(s) to call. If multiple models are specified, the output will be sampled from the
                cheapest model that has capacity. All models must be from the same class (e.g. OpenAI Base,
                OpenAI Chat, or Anthropic Chat). Anthropic chat will error if multiple models are passed in.
                Passing in multiple models could speed up the response time if one of the models is overloaded.
            prompt: The prompt to send to the model(s). Type should match what's expected by the model(s).
            max_tokens: The maximum number of tokens to request from the API (argument added to
                standardize the Anthropic and OpenAI APIs, which have different names for this).
            print_prompt_and_response: Whether to print the prompt and response to stdout.
            n: The number of completions to request.
            max_attempts_per_api_call: Passed to the underlying API call. If the API call fails (e.g. because the
                API is overloaded), it will be retried this many times. If still fails, an exception will be raised.
            num_candidates_per_completion: How many candidate completions to generate for each desired completion. n*num_candidates_per_completion completions will be generated, then is_valid is applied as a filter, then the remaining completions are returned up to a maximum of n.
            parse_fn: post-processing on the generated response
            save_path: cache path
            use_cache: whether to load from the cache or overwrite it
        """
        assert (
            "max_tokens_to_sample" not in kwargs
        ), "max_tokens_to_sample should be passed in as max_tokens."

        if isinstance(model_ids, str):
            model_ids = [model_ids]
            # # trick to double rate limit for most recent model only

        def model_id_to_class(model_id: str) -> ModelAPIProtocol:
            # NEW: Route base models to vLLM if enabled
            if self.use_vllm and model_id in BASE_MODELS:
                return self._vllm_client
            elif model_id in GPT_CHAT_MODELS or "ft:gpt-3.5-turbo" in model_id:
                return self._openai_chat
            raise ValueError(f"Invalid model id: {model_id}")

        model_classes = [model_id_to_class(model_id) for model_id in model_ids]

        if len(set(str(type(x)) for x in model_classes)) != 1:
            raise ValueError("All model ids must be of the same type.")

        max_tokens = (
            kwargs.get("max_tokens") if kwargs.get("max_tokens") is not None else 2000
        )
        model_class = model_classes[0]
      
        kwargs["max_tokens"] = max_tokens
        num_candidates = 1
        
        responses = await model_class(
            model_ids,
            prompt,
            print_prompt_and_response,
            max_attempts_per_api_call,
            n=num_candidates,
            **kwargs,
        )
         
        modified_responses = []
        for response in responses:
            self.running_cost += response["response"]["cost"]
            if kwargs.get("metadata") is not None:
                response["metadata"] = kwargs.get("metadata")
            if parse_fn is not None:
                response = parse_fn(response)

            self.model_timings.setdefault(response["response"]["model_id"], []).append(
                response["response"]["api_duration"]
            )
            self.model_wait_times.setdefault(
                response["response"]["model_id"], []
            ).append(
                response["response"]["duration"] - response["response"]["api_duration"]
            )
            modified_responses.append(response)

        return modified_responses

    def reset_cost(self):
        self.running_cost = 0

    def shutdown(self):
        """Gracefully shutdown the vLLM engine."""
        if self.use_vllm and hasattr(self, '_vllm_client') and self._vllm_client is not None:
            self._vllm_client.shutdown()
