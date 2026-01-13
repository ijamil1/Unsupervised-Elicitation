"""
vLLM Client Module for Self-Hosted LLM Inference

This module provides a client for interacting with self-hosted vLLM servers.
Key features:
- No rate limiting (self-hosted)
- Batched inference support for efficiency
- OpenAI-compatible API
- Logprobs extraction for ICM algorithm
"""

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
    Supports batched inference for maximum efficiency.
    """

    base_url: str = attrs.field()
    print_prompt_and_response: bool = False
    timeout: int = 300  # Longer timeout for batch processing

    def __attrs_post_init__(self):
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

        Args:
            model_ids: List of model IDs (vLLM serves one model, uses first)
            prompt: Single prompt string, list of prompts, or chat messages
            print_prompt_and_response: Whether to print I/O
            max_attempts: Number of retry attempts
            **kwargs: Additional parameters (max_tokens, temperature, etc.)

        Returns:
            List of LLMResponse objects
        """
        model_id = model_ids[0]  # vLLM only serves one model at a time

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

        # Make request with retries
        for attempt in range(max_attempts):
            try:
                responses = await self._make_vllm_request(
                    model_id, prompts, start_time, **kwargs
                )

                # If original input was single prompt, return wrapped response
                if single_prompt:
                    return [{"prompt": prompts[0], "response": responses[0].to_dict()}]
                else:
                    return [{"prompt": p, "response": r.to_dict()} for p, r in zip(prompts, responses)]

            except Exception as e:
                LOGGER.warning(f"vLLM request failed (attempt {attempt+1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1.5 ** attempt)
                else:
                    raise RuntimeError(f"Failed to get response from vLLM after {max_attempts} attempts: {e}")

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
        This is the key optimization that eliminates N separate API calls.

        Args:
            model_id: Model identifier
            prompts: List of prompt strings (can be 1 or many)
            start_time: Timestamp when request started
            **kwargs: API parameters

        Returns:
            List of LLMResponse objects, one per prompt
        """
        api_start = time.time()

        # Prepare request payload
        # Note: vLLM uses 'logprobs' parameter to specify top-k logprobs
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

        LOGGER.debug(f"Making vLLM request with {len(prompts)} prompts")

        # Make async HTTP request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"vLLM request failed with status {response.status}: {error_text}")

                result = await response.json()

        api_duration = time.time() - api_start
        duration = time.time() - start_time

        LOGGER.debug(f"vLLM request completed in {api_duration:.2f}s for {len(prompts)} prompts")

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


def create_vllm_client(base_url: str) -> VLLMClient:
    """
    Factory function to create a vLLM client.

    Args:
        base_url: URL of vLLM server (e.g., http://localhost:8000)

    Returns:
        VLLMClient instance
    """
    return VLLMClient(base_url=base_url)
