# %%
import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime
from itertools import cycle
from traceback import format_exc
from typing import Optional, Union

import attrs
import openai
from openai import AsyncOpenAI
import requests
import tiktoken
from tenacity import retry, stop_after_attempt, wait_fixed
from termcolor import cprint

from core.llm_api.base_llm import (
    PRINT_COLORS,
    LLMResponse,
    ModelAPIProtocol,
    messages_to_single_prompt,
)

OAIChatPrompt = list[dict[str, str]]
OAIBasePrompt = Union[str, list[str]]
LOGGER = logging.getLogger(__name__)

def count_tokens(text: str) -> int:
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def price_per_token(model_id: str) -> tuple[float, float]:
    """
    Returns the (input token, output token) price for the given model id.
    """
    if model_id == "gpt-4-1106-preview":
        prices = 0.01, 0.03
    elif model_id == "gpt-3.5-turbo-1106":
        prices = 0.001, 0.002
    elif model_id.startswith("gpt-4"):
        prices = 0.03, 0.06
    elif model_id.startswith("gpt-4-32k"):
        prices = 0.06, 0.12
    elif model_id.startswith("gpt-3.5-turbo-16k"):
        prices = 0.003, 0.004
    elif model_id.startswith("gpt-3.5-turbo"):
        prices = 0.0015, 0.002
    elif model_id == "davinci-002":
        prices = 0.002, 0.002
    elif model_id == "babbage-002":
        prices = 0.0004, 0.0004
    elif model_id == "text-davinci-003" or model_id == "text-davinci-002":
        prices = 0.02, 0.02
    elif "ft:gpt-3.5-turbo" in model_id:
        prices = 0.012, 0.016
    elif "llama" in model_id.lower() or "mixtral" in model_id.lower():
        prices = 0.0015, 0.002
    elif "o1" in model_id.lower():
        prices = 0.01, 0.03
    else:
        prices = 0, 0
        # raise ValueError(f"Invalid model id: {model_id}")

    return tuple(price / 1000 for price in prices)


@attrs.define()
class Resource:
    """
    A resource that is consumed over time and replenished at a constant rate.
    """

    refresh_rate: float = (
        attrs.field()
    )  # How many units of the resource are replenished per minute
    value: float = attrs.field(init=False)
    total: float = 0
    throughput: float = 0
    last_update_time: float = attrs.field(init=False, factory=time.time)
    start_time: float = attrs.field(init=False, factory=time.time)

    def __attrs_post_init__(self):
        self.value = self.refresh_rate

    def _replenish(self):
        """
        Updates the value of the resource based on the time since the last update.
        """
        curr_time = time.time()
        self.value = min(
            self.refresh_rate,
            self.value + (curr_time - self.last_update_time) * self.refresh_rate / 60,
        )
        self.last_update_time = curr_time
        self.throughput = self.total / (curr_time - self.start_time) * 60

    def geq(self, amount: float) -> bool:
        self._replenish()
        return self.value >= amount

    def consume(self, amount: float):
        """
        Consumes the given amount of the resource.
        """
        assert self.geq(
            amount
        ), f"Resource does not have enough capacity to consume {amount} units"
        self.value -= amount
        self.total += amount


@attrs.define
class OpenAIModel(ModelAPIProtocol):
    frac_rate_limit: float
    organization: str
    print_prompt_and_response: bool = False
    model_ids: set[str] = attrs.field(init=False, default=attrs.Factory(set))

    # OpenAI clients (v2.x API)
    llama_client: AsyncOpenAI = attrs.field(init=False)  # For Llama models with custom base URL

    # rate limit
    token_capacity: dict[str, Resource] = attrs.field(
        init=False, default=attrs.Factory(dict)
    )
    request_capacity: dict[str, Resource] = attrs.field(
        init=False, default=attrs.Factory(dict)
    )
    lock_add: asyncio.Lock = attrs.field(
        init=False, default=attrs.Factory(asyncio.Lock)
    )
    lock_consume: asyncio.Lock = attrs.field(
        init=False, default=attrs.Factory(asyncio.Lock)
    )

    def __attrs_post_init__(self):
        # Initialize Llama client with custom base URL
        llama_api_base = os.environ.get('NEW_LLAMA_API_BASE', 'https://api.hyperbolic.xyz/v1')
        self.llama_client = AsyncOpenAI(
            api_key=openai.api_key,
            base_url=llama_api_base
        )

    @staticmethod
    def _assert_valid_id(model_id: str):
        raise NotImplementedError

    @staticmethod
    async def _get_dummy_response_header(model_id: str):
        raise NotImplementedError

    @staticmethod
    def _count_prompt_token_capacity(prompt, **kwargs) -> int:
        raise NotImplementedError

    @staticmethod
    def _print_prompt_and_response(prompt, responses):
        raise NotImplementedError

    @staticmethod
    def _create_prompt_history_file(prompt):
        filename = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}_prompt.txt"
        with open(os.path.join("prompt_history", filename), "w") as f:
            json_str = json.dumps(prompt, indent=4)
            json_str = json_str.replace("\\n", "\n")
            f.write(json_str)

        return filename

    @staticmethod
    def _add_response_to_prompt_file(prompt_file, responses):
        with open(os.path.join("prompt_history", prompt_file), "a") as f:
            f.write("\n\n======RESPONSE======\n\n")
            json_str = json.dumps(
                [response.to_dict() for response in responses], indent=4
            )
            json_str = json_str.replace("\\n", "\n")
            f.write(json_str)

    async def add_model_id(self, model_id: str):
        self._assert_valid_id(model_id)
        if model_id in self.model_ids:
            return

        # make dummy request to get token and request capacity
        model_metadata = await self._get_dummy_response_header(model_id)
        token_capacity = int(model_metadata["x-ratelimit-limit-tokens"])
        request_capacity = int(model_metadata["x-ratelimit-limit-requests"])
        print(
            f"got capacities for model {model_id}: {token_capacity}, {request_capacity}"
        )
        tokens_consumed = token_capacity - int(
            model_metadata["x-ratelimit-remaining-tokens"]
        )
        requests_consumed = request_capacity - int(
            model_metadata["x-ratelimit-remaining-requests"]
        )
        print(
            f"consumed capacities for model {model_id}: {tokens_consumed}, {requests_consumed}"
        )
        token_cap = token_capacity * self.frac_rate_limit
        request_cap = request_capacity * self.frac_rate_limit
        if model_id in BASE_MODELS:
            token_cap *= (
                10000  # openai does not track token limit so we can increase it
            )

        print(f"setting cap for model {model_id}: {token_cap}, {request_cap}")
        self.model_ids.add(model_id)
        token_capacity = Resource(token_cap)
        request_capacity = Resource(request_cap)
        token_capacity.consume(min(token_cap, tokens_consumed))
        request_capacity.consume(min(request_cap, requests_consumed))
        self.token_capacity[model_id] = token_capacity
        self.request_capacity[model_id] = request_capacity

    async def __llama_call__(
        self,
        model_ids: list[str],
        prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        **kwargs,
    ) -> list[LLMResponse]:
        kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in ("save_path", "metadata", "api_base")  # Remove api_base from kwargs
        }

        start = time.time()

        async def attempt_api_call():
            for model_id in cycle(model_ids):
                return await asyncio.wait_for(
                    self._make_api_call_llama(prompt, model_id, start, **kwargs),
                    timeout=100,  # cloudflare has a 100-second limit for a connection to remain open:  https://docs.runpod.io/pods/configuration/expose-ports
                )

        model_ids.sort(
            key=lambda model_id: price_per_token(model_id)[0]
        )  # Default to cheapest model
        model_id = model_ids[0]
        prompt = self._process_prompt(prompt)
        # prompt_file = self._create_prompt_history_file(prompt)
        responses: Optional[list[LLMResponse]] = None
        for i in range(max_attempts):
            try:
                responses = await attempt_api_call()
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                print('error_info', error_info)
                LOGGER.warn(
                    f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})"
                )
                await asyncio.sleep(1.5**i)
            else:
                break

        if responses is None:
            raise RuntimeError(
                f"Failed to get a response from the API after {max_attempts} attempts."
            )

        if self.print_prompt_and_response or print_prompt_and_response:
            self._print_prompt_and_response(prompt, responses)

        end = time.time()
        LOGGER.debug(f"Completed call to {model_id} in {end - start}s.")
        return [
            {"prompt": prompt, "response": response.to_dict()} for response in responses
        ]

    async def __call__(
        self,
        model_ids: list[str],
        prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        **kwargs,
    ) -> list[LLMResponse]:
        if "gpt" not in model_ids[0]:
            return await self.__llama_call__(
                model_ids, prompt, print_prompt_and_response, max_attempts, **kwargs
            )


_GPT_4_MODELS = [
    "gpt-4o",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-0125-preview",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-4-1106-preview",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-11-20",
    "o1-preview-2024-09-12",
    "o1-mini-2024-09-12",
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.2-3b-instruct",
    "meta-llama/llama-3.2-1b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-7b-instruct",
    "meta-llama/llama-3-8b-instruct",
    "allenai/olmo-7b-instruct",
    "01-ai/yi-large",
    "meta-llama/llama-2-70b-chat",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "qwen/qwen-2.5-7b-instruct",
    "openai/gpt-4o",
    "openchat/openchat-7b",
    "ai21/jamba-instruct",
    "neversleep/llama-3.1-lumimaid-8b",
    "mistralai/mixtral-8x7b-instruct:nitro",
    "deepseek/deepseek-r1",
    "deepseek/deepseek-r1-distill-llama-70b",
    "minimax/minimax-01",
    "microsoft/phi-4",
    "qwen/qvq-72b-preview",
]
_GPT_TURBO_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
]
GPT_CHAT_MODELS = set(_GPT_4_MODELS + _GPT_TURBO_MODELS)


class OpenAIChatModel(OpenAIModel):
    def _process_prompt(self, prompt: OAIChatPrompt) -> OAIChatPrompt:
        return prompt

    def _assert_valid_id(self, model_id: str):
        if "ft:" in model_id:
            model_id = model_id.split(":")[1]
        assert model_id in GPT_CHAT_MODELS, f"Invalid model id: {model_id}"

    @retry(stop=stop_after_attempt(8), wait=wait_fixed(2))
    async def _get_dummy_response_header(self, model_id: str):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}",
            "OpenAI-Organization": self.organization,
        }
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Say 1"}],
        }
        response = requests.post(url, headers=headers, json=data)
        if "x-ratelimit-limit-tokens" not in response.headers:
            raise RuntimeError("Failed to get dummy response header")
        return response.headers

    @staticmethod
    def _count_prompt_token_capacity(prompt: OAIChatPrompt, **kwargs) -> int:
        # The magic formula is: .25 * (total number of characters) + (number of messages) + (max_tokens, or 15 if not specified)
        BUFFER = 5  # A bit of buffer for some error margin
        MIN_NUM_TOKENS = 20

        num_tokens = 0
        for message in prompt:
            num_tokens += 1
            num_tokens += len(message["content"]) / 4

        return max(
            MIN_NUM_TOKENS,
            int(num_tokens + BUFFER)
            + kwargs.get("n", 1) * kwargs.get("max_tokens", 15),
        )

    async def _make_api_call_llama(
        self, prompt: OAIChatPrompt, model_id, start_time, **params
    ) -> list[LLMResponse]:
        """Make API call for Llama chat models using custom base URL client."""
        LOGGER.debug(f"Making {model_id} call (Llama Chat)")
        print(f"Making {model_id} call (Llama Chat) in OpenAIChatModel")
        
        api_start = time.time()
        prompt_ = [{"role": "user", "content": prompt}]

        api_response = await self.llama_client.chat.completions.create(
            messages=prompt_,
            model=model_id,
            **params
        )

        api_duration = time.time() - api_start
        duration = time.time() - start_time
        return [
            LLMResponse(
                model_id=model_id,
                completion=choice.message.content
                if "tools" not in params
                else choice.message.tool_calls[0].function.arguments,
                stop_reason=choice.finish_reason,
                api_duration=api_duration,
                duration=duration,
                cost=0,  # No cost for self-hosted Llama
                logprobs=choice.logprobs.top_logprobs
                if choice.logprobs is not None
                else None,
            )
            for choice in api_response.choices
        ]

    @staticmethod
    def _print_prompt_and_response(
        prompts: OAIChatPrompt, responses: list[LLMResponse]
    ):
        for prompt in prompts:
            role, text = prompt["role"], prompt["content"]
            cprint(f"=={role.upper()}:", "white")
            cprint(text, PRINT_COLORS[role])
        for i, response in enumerate(responses):
            if len(responses) > 1:
                cprint(f"==RESPONSE {i + 1} ({response.model_id}):", "white")
            cprint(response.completion, PRINT_COLORS["assistant"], attrs=["bold"])
        print()


BASE_MODELS = {
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Meta-Llama-3.1-405B",
    "meta-llama/Meta-Llama-3.1-70B",
    "meta-llama/Meta-Llama-3.1-8B"
}



