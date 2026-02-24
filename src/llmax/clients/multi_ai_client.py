"""This module contains the MultiAIClient class.

This class is used to interface with multiple LLMs and AI models, supporting both
synchronous and asynchronous operations.
"""

import asyncio
import base64
import json
import threading
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from io import BufferedReader, BytesIO
from queue import Queue
from typing import Any, Literal

import httpx
from google import genai  # type: ignore[reportMissingTypeStubs]
from google.genai import types as genai_types  # type: ignore[reportMissingTypeStubs]
from openai import NOT_GIVEN, BadRequestError, RateLimitError
from openai.types import CompletionUsage, Embedding
from openai.types.audio import Transcription, TranscriptionVerbose
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDeltaToolCall
from openai.types.responses import ParsedResponse

from llmax.external_clients.clients import Client, get_aclient, get_client
from llmax.messages import Messages
from llmax.messages.message import Message
from llmax.models.deployment import Deployment
from llmax.models.fake import fake_llm
from llmax.models.models import (
    ANTHROPIC_MODELS,
    GEMINI_MODELS,
    MISTRAL_MODELS,
    Model,
    SpeechModelAllowVerboseJson,
)
from llmax.usage import ModelUsage, tokens
from llmax.utils import (
    StreamedItem,
    StreamItemContent,
    StreamItemOutput,
    ToolItem,
    ToolItemContent,
    logger,
)
from llmax.utils.types import ModelItemContent


async def _default_get_usage() -> float:
    return 0.0


async def _default_increment_usage(
    _usage: float,
    _model: Model,
    _user_id: str,
    _cost: float,
    _limit: float | None,
    _retries: int,
    _window: int,
    _context: str,
    _action: str,
    _mode: str,
    _code: int,
) -> bool:
    return True


class MultiAIClient:
    """Class to interface with multiple LLMs and AI models.

    This class supports both synchronous and asynchronous operations for obtaining
    chat completions, streaming responses, and generating text embeddings, with
    an emphasis on tracking and managing API usage.

    Attributes:
        deployments: A mapping from models to their deployment objects.
        get_usage: A function to get the current usage.
        increment_usage: A function to increment usage.
        total_usage: The total usage accumulated by the client. (Mainly for dev purposes, does not handles errors properly)
    """

    def __init__(  # noqa: PLR0913
        self,
        deployments: dict[Model, Deployment],
        fallback_models: dict[Model, list[Model]],
        get_usage: Callable[[], Awaitable[float]] = _default_get_usage,
        increment_usage: Callable[
            [float, Model, str, float, float | None, int, int, str, str, str, int],
            Awaitable[bool],
        ] = _default_increment_usage,
        httpx_client: httpx.Client | None = None,
        httpx_aclient: httpx.AsyncClient | None = None,
    ) -> None:
        """Initializes the MultiAIClient class.

        Args:
            deployments: A mapping from models to their deployment objects.
            fallback_models: The fallbacks models to use when not available.
            get_usage: A function to get the current usage.
            increment_usage: A function to increment usage.
            httpx_client: Optional shared sync httpx client to prevent OpenAI SDK memory leaks.
            httpx_aclient: Optional shared async httpx client to prevent OpenAI SDK memory leaks.
        """
        self.deployments = deployments
        self._get_usage = get_usage
        self._increment_usage = increment_usage
        self._httpx_client = httpx_client
        self._httpx_aclient = httpx_aclient
        self.total_usage: float = 0
        self.usages: list[ModelUsage] = []
        self.fallback_models = fallback_models

        self._clients: dict[Model, Client] = {}
        self._aclients: dict[Model, Client] = {}

    def resolve_model(self, models: list[Model]) -> Model:
        """Returns the model that would be used for the given model list, without creating a client.

        Args:
            models: The models to resolve.

        Returns:
            The model that would be used (either a direct match or a fallback).

        Raises:
            ValueError: If no deployment is available for any of the models or their fallbacks.
        """
        for model in models:
            if model in self.deployments:
                return model

        fallback_models: set[Model] = {
            m for model in models for m in self.fallback_models[model]
        }

        for model in fallback_models:
            if model in self.deployments:
                return model

        message = f"No model deployments are available from the provided list : {models}, {fallback_models}"
        logger.warning(message)
        raise ValueError(message)

    def client(self, models: list[Model]) -> Client:
        """Returns the client for the given model, creating it if necessary.

        Args:
            models: The models for which to get the client.

        Returns:
            The client object for the specified model.
        """
        for model in models:
            if model in self.deployments:
                if model not in self._clients:
                    self._clients[model] = get_client(
                        self.deployments[model],
                        http_client=self._httpx_client,
                    )
                return self._clients[model], model

        fallback_models: set[Model] = {
            m for model in models for m in self.fallback_models[model]
        }

        for model in fallback_models:
            if model in self.deployments:
                if model not in self._clients:
                    self._clients[model] = get_client(
                        self.deployments[model],
                        http_client=self._httpx_client,
                    )
                return self._clients[model], model

        message = f"No model deployments are available from the provided list : {models}, {fallback_models}"
        logger.warning(message)
        raise ValueError(message)

    def aclient(self, models: list[Model]) -> Client:
        """Returns the asynchronous client for the given model, creating it if necessary.

        Args:
            models: The model for which to get the client.

        Returns:
            The asynchronous client object for the specified model.
        """
        for model in models:
            if model in self.deployments:
                if model not in self._aclients:
                    self._aclients[model] = get_aclient(
                        self.deployments[model],
                        http_client=self._httpx_aclient,
                    )
                return self._aclients[model], model

        fallback_models: set[Model] = {
            m for model in models for m in self.fallback_models[model]
        }

        for model in fallback_models:
            if model in self.deployments:
                if model not in self._aclients:
                    self._aclients[model] = get_aclient(
                        self.deployments[model],
                        http_client=self._httpx_aclient,
                    )
                return self._aclients[model], model

        message = f"No model deployments are available from the provided list : {models}, {fallback_models}"
        logger.warning(message)
        raise ValueError(message)

    @staticmethod
    def clean_kwargs(
        kwargs: dict[str, Any],
        deployment: Deployment,
    ) -> dict[str, Any]:
        """Clean kwargs to avoid errors."""
        if "temperature" in kwargs and deployment.model in {"o3-mini", "o3-mini-high"}:
            logger.warning("Temperature is not supported for this model.")
            kwargs.pop("temperature")
        if "text_format" in kwargs and deployment.model not in {
            "gpt'4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-turbo",
        }:
            logger.warning("Text format is not supported for this model.")
            kwargs.pop("text_format")
        if "text_format" in kwargs and deployment.api_version < "2025-03-01-preview":
            logger.warning("Text format is not supported for this API version.")
            kwargs.pop("text_format")
        if "reasoning_effort" in kwargs and deployment.model == "o3-mini-high":
            logger.warning("Reasoning effort is not supported for this model.")
            kwargs.pop("reasoning_effort")

        if "stream_options" in kwargs and deployment.model in MISTRAL_MODELS:
            kwargs["stream_options"] = NOT_GIVEN

        if "max_completion_tokens" in kwargs and deployment.model in MISTRAL_MODELS:
            kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")

        return kwargs

    def _create_chat(
        self,
        messages: Messages,
        models: list[Model],
        **kwargs: Any,
    ) -> tuple[ChatCompletion | ParsedResponse[Any], Model]:
        """Synchronously creates chat completions for the given messages and model.

        Args:
            messages: The list of messages to process.
            models: The model to use for generating completions.
            kwargs: More args.

        Returns:
            ChatCompletion: The completion response from the API.
        """
        client, model = self.client(models)
        deployment = self.deployments[model]

        kwargs = MultiAIClient.clean_kwargs(kwargs, deployment)

        if "text_format" in kwargs:
            return client.responses.parse(
                input=messages,
                model=deployment.deployment_name,
                **kwargs,
            )

        return client.chat.completions.create(
            messages=messages,
            model=deployment.deployment_name,
            **kwargs,
        ), model

    async def _acreate_chat(
        self,
        messages: Messages,
        models: list[Model],
        **kwargs: Any,
    ) -> tuple[ChatCompletion | ParsedResponse[Any], Model]:
        """Asynchronously creates chat completions for the given messages and model.

        Args:
            messages: The list of messages to process.
            models: The model to use for generating completions.
            kwargs: More args.

        Returns:
            ChatCompletion: The completion response from the API.
        """
        aclient, model = self.aclient(models)
        deployment = self.deployments[model]

        kwargs = MultiAIClient.clean_kwargs(kwargs, deployment)

        if "text_format" in kwargs:
            return await aclient.responses.parse(
                input=messages,
                model=deployment.deployment_name,
                **kwargs,
            )

        response = await aclient.chat.completions.create(
            messages=messages,
            model=deployment.deployment_name,
            **kwargs,
        )
        return response, model

    def invoke(
        self,
        messages: Messages,
        models: list[Model],
        system: str | None = None,
        delay: float = 0.0,
        tries: int = 1,
        **kwargs: Any,
    ) -> ChatCompletion | ParsedResponse[Any]:
        """Synchronously invokes the API to get chat completions, tracking usage.

        Args:
            messages: The list of messages for the chat.
            models: The model to use for generating the chat completions.
            system: A string that will be passed as a system prompt.
            delay : How log to wait between each try (in s).
            tries : How many tries we can endure with rate limits.
            kwargs: More args.

        Returns:
            ChatCompletion: The API response containing the chat completions.
        """
        logger.error(
            "[bold purple][LLMAX][/bold purple] Deprecated function `invoke`, use the async one instead.",
        )
        kwargs.pop("operation", "")
        if system:
            messages = add_system_message(
                messages=messages,
                system=system,
            )
        response: ChatCompletion | ParsedResponse[Any] | None = None
        for _ in range(tries):
            try:
                response, _model_used = self._create_chat(
                    messages,
                    models,
                    **kwargs,
                    stream=False,
                )
                break
            except RateLimitError as e:
                time.sleep(delay)
                logger.debug(f"Rate limit error: {e}")

        if response is None:
            message = "Rate Limit error"
            raise ValueError(message)

        return response

    def invoke_to_str(
        self,
        messages: Messages,
        model: Model | list[Model],
        system: str | None = None,
        delay: float = 0.0,
        tries: int = 1,
        **kwargs: Any,
    ) -> str | None:
        """Convenience method to invoke the API and return the first response as a string.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for the chat completions.
            system: A string that will be passed as a system prompt.
            delay : How log to wait between each try (in s).
            tries : How many tries we can endure with rate limits.
            kwargs: More args.

        Returns:
            str | None: The content of the first choice in the response, if available.
        """
        response = self.invoke(
            messages,
            model if isinstance(model, list) else [model],
            system=system,
            delay=delay,
            tries=tries,
            **kwargs,
        )
        if isinstance(response, ChatCompletion):
            return response.choices[0].message.content if response.choices else None
        if not response:
            return None
        return response.output[0].content[0].text if response.output else None  # type: ignore

    async def ainvoke(
        self,
        messages: Messages,
        models: list[Model],
        system: str | None = None,
        delay: float = 0.0,
        tries: int = 1,
        **kwargs: Any,
    ) -> ChatCompletion | ParsedResponse[Any]:
        """Asynchronously invokes the API to get chat completions, tracking usage.

        Args:
            messages: The list of messages for the chat.
            models: The model to use for generating the chat completions.
            system: A string that will be passed as a system prompt.
            delay : How log to wait between each try (in s).
            tries : How many tries we can endure with rate limits.
            kwargs: More args.

        Returns:
            ChatCompletion: The API response containing the chat completions.
        """
        start_time = time.time()
        operation: str = kwargs.pop("operation", "")
        if system:
            messages = add_system_message(
                messages=messages,
                system=system,
            )

        response: ChatCompletion | ParsedResponse[Any] | None = None
        model = models[0]
        for _ in range(tries):
            try:
                response, model = await self._acreate_chat(
                    messages,
                    models,
                    **kwargs,
                    stream=False,
                )
                break
            except RateLimitError as e:
                await asyncio.sleep(delay)
                logger.debug(f"Rate limit error: {e}")

        if response is None:
            message = "Rate Limit error"
            raise ValueError(message)

        duration = time.time() - start_time
        if not response.usage:
            message = "No usage for this request"
            raise ValueError(message)
        deployment = self.deployments[model]
        response_usage = (
            response.usage
            if isinstance(response, ChatCompletion)
            else CompletionUsage(
                completion_tokens=response.usage.output_tokens,
                prompt_tokens=response.usage.input_tokens,
                total_tokens=response.usage.total_tokens,
            )
        )
        usage = ModelUsage(deployment, self._increment_usage, response_usage)

        cost = await usage.apply(
            operation=operation,
            duration=duration,
            ttft=None,
        )
        self.total_usage += cost
        self.usages.append(usage)

        return response

    async def ainvoke_to_str(
        self,
        messages: Messages,
        model: Model | list[Model],
        system: str | None = None,
        delay: float = 0.0,
        tries: int = 1,
        **kwargs: Any,
    ) -> str | None:
        """Asynchronously invokes the API and returns the first response as a string.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for the chat completions.
            system: A string that will be passed as a system prompt.
            delay : How log to wait between each try (in s).
            tries : How many tries we can endure with rate limits.
            kwargs: More args.

        Returns:
            str | None: The content of the first choice in the response, if available.
        """
        response = await self.ainvoke(
            messages,
            model if isinstance(model, list) else [model],
            delay=delay,
            tries=tries,
            system=system,
            **kwargs,
        )
        if isinstance(response, ChatCompletion):
            return response.choices[0].message.content if response.choices else None
        return response.output[0].content[0].text if response.output else None  # type: ignore

    async def ainvoke_get_tools(
        self,
        messages: Messages,
        model: list[Model] | Model,
        system: str | None = None,
        delay: float = 0.0,
        tries: int = 1,
        **kwargs: Any,
    ) -> tuple[str | None, list[ChatCompletionMessageToolCall]]:
        """Asynchronously invokes the API and returns the first response as a string.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for the chat completions.
            system: A string that will be passed as a system prompt.
            delay : How log to wait between each try (in s).
            tries : How many tries we can endure with rate limits.
            max_tool_calls: Maximum number of tool call before stopping.
            kwargs: More args.

        Returns:
            str | None: The content of the first choice in the response, if available.
        """
        response = await self.ainvoke(
            messages,
            model if isinstance(model, list) else [model],
            delay=delay,
            tries=tries,
            system=system,
            **kwargs,
        )
        output_str = response.choices[0].message.content if response.choices else None  # type: ignore
        final_tool_calls = response.choices[0].message.tool_calls or []  # type: ignore

        return output_str, final_tool_calls  # type: ignore

    async def ainvoke_with_tools(  # noqa: D417, PLR0913
        self,
        messages: Messages,
        model: Model | list[Model],
        execute_tools: Callable[[str, str], Awaitable[str]],
        system: str | None = None,
        delay: float = 0.0,
        max_tool_calls: int = 4,
        tries: int = 1,
        **kwargs: Any,
    ) -> str | None:
        """Asynchronously invokes the API and returns the first response as a string.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for the chat completions.
            system: A string that will be passed as a system prompt.
            delay : How log to wait between each try (in s).
            tries : How many tries we can endure with rate limits.
            max_tool_calls: Maximum number of tool call before stopping.
            kwargs: More args.

        Returns:
            str | None: The content of the first choice in the response, if available.
        """
        output_str = None
        tries = 0

        while tries < max_tool_calls:
            tries += 1
            response = await self.ainvoke(
                messages,
                model if isinstance(model, list) else [model],
                delay=delay,
                tries=tries,
                system=system,
                **kwargs,
            )
            output_str = (
                response.choices[0].message.content if response.choices else None  # type: ignore
            )
            final_tool_calls = response.choices[0].message.tool_calls or []  # type: ignore

            if len(final_tool_calls) == 0:
                return output_str

            if output_str:
                await execute_tools(
                    "llm_output",
                    output_str,
                )
                messages.append({"role": "assistant", "content": output_str})

            tool_coros = []
            for tool in final_tool_calls:
                function_name = tool.function.name  # type: ignore
                function_args = tool.function.arguments  # type: ignore
                logger.info(
                    f"Tool called for function `{function_name}` with the args `{function_args}`",
                )
                tool_coros.append(execute_tools(function_name, function_args))

            results = await asyncio.gather(*tool_coros)

            for tool, resultat in zip(final_tool_calls, results, strict=True):
                messages.append(
                    parse_tool_call(
                        tool,  # type: ignore
                        model[0] if isinstance(model, list) else model,
                    ),
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool.id,
                        "content": str(resultat),
                    },
                )

        return output_str

    def stream(
        self,
        messages: Messages,
        model: Model | list[Model],
        system: str | None = None,
        **kwargs: Any,
    ) -> Generator[ChatCompletionChunk | CompletionUsage | Model, None, None]:
        """Streams chat completions, allowing responses to be received in chunks.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for generating the chat completions.
            system: A string that will be passed as a system prompt.
            kwargs: More args.

        Yields:
            ChatCompletionChunk: Individual chunks of the completion response.
        """
        model_used = model[0] if isinstance(model, list) else model
        if system:
            messages = add_system_message(
                messages=messages,
                system=system,
            )
        try:
            if model == "o3-mini-high":
                response, model_used = self._create_chat(
                    messages,
                    ["o3-mini"],
                    **kwargs,
                    stream=True,
                    reasoning_effort="high",
                    stream_options={"include_usage": True},
                )
            else:
                response, model_used = self._create_chat(
                    messages,
                    model if isinstance(model, list) else [model],
                    **kwargs,
                    stream=True,
                    stream_options={"include_usage": True},
                )
        except BadRequestError as e:
            logger.error(
                f"[bold purple][LLMAX][/bold purple] BadRequestError in stream for model {model}: {e}",
            )
            return
        except Exception as e:
            logger.error(
                f"[bold purple][LLMAX][/bold purple] Unexpected error in stream for model {model}: {e}",
            )
            return

        yield model_used

        try:
            for chunk in response:
                if isinstance(chunk.usage, CompletionUsage):  # type: ignore
                    yield chunk.usage  # type: ignore
                try:
                    if len(chunk.choices) == 0:  # type: ignore
                        continue
                except Exception as e:
                    logger.debug(f"Error in llmax streaming : {e}")
                    continue
                yield chunk  # type: ignore
        except Exception as e:
            logger.error(
                f"[bold purple][LLMAX][/bold purple] Error iterating stream chunks for model {model}: {e}",
            )
            return

    async def stream_output_smooth(  # noqa: C901, PLR0915
        self,
        messages: Messages,
        model: list[Model] | Model,
        smooth_duration: int,
        system: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamedItem, None]:
        """Streams formatted output from the chat completions.

        This method formats each chunk received from `stream` into a specific
        string format before yielding it.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for generating the chat completions.
            system: A string that will be passed as a system prompt.
            smooth_duration: The duration in ms to wait before trying to send another chunk.
            kwargs: More args.

        Yields:
            str: Formatted output for each chunk.
        """
        output_queue: Queue[list[Choice] | Exception | None] = Queue()
        chunk_usage: CompletionUsage = CompletionUsage(
            completion_tokens=0,
            prompt_tokens=0,
            total_tokens=0,
        )

        # Track timing for usage
        start = time.time()
        ttft = None
        model_used = model[0] if isinstance(model, list) else model

        operation: str = kwargs.pop("operation", "")

        for chunk in fake_llm(
            "",
            stream=False,
            send_empty=True,
        ):
            yield StreamItemContent(content=chunk)

        def collect_chunks() -> None:
            nonlocal chunk_usage, ttft, model_used
            chunk_count = 0
            try:
                for completion_chunk in self.stream(
                    messages,
                    model,
                    system=system,
                    **kwargs,
                ):
                    chunk_count += 1
                    if isinstance(completion_chunk, CompletionUsage):
                        chunk_usage = completion_chunk
                    elif isinstance(completion_chunk, ChatCompletionChunk):
                        if (
                            ttft is None
                            and len(completion_chunk.choices) > 0
                            and completion_chunk.choices[0].delta.content
                        ):
                            ttft = time.time() - start
                        output_queue.put(completion_chunk.choices)
                    else:
                        model_used = completion_chunk

            except Exception as e:
                logger.error(
                    f"[bold purple][LLMAX][/bold purple] Exception in stream collection for model {model}: {e}",
                )
                output_queue.put(e)
            finally:
                if chunk_count == 0:
                    logger.warning(
                        f"[bold purple][LLMAX][/bold purple] No chunks received for model {model}",
                    )
                output_queue.put(None)

        collector = threading.Thread(target=collect_chunks)
        collector.start()

        final_tool_calls: dict[int, ChoiceDeltaToolCall] = {}
        final_output = ""
        loop = asyncio.get_running_loop()

        while True:
            item = await loop.run_in_executor(None, output_queue.get)

            if item is None:
                break

            if isinstance(item, Exception):
                logger.error(f"Exception in streaming thread: {item}")
                break

            choices = item
            if len(choices) == 0:
                continue

            if choices[0].delta.content:
                chunk_str = choices[0].delta.content
                final_output += chunk_str
                yield StreamItemContent(content=stream_chunk(chunk_str, "text"))

            for tool_call in choices[0].delta.tool_calls or []:
                index = tool_call.index
                if index is None:  # type: ignore
                    index = 1

                if index not in final_tool_calls:
                    final_tool_calls[index] = tool_call
                elif (
                    tool_call.function is not None
                    and final_tool_calls[index].function is not None
                ):
                    current_args = final_tool_calls[index].function.arguments or ""  # type: ignore
                    new_args = tool_call.function.arguments or ""
                    final_tool_calls[index].function.arguments = current_args + new_args  # type: ignore

            await asyncio.sleep(smooth_duration / 1000)

        yield ModelItemContent(model_used=model_used)
        await loop.run_in_executor(None, collector.join)

        deployment = self.deployments[model_used]
        usage = ModelUsage(deployment, self._increment_usage, chunk_usage)

        duration = time.time() - start

        cost = await usage.apply(operation=operation, ttft=ttft, duration=duration)
        self.total_usage += cost
        self.usages.append(usage)

        yield StreamItemOutput(tools=final_tool_calls, output=final_output)

    async def stream_output(
        self,
        messages: Messages,
        model: list[Model] | Model,
        system: str | None = None,
        smooth_duration: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamedItem, None]:
        """Streams formatted output from the chat completions.

        This method formats each chunk received from `stream` into a specific
        string format before yielding it.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for generating the chat completions.
            system: A string that will be passed as a system prompt.
            smooth_duration: The duration in ms to wait before trying to send another chunk.
            kwargs: More args.

        Yields:
            str: Formatted output for each chunk.
        """
        async for chunk in self.stream_output_smooth(
            messages=messages,
            model=model,
            system=system,
            smooth_duration=smooth_duration or 0,
            **kwargs,
        ):
            yield chunk

    async def stream_output_with_tools(  # noqa: C901, PLR0912, PLR0913
        self,
        messages: Messages,
        model: list[Model] | Model,
        execute_tools: Callable[[str, str, str], AsyncGenerator[ToolItem, None]],
        system: str | None = None,
        smooth_duration: int | None = None,
        max_tool_calls: int = 4,
        max_tokens_before_tool_use: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Streams formatted output from the chat completions.

        This method formats each chunk received from `stream` into a specific
        string format before yielding it.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for generating the chat completions.
            system: A string that will be passed as a system prompt.
            smooth_duration: The duration in ms to wait before trying to send another chunk.
            execute_tools: how to execute the tools given.
            max_tool_calls: maximum number of call to a tool before stopping.
            max_tokens_before_tool_use: To limit the usage of tools if there are too many tokens and force an answer
            kwargs: More args.

        Yields:
            str: Formatted output for each chunk.
        """
        tries = 0
        tools = kwargs.pop("tools", None)

        while tries < max_tool_calls:
            tries += 1
            final_tool_calls = {}
            output_str = ""

            token_count = sum(
                tokens.count(m["content"]) for m in messages if "content" in m
            )
            allow_tools = (
                max_tokens_before_tool_use is None
                or token_count <= max_tokens_before_tool_use
            )

            if not allow_tools:
                messages.append(
                    {
                        "role": "user",
                        "content": "You can no longer call any tools due to the size of the context. With the curent information, answer the user question directly.",
                    },
                )

            model_used = model[0] if isinstance(model, list) else model

            async for item in self.stream_output(
                messages=messages,
                model=model if isinstance(model, list) else [model],
                system=system,
                smooth_duration=smooth_duration,
                tools=tools if allow_tools else None,
                **kwargs,
            ):
                if isinstance(item, StreamItemContent):
                    yield item.content
                elif isinstance(item, ModelItemContent):
                    model_used = item.model_used
                else:
                    final_tool_calls, output_str = item.tools, item.output

            finished = len(final_tool_calls) == 0
            if finished:
                break

            retrigger_stream = True

            if output_str:
                messages.append({"role": "assistant", "content": output_str})

            queue = asyncio.Queue()

            async def run_tool(tool: ChoiceDeltaToolCall, queue: asyncio.Queue) -> None:  # type: ignore
                tool_id = tool.id
                function_name = tool.function.name  # type: ignore
                function_args = tool.function.arguments  # type: ignore
                if not all([tool_id, function_name, function_args]):
                    return
                logger.info(
                    f"Tool called for function `{function_name}` with args `{function_args}`",
                )
                async for res in execute_tools(function_name, function_args, tool_id):  # type: ignore
                    await queue.put((tool, res))
                await queue.put((tool, None))

            tasks = [
                asyncio.create_task(run_tool(tool, queue))
                for tool in final_tool_calls.values()
            ]
            finished_tools = 0

            while finished_tools < len(tasks):
                tool, res = await queue.get()
                if res is None:
                    finished_tools += 1
                    continue

                if isinstance(res, ToolItemContent):
                    yield res.content
                else:
                    tool_result, tool_retrigger = res.output, res.redo
                    update_messages_tools(
                        tool,
                        tool_result,
                        messages,
                        model_used,
                    )
                    if not tool_retrigger:
                        retrigger_stream = False

            for res in fake_llm("\n", stream=False):
                yield res

            if not retrigger_stream:
                break

    async def aembedder(
        self,
        texts: list[str],
        model: Model | list[Model],
        **kwargs: Any,
    ) -> list[Embedding]:
        """Asynchronously obtains vector embeddings for a list of texts.

        Args:
            texts: The texts to generate embeddings for.
            model: The embedding model.
            kwargs: Additional arguments.

        Returns:
            List[Embedding]: The embeddings for each text.
        """
        start = time.time()
        operation: str = kwargs.pop("operation", "")
        texts = [text.replace("\n", " ") for text in texts]

        client, model_used = self.aclient(model if isinstance(model, list) else [model])
        deployment = self.deployments[model_used]

        response = await client.embeddings.create(
            input=texts,
            model=deployment.deployment_name,
        )

        duration = time.time() - start

        usage = ModelUsage(deployment, self._increment_usage)
        usage.add_tokens(prompt_tokens=response.usage.prompt_tokens)

        cost = await usage.apply(operation=operation, duration=duration, ttft=None)
        self.total_usage += cost
        self.usages.append(usage)

        embeddings = response.data
        return embeddings

    def embedder(
        self,
        texts: list[str],
        model: Model | list[Model],
        **kwargs: Any,
    ) -> list[Embedding]:
        """Obtains vector embeddings for a list of texts asynchronously.

        Args:
            texts: The texts to generate embeddings for.
            model: The embedding model.
            kwargs: More args.

        Returns:
            list[Embedding]: The embeddings for each text.
        """
        logger.error(
            "[bold purple][LLMAX][/bold purple] Deprecated function `embedder`, use the async one instead.",
        )
        kwargs.pop("operation", "")
        texts = [text.replace("\n", " ") for text in texts]

        client, model_used = self.client(model if isinstance(model, list) else [model])
        deployment = self.deployments[model_used]

        response = client.embeddings.create(
            input=texts,
            model=deployment.deployment_name,
        )

        embeddings = response.data
        return embeddings

    def speech_to_text(
        self,
        file: BufferedReader,
        model: Model | list[Model],
        **kwargs: Any,
    ) -> TranscriptionVerbose:
        """Synchronously processes audio data for speech-to-text using the Whisper model.

        Args:
            file: The audio data to process.
            model: The model to use for processing the audio.
            kwargs: Additional arguments to pass to the API.

        Returns:
            Any: The response from the API.
        """
        logger.error(
            "[bold purple][LLMAX][/bold purple] Deprecated function `speech_to_text`, use the async one instead.",
        )
        kwargs.pop("operation", "")
        client, model_used = self.client(model if isinstance(model, list) else [model])
        deployment = self.deployments[model_used]

        response = client.audio.transcriptions.create(
            file=file,
            model=deployment.deployment_name,
            response_format="verbose_json",
            **kwargs,
        )

        return response

    async def aspeech_to_text(
        self,
        file: BytesIO,
        model: Model | list[Model],
        response_format: Literal["json", "verbose_json"] = "verbose_json",
        duration: float | None = None,
        **kwargs: Any,
    ) -> TranscriptionVerbose | Transcription:
        """Asynchronously processes audio data for speech-to-text using the Whisper model.

        Args:
            file: The audio data to process.
            model: The model to use for processing the audio.
            response_format: for gpt-4o-transcribe, you cannot pass verbose_json.
            duration: duration of the audio, if known.
            kwargs: Additional arguments to pass to the API.

        Returns:
            Any: The response from the API.
        """
        start = time.time()
        operation: str = kwargs.pop("operation", "")
        aclient, model_used = self.aclient(
            model if isinstance(model, list) else [model],
        )
        deployment = self.deployments[model_used]
        if (
            response_format == "verbose_json"
            and model_used not in SpeechModelAllowVerboseJson
        ):
            response_format = "json"

        response = await aclient.audio.transcriptions.create(
            file=file,
            model=deployment.deployment_name,
            response_format=response_format,
            **kwargs,
        )
        duration = time.time() - start

        usage = ModelUsage(deployment, self._increment_usage)
        response_duration: float | None = None
        if isinstance(response, TranscriptionVerbose):
            response_duration = response.duration
        if isinstance(response, Transcription) and duration:
            response_duration = 360

        if not response_duration:
            message = "You need to specify the duration in json mode."
            raise ValueError(message)

        usage.add_audio_duration(response_duration)

        cost = await usage.apply(
            operation=operation,
            duration=duration,
            ttft=None,
        )
        self.total_usage += cost
        self.usages.append(usage)

        return response

    async def text_to_image(
        self,
        model: Model | list[Model],
        prompt: str,
        size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024",
        quality: Literal["low", "medium", "high", "auto"] = "medium",
        n: int = 1,
        **kwargs: Any,
    ) -> bytes:
        """Generate images from a text prompt using the specified model."""
        start = time.time()
        operation: str = kwargs.pop("operation", "")
        client, model_used = self.aclient(model if isinstance(model, list) else [model])
        deployment = self.deployments[model_used]

        image_bytes: bytes | None = None

        if model_used in GEMINI_MODELS:
            client = genai.Client(api_key=deployment.api_key)
            aclient = client.aio
            response = await aclient.models.generate_content(
                model=model_used,
                contents=[prompt],
            )
            image_bytes = _extract_image_from_gemini_response(response)
        else:
            response = await client.images.generate(
                model=model_used,
                prompt=prompt,
                size=size,
                quality=quality,
                n=n,
            )
            image_base64 = response.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)

        duration = time.time() - start

        usage = ModelUsage(deployment, self._increment_usage)
        usage.add_image(
            quality=quality,
            size=size,
            n=n,
        )

        cost = await usage.apply(
            operation=operation,
            duration=duration,
            ttft=None,
        )
        self.total_usage += cost
        self.usages.append(usage)

        if image_bytes is None:
            message = "Couldn't generate an image"
            raise ValueError(message)

        return image_bytes

    async def edit_image(  # noqa: PLR0913
        self,
        model: Model | list[Model],
        prompt: str,
        image: tuple[str, bytes, str],
        size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024",
        quality: Literal["low", "medium", "high", "auto"] = "medium",
        n: int = 1,
        **kwargs: Any,
    ) -> bytes:
        """Edit an image using the specified model and a text prompt."""
        start = time.time()
        operation: str = kwargs.pop("operation", "")
        client, model_used = self.aclient(model if isinstance(model, list) else [model])
        deployment = self.deployments[model_used]
        image_bytes: bytes | None = None

        if model_used in GEMINI_MODELS:
            client = genai.Client(api_key=deployment.api_key)
            aclient = client.aio

            # Convert input image to Part
            _filename, file_bytes, mime_type = image
            image_part = genai_types.Part.from_bytes(
                data=file_bytes,
                mime_type=mime_type,
            )
            text_part = genai_types.Part.from_text(text=prompt)

            content = genai_types.Content(parts=[text_part, image_part])
            response = await aclient.models.generate_content(
                model=model_used,
                contents=[content],
            )
            image_bytes = _extract_image_from_gemini_response(response)
        else:
            response = await client.images.edit(
                model=deployment.deployment_name,
                prompt=prompt,
                image=image,
                size=size,
                quality=quality,
                n=n,
            )
            image_base64 = response.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)

        duration = time.time() - start

        usage = ModelUsage(deployment, self._increment_usage)
        usage.add_image(
            quality=quality,
            size=size,
            n=n,
        )
        cost = await usage.apply(
            operation=operation,
            duration=duration,
            ttft=None,
        )
        self.total_usage += cost
        self.usages.append(usage)

        if image_bytes is None:
            message = "Couldn't generate an edited image"
            raise ValueError(message)

        return image_bytes

    def text_to_speech(
        self,
        text: str,
        model: Model | list[Model],
        path: str,
        **kwargs: Any,
    ) -> str:
        """Generate audio from text using the specified model.

        Parameters:
        - text (str): The text to be converted to speech.
        - model (Model): The model to be used for generating audio.
        - **kwargs (Any): Additional keyword arguments for further customization.

        Returns:
        - str: The URL of the generated audio file.

        Raises:
        - Any relevant exceptions that may occur during the audio generation process.
        """
        logger.error(
            "[bold purple][LLMAX][/bold purple] Deprecated function `text_to_speech`, create the async one instead.",
        )
        kwargs.pop("operation", "")
        client, model_used = self.client(model if isinstance(model, list) else [model])
        deployment = self.deployments[model_used]
        response = client.audio.speech.create(
            model=deployment.deployment_name,
            input=text,
            voice="alloy",
            **kwargs,
        )
        response.stream_to_file(path)

        return path


def _extract_image_from_gemini_response(response: Any) -> bytes | None:
    """Extract image bytes from Gemini API response."""
    if not response.candidates:
        return None
    for candidate in response.candidates:
        content = candidate.content
        if not content or not content.parts:
            continue
        for part in content.parts:
            if part.inline_data:
                image_bytes = part.inline_data.data
                if image_bytes:
                    return image_bytes
    return None


def add_system_message(
    messages: Messages,
    system: str,
) -> Messages:
    """Adds a system message at the start of the messages.

    It should take into account the model name to correctly name the system.

    Args:
        messages: The list of messages for the chat.
        model: The model to use for generating the chat completions.
        system: A string that will be passed as a system prompt.

    Returns:
        Messages: The same initial list with the system message inserted at index 0.
    """
    messages.insert(0, {"role": "system", "content": system})
    return messages


def stream_chunk(chunk: str, stream_part_type: str = "text") -> str:
    """Format the chunk to the correct format for vercel sdk."""
    code = get_stream_part_code(stream_part_type)
    formatted_stream_part = f"{code}:{json.dumps(chunk, separators=(',', ':'))}\n\n"
    return formatted_stream_part


def get_stream_part_code(stream_part_type: str) -> str:
    """Converts the type to a number."""
    stream_part_types = {
        "text": "0",
        "function_call": "1",
        "data": "2",
        "error": "3",
        "assistant_message": "4",
        "assistant_data_stream_part": "5",
        "data_stream_part": "6",
        "message_annotations_stream_part": "7",
    }
    return stream_part_types[stream_part_type]


def parse_tool_call(
    tool_call: ChoiceDeltaToolCall | ChatCompletionMessageToolCall,
    model: Model,
) -> dict[str, Any]:
    """Returns the tool and the correct format for the llm."""
    call_id = tool_call.id
    function_name = tool_call.function.name if tool_call.function else ""

    if model in ANTHROPIC_MODELS:
        formatted_call = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": function_name,
                    "input": json.loads(str(tool_call.function.arguments)),  # type: ignore
                },
            ],
        }
        return formatted_call

    if model in MISTRAL_MODELS:
        formatted_call = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,  # type: ignore
                        "arguments": str(tool_call.function.arguments),  # type: ignore
                    },
                },
            ],
        }
        return formatted_call

    formatted_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": tool_call.function.arguments
                    if tool_call.function
                    else "",
                },
            },
        ],
    }

    return formatted_call


def update_messages_tools(
    tool_call: ChoiceDeltaToolCall | ChatCompletionMessageToolCall,
    tool_result: str | None,
    messages: list[Message],
    model: Model,
) -> None:
    """Update the messages with the tool called."""
    messages.append(parse_tool_call(tool_call, model))

    if model in ANTHROPIC_MODELS:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,  # This matches the tool_use.id from Claude
                        "content": str(tool_result),
                    },
                ],
            },
        )
        return

    if model in MISTRAL_MODELS:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_result),
            },
        )
        return

    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(tool_result),
        },
    )
