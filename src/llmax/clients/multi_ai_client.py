"""This module contains the MultiAIClient class.

This class is used to interface with multiple LLMs and AI models, supporting both
synchronous and asynchronous operations.
"""

import asyncio
import json
import threading
import time
from collections.abc import AsyncGenerator, Awaitable, Generator
from io import BufferedReader, BytesIO
from queue import Queue
from typing import Any, Callable, Literal

from openai import NOT_GIVEN, BadRequestError, RateLimitError
from openai.types import CompletionUsage, Embedding
from openai.types.audio import TranscriptionVerbose
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDeltaToolCall

from llmax.external_clients.clients import Client, get_aclient, get_client
from llmax.messages import Messages
from llmax.models.deployment import Deployment
from llmax.models.fake import fake_llm
from llmax.models.models import (
    ANTHROPIC_MODELS,
    COHERE_MODELS,
    GEMINI_MODELS,
    META_MODELS,
    MISTRAL_MODELS,
    OPENAI_MODELS,
    Model,
)
from llmax.usage import ModelUsage
from llmax.utils import (
    StreamedItem,
    StreamItemContent,
    StreamItemOutput,
    ToolItem,
    ToolItemContent,
    logger,
)


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

    def __init__(
        self,
        deployments: dict[Model, Deployment],
        get_usage: Callable[[], float] = lambda: 0.0,
        increment_usage: Callable[
            [float, Model, str, float, float | None, int, int, str, str, str],
            bool,
        ] = lambda _1, _2, _3, _4, _5, _6, _7, _8, _9, _10: True,
    ) -> None:
        """Initializes the MultiAIClient class.

        Args:
            deployments: A mapping from models to their deployment objects.
            get_usage: A function to get the current usage.
            increment_usage: A function to increment usage.
        """
        self.deployments = deployments
        self._get_usage = get_usage
        self._increment_usage = increment_usage
        self.total_usage: float = 0
        self.usages: list[ModelUsage] = []

        self._clients: dict[Model, Client] = {}
        self._aclients: dict[Model, Client] = {}

    def client(self, model: Model) -> Client:
        """Returns the client for the given model, creating it if necessary.

        Args:
            model: The model for which to get the client.

        Returns:
            The client object for the specified model.
        """
        if model not in self._clients:
            self._clients[model] = get_client(self.deployments[model])
        return self._clients[model]

    def aclient(self, model: Model) -> Client:
        """Returns the asynchronous client for the given model, creating it if necessary.

        Args:
            model: The model for which to get the client.

        Returns:
            The asynchronous client object for the specified model.
        """
        if model not in self._aclients:
            self._aclients[model] = get_aclient(self.deployments[model])
        return self._aclients[model]

    def _create_chat(
        self,
        messages: Messages,
        model: Model,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Synchronously creates chat completions for the given messages and model.

        Args:
            messages: The list of messages to process.
            model: The model to use for generating completions.
            kwargs: More args.

        Returns:
            ChatCompletion: The completion response from the API.
        """
        client = self.client(model)
        deployment = self.deployments[model]

        return client.chat.completions.create(
            messages=messages,
            model=deployment.deployment_name,
            **kwargs,
        )

    async def _acreate_chat(
        self,
        messages: Messages,
        model: Model,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Asynchronously creates chat completions for the given messages and model.

        Args:
            messages: The list of messages to process.
            model: The model to use for generating completions.
            kwargs: More args.

        Returns:
            ChatCompletion: The completion response from the API.
        """
        aclient = self.aclient(model)
        result = await aclient.chat.completions.create(
            messages=messages,
            model=self.deployments[model].deployment_name,
            **kwargs,
        )
        return result

    def invoke(
        self,
        messages: Messages,
        model: Model,
        system: str | None = None,
        delay: float = 0.0,
        tries: int = 1,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Synchronously invokes the API to get chat completions, tracking usage.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for generating the chat completions.
            system: A string that will be passed as a system prompt.
            delay : How log to wait between each try (in s).
            tries : How many tries we can endure with rate limits.
            kwargs: More args.

        Returns:
            ChatCompletion: The API response containing the chat completions.
        """
        start_time = time.time()
        operation: str = kwargs.pop("operation", "")
        if "temperature" in kwargs and model in {"o3-mini", "o3-mini-high"}:
            kwargs.pop("temperature")
        if system:
            messages = add_system_message(
                messages=messages,
                model=model,
                system=system,
            )
        response: ChatCompletion | None = None
        for _ in range(tries):
            try:
                response = self._create_chat(messages, model, **kwargs, stream=False)
                break
            except RateLimitError as e:
                time.sleep(delay)
                logger.debug(f"Rate limit error: {e}")

        if response is None:
            message = "Rate Limit error"
            raise ValueError(message)

        duration = time.time() - start_time
        if not response.usage:
            message = "No usage for this request"
            raise ValueError(message)
        deployment = self.deployments[model]
        usage = ModelUsage(deployment, self._increment_usage, response.usage)

        cost = usage.apply(
            operation=operation,
            duration=duration,
            ttft=None,
        )
        self.total_usage += cost
        self.usages.append(usage)

        return response

    def invoke_to_str(
        self,
        messages: Messages,
        model: Model,
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
            model,
            system=system,
            delay=delay,
            tries=tries,
            **kwargs,
        )
        return response.choices[0].message.content if response.choices else None

    async def ainvoke(
        self,
        messages: Messages,
        model: Model,
        system: str | None = None,
        delay: float = 0.0,
        tries: int = 1,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Asynchronously invokes the API to get chat completions, tracking usage.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for generating the chat completions.
            system: A string that will be passed as a system prompt.
            delay : How log to wait between each try (in s).
            tries : How many tries we can endure with rate limits.
            kwargs: More args.

        Returns:
            ChatCompletion: The API response containing the chat completions.
        """
        start_time = time.time()
        operation: str = kwargs.pop("operation", "")
        if "temperature" in kwargs and model in {"o3-mini", "o3-mini-high"}:
            kwargs.pop("temperature")
        if system:
            messages = add_system_message(
                messages=messages,
                model=model,
                system=system,
            )

        response: ChatCompletion | None = None
        for _ in range(tries):
            try:
                response = await self._acreate_chat(
                    messages,
                    model,
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
        usage = ModelUsage(deployment, self._increment_usage, response.usage)

        cost = usage.apply(
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
        model: Model,
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
            model,
            delay=delay,
            tries=tries,
            system=system,
            **kwargs,
        )
        return response.choices[0].message.content if response.choices else None

    async def ainvoke_with_tools(
        self,
        messages: Messages,
        model: Model,
        execute_tools: Callable[[str, str], Awaitable[str]],
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
        output_str = None
        tries = 0
        max_tries = 4

        while tries < max_tries:
            tries += 1
            response = await self.ainvoke(
                messages,
                model,
                delay=delay,
                tries=tries,
                system=system,
                **kwargs,
            )
            output_str = (
                response.choices[0].message.content if response.choices else None
            )
            final_tool_calls = response.choices[0].message.tool_calls or []

            if len(final_tool_calls) == 0:
                return output_str

            if output_str:
                messages.append({"role": "assistant", "content": output_str})

            for tool in final_tool_calls:
                function_name = tool.function.name
                function_args = tool.function.arguments
                logger.info(
                    f"Tool called for function `{function_name}` with the args `{function_args}`",
                )

                resultat = await execute_tools(
                    function_name,
                    function_args,
                )

                messages.append(parse_tool_call(tool))
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
        model: Model,
        system: str | None = None,
        **kwargs: Any,
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Streams chat completions, allowing responses to be received in chunks.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for generating the chat completions.
            system: A string that will be passed as a system prompt.
            kwargs: More args.

        Yields:
            ChatCompletionChunk: Individual chunks of the completion response.
        """
        start = time.time()
        ttft = None
        operation: str = kwargs.pop("operation", "")
        if "temperature" in kwargs and model in {"o3-mini", "o3-mini-high"}:
            kwargs.pop("temperature")
        if system:
            messages = add_system_message(
                messages=messages,
                model=model,
                system=system,
            )
        try:
            if model == "o3-mini-high":
                response = self._create_chat(
                    messages,
                    "o3-mini",
                    **kwargs,
                    stream=True,
                    reasoning_effort="high",
                    stream_options={"include_usage": True},
                )
                model = "o3-mini"
            else:
                response = self._create_chat(
                    messages,
                    model,
                    **kwargs,
                    stream=True,
                    stream_options=NOT_GIVEN
                    if model in MISTRAL_MODELS
                    else {"include_usage": True},
                )
        except BadRequestError:
            return
        deployment = self.deployments[model]

        chunk_usage: CompletionUsage = CompletionUsage(
            completion_tokens=0,
            prompt_tokens=0,
            total_tokens=0,
        )

        for chunk in response:
            if chunk.usage:
                chunk_usage = chunk.usage
            try:
                if len(chunk.choices) == 0:  # type: ignore
                    continue
                if chunk.choices[0].delta.content and ttft is None:  # type: ignore
                    ttft = time.time() - start
            except Exception as e:
                logger.debug(f"Error in llmax streaming : {e}")
            yield chunk  # type: ignore

        usage = ModelUsage(deployment, self._increment_usage, chunk_usage)

        duration = time.time() - start

        cost = usage.apply(operation=operation, ttft=ttft, duration=duration)
        self.total_usage += cost
        self.usages.append(usage)

    async def stream_output_smooth(
        self,
        messages: Messages,
        model: Model,
        smooth_duration: int,
        system: str | None = None,
        beta: bool = True,
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
            beta: Whether to use the beta chat for vercel streaming
            kwargs: More args.

        Yields:
            str: Formatted output for each chunk.
        """
        output_queue: Queue[list[Choice] | Exception | None] = Queue()

        for chunk in fake_llm(
            "",
            stream=False,
            send_empty=True,
            beta=beta,
        ):
            yield StreamItemContent(content=chunk)

        def collect_chunks() -> None:
            try:
                for completion_chunk in self.stream(
                    messages,
                    model,
                    system=system,
                    **kwargs,
                ):
                    output_queue.put(completion_chunk.choices)
            except Exception as e:
                output_queue.put(e)
            finally:
                output_queue.put(None)

        collector = threading.Thread(target=collect_chunks)
        collector.start()

        final_tool_calls: dict[int, ChoiceDeltaToolCall] = {}
        final_output = ""
        loop = asyncio.get_running_loop()

        while True:
            item = await loop.run_in_executor(None, output_queue.get)

            if isinstance(item, Exception):
                logger.error(f"Exception in streaming thread: {item}")
                break

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
                if index not in final_tool_calls:
                    final_tool_calls[index] = tool_call
                elif (
                    tool_call.function is not None
                    and final_tool_calls[index].function is not None
                ):
                    current_args = final_tool_calls[index].function.arguments or ""
                    new_args = tool_call.function.arguments or ""
                    final_tool_calls[index].function.arguments = current_args + new_args

            await asyncio.sleep(smooth_duration / 1000)

        await loop.run_in_executor(None, collector.join)

        yield StreamItemOutput(tools=final_tool_calls, output=final_output)

    async def stream_output(
        self,
        messages: Messages,
        model: Model,
        system: str | None = None,
        smooth_duration: int | None = None,
        beta: bool = True,
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
            beta: whether or not to use the new chat version of vercel.
            kwargs: More args.

        Yields:
            str: Formatted output for each chunk.
        """
        async for chunk in self.stream_output_smooth(
            messages=messages,
            model=model,
            system=system,
            smooth_duration=smooth_duration or 0,
            beta=beta,
            **kwargs,
        ):
            yield chunk

    async def stream_output_with_tools(  # noqa: PLR0913
        self,
        messages: Messages,
        model: Model,
        execute_tools: Callable[[str, str], AsyncGenerator[ToolItem, None]],
        system: str | None = None,
        smooth_duration: int | None = None,
        beta: bool = True,
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
            beta: whether or not to use the new chat version of vercel.
            execute_tools: how to execute the tools given.
            kwargs: More args.

        Yields:
            str: Formatted output for each chunk.
        """
        tries = 0
        max_tries = 4

        while tries < max_tries:
            tries += 1
            final_tool_calls = {}
            output_str = ""
            async for item in self.stream_output(
                messages=messages,
                model=model,
                system=system,
                smooth_duration=smooth_duration,
                beta=beta,
                **kwargs,
            ):
                # Check the tag on the yielded item.
                if isinstance(item, StreamItemContent):
                    yield item.content
                else:
                    final_tool_calls, output_str = item.tools, item.output

            finished = len(final_tool_calls) == 0
            if finished:
                break

            retrigger_stream = True

            messages.append({"role": "assistant", "content": output_str})

            for tool in final_tool_calls.values():
                if tool.function is None:
                    continue
                function_name = tool.function.name
                function_args = tool.function.arguments
                if function_args is None or function_name is None:
                    continue

                logger.info(
                    f"Tool called for function `{function_name}` with the args `{function_args}`",
                )

                tool_result = None
                tool_retrigger = False

                # Consume the async execute_tools generator.
                async for res in execute_tools(function_name, function_args):
                    if isinstance(res, ToolItemContent):
                        yield res.content
                    else:
                        tool_result, tool_retrigger = res.output, res.redo

                if not tool_retrigger:
                    retrigger_stream = False

                messages.append(parse_tool_call(tool))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool.id,
                        "content": str(tool_result),
                    },
                )

            for res in fake_llm("\n", stream=False):
                yield res

            if not retrigger_stream:
                break

    def embedder(
        self,
        texts: list[str],
        model: Model,
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
        start = time.time()
        operation: str = kwargs.pop("operation", "")
        texts = [text.replace("\n", " ") for text in texts]

        client = self.client(model)
        deployment = self.deployments[model]

        response = client.embeddings.create(
            input=texts,
            model=deployment.deployment_name,
        )
        duration = time.time() - start

        usage = ModelUsage(deployment, self._increment_usage)
        usage.add_tokens(prompt_tokens=response.usage.prompt_tokens)

        cost = usage.apply(operation=operation, duration=duration, ttft=None)
        self.total_usage += cost
        self.usages.append(usage)

        embeddings = response.data
        return embeddings

    def speech_to_text(
        self,
        file: BufferedReader,
        model: Model,
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
        start = time.time()
        operation: str = kwargs.pop("operation", "")
        client = self.client(model)
        deployment = self.deployments[model]

        response = client.audio.transcriptions.create(
            file=file,
            model=deployment.deployment_name,
            response_format="verbose_json",
            **kwargs,
        )
        duration = time.time() - start

        usage = ModelUsage(deployment, self._increment_usage)
        usage.add_audio_duration(response.duration)

        cost = usage.apply(
            operation=operation,
            duration=duration,
            ttft=None,
        )
        self.total_usage += cost
        self.usages.append(usage)

        return response

    async def aspeech_to_text(
        self,
        file: BytesIO,
        model: Model,
        **kwargs: Any,
    ) -> TranscriptionVerbose:
        """Asynchronously processes audio data for speech-to-text using the Whisper model.

        Args:
            file: The audio data to process.
            model: The model to use for processing the audio.
            kwargs: Additional arguments to pass to the API.

        Returns:
            Any: The response from the API.
        """
        start = time.time()
        operation: str = kwargs.pop("operation", "")
        aclient = self.aclient(model)
        deployment = self.deployments[model]

        response = await aclient.audio.transcriptions.create(
            file=file,
            model=deployment.deployment_name,
            response_format="verbose_json",
            **kwargs,
        )
        duration = time.time() - start

        usage = ModelUsage(deployment, self._increment_usage)
        usage.add_audio_duration(response.duration)

        cost = usage.apply(
            operation=operation,
            duration=duration,
            ttft=None,
        )
        self.total_usage += cost
        self.usages.append(usage)

        return response

    def text_to_image(
        self,
        model: Model,
        prompt: str,
        size: Literal["1024x1024", "1024x1792", "1792x1024"] = "1024x1024",
        quality: Literal["standard", "hd"] = "standard",
        n: int = 1,
        **kwargs: Any,
    ) -> str:
        """Generate images from a text prompt using the specified model.

        Parameters:
        - model (Model): The model to be used for generating images.
        - prompt (str): The text prompt that describes the image to be generated.
        - size (Literal["1024x1024", "1024x1792", "1792x1024"]): The size of the generated image.
        Default is "1024x1024".
        - quality (Literal["standard", "hd"]): The quality of the generated image.
        Default is "hd".
        - n (int): The number of images to generate. Default is 1.
        - **kwargs (Any): Additional keyword arguments for further customization.

        Returns:
        - None: This function does not return any value. It performs the image generation
        operation and may handle side effects like saving or displaying the generated
        images based on the implementation.

        Raises:
        - Any relevant exceptions that may occur during the image generation process.
        """
        start = time.time()
        operation: str = kwargs.pop("operation", "")
        if "temperature" in kwargs and model in {"o3-mini", "o3-mini-high"}:
            kwargs.pop("temperature")
        client = self.client(model)
        deployment = self.deployments[model]

        response = client.images.generate(
            model=deployment.deployment_name,
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
        )
        duration = time.time() - start

        usage = ModelUsage(deployment, self._increment_usage)
        usage.add_image(
            quality=quality,
            size=size,
            n=n,
        )

        cost = usage.apply(
            operation=operation,
            duration=duration,
            ttft=None,
        )
        self.total_usage += cost
        self.usages.append(usage)

        return response.data[0].url

    def text_to_speech(
        self,
        text: str,
        model: Model,
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
        operation: str = kwargs.pop("operation", "")
        start = time.time()
        client = self.client(model)
        deployment = self.deployments[model]
        response = client.audio.speech.create(
            model=deployment.deployment_name,
            input=text,
            voice="alloy",
            **kwargs,
        )
        response.stream_to_file(path)
        duration = time.time() - start
        usage = ModelUsage(deployment, self._increment_usage)
        usage.add_tts(text)
        cost = usage.apply(operation=operation, duration=duration, ttft=None)
        self.total_usage += cost
        self.usages.append(usage)
        return path


def add_system_message(
    messages: Messages,
    model: Model,
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
    match model:
        case model if model in OPENAI_MODELS:
            messages.insert(0, {"role": "system", "content": system})
        case model if model in COHERE_MODELS:
            messages.insert(0, {"role": "system", "content": system})
        case model if model in META_MODELS:
            messages.insert(0, {"role": "system", "content": system})
        case model if model in GEMINI_MODELS:
            messages.insert(0, {"role": "system", "content": system})
        case model if model in MISTRAL_MODELS:
            messages.insert(0, {"role": "system", "content": system})
        case model if model in ANTHROPIC_MODELS:
            messages.insert(0, {"role": "system", "content": system})
        case _:
            logger.debug(
                f"[bold purple][LLMAX][/bold purple] The model specified, {model}, does not understand system mode.",
            )
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
) -> dict[str, Any]:
    """Returns the tool and the correct format for the llm."""
    call_id = tool_call.id
    function_name = tool_call.function.name if tool_call.function else ""

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
