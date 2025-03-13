"""This module contains the MultiAIClient class.

This class is used to interface with multiple LLMs and AI models, supporting both
synchronous and asynchronous operations.
"""

import asyncio
import json
import threading
import time
from collections.abc import Generator
from io import BufferedReader, BytesIO
from queue import Queue
from typing import Any, Callable, Literal

from openai import BadRequestError, RateLimitError
from openai.types import Embedding
from openai.types.audio import TranscriptionVerbose
from openai.types.chat import ChatCompletion, ChatCompletionChunk

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
from llmax.usage import ModelUsage, tokens
from llmax.utils import logger


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
        if system:
            messages = add_system_message(
                messages=messages,
                model=model,
                system=system,
            )
        try:
            response = self._create_chat(messages, model, **kwargs, stream=True)
        except BadRequestError:
            return
        deployment = self.deployments[model]
        usage = ModelUsage(deployment, self._increment_usage)
        usage.add_messages(messages)
        answer = ""

        for chunk in response:
            try:
                if chunk.choices[0].delta.content:  # type: ignore
                    if ttft is None:
                        ttft = time.time() - start
                    answer += str(chunk.choices[0].delta.content)  # type: ignore
            except Exception as e:
                logger.debug(f"Error in llmax streaming : {e}")
            yield chunk  # type: ignore

        duration = time.time() - start
        usage.add_tokens(completion_tokens=tokens.count(answer))

        cost = usage.apply(operation=operation, ttft=ttft, duration=duration)
        self.total_usage += cost
        self.usages.append(usage)

    def stream_output_smooth(
        self,
        messages: Messages,
        model: Model,
        smooth_duration: int,
        system: str | None = None,
        beta: bool = True,
        **kwargs: Any,
    ) -> Generator[str, None, str]:
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
        output = ""
        output_queue = Queue()
        yield from fake_llm(
            "",
            stream=False,
            send_empty=True,
            beta=beta,
        )

        def collect_chunks() -> None:
            for completion_chunk in self.stream(
                messages,
                model,
                system=system,
                **kwargs,
            ):
                choices = completion_chunk.choices
                if choices and (content := choices[0].delta.content):
                    chunk_data = {
                        "chunk": completion_chunk.model_dump_json(exclude_unset=True),
                        "content": content,
                    }
                    output_queue.put(chunk_data)

            # Marquer la fin du streaming
            output_queue.put(None)

        # Démarrer le thread de collecte
        collector = threading.Thread(target=collect_chunks)
        collector.start()

        while True:
            if not output_queue.empty():
                chunk_data = output_queue.get()

                # Vérifier si c'est la fin du streaming
                if chunk_data is None:
                    if not beta:
                        yield "data: [DONE]\n\n"
                    break

                # Yield le chunk
                if beta:
                    content = chunk_data["content"]
                    yield stream_chunk(content, "text")
                else:
                    content, chunk = chunk_data["content"], chunk_data["chunk"]
                    yield f"data: {chunk}\n\n"
                output += content

            # Attendre 10ms avant le prochain check
            time.sleep(smooth_duration / 1000)

        # Attendre que le thread de collecte se termine
        collector.join()
        return output

    def stream_output_base(  # noqa: D417
        self,
        messages: Messages,
        model: Model,
        system: str | None = None,
        beta: bool = True,
        **kwargs: Any,
    ) -> Generator[str, None, str]:
        """Streams formatted output from the chat completions.

        This method formats each chunk received from `stream` into a specific
        string format before yielding it.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for generating the chat completions.
            system: A string that will be passed as a system prompt.
            kwargs: More args.

        Yields:
            str: Formatted output for each chunk.
        """
        if beta:
            output = ""
            yield from fake_llm("", stream=False, send_empty=True, beta=beta)
            for completion_chunk in self.stream(
                messages,
                model,
                system=system,
                **kwargs,
            ):
                content = completion_chunk.choices[0].delta.content or ""
                if not content:
                    continue
                yield stream_chunk(content, "text")
                output += content
            return output

        output = ""
        yield from fake_llm("", stream=False, send_empty=True)
        for completion_chunk in self.stream(messages, model, system=system, **kwargs):
            yield f"data: {completion_chunk.model_dump_json(exclude_unset=True)}\n\n"
            output += (
                content
                if (choices := completion_chunk.choices)
                and (content := choices[0].delta.content)
                else ""
            )
        yield "data: [DONE]\n\n"
        return output

    def stream_output(
        self,
        messages: Messages,
        model: Model,
        system: str | None = None,
        smooth_duration: int | None = None,
        beta: bool = True,
        **kwargs: Any,
    ) -> Generator[str, None, str]:
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
        if smooth_duration is None:
            output = yield from self.stream_output_base(
                messages=messages,
                model=model,
                system=system,
                beta=beta,
                **kwargs,
            )
            return output

        output = yield from self.stream_output_smooth(
            messages=messages,
            model=model,
            system=system,
            smooth_duration=smooth_duration,
            beta=beta,
            **kwargs,
        )
        return output

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
            pass
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
