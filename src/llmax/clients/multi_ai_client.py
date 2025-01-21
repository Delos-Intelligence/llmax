"""This module contains the MultiAIClient class.

This class is used to interface with multiple LLMs and AI models, supporting both
synchronous and asynchronous operations.
"""

import threading
import time
from collections.abc import Generator
from io import BufferedReader, BytesIO
from queue import Queue
from typing import Any, Callable, Literal

from openai.types import Embedding
from openai.types.audio import Transcription
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
        increment_usage: Callable[[float, Model, str], bool] = lambda _1, _2, _3: True,
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
        **kwargs: Any,
    ) -> ChatCompletion:
        """Synchronously invokes the API to get chat completions, tracking usage.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for generating the chat completions.
            system: A string that will be passed as a system prompt.
            kwargs: More args.

        Returns:
            ChatCompletion: The API response containing the chat completions.
        """
        operation: str = kwargs.pop("operation", "")
        if system:
            messages = add_system_message(
                messages=messages,
                model=model,
                system=system,
            )
        response = self._create_chat(messages, model, **kwargs, stream=False)
        if not response.usage:
            message = "No usage for this request"
            raise ValueError(message)
        deployment = self.deployments[model]
        usage = ModelUsage(deployment, self._increment_usage, response.usage)

        cost = usage.apply(operation=operation)
        self.total_usage += cost
        self.usages.append(usage)

        return response

    def invoke_to_str(
        self,
        messages: Messages,
        model: Model,
        system: str | None = None,
        **kwargs: Any,
    ) -> str | None:
        """Convenience method to invoke the API and return the first response as a string.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for the chat completions.
            system: A string that will be passed as a system prompt.
            kwargs: More args.

        Returns:
            str | None: The content of the first choice in the response, if available.
        """
        response = self.invoke(messages, model, system=system, **kwargs)
        return response.choices[0].message.content if response.choices else None

    async def ainvoke(
        self,
        messages: Messages,
        model: Model,
        system: str | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Asynchronously invokes the API to get chat completions, tracking usage.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for generating the chat completions.
            system: A string that will be passed as a system prompt.
            kwargs: More args.

        Returns:
            ChatCompletion: The API response containing the chat completions.
        """
        operation: str = kwargs.pop("operation", "")
        if system:
            messages = add_system_message(
                messages=messages,
                model=model,
                system=system,
            )
        response = await self._acreate_chat(messages, model, **kwargs, stream=False)
        if not response.usage:
            message = "No usage for this request"
            raise ValueError(message)
        deployment = self.deployments[model]
        usage = ModelUsage(deployment, self._increment_usage, response.usage)

        cost = usage.apply(operation=operation)
        self.total_usage += cost
        self.usages.append(usage)

        return response

    async def ainvoke_to_str(
        self,
        messages: Messages,
        model: Model,
        system: str | None = None,
        **kwargs: Any,
    ) -> str | None:
        """Asynchronously invokes the API and returns the first response as a string.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for the chat completions.
            system: A string that will be passed as a system prompt.
            kwargs: More args.

        Returns:
            str | None: The content of the first choice in the response, if available.
        """
        response = await self.ainvoke(messages, model, system=system, **kwargs)
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
        operation: str = kwargs.pop("operation", "")
        if system:
            messages = add_system_message(
                messages=messages,
                model=model,
                system=system,
            )
        response = self._create_chat(messages, model, **kwargs, stream=True)
        deployment = self.deployments[model]
        usage = ModelUsage(deployment, self._increment_usage)
        usage.add_messages(messages)
        answer = ""

        for chunk in response:
            try:
                if chunk.choices[0].delta.content:  # type: ignore
                    answer += str(chunk.choices[0].delta.content)  # type: ignore
            except Exception as e:
                logger.debug(f"Error in llmax streaming : {e}")
            yield chunk  # type: ignore

        usage.add_tokens(completion_tokens=tokens.count(answer))

        cost = usage.apply(operation=operation)
        self.total_usage += cost
        self.usages.append(usage)

    def stream_output_smooth(
        self,
        messages: Messages,
        model: Model,
        smooth_duration: int,
        system: str | None = None,
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
            kwargs: More args.

        Yields:
            str: Formatted output for each chunk.
        """
        output = ""
        output_queue = Queue()
        yield from fake_llm("", stream=False, send_empty=True)

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
                    yield "data: [DONE]\n\n"
                    break

                # Yield le chunk
                content, chunk = chunk_data["content"], chunk_data["chunk"]
                yield f"data: {chunk}\n\n"
                output += content

            # Attendre 10ms avant le prochain check
            time.sleep(smooth_duration / 1000)

        # Attendre que le thread de collecte se termine
        collector.join()
        return output

    def stream_output_base(
        self,
        messages: Messages,
        model: Model,
        system: str | None = None,
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
            kwargs: More args.

        Yields:
            str: Formatted output for each chunk.
        """
        if smooth_duration is None:
            output = yield from self.stream_output_base(
                messages=messages,
                model=model,
                system=system,
                **kwargs,
            )
            return output

        output = yield from self.stream_output_smooth(
            messages=messages,
            model=model,
            system=system,
            smooth_duration=smooth_duration,
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
        operation: str = kwargs.pop("operation", "")
        texts = [text.replace("\n", " ") for text in texts]

        client = self.client(model)
        deployment = self.deployments[model]

        response = client.embeddings.create(
            input=texts,
            model=deployment.deployment_name,
        )

        usage = ModelUsage(deployment, self._increment_usage)
        usage.add_tokens(prompt_tokens=response.usage.prompt_tokens)

        cost = usage.apply(operation=operation)
        self.total_usage += cost
        self.usages.append(usage)

        embeddings = response.data
        return embeddings

    def speech_to_text(
        self,
        file: BufferedReader,
        model: Model,
        **kwargs: Any,
    ) -> str:
        """Synchronously processes audio data for speech-to-text using the Whisper model.

        Args:
            file: The audio data to process.
            model: The model to use for processing the audio.
            kwargs: Additional arguments to pass to the API.

        Returns:
            Any: The response from the API.
        """
        operation: str = kwargs.pop("operation", "")
        client = self.client(model)
        deployment = self.deployments[model]

        response = client.audio.transcriptions.create(
            file=file,
            model=deployment.deployment_name,
            response_format="verbose_json",
            **kwargs,
        )

        usage = ModelUsage(deployment, self._increment_usage)
        usage.add_audio_duration(response.duration)

        cost = usage.apply(operation=operation)
        self.total_usage += cost
        self.usages.append(usage)

        return response

    async def aspeech_to_text(
        self,
        file: BytesIO,
        model: Model,
        **kwargs: Any,
    ) -> Transcription:
        """Asynchronously processes audio data for speech-to-text using the Whisper model.

        Args:
            file: The audio data to process.
            model: The model to use for processing the audio.
            kwargs: Additional arguments to pass to the API.

        Returns:
            Any: The response from the API.
        """
        operation: str = kwargs.pop("operation", "")
        aclient = self.aclient(model)
        deployment = self.deployments[model]

        response = await aclient.audio.transcriptions.create(
            file=file,
            model=deployment.deployment_name,
            response_format="verbose_json",
            **kwargs,
        )

        usage = ModelUsage(deployment, self._increment_usage)
        usage.add_audio_duration(response.duration)

        cost = usage.apply(operation=operation)
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

        usage = ModelUsage(deployment, self._increment_usage)
        usage.add_image(
            quality=quality,
            size=size,
            n=n,
        )

        cost = usage.apply(operation=operation)
        self.total_usage += cost
        self.usages.append(usage)

        return response.data[0].url


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
