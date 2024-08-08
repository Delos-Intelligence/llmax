"""This module contains the MultiAIClient class.

This class is used to interface with multiple LLMs and AI models, supporting both
synchronous and asynchronous operations.
"""

from io import BufferedReader, BytesIO
from typing import Any, Callable, Generator

from openai.types import Embedding
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from llmax.external_clients.clients import Client, get_aclient, get_client
from llmax.messages import Messages
from llmax.models.deployment import Deployment
from llmax.models.fake import fake_llm
from llmax.models.models import Model
from llmax.usage import ModelUsage


class MultiAIClient:
    """Class to interface with multiple LLMs and AI models.

    This class supports both synchronous and asynchronous operations for obtaining
    chat completions, streaming responses, and generating text embeddings, with
    an emphasis on tracking and managing API usage.

    Attributes:
        deployments: A mapping from models to their deployment objects.
        get_usage: A function to get the current usage.
        increment_usage: A function to increment usage.
    """

    def __init__(
        self,
        deployments: dict[Model, Deployment],
        get_usage: Callable[[], float] = lambda: 0.0,
        increment_usage: Callable[[float, Model], bool] = lambda _1, _2: True,
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
        **kwargs: Any,
    ) -> ChatCompletion:
        """Synchronously invokes the API to get chat completions, tracking usage.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for generating the chat completions.
            kwargs: More args.

        Returns:
            ChatCompletion: The API response containing the chat completions.
        """
        response = self._create_chat(messages, model, **kwargs, stream=False)
        if not response.usage:
            message = "No usage for this request"
            raise ValueError(message)
        deployment = self.deployments[model]
        usage = ModelUsage(deployment, self._increment_usage, response.usage)
        usage.apply()
        return response

    def invoke_to_str(
        self,
        messages: Messages,
        model: Model,
        **kwargs: Any,
    ) -> str | None:
        """Convenience method to invoke the API and return the first response as a string.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for the chat completions.
            kwargs: More args.

        Returns:
            str | None: The content of the first choice in the response, if available.
        """
        response = self.invoke(messages, model, **kwargs)
        return response.choices[0].message.content if response.choices else None

    async def ainvoke(
        self,
        messages: Messages,
        model: Model,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Asynchronously invokes the API to get chat completions, tracking usage.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for generating the chat completions.
            kwargs: More args.

        Returns:
            ChatCompletion: The API response containing the chat completions.
        """
        response = await self._acreate_chat(messages, model, **kwargs, stream=False)
        if not response.usage:
            message = "No usage for this request"
            raise ValueError(message)
        deployment = self.deployments[model]
        usage = ModelUsage(deployment, self._increment_usage, response.usage)
        usage.apply()
        return response

    async def ainvoke_to_str(
        self,
        messages: Messages,
        model: Model,
        **kwargs: Any,
    ) -> str | None:
        """Asynchronously invokes the API and returns the first response as a string.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for the chat completions.
            kwargs: More args.

        Returns:
            str | None: The content of the first choice in the response, if available.
        """
        response = await self.ainvoke(messages, model, **kwargs)
        return response.choices[0].message.content if response.choices else None

    def stream(
        self,
        messages: Messages,
        model: Model,
        **kwargs: Any,
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Streams chat completions, allowing responses to be received in chunks.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for generating the chat completions.
            kwargs: More args.

        Yields:
            ChatCompletionChunk: Individual chunks of the completion response.
        """
        response = self._create_chat(messages, model, **kwargs, stream=True)
        deployment = self.deployments[model]
        usage = ModelUsage(deployment, self._increment_usage)
        usage.add_messages(messages)

        for chunk in response:
            usage.add_tokens(completion_tokens=1)
            yield chunk  # type: ignore

        usage.apply()

    def stream_output(
        self,
        messages: Messages,
        model: Model,
        **kwargs: Any,
    ) -> Generator[str, None, str]:
        """Streams formatted output from the chat completions.

        This method formats each chunk received from `stream` into a specific
        string format before yielding it.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for generating the chat completions.
            kwargs: More args.

        Yields:
            str: Formatted output for each chunk.
        """
        output = ""
        yield from fake_llm("", stream=False, send_empty=True)
        for completion_chunk in self.stream(messages, model, **kwargs):
            yield f"data: {completion_chunk.model_dump_json(exclude_unset=True)}\n\n"
            output += (
                content
                if (choices := completion_chunk.choices)
                and (content := choices[0].delta.content)
                else ""
            )
        yield "data: [DONE]\n\n"
        return output

    def embedder(
        self,
        texts: list[str],
        model: Model,
    ) -> list[Embedding]:
        """Obtains vector embeddings for a list of texts asynchronously.

        Args:
            texts: The texts to generate embeddings for.
            model: The embedding model.

        Returns:
            list[Embedding]: The embeddings for each text.
        """
        texts = [text.replace("\n", " ") for text in texts]

        client = self.client(model)
        deployment = self.deployments[model]

        response = client.embeddings.create(
            input=texts,
            model=deployment.deployment_name,
        )

        usage = ModelUsage(deployment, self._increment_usage)
        usage.add_tokens(prompt_tokens=response.usage.prompt_tokens)
        usage.apply()

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
        client = self.client(model)
        deployment = self.deployments[model]

        response = client.audio.transcriptions.create(
            file=file,
            model=deployment.deployment_name,
            **kwargs,
        )

        return response

    async def aspeech_to_text(
        self,
        file: BytesIO,
        model: Model,
        **kwargs: Any,
    ) -> str:
        """Asynchronously processes audio data for speech-to-text using the Whisper model.

        Args:
            file: The audio data to process.
            model: The model to use for processing the audio.
            kwargs: Additional arguments to pass to the API.

        Returns:
            Any: The response from the API.
        """
        aclient = self.aclient(model)
        deployment = self.deployments[model]

        response = await aclient.audio.transcriptions.create(
            file=file,
            model=deployment.deployment_name,
            **kwargs,
        )

        """usage = ModelUsage(deployment, self._increment_usage)
        usage.add_audio_duration(self.calculate_duration(audio_file=file))
        usage.apply()"""

        return response
