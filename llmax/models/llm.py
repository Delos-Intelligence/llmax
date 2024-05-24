import random
import time
from typing import Any, Callable, Generator

from openai.types import Embedding
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from llmax import clients
from llmax.usage import ModelUsage

from .deployment import Deployment
from .fake import fake_llm
from .models import Model

Messages = list


class MultiAIClient:
    """Class to interface with multiple LLMs and AI models.

    This class supports both synchronous and asynchronous operations for obtaining
    chat completions, streaming responses, and generating text embeddings, with
    an emphasis on tracking and managing API usage.

    Attributes:
        client: The client for synchronous API calls.
        aclient: The client for asynchronous API calls.
        deployments: A mapping of models to deployment strings.
        _get_usage: Function to get current usage.
        _increment_usage: Function to increment usage by a given amount.
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

        self.clients: dict[Model, Any] = {
            model: clients.get_client(deployment)
            for model, deployment in deployments.items()
        }

        self.aclients: dict[Model, Any] = {
            model: clients.get_aclient(deployment)
            for model, deployment in deployments.items()
        }

    def _create_chat(
        self,
        messages: Messages,
        model: Model,
        **kwargs,
    ) -> ChatCompletion:
        """Synchronously creates chat completions for the given messages and model.

        Args:
            messages: The list of messages to process.
            model: The model to use for generating completions.

        Returns:
            ChatCompletion: The completion response from the API.
        """
        client = self.clients[model]
        return client.chat.completions.create(
            messages=messages,
            model=self.deployments[model].deployment_name,
            **kwargs,
        )

    async def _acreate_chat(
        self,
        messages: Messages,
        model: Model,
        **kwargs,
    ) -> ChatCompletion:
        """Asynchronously creates chat completions for the given messages and model.

        Args:
            messages: The list of messages to process.
            model: The model to use for generating completions.

        Returns:
            ChatCompletion: The completion response from the API.
        """
        aclient = self.aclients[model]
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

        Returns:
            ChatCompletion: The API response containing the chat completions.
        """
        response = self._create_chat(messages, model, **kwargs, stream=False)
        assert response.usage, "No usage for this request"
        deployment = self.deployments[model]
        usage = ModelUsage(deployment, self._increment_usage, response.usage)
        usage.apply()
        return response

    def invoke_to_str(
        self,
        messages: Messages,
        model: Model,
        **kwargs,
    ) -> str | None:
        """Convenience method to invoke the API and return the first response as a string.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for the chat completions.

        Returns:
            str | None: The content of the first choice in the response, if available.
        """
        response = self.invoke(messages, model, **kwargs)
        return response.choices[0].message.content if response.choices else None

    async def ainvoke(
        self,
        messages: Messages,
        model: Model,
        **kwargs,
    ) -> ChatCompletion:
        """Asynchronously invokes the API to get chat completions, tracking usage.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for generating the chat completions.

        Returns:
            ChatCompletion: The API response containing the chat completions.
        """
        response = await self._acreate_chat(messages, model, **kwargs, stream=False)
        assert response.usage, "No usage for this request"
        deployment = self.deployments[model]
        usage = ModelUsage(deployment, self._increment_usage, response.usage)
        usage.apply()
        return response

    async def ainvoke_to_str(
        self,
        messages: Messages,
        model: Model,
        **kwargs,
    ) -> str | None:
        """Asynchronously invokes the API and returns the first response as a string.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for the chat completions.

        Returns:
            str | None: The content of the first choice in the response, if available.
        """
        response = await self.ainvoke(messages, model, **kwargs)
        return response.choices[0].message.content if response.choices else None

    def stream(
        self,
        messages: Messages,
        model: Model,
        **kwargs,
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Streams chat completions, allowing responses to be received in chunks.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for generating the chat completions.

        Yields:
            ChatCompletionChunk: Individual chunks of the completion response.
        """
        response = self._create_chat(messages, model, **kwargs, stream=True)
        deployment = self.deployments[model]
        usage = ModelUsage(deployment, self._increment_usage)
        usage.add_messages(messages)

        for chunk in response:
            usage.add_tokens(completion_tokens=1)
            yield chunk

        usage.apply()

    def stream_output(
        self,
        messages: Messages,
        model: Model,
        **kwargs,
    ) -> Generator[str, None, str]:
        """Streams formatted output from the chat completions.

        This method formats each chunk received from `stream` into a specific
        string format before yielding it.

        Args:
            messages: The list of messages for the chat.
            model: The model to use for generating the chat completions.

        Yields:
            str: Formatted output for each chunk.
        """
        output = ""
        yield from fake_llm("", stream=False, send_empty=True)
        for completion_chunk in self.stream(messages, model, **kwargs):
            yield f"data: {completion_chunk.model_dump_json(exclude_unset=True)}\n\n"
            if model == "gpt-4":
                time.sleep(random.uniform(0.05, 0.15))
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

        client = self.clients[model]
        response = client.embeddings.create(input=texts, model=self.deployments[model])

        deployment = self.deployments[model]
        usage = ModelUsage(deployment, self._increment_usage)
        usage.add_tokens(prompt_tokens=response.usage.prompt_tokens)
        usage.apply()

        embeddings = response.data
        return embeddings
