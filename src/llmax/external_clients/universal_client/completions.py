"""Completion model for universal client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Generator, Iterable

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

    from llmax.external_clients.universal_client.chat_completion_message import (
        ChatCompletionMessage,
    )
    from llmax.models.deployment import Deployment
    from llmax.models.models import Model


class Completions:
    """Completions for the universal client."""

    client: Any
    completion_call: Callable[
        ...,
        ChatCompletion | None | Generator[ChatCompletionChunk, None, None],
    ]
    deployment: Deployment

    def __init__(
        self,
        client: Any,
        completion_call: Callable[
            ...,
            ChatCompletion | None | Generator[ChatCompletionChunk, None, None],
        ],
        deployment: Deployment,
    ) -> None:
        """Construct a completions object based on OpenAI client model."""
        self.client = client
        self.completion_call = completion_call
        self.deployment = deployment

    def create(
        self,
        messages: Iterable[ChatCompletionMessage],
        model: Model,  # noqa: ARG002
        *args: Any,
        **kwargs: Any,
    ) -> ChatCompletion | Generator[ChatCompletionChunk, None, None] | None:
        """Create function to create a completion item."""
        return self.completion_call(
            self.client,
            messages,
            self.deployment.deployment_name,
            *args,
            **kwargs,
        )
