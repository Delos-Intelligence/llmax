"""Chat model for the universal client."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Generator

from .completions import Completions

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

    from llmax.models.deployment import Deployment

__all__ = ["Chat"]


class Chat:
    """Chat item for the universal client."""

    client: Any
    completion_call: Callable[
        ...,
        None | ChatCompletion | Generator[ChatCompletionChunk, None, None],
    ]
    deployment: Deployment

    def __init__(
        self,
        client: Any,
        completion_call: Callable[
            ...,
            None | ChatCompletion | Generator[ChatCompletionChunk, None, None],
        ],
        deployment: Deployment,
    ) -> None:
        """Construct a chat instance based on OpenAI client model."""
        self.client = client
        self.completion_call = completion_call
        self.deployment = deployment

    @cached_property
    def completions(self) -> Completions:
        """Format the input to completions."""
        return Completions(self.client, self.completion_call, self.deployment)
