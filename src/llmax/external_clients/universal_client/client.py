"""Universal client client part."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Generator

from llmax.external_clients.universal_client.chat import Chat

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

    from llmax.models.deployment import Deployment


class UniversalClient:
    """Universal client."""

    internal_client: Any
    chat: Chat
    deployment: Deployment

    def __init__(
        self,
        client_creation: Callable[..., Any],
        completion_call: Callable[
            ...,
            ChatCompletion | None | Generator[ChatCompletionChunk, None, None],
        ],
        deployment: Deployment,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Construct a new synchronous client instance based on OpenAI client model."""
        self.internal_client = client_creation(*args, **kwargs)
        self.chat = Chat(self.internal_client, completion_call, deployment)
