"""Universal client client part."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from llmax.external_clients.universal_client.chat import AsyncChat, Chat

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Generator

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


class AsyncUniversalClient:
    """Async universal client."""

    def __init__(
        self,
        client_creation: Callable[..., Awaitable[Any]],
        completion_call: Callable[
            ...,
            Awaitable[
                ChatCompletion | AsyncGenerator[ChatCompletionChunk, None] | None
            ],
        ],
        deployment: Deployment,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Construct a new asynchronous client instance based on OpenAI client model."""
        self.client_creation = client_creation
        self.client_creation_args = args
        self.client_creation_kwargs = kwargs
        self.completion_call = completion_call
        self.deployment = deployment
        self._internal_client = None
        self._chat = None

    async def _ensure_client(self) -> Any:
        """Ensure the internal client is initialized."""
        if self._internal_client is None:
            self._internal_client = await self.client_creation(
                *self.client_creation_args,
                **self.client_creation_kwargs,
            )
        return self._internal_client

    async def get_chat(self) -> AsyncChat:
        """Get the chat interface, ensuring the client is initialized."""
        if self._chat is None:
            client = await self._ensure_client()
            self._chat = AsyncChat(client, self.completion_call, self.deployment)
        return self._chat

    async def close(self) -> None:
        """Close the internal client and release resources."""
        if self._internal_client is not None:
            # For aioboto3 client
            await self._internal_client.__aexit__(None, None, None)
            self._internal_client = None
            self._chat = None

    async def __aenter__(self) -> AsyncUniversalClient:  # noqa: PYI034
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Exit the async context manager and cleanup resources."""
        await self.close()
