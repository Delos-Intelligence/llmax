# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, Generator, Optional

from openai.types.chat import ChatCompletion, ChatCompletionChunk

from llmax.models.deployment import Deployment

from .completions import Completions

__all__ = ["Chat"]


class Chat:
    client: Any
    completion_call: Callable[..., Optional[ChatCompletion] | Optional[Generator[ChatCompletionChunk, None, None]]]
    deployment: Deployment

    def __init__(
        self,
        client: Any,
        completion_call: Callable[..., Optional[ChatCompletion] | Optional[Generator[ChatCompletionChunk, None, None]]],
        deployment: Deployment,
    ) -> None:
        """Construct a chat instance based on OpenAI client model."""
        self.client = client
        self.completion_call = completion_call
        self.deployment = deployment

    @cached_property
    def completions(self) -> Completions:
        return Completions(self.client, self.completion_call, self.deployment)


# class AsyncChat(AsyncAPIResource):
#     @cached_property
#     def completions(self) -> AsyncCompletions:
#         return AsyncCompletions(self._client)

#     @cached_property
#     def with_raw_response(self) -> AsyncChatWithRawResponse:
#         """
#         This property can be used as a prefix for any HTTP method call to return the
#         the raw response object instead of the parsed content.

#         For more information, see https://www.github.com/openai/openai-python#accessing-raw-response-data-eg-headers
#         """
#         return AsyncChatWithRawResponse(self)

#     @cached_property
#     def with_streaming_response(self) -> AsyncChatWithStreamingResponse:
#         """
#         An alternative to `.with_raw_response` that doesn't eagerly read the response body.

#         For more information, see https://www.github.com/openai/openai-python#with_streaming_response
#         """
#         return AsyncChatWithStreamingResponse(self)


# class ChatWithRawResponse:
#     def __init__(self, chat: Chat) -> None:
#         self._chat = chat

#     @cached_property
#     def completions(self) -> CompletionsWithRawResponse:
#         return CompletionsWithRawResponse(self._chat.completions)


# class AsyncChatWithRawResponse:
#     def __init__(self, chat: AsyncChat) -> None:
#         self._chat = chat

#     @cached_property
#     def completions(self) -> AsyncCompletionsWithRawResponse:
#         return AsyncCompletionsWithRawResponse(self._chat.completions)


# class ChatWithStreamingResponse:
#     def __init__(self, chat: Chat) -> None:
#         self._chat = chat

#     @cached_property
#     def completions(self) -> CompletionsWithStreamingResponse:
#         return CompletionsWithStreamingResponse(self._chat.completions)


# class AsyncChatWithStreamingResponse:
#     def __init__(self, chat: AsyncChat) -> None:
#         self._chat = chat

#     @cached_property
#     def completions(self) -> AsyncCompletionsWithStreamingResponse:
#         return AsyncCompletionsWithStreamingResponse(self._chat.completions)
