from __future__ import annotations

from typing import Any, Callable, Generator, Iterable, Optional

# from ..._utils import (
#     maybe_transform,
#     async_maybe_transform,
# )
# from ..._streaming import Stream, AsyncStream
# from ...types.chat import (
#     completion_create_params,
# )
from llmax.external_clients.universal_client.chat_completion import ChatCompletion
from llmax.external_clients.universal_client.chat_completion_chunk import ChatCompletionChunk
from llmax.external_clients.universal_client.chat_completion_message import ChatCompletionMessage
from llmax.models.deployment import Deployment
from llmax.models.models import Model

# __all__ = ["Completions", "AsyncCompletions"]


class Completions:
    client: Any
    completion_call: Callable[..., Optional[ChatCompletion] | Optional[Generator[ChatCompletionChunk, None, None]]]
    deployment: Deployment

    def __init__(
        self,
        client: Any,
        completion_call: Callable[...,Optional[ChatCompletion] | Optional[Generator[ChatCompletionChunk, None, None]]],
        deployment: Deployment,
    ) -> None:
        """Construct a completions object based on OpenAI client model."""
        self.client = client
        self.completion_call = completion_call
        self.deployment = deployment

    def create(
        self,
        messages: Iterable[ChatCompletionMessage],
        model: Model,
        *args: Any,
        **kwargs: Any,
    ) -> Optional[ChatCompletion] | Optional[Generator[ChatCompletionChunk, None, None]]:
        res = model
        return self.completion_call(self.client, messages, self.deployment.deployment_name, *args, **kwargs)

    # def create(
    #     self,
    #     *,
    #     messages: Iterable[ChatCompletionMessageParam],
    #     model: Union[str, ChatModel],
    #     max_completion_tokens: Optional[int] = None,
    #     max_tokens: Optional[int] = None,
    #     stream: bool = False,
    #     temperature: Optional[float] = None,
    #     timeout: float | None = None,
    # ) -> ChatCompletion | Stream[ChatCompletionChunk]:
    #     return self._post(
    #         "/chat/completions",
    #         body=maybe_transform(
    #             {
    #                 "messages": messages,
    #                 "model": model,
    #                 "max_completion_tokens": max_completion_tokens,
    #                 "max_tokens": max_tokens,
    #                 "stream": stream,
    #                 "temperature": temperature,
    #             },
    #             completion_create_params.CompletionCreateParams,
    #         ),
    #         cast_to=ChatCompletion,
    #         stream=stream or False,
    #         stream_cls=Stream[ChatCompletionChunk],
    #     )
