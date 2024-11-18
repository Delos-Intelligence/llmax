from __future__ import annotations

from typing import Union, Iterable, Optional

from ..._utils import (
    maybe_transform,
    async_maybe_transform,
)
# from ..._streaming import Stream, AsyncStream
# from ...types.chat import (
#     completion_create_params,
# )
from chat_completion import ChatCompletion
from chat_completion_message_param import ChatCompletionMessageParam

from llmax.models.models import MODELS

# __all__ = ["Completions", "AsyncCompletions"]


class Completions:
    def create(
        self,
        *,
        messages: Iterable[ChatCompletionMessageParam],
        model: MODELS,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: float | None = None,
    # ) -> ChatCompletion | Stream[ChatCompletionChunk]:
    ) -> ChatCompletion:
        return self._post(
            "/chat/completions",
            body=maybe_transform(
                {
                    "messages": messages,
                    "model": model,
                    "max_completion_tokens": max_completion_tokens,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                completion_create_params.CompletionCreateParams,
            ),
            cast_to=ChatCompletion,
            stream_cls=Stream[ChatCompletionChunk],
        )
    
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
