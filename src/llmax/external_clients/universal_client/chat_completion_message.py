"""Chat completion message models."""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel

__all__ = ["ChatCompletionMessage"]


class ChatCompletionAssistantMessage(BaseModel):
    content: str
    """The contents of the assistant message."""

    role: Literal["assistant"] = "assistant"
    """The role of the messages author, in this case `assistant`."""


class ChatCompletionSystemMessage(BaseModel):
    content: list[str] | str
    """The contents of the system message."""

    role: Literal["system"]
    """The role of the messages author, in this case `system`."""


class ChatCompletionUserMessage(BaseModel):
    content: list[str] | str
    """The contents of the user message."""

    role: Literal["user"]
    """The role of the messages author, in this case `user`."""


ChatCompletionMessage = Union[
    ChatCompletionSystemMessage,
    ChatCompletionUserMessage,
    ChatCompletionAssistantMessage,
]
