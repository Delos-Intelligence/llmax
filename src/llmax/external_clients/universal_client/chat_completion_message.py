"""Chat completion message models."""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel

__all__ = ["ChatCompletionMessage"]


class ChatCompletionAssistantMessage(BaseModel):
    content: str | list[dict[str, Any]]
    """The contents of the assistant message."""

    role: Literal["assistant"] = "assistant"
    """The role of the messages author, in this case `assistant`."""


class ChatCompletionSystemMessage(BaseModel):
    content: list[str] | str
    """The contents of the system message."""

    role: Literal["system"]
    """The role of the messages author, in this case `system`."""


class ChatCompletionUserMessage(BaseModel):
    content: list[str] | str | list[dict[str, Any]]
    """The contents of the user message."""

    role: Literal["user"]
    """The role of the messages author, in this case `user`."""


class ToolCall(BaseModel):
    function: FunctionCall
    id: str
    type: Literal["function"]
    index: int = 0


class FunctionCall(BaseModel):
    arguments: str | None = None
    name: str


class ChatCompletionToolMessage(BaseModel):
    content: str | None = None
    role: Literal["tool"] = "tool"
    tool_call_id: str


ChatCompletionMessage = Union[
    ChatCompletionSystemMessage,
    ChatCompletionUserMessage,
    ChatCompletionAssistantMessage,
    ChatCompletionToolMessage,
]
