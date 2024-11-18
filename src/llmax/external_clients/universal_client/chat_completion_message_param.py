from __future__ import annotations

from typing import Literal, Union


__all__ = ["ChatCompletionMessageParam"]


class ChatCompletionAssistantMessageParam:
    content: str
    """The contents of the system message."""

    role: Literal["system"]
    """The role of the messages author, in this case `system`."""


class ChatCompletionSystemMessageParam:
    content: str
    """The contents of the system message."""

    role: Literal["system"]
    """The role of the messages author, in this case `system`."""


class ChatCompletionUserMessageParam:
    content: str
    """The contents of the system message."""

    role: Literal["system"]
    """The role of the messages author, in this case `system`."""


ChatCompletionMessageParam: Union[
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
]