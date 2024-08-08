"""Message type."""

from typing import Any, Literal

from pydantic import BaseModel


class Message(BaseModel):
    """Message."""

    role: Literal["user", "assistant", "system"]
    content: Any
