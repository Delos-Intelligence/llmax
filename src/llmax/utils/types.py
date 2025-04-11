"""Types for multiai clients."""

from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from pydantic import BaseModel


class StreamItemContent(BaseModel):
    """The content of a stream item."""

    content: str


class StreamItemOutput(BaseModel):
    """The output of a stream item."""

    tools: dict[int, ChoiceDeltaToolCall]
    output: str
