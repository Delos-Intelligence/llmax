"""Types for multiai clients."""  # noqa: A005

from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from pydantic import BaseModel

from llmax.models import Model


class StreamItemContent(BaseModel):
    """The content of a stream item."""

    content: str


class ModelItemContent(BaseModel):
    """The content designing the model used."""

    model_used: Model


class StreamItemOutput(BaseModel):
    """The output of a stream item."""

    tools: dict[int, ChoiceDeltaToolCall]
    output: str


class ToolItemContent(BaseModel):
    """The content of an item that is yielded by the tool."""

    content: str


class ToolStreamItemOutput(BaseModel):
    """The output of an item that is yielded by the tool."""

    redo: bool
    output: str | None


ToolItem = ToolItemContent | ToolStreamItemOutput
StreamedItem = StreamItemContent | StreamItemOutput | ModelItemContent
