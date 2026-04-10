"""Fake LLM messages."""

import json
import random
import time
from collections.abc import Generator
from typing import Any

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta


def to_completion_chunk(message: str, model: str) -> ChatCompletionChunk:
    """Convert a message to a completion chunk."""
    return ChatCompletionChunk(
        id="fake",
        choices=[
            Choice(
                delta=ChoiceDelta(content=message + ""),
                finish_reason="stop",
                index=0,
            ),
        ],
        created=1708363210,
        model=model,
        object="chat.completion.chunk",
    )


def fake_llm(
    message: str,
    stream: bool = True,
    send_empty: bool = False,
) -> Generator[str, None, None]:
    """Generate fake LLM messages."""
    if send_empty:
        yield f"0: {json.dumps('', separators=(',', ':'))}\n\n"
        return

    if stream:
        for word in message.split(" "):
            yield f"0: {json.dumps(word, separators=(',', ':'))}\n\n"
            time.sleep(0.1 * random.random())
    else:
        yield f"0: {json.dumps(message, separators=(',', ':'))}\n\n"
    return


def fake_tool_llm(
    tool_name: str,
    tool_id: str,
    args: dict[str, Any],
) -> str:
    """Generate fake LLM messages compatible with useChat."""
    tool_call = {
        "toolCallId": tool_id,
        "toolName": tool_name,
        "args": args,
    }

    return f"9:{json.dumps(tool_call, separators=(',', ':'))}\n\n"
