"""Fake LLM messages."""

import json
import random
import time
from typing import Generator

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

from llmax.models.models import Model


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
    model: Model = "gpt-4-turbo",  # noqa: ARG001
    stream: bool = True,
    done: bool = False,  # noqa: ARG001
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
