"""Fake LLM messages."""

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
                delta=ChoiceDelta(content=message + " "),
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
    model: Model = "gpt-4-turbo",
    stream: bool = True,
    done: bool = False,
    send_empty: bool = False,
) -> Generator[str, None, None]:
    """Generate fake LLM messages."""
    if send_empty:
        completion_chunk = to_completion_chunk("", model)
        yield f"data: {completion_chunk.model_dump_json(exclude_unset=True)}\n\n"
        return

    if stream:
        for word in message.split(" "):
            completion_chunk = to_completion_chunk(word, model)
            yield f"data: {completion_chunk.model_dump_json(exclude_unset=True)}\n\n"
            time.sleep(0.1 * random.random())
    else:
        completion_chunk = to_completion_chunk(message, model)
        yield f"data: {completion_chunk.model_dump_json(exclude_unset=True)}\n\n"
    if done:
        yield "data: [DONE]\n\n"
