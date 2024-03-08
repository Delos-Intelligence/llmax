import random
import time
from typing import Generator

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta


def to_completion_chunk(message: str, model: str) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="fake",
        choices=[
            Choice(
                delta=ChoiceDelta(content=message + " "),
                finish_reason="stop",
                index=0,
            )
        ],
        created=1708363210,
        model=model,
        object="chat.completion.chunk",
    )


def fake_llm(
    message: str, model="gpt-4", stream=True, done=False, send_empty=False
) -> Generator[str, None, None]:
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
