"""Token-related utilities."""

import tiktoken
from openai.types.chat import ChatCompletionMessage


def count(string: str) -> int:
    """Count the number of tokens in a string."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(repr(string), disallowed_special=()))


def truncate(
    messages: list[ChatCompletionMessage],
    max_tokens: int,
) -> list[ChatCompletionMessage]:
    """Truncate messages to a maximum token length."""
    tokens = 0
    for i, message in enumerate(messages):
        tokens += count(repr(message))
        if tokens > max_tokens:
            return messages[:i]
    return messages
