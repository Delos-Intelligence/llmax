import tiktoken
from openai.types.chat import ChatCompletionMessage


def count(string: str, model_name: str = "") -> int:
    if model_name:
        enc = tiktoken.encoding_for_model(model_name)
    else:
        enc = tiktoken.get_encoding("cl100k_base")

    return len(enc.encode(repr(string)))


def truncate(
    messages: list[ChatCompletionMessage], max_tokens: int
) -> list[ChatCompletionMessage]:
    tokens = 0
    for i, message in enumerate(messages):
        tokens += count(repr(message))
        if tokens > max_tokens:
            return messages[:i]
    return messages
