from pydantic import BaseModel
from typing import List, Literal, Optional


from llmax.external_clients.universal_client.chat_completion import Choice
from llmax.external_clients.universal_client.completion_usage import CompletionUsage


class ChatCompletionChunk(BaseModel):
    id: str
    """A unique identifier for the chat completion. Each chunk has the same ID."""

    choices: List[Choice]
    """A list of chat completion choices.

    Can contain more than one elements if `n` is greater than 1. Can also be empty
    for the last chunk if you set `stream_options: {"include_usage": true}`.
    """

    created: int
    """The Unix timestamp (in seconds) of when the chat completion was created.

    Each chunk has the same timestamp.
    """

    model: str
    """The model to generate the completion."""

    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    """The object type, which is always `chat.completion.chunk`."""

    service_tier: Optional[Literal["scale", "default"]] = None
    """The service tier used for processing the request.

    This field is only included if the `service_tier` parameter is specified in the
    request.
    """

    system_fingerprint: Optional[str] = None
    """
    This fingerprint represents the backend configuration that the model runs with.
    Can be used in conjunction with the `seed` request parameter to understand when
    backend changes have been made that might impact determinism.
    """

    usage: Optional[CompletionUsage] = None
    """
    An optional field that will only be present when you set
    `stream_options: {"include_usage": true}` in your request. When present, it
    contains a null value except for the last chunk which contains the token usage
    statistics for the entire request.
    """
