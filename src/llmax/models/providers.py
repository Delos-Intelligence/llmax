"""Supported providers."""

import typing
from typing import Literal

Provider = Literal[
    "azure",
    "openai",
    "mistral",
    "anthropic",
    "aws-bedrock",
    "gcp-vertexai",
    "scaleway",
    "gemini",
    "openrouter",
    "elevenlabs",
]


PROVIDERS: tuple[Provider, ...] = typing.get_args(Provider)
