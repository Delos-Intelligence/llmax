"""Supported providers."""

import typing
from typing import Literal

Provider = Literal[
    "azure",
    "openai",
    "mistral",
    "aws-bedrock",
    "gcp-vertexai",
]


PROVIDERS: tuple[Provider, ...] = typing.get_args(Provider)
