"""Supported providers."""

import typing
from typing import Literal

Provider = Literal[
    "azure",
    "openai",
    "mistral",
    "aws-bedrock",
    "gcp-vertexai",
    "scaleway",
]


PROVIDERS: tuple[Provider, ...] = typing.get_args(Provider)
