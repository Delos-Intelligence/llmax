"""Supported providers."""

import typing
from typing import Literal

Provider = Literal[
    "azure",
    "openai",
    "mistral",
]


PROVIDERS: tuple[Provider, ...] = typing.get_args(Provider)
