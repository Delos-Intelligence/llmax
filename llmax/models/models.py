"""Supported models."""

import typing
from typing import Literal

OpenAIModels = Literal[
    "ada-v2",
    "gpt-3.5",
    "gpt-4",
    "gpt-4-turbo",
    "text-embedding-3-large",
    "text-embedding-3-small",
]

MistralModels = Literal[
    "mistral-large",
    "mistral-small",
]

Model = OpenAIModels | MistralModels


OPENAI_MODELS: tuple[OpenAIModels, ...] = typing.get_args(OpenAIModels)
MISTRAL_MODELS: tuple[MistralModels, ...] = typing.get_args(MistralModels)
MODELS: tuple[Model, ...] = OPENAI_MODELS + MISTRAL_MODELS
