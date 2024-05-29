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

CohereModels = Literal[
    "command-r",
    "command-r-plus",
]

MetaModels = Literal["llama-3-70b-instruct",]

Model = OpenAIModels | MistralModels | CohereModels | MetaModels


OPENAI_MODELS: tuple[OpenAIModels, ...] = typing.get_args(OpenAIModels)
MISTRAL_MODELS: tuple[MistralModels, ...] = typing.get_args(MistralModels)
COHERE_MODELS: tuple[CohereModels, ...] = typing.get_args(CohereModels)
META_MODELS: tuple[MetaModels, ...] = typing.get_args(MetaModels)

MODELS: tuple[Model, ...] = OPENAI_MODELS + MISTRAL_MODELS + COHERE_MODELS + META_MODELS
