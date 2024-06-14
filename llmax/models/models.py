"""Supported models."""

from typing import Literal, Union, get_args

CohereModel = Literal[
    "command-r",
    "command-r-plus",
]

MetaModel = Literal["llama-3-70b-instruct",]

MistralModel = Literal[
    "mistral-large",
    "mistral-small",
]


OpenAIModel = Literal[
    "ada-v2",
    "gpt-3.5",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4o",
    "text-embedding-3-large",
    "text-embedding-3-small",
]


COHERE_MODELS: tuple[CohereModel, ...] = get_args(CohereModel)
META_MODELS: tuple[MetaModel, ...] = get_args(MetaModel)
MISTRAL_MODELS: tuple[MistralModel, ...] = get_args(MistralModel)
OPENAI_MODELS: tuple[OpenAIModel, ...] = get_args(OpenAIModel)

Model = Union[
    CohereModel,
    MetaModel,
    MistralModel,
    OpenAIModel,
]
MODELS: tuple[Model, ...] = get_args(Model)
