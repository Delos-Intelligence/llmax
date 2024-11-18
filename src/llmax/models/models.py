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
    "whisper-1",
    "gpt-4o-mini",
    "o1-preview",
]

AnthropicModel = Literal[
    "claude-3.5-sonnet",
]


LLModel = Union[CohereModel, MetaModel, MistralModel, OpenAIModel, AnthropicModel]

SpeechToTextModel = Literal["whisper-1",]

Model = Union[LLModel, SpeechToTextModel]

COHERE_MODELS: tuple[CohereModel, ...] = get_args(CohereModel)
META_MODELS: tuple[MetaModel, ...] = get_args(MetaModel)
MISTRAL_MODELS: tuple[MistralModel, ...] = get_args(MistralModel)
OPENAI_MODELS: tuple[OpenAIModel, ...] = get_args(OpenAIModel)
ANTHROPIC_MODELS: tuple[AnthropicModel, ...] = get_args(AnthropicModel)

LLMS: tuple[LLModel, ...] = get_args(LLModel)
AUDIO: tuple[SpeechToTextModel, ...] = get_args(SpeechToTextModel)

MODELS: tuple[Model, ...] = get_args(Model)
