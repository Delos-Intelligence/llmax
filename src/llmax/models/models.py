"""Supported models."""

from typing import Literal, Union, get_args

CohereModel = Literal[
    "command-r",
    "command-r-plus",
]

MetaModel = Literal["llama-3-70b-instruct",]

GeminiModel = Literal["google/gemini-1.5-flash-002", "google/gemini-1.5-pro-002"]

MistralModel = Literal[
    "mistral-large",
    "mistral-small",
    "mistral-large-2411",
]


OpenAIModel = Literal[
    "ada-v2",
    "dall-e-3",
    "gpt-3.5",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4.1",
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "o1-preview",
    "o3-mini",
    "o3-mini-high",
    "text-embedding-3-large",
    "text-embedding-3-small",
    "tts-1",
    "whisper-1",
]

AnthropicModel = Literal[
    "claude-3.5-sonnet",
    "claude-3-haiku",
    "claude-3.7-sonnet",
]


LLModel = Union[
    CohereModel,
    MetaModel,
    MistralModel,
    OpenAIModel,
    AnthropicModel,
    GeminiModel,
]

SpeechToTextModel = Literal["whisper-1",]

ImageGenerationModel = Literal["dall-e-3",]

TextToSpeechModel = Literal["tts-1",]

Model = Union[LLModel, SpeechToTextModel, ImageGenerationModel]

COHERE_MODELS: tuple[CohereModel, ...] = get_args(CohereModel)
META_MODELS: tuple[MetaModel, ...] = get_args(MetaModel)
GEMINI_MODELS: tuple[GeminiModel, ...] = get_args(GeminiModel)
MISTRAL_MODELS: tuple[MistralModel, ...] = get_args(MistralModel)
OPENAI_MODELS: tuple[OpenAIModel, ...] = get_args(OpenAIModel)
ANTHROPIC_MODELS: tuple[AnthropicModel, ...] = get_args(AnthropicModel)

LLMS: tuple[LLModel, ...] = get_args(LLModel)
AUDIO: tuple[SpeechToTextModel, ...] = get_args(SpeechToTextModel)
IMAGE: tuple[ImageGenerationModel, ...] = get_args(ImageGenerationModel)
TEXTTOAUDIO: tuple[TextToSpeechModel, ...] = get_args(TextToSpeechModel)

MODELS: tuple[Model, ...] = get_args(Model)
