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
    "dall-e-3",
    "tts-1",
]

AnthropicModel = Literal[
    "claude-3.5-sonnet",
    "claude-3-haiku",
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
