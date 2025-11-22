"""Supported models."""

from typing import Literal, Union, get_args

CohereModel = Literal[
    "command-r",
    "command-r-plus",
]

MetaModel = Literal[
    "llama-3-70b-instruct",
    "llama-4-scout-17b-16e-instruct",
    "llama-4-maverick-17b-128e-instruct-fp8",
]

GeminiModel = Literal["google/gemini-1.5-flash-002", "google/gemini-1.5-pro-002"]

MistralModel = Literal[
    "mistral-large",
    "mistral-small",
    "mistral-large-2411",
]


OpenAIModel = Literal[
    "ada-v2",
    "dall-e-3",
    "gpt-image-1",
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
    "gpt-4o-transcribe",
    "text-embedding-3-large",
    "text-embedding-3-small",
    "tts-1",
    "whisper-1",
    "gpt-5",
    "gpt-5-chat",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-oss-120b",
]

AnthropicModel = Literal[
    "claude-3.5-sonnet",
    "claude-3-haiku",
    "claude-3.7-sonnet",
    "claude-4-sonnet",
    "claude-4.5-sonnet",
]

ScalewayModel = Literal[
    "scaleway/qwen3-235b-a22b-instruct-2507",
    "scaleway/gpt-oss-120b",
    "scaleway/gemma-3-27b-it",
    "scaleway/whisper-large-v3",
    "scaleway/voxtral-small-24b-2507",
    "scaleway/mistral-small-3.2-24b-instruct-2506",
    "scaleway/llama-3.3-70b-instruct",
    "scaleway/deepseek-r1-distill-llama-70b",
]


LLModel = Union[
    CohereModel,
    MetaModel,
    MistralModel,
    OpenAIModel,
    AnthropicModel,
    GeminiModel,
    ScalewayModel,
]

SpeechToTextModel = Literal[
    "whisper-1",
    "gpt-4o-transcribe",
    "scaleway/whisper-large-v3",
]

ImageGenerationModel = Literal["dall-e-3", "gpt-image-1"]

TextToSpeechModel = Literal["tts-1",]

Model = Union[LLModel, SpeechToTextModel, ImageGenerationModel]

COHERE_MODELS: tuple[CohereModel, ...] = get_args(CohereModel)
META_MODELS: tuple[MetaModel, ...] = get_args(MetaModel)
GEMINI_MODELS: tuple[GeminiModel, ...] = get_args(GeminiModel)
MISTRAL_MODELS: tuple[MistralModel, ...] = get_args(MistralModel)
OPENAI_MODELS: tuple[OpenAIModel, ...] = get_args(OpenAIModel)
ANTHROPIC_MODELS: tuple[AnthropicModel, ...] = get_args(AnthropicModel)
SCALEWAY_MODELS: tuple[ScalewayModel, ...] = get_args(ScalewayModel)

# Models that require special JSON format handling
QWEN_SCALEWAY_MODELS: tuple[str, ...] = ("scaleway/qwen3-235b-a22b-instruct-2507",)

LLMS: tuple[LLModel, ...] = get_args(LLModel)
AUDIO: tuple[SpeechToTextModel, ...] = get_args(SpeechToTextModel)
IMAGE: tuple[ImageGenerationModel, ...] = get_args(ImageGenerationModel)
TEXTTOAUDIO: tuple[TextToSpeechModel, ...] = get_args(TextToSpeechModel)

MODELS: tuple[Model, ...] = get_args(Model)
