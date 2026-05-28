"""Supported models."""

from typing import Literal, get_args

MetaModel = Literal[
    "llama-3-70b-instruct",
    "llama-4-scout-17b-16e-instruct",
    "llama-4-maverick-17b-128e-instruct-fp8",
]

GeminiModel = Literal[
    "gemini-3-pro-preview",
    "gemini-3-pro-image-preview",
    "veo-3.1-generate-preview",
    "veo-3.1-fast-generate-preview",
    "veo-3.1-lite-generate-preview",
]

MistralModel = Literal[
    "mistral-large",
    "mistral-small",
    "mistral-large-2411",
]

OpenAIModel = Literal[
    "ada-v2",
    "gpt-image-1",
    "gpt-image-2",
    "gpt-4.1",
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-transcribe",
    "text-embedding-3-large",
    "text-embedding-3-small",
    "tts-1",
    "whisper-1",
    "gpt-5",
    "gpt-5.1",
    "gpt-5.4",
    "gpt-5.5",
    "gpt-5-chat",
    "gpt-5-mini",
    "gpt-5-nano",
]

AnthropicModel = Literal[
    "claude-4.5-haiku",
    "claude-4.5-sonnet",
    "claude-4.5-opus",
    "claude-4.6-opus",
    "claude-4.6-sonnet",
    "claude-4.7-opus",
]

OpenRouterModel = Literal[
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "glm-4.7",
    "glm-5.1",
    "llama-4-maverick",
    "qwen3.6-plus",
    "kimi-k2.5",
    "kimi-k2.6",
]

ScalewayModel = Literal[
    "qwen3-235b-a22b-instruct-2507",
    "qwen3.5-397b-a17b",
    "gpt-oss-120b",
    "gemma-3-27b-it",
    "gemma-4-26b-a4b-it",
    "whisper-large-v3",
    "voxtral-small-24b-2507",
    "mistral-small-3.2-24b-instruct-2506",
    "llama-3.3-70b-instruct",
    "deepseek-r1-distill-llama-70b",
    "llama-3.1-8b-instruct",
    "bge-multilingual-gemma2",
    "devstral-2-123b-instruct-2512",
]

LLModel = (
    MetaModel
    | MistralModel
    | OpenAIModel
    | AnthropicModel
    | GeminiModel
    | ScalewayModel
    | OpenRouterModel
)

SpeechToTextModel = Literal[
    "whisper-1",
    "gpt-4o-transcribe",
    "whisper-large-v3",
]

ImageGenerationModel = Literal[
    "gpt-image-1",
    "gpt-image-2",
    "gemini-3-pro-image-preview",
]

VideoGenerationModel = Literal[
    "veo-3.1-generate-preview",
    "veo-3.1-fast-generate-preview",
    "veo-3.1-lite-generate-preview",
]

TextToAudioModel = Literal[
    "tts-1",
    "eleven_turbo_v2_5",
    "eleven_multilingual_v2",
    "eleven_v3",
]

AudioIsolationModel = Literal["eleven_audio_isolation"]

Model = (
    LLModel
    | SpeechToTextModel
    | ImageGenerationModel
    | VideoGenerationModel
    | TextToAudioModel
    | AudioIsolationModel
)

META_MODELS: tuple[MetaModel, ...] = get_args(MetaModel)
GEMINI_MODELS: tuple[GeminiModel, ...] = get_args(GeminiModel)
MISTRAL_MODELS: tuple[MistralModel, ...] = get_args(MistralModel)
OPENAI_MODELS: tuple[OpenAIModel, ...] = get_args(OpenAIModel)
ANTHROPIC_MODELS: tuple[AnthropicModel, ...] = get_args(AnthropicModel)
SCALEWAY_MODELS: tuple[ScalewayModel, ...] = get_args(ScalewayModel)
OPENROUTER_MODELS: tuple[OpenRouterModel, ...] = get_args(OpenRouterModel)
ELEVEN_LABS_MODELS: tuple[TextToAudioModel | AudioIsolationModel, ...] = (
    *get_args(TextToAudioModel),
    *get_args(AudioIsolationModel),
)

LLMS: tuple[LLModel, ...] = get_args(LLModel)
AUDIO: tuple[SpeechToTextModel, ...] = get_args(SpeechToTextModel)
IMAGE: tuple[ImageGenerationModel, ...] = get_args(ImageGenerationModel)
VIDEO: tuple[VideoGenerationModel, ...] = get_args(VideoGenerationModel)
TTS: tuple[TextToAudioModel, ...] = get_args(TextToAudioModel)
AUDIO_ISOLATION: tuple[AudioIsolationModel, ...] = get_args(AudioIsolationModel)

MODELS: tuple[Model, ...] = get_args(Model)

SpeechModelAllowVerboseJson: set[SpeechToTextModel] = {"whisper-1"}
