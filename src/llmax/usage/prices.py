"""Prices for models and providers."""

from llmax.models import Model, Provider

from .exceptions import PriceNotFoundError

PROMPT_PRICES_PER_1M: dict[Model, float | dict[Provider, float]] = {
    "gpt-4.1": 2.0,
    "gpt-4.1-mini": 0.4,
    "gpt-4.1-nano": 0.1,
    "gpt-4o": {"openai": 2.5, "azure": 2.75},
    "gpt-4o-mini": {"openai": 0.15, "azure": 0.165},
    "ada-v2": 0.1,
    "claude-4.5-haiku": 1.0,
    "claude-4.5-sonnet": 3.0,
    "claude-4.6-sonnet": 3.0,
    "claude-4.5-opus": 5.0,
    "claude-4.6-opus": 5.0,
    "claude-4.7-opus": 5.0,
    "claude-4.8-opus": 5.0,
    "llama-3-70b-instruct": 3.78,  # To be checked
    "llama-4-scout-17b-16e-instruct": 0.275,  # To be checked
    "llama-4-maverick-17b-128e-instruct-fp8": 0.275,  # To be checked
    "mistral-large": 4.0,
    "mistral-large-2411": 2.0,
    "mistral-small": 1.0,
    "text-embedding-3-large": 0.13,
    "text-embedding-3-small": 0.02,
    "bge-multilingual-gemma2": {"scaleway": 0.115},  # €0.10 → $0.115
    "gpt-5": 1.25,
    "gpt-5.1": 1.25,
    "gpt-5.4": 2.5,
    "gpt-5.5": 5.0,
    "gpt-5-chat": 1.25,
    "gpt-5-mini": 0.25,
    "gpt-5-nano": 0.05,
    "llama-3.1-8b-instruct": {"scaleway": 0.23, "openrouter": 0.02},
    "qwen3-235b-a22b-instruct-2507": {"scaleway": 0.8625, "openrouter": 0.071},
    "qwen3.5-397b-a17b": {"scaleway": 0.66, "openrouter": 0.39},
    "gpt-oss-120b": {"scaleway": 0.1725, "openrouter": 0.039},
    "gemma-3-27b-it": {"scaleway": 0.2875, "openrouter": 0.08},
    "gemma-4-26b-a4b-it": {"scaleway": 0.2875, "openrouter": 0.06},
    "voxtral-small-24b-2507": {"scaleway": 0.1725, "openrouter": 0.10},
    "mistral-small-3.2-24b-instruct-2506": {"scaleway": 0.1725, "openrouter": 0.075},
    "llama-3.3-70b-instruct": {"scaleway": 1.035, "openrouter": 0.10},
    "deepseek-r1-distill-llama-70b": {"scaleway": 1.035, "openrouter": 0.70},
    "devstral-2-123b-instruct-2512": {"scaleway": 0.46, "openrouter": 0.40},
    "mistral-large-3": {"scaleway": 0.50, "openrouter": 0.50},
    "mistral-medium-3.5-128b": {"scaleway": 1.725, "openrouter": 1.50},
    "pixtral-12b-2409": {"scaleway": 0.23},  # €0.20 → $0.23
    "holo2-30b-a3b": {"scaleway": 0.345},  # €0.30 → $0.345
    "qwen3-embedding-8b": {"scaleway": 0.115},  # €0.10 → $0.115
    "qwen3.6-35b-a3b": {"scaleway": 0.2875, "openrouter": 0.14},
    "qwen3-coder-30b-a3b-instruct": {"scaleway": 0.23},  # €0.20 → $0.23
    "gemini-3-pro-preview": 2,
    "deepseek-v4-pro": {"openrouter": 0.435},
    "deepseek-v4-flash": {"openrouter": 0.14},
    "glm-4.7": {"openrouter": 0.38},
    "glm-5.1": {"openrouter": 1.05},
    "glm-5.2": {"openrouter": 1.40},
    "llama-4-maverick": {"openrouter": 0.15},
    "qwen3.6-plus": {"openrouter": 0.325},
    "kimi-k2.5": {"openrouter": 0.44},
    "kimi-k2.6": {"openrouter": 0.75},
}

CACHED_PROMPT_PRICES_PER_1M: dict[Model, float | dict[Provider, float]] = {
    "gpt-4.1": 0.5,
    "gpt-4.1-mini": 0.1,
    "gpt-4.1-nano": 0.03,
    "gpt-4o": {"openai": 1.25, "azure": 1.375},
    "gpt-4o-mini": {"openai": 0.075, "azure": 0.083},
    "gpt-5": 0.125,
    "gpt-5.1": 0.125,
    "gpt-5.4": 0.25,
    "gpt-5.5": 0.5,
    "gpt-5-chat": 0.125,
    "gpt-5-mini": 0.025,
    "gpt-5-nano": 0.005,
    "gpt-oss-120b": 0.85,
    "claude-4.5-haiku": 0.1,
    "claude-4.5-sonnet": 0.30,
    "claude-4.6-sonnet": 0.30,
    "claude-4.5-opus": 0.50,
    "claude-4.6-opus": 0.50,
    "claude-4.7-opus": 0.50,
    "claude-4.8-opus": 0.50,
    "deepseek-v4-pro": {"openrouter": 0.10875},  # input * 0.25
    "deepseek-v4-flash": {"openrouter": 0.035},  # input * 0.25
    "glm-5.2": {"openrouter": 0.26},
}

COMPLETION_PRICES_PER_1M: dict[Model, float | dict[Provider, float]] = {
    "gpt-4.1": 8.0,
    "gpt-4.1-mini": 1.6,
    "gpt-4.1-nano": 0.4,
    "gpt-4o": {"openai": 10.0, "azure": 11.0},
    "gpt-4o-mini": {"openai": 0.60, "azure": 0.66},
    "claude-4.5-haiku": 5.0,
    "claude-4.5-sonnet": 15.0,
    "claude-4.6-sonnet": 15.0,
    "claude-4.5-opus": 25.0,
    "claude-4.6-opus": 25.0,
    "claude-4.7-opus": 25.0,
    "claude-4.8-opus": 25.0,
    "llama-3-70b-instruct": 11.34,
    "llama-4-scout-17b-16e-instruct": 1.1,  # To be checked
    "llama-4-maverick-17b-128e-instruct-fp8": 1.1,  # To be checked
    "mistral-large": 12.0,
    "mistral-large-2411": 6.0,
    "mistral-small": 3.0,
    "gpt-5": 10.00,
    "gpt-5.1": 10.00,
    "gpt-5.4": 15.00,
    "gpt-5.5": 30.00,
    "gpt-5-chat": 10.00,
    "gpt-5-mini": 2.00,
    "gpt-5-nano": 0.40,
    "llama-3.1-8b-instruct": {"scaleway": 0.23, "openrouter": 0.03},
    "qwen3.5-397b-a17b": {"scaleway": 3.96, "openrouter": 2.34},
    "qwen3-235b-a22b-instruct-2507": {"scaleway": 2.5875, "openrouter": 0.10},
    "gpt-oss-120b": {"scaleway": 0.69, "openrouter": 0.18},
    "gemma-3-27b-it": {"scaleway": 0.575, "openrouter": 0.16},
    "gemma-4-26b-a4b-it": {"scaleway": 0.575, "openrouter": 0.33},
    "voxtral-small-24b-2507": {"scaleway": 0.4025, "openrouter": 0.30},
    "mistral-small-3.2-24b-instruct-2506": {"scaleway": 0.4025, "openrouter": 0.20},
    "llama-3.3-70b-instruct": {"scaleway": 1.035, "openrouter": 0.32},
    "deepseek-r1-distill-llama-70b": {"scaleway": 1.035, "openrouter": 0.80},
    "devstral-2-123b-instruct-2512": {"scaleway": 2.30, "openrouter": 2.00},
    "mistral-large-3": {"scaleway": 1.50, "openrouter": 1.50},
    "mistral-medium-3.5-128b": {"scaleway": 8.625, "openrouter": 7.50},
    "pixtral-12b-2409": {"scaleway": 0.23},  # €0.20 → $0.23
    "holo2-30b-a3b": {"scaleway": 0.805},  # €0.70 → $0.805
    "qwen3.6-35b-a3b": {"scaleway": 1.725, "openrouter": 1.00},
    "qwen3-coder-30b-a3b-instruct": {"scaleway": 0.92},  # €0.80 → $0.92
    "gemini-3-pro-preview": 12,
    "deepseek-v4-pro": {"openrouter": 0.87},
    "deepseek-v4-flash": {"openrouter": 0.28},
    "glm-4.7": {"openrouter": 1.74},
    "glm-5.1": {"openrouter": 3.50},
    "glm-5.2": {"openrouter": 4.40},
    "llama-4-maverick": {"openrouter": 0.60},
    "qwen3.6-plus": {"openrouter": 1.95},
    "kimi-k2.5": {"openrouter": 2.00},
    "kimi-k2.6": {"openrouter": 3.50},
}

TRANSCRIPTION_PRICES_PER_1M: dict[Model, float | dict[Provider, float]] = {
    "whisper-1": 0.006,
    "gpt-4o-transcribe": 0.006,
    "whisper-large-v3": {"scaleway": 0.00345},  # €0.003/minute → $0.00345/minute
    "eleven_audio_isolation": {"elevenlabs": 0.10},  # $0.10/minute
    "eleven_dubbing": {"elevenlabs": 0.10},  # $0.10/minute of source video
}

GENERATION_PRICE_BASE: dict[Model, float | dict[Provider, float]] = {
    "gpt-image-1": 0.08,
}

IMAGE_GENERATION_PRICES_BY_QUALITY: dict[Model, dict[str, float]] = {
    "gpt-image-2": {
        "low": 0.006,
        "medium": 0.053,
        "high": 0.211,
        "auto": 0.053,
    },
    "gemini-3-pro-image-preview": {
        "low": 0.134,
        "medium": 0.134,
        "high": 0.240,
        "auto": 0.134,
    },
}

GENERATION_PRICES_PER_1M: dict[Model, float | dict[Provider, float]] = {
    "tts-1": 0.000015,
    "eleven_turbo_v2_5": {"elevenlabs": 0.00006},
    "eleven_multilingual_v2": {"elevenlabs": 0.00012},
    "eleven_v3": {"elevenlabs": 0.00012},
}

VIDEO_GENERATION_PRICES_PER_SECOND: dict[Model, dict[str, tuple[float, float]]] = {
    "veo-3.1-generate-preview": {
        "720p": (0.20, 0.40),
        "1080p": (0.20, 0.40),
        "4k": (0.40, 0.60),
    },
    "veo-3.1-fast-generate-preview": {
        "720p": (0.08, 0.10),
        "1080p": (0.10, 0.12),
        "4k": (0.25, 0.30),
    },
    "veo-3.1-lite-generate-preview": {
        "720p": (0.05, 0.05),
        "1080p": (0.08, 0.08),
    },
}


def _fetch_price(
    prices: dict[Model, float | dict[Provider, float]],
    model: Model,
    provider: Provider,
) -> float:
    """Fetch price for a given model and provider from the provided dictionary."""
    if model not in prices:
        raise PriceNotFoundError(model)

    prices_for_model = prices[model]
    if isinstance(prices_for_model, dict):
        if provider not in prices_for_model:
            raise PriceNotFoundError(model, provider)
        return prices_for_model[provider]

    return prices_for_model


def get_prompt_price(model: Model, provider: Provider) -> float:
    """Get the prompt price for a model and provider."""
    return _fetch_price(PROMPT_PRICES_PER_1M, model, provider)


def get_cached_prompt_price(model: Model, provider: Provider) -> float:
    """Get the cached prompt price for a model and provider."""
    try:
        return _fetch_price(CACHED_PROMPT_PRICES_PER_1M, model, provider)
    except PriceNotFoundError:
        return _fetch_price(PROMPT_PRICES_PER_1M, model, provider)


def get_completion_price(model: Model, provider: Provider) -> float:
    """Get the completion price for a model and provider."""
    return _fetch_price(COMPLETION_PRICES_PER_1M, model, provider)


def get_stt_price(model: Model, provider: Provider) -> float:
    """Get the audio price for a model and provider."""
    return _fetch_price(TRANSCRIPTION_PRICES_PER_1M, model, provider)


def get_tti_price(model: Model, provider: Provider) -> float:
    """Get the generation price for a model and provider."""
    return _fetch_price(GENERATION_PRICE_BASE, model, provider)


def get_tti_price_by_quality(model: Model, quality: str) -> float | None:
    """Get per-image price for models with quality-based pricing. Returns None if not applicable."""
    if model not in IMAGE_GENERATION_PRICES_BY_QUALITY:
        return None
    return IMAGE_GENERATION_PRICES_BY_QUALITY[model].get(quality, IMAGE_GENERATION_PRICES_BY_QUALITY[model]["medium"])


def get_tts_price(model: Model, provider: Provider) -> float:
    """Get the generation price for a model and provider."""
    return _fetch_price(GENERATION_PRICES_PER_1M, model, provider)


def get_ttv_price(model: Model, resolution: str, with_audio: bool) -> float:
    """Get the text-to-video price per second for a model, resolution, and audio flag."""
    if model not in VIDEO_GENERATION_PRICES_PER_SECOND:
        raise PriceNotFoundError(model)
    resolutions = VIDEO_GENERATION_PRICES_PER_SECOND[model]
    if resolution not in resolutions:
        raise PriceNotFoundError(model)
    video_price, audio_price = resolutions[resolution]
    return audio_price if with_audio else video_price
