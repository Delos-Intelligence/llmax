"""Prices for models and providers."""

from llmax.models import Model, Provider

from .exceptions import PriceNotFoundError

PROMPT_PRICES_PER_1M: dict[Model, float | dict[Provider, float]] = {
    "gpt-4.1": 2.0,
    "gpt-4.1-mini": 0.4,
    "gpt-4.1-nano": 0.1,
    "gpt-4o": {"openai": 2.5, "azure": 2.75},
    "gpt-4o-mini": {"openai": 0.15, "azure": 0.165},
    "o3-mini": {"openai": 1.10, "azure": 1.21},
    "o3-mini-high": {"openai": 1.10, "azure": 1.21},
    "ada-v2": 0.1,
    "gpt-3.5": 0.5,
    "gpt-4-turbo": 10.0,
    "claude-3-haiku": 0.25,
    "claude-4.5-haiku": 1.0,
    "claude-3.5-sonnet": 3.0,
    "claude-3.7-sonnet": 3.0,
    "claude-4-sonnet": 3.0,
    "claude-4.5-sonnet": 3.0,
    "claude-4.6-sonnet": 3.0,
    "claude-4.5-opus": 5.0,
    "claude-4.6-opus": 5.0,
    "claude-4.7-opus": 5.0,
    "command-r": 0.5,
    "command-r-plus": 3.0,
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
    "llama-3.1-8b-instruct": {"scaleway": 0.23},  # €0.20 → $0.23
    "qwen3-235b-a22b-instruct-2507": {"scaleway": 0.8625},  # €0.75 → $0.8625
    "qwen3.5-397b-a17b": {"scaleway": 0.66},
    "gpt-oss-120b": {"scaleway": 0.1725},  # €0.15 → $0.1725
    "gemma-3-27b-it": {"scaleway": 0.2875},  # €0.25 → $0.2875
    "voxtral-small-24b-2507": {"scaleway": 0.1725},  # €0.15 → $0.1725
    "mistral-small-3.2-24b-instruct-2506": {"scaleway": 0.1725},  # €0.15 → $0.1725
    "llama-3.3-70b-instruct": {"scaleway": 1.035},  # €0.90 → $1.035
    "deepseek-r1-distill-llama-70b": {"scaleway": 1.035},  # €0.90 → $1.035
    "devstral-2-123b-instruct-2512": {"scaleway": 0.46},  # €0.40 → $0.46
    "grok-4-1-fast": 0.2,
    "gemini-3-pro-preview": 2,
    "gemini-3.1-flash-lite-preview": 0.25,
    "deepseek-v4-pro": {"openrouter": 0.435},
    "deepseek-v4-flash": {"openrouter": 0.14},
    "glm-4.6": {"openrouter": 0.39},
    "glm-4.7": {"openrouter": 0.38},
    "llama-4-maverick": {"openrouter": 0.15},
    "qwen3.6-plus": {"openrouter": 0.325},
}

CACHED_PROMPT_PRICES_PER_1M: dict[Model, float | dict[Provider, float]] = {
    "gpt-4.1": 0.5,
    "gpt-4.1-mini": 0.1,
    "gpt-4.1-nano": 0.03,
    "gpt-4o": {"openai": 1.25, "azure": 1.375},
    "gpt-4o-mini": {"openai": 0.075, "azure": 0.083},
    "o3-mini": {"openai": 0.55, "azure": 0.605},
    "o3-mini-high": {"openai": 0.55, "azure": 0.605},
    "gpt-5": 0.125,
    "gpt-5.1": 0.125,
    "gpt-5.4": 0.25,
    "gpt-5.5": 0.5,
    "gpt-5-chat": 0.125,
    "gpt-5-mini": 0.025,
    "gpt-5-nano": 0.005,
    "gpt-oss-120b": 0.85,
    "claude-3-haiku": 0.03,
    "claude-4.5-haiku": 0.1,
    "claude-3.5-sonnet": 0.30,
    "claude-3.7-sonnet": 0.30,
    "claude-4-sonnet": 0.30,
    "claude-4.5-sonnet": 0.30,
    "claude-4.6-sonnet": 0.30,
    "claude-4.5-opus": 0.50,
    "claude-4.6-opus": 0.50,
    "claude-4.7-opus": 0.50,
    "deepseek-v4-pro": {"openrouter": 0.10875},  # input * 0.25
    "deepseek-v4-flash": {"openrouter": 0.035},  # input * 0.25
}

COMPLETION_PRICES_PER_1M: dict[Model, float | dict[Provider, float]] = {
    "gpt-4.1": 8.0,
    "gpt-4.1-mini": 1.6,
    "gpt-4.1-nano": 0.4,
    "gpt-4o": {"openai": 10.0, "azure": 11.0},
    "gpt-4o-mini": {"openai": 0.60, "azure": 0.66},
    "o3-mini": {"openai": 4.40, "azure": 4.84},
    "o3-mini-high": {"openai": 4.40, "azure": 4.84},
    "gpt-3.5": 1.5,
    "gpt-4-turbo": 30.0,
    "claude-3-haiku": 1.25,
    "claude-4.5-haiku": 5.0,
    "claude-3.5-sonnet": 15.0,
    "claude-3.7-sonnet": 15.0,
    "claude-4-sonnet": 15.0,
    "claude-4.5-sonnet": 15.0,
    "claude-4.6-sonnet": 15.0,
    "claude-4.5-opus": 25.0,
    "claude-4.6-opus": 25.0,
    "claude-4.7-opus": 25.0,
    "command-r": 1.5,
    "command-r-plus": 15.0,
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
    "llama-3.1-8b-instruct": {"scaleway": 0.23},  # €0.20 → $0.23
    "qwen3.5-397b-a17b": {"scaleway": 3.96},
    "qwen3-235b-a22b-instruct-2507": {"scaleway": 2.5875},  # €2.25 → $2.5875
    "gpt-oss-120b": {"scaleway": 0.69},  # €0.60 → $0.69
    "gemma-3-27b-it": {"scaleway": 0.575},  # €0.50 → $0.575
    "voxtral-small-24b-2507": {"scaleway": 0.4025},  # €0.35 → $0.4025
    "mistral-small-3.2-24b-instruct-2506": {"scaleway": 0.4025},  # €0.35 → $0.4025
    "llama-3.3-70b-instruct": {"scaleway": 1.035},  # €0.90 → $1.035
    "deepseek-r1-distill-llama-70b": {"scaleway": 1.035},  # €0.90 → $1.035
    "devstral-2-123b-instruct-2512": {"scaleway": 2.30},  # €2.00 → $2.30
    "grok-4-1-fast": 0.5,
    "gemini-3-pro-preview": 12,
    "gemini-3.1-flash-lite-preview": 1.5,
    "deepseek-v4-pro": {"openrouter": 0.87},
    "deepseek-v4-flash": {"openrouter": 0.28},
    "glm-4.6": {"openrouter": 1.90},
    "glm-4.7": {"openrouter": 1.74},
    "llama-4-maverick": {"openrouter": 0.60},
    "qwen3.6-plus": {"openrouter": 1.95},
}

TRANSCRIPTION_PRICES_PER_1M: dict[Model, float | dict[Provider, float]] = {
    "whisper-1": 0.006,
    "gpt-4o-transcribe": 0.006,
    "whisper-large-v3": {"scaleway": 0.00345},  # €0.003/minute → $0.00345/minute
}

GENERATION_PRICE_BASE: dict[Model, float | dict[Provider, float]] = {
    "dall-e-3": 0.04,
    "gpt-image-1": 0.08,
    "gpt-image-2": 0.08,
    "gemini-3-pro-image-preview": 0.08,
}

GENERATION_PRICES_PER_1M: dict[Model, float | dict[Provider, float]] = {
    "tts-1": 0.000015,
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
