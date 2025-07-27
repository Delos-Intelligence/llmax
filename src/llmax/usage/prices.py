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
    "claude-3.5-sonnet": 3.0,
    "claude-3.7-sonnet": 3.0,
    "command-r": 0.5,
    "command-r-plus": 3.0,
    "google/gemini-1.5-flash-002": 0.075,
    "google/gemini-1.5-pro-002": 1.25,
    "llama-3-70b-instruct": 3.78,
    "mistral-large": 4.0,
    "mistral-large-2411": 2.0,
    "mistral-small": 1.0,
    "text-embedding-3-large": 0.13,
    "text-embedding-3-small": 0.02,
}

CACHED_PROMPT_PRICES_PER_1M: dict[Model, float | dict[Provider, float]] = {
    "gpt-4.1": 0.5,
    "gpt-4.1-mini": 0.1,
    "gpt-4.1-nano": 0.03,
    "gpt-4o": {"openai": 1.25, "azure": 1.375},
    "gpt-4o-mini": {"openai": 0.075, "azure": 0.083},
    "o3-mini": {"openai": 0.55, "azure": 0.605},
    "o3-mini-high": {"openai": 0.55, "azure": 0.605},
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
    "claude-3.5-sonnet": 15.0,
    "claude-3.7-sonnet": 15.0,
    "command-r": 1.5,
    "command-r-plus": 15.0,
    "google/gemini-1.5-flash-002": 0.3,
    "google/gemini-1.5-pro-002": 5.0,
    "llama-3-70b-instruct": 11.34,
    "mistral-large": 12.0,
    "mistral-large-2411": 6.0,
    "mistral-small": 3.0,
}

TRANSCRIPTION_PRICES_PER_1M: dict[Model, float | dict[Provider, float]] = {
    "whisper-1": 0.006,
    "gpt-4o-transcribe": 0.006,
}

GENERATION_PRICE_BASE: dict[Model, float | dict[Provider, float]] = {
    "dall-e-3": 0.04,
}

GENERATION_PRICES_PER_1M: dict[Model, float | dict[Provider, float]] = {
    "tts-1": 0.000015,
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
