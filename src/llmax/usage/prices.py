"""Prices for models and providers."""

from llmax.models import Model, Provider

from .exceptions import PriceNotFoundError

PROMPT_PRICES_PER_1K: dict[Model, float | dict[Provider, float]] = {
    "gpt-4o-mini": 0.000165,
    "gpt-4o": 0.00500,
    "gpt-4-turbo": 0.01,
    "gpt-3.5": 0.0005,
    "text-embedding-3-small": 0.00002,
    "text-embedding-3-large": 0.00013,
    "ada-v2": 0.00010,
    "mistral-large": 0.004,
    "mistral-small": 0.001,
    "command-r": 0.0005,
    "command-r-plus": 0.003,
    "llama-3-70b-instruct": 0.00378,
    "claude-3.5-sonnet": 0.003,
    "claude-3-haiku": 0.00025,
    "google/gemini-1.5-flash-002": 0.000075,
    "google/gemini-1.5-pro-002": 0.00125,
}

COMPLETION_PRICES_PER_1K: dict[Model, float | dict[Provider, float]] = {
    "gpt-4o-mini": 0.00066,
    "gpt-4o": 0.015,
    "gpt-4-turbo": 0.03,
    "gpt-3.5": 0.0015,
    "mistral-large": 0.012,
    "mistral-small": 0.003,
    "command-r": 0.0015,
    "command-r-plus": 0.015,
    "llama-3-70b-instruct": 0.01134,
    "claude-3.5-sonnet": 0.015,
    "claude-3-haiku": 0.00125,
    "google/gemini-1.5-flash-002": 0.0003,
    "google/gemini-1.5-pro-002": 0.005,
}

TRANSCRIPTION_PRICES_PER_1M: dict[Model, float | dict[Provider, float]] = {
    "whisper-1": 0.006,
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
    return _fetch_price(PROMPT_PRICES_PER_1K, model, provider)


def get_completion_price(model: Model, provider: Provider) -> float:
    """Get the completion price for a model and provider."""
    return _fetch_price(COMPLETION_PRICES_PER_1K, model, provider)


def get_stt_price(model: Model, provider: Provider) -> float:
    """Get the audio price for a model and provider."""
    return _fetch_price(TRANSCRIPTION_PRICES_PER_1M, model, provider)


def get_tti_price(model: Model, provider: Provider) -> float:
    """Get the generation price for a model and provider."""
    return _fetch_price(GENERATION_PRICE_BASE, model, provider)


def get_tts_price(model: Model, provider: Provider) -> float:
    """Get the generation price for a model and provider."""
    return _fetch_price(GENERATION_PRICES_PER_1M, model, provider)
