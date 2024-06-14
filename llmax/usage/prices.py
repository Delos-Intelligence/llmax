"""Prices for models and providers."""

from llmax.models import Model, Provider

PROMPT_PRICES_PER_1K: dict[Model, float | dict[Provider, float]] = {
    "gpt-4o": 0.005,
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
}

COMPLETION_PRICES_PER_1K: dict[Model, float | dict[Provider, float]] = {
    "gpt-4o": 0.015,
    "gpt-4-turbo": 0.03,
    "gpt-3.5": 0.0015,
    "mistral-large": 0.012,
    "mistral-small": 0.003,
    "command-r": 0.0015,
    "command-r-plus": 0.015,
    "llama-3-70b-instruct": 0.01134,
}


def get_prompt_price(model: Model, provider: Provider) -> float:
    """Get the prompt price for a model and provider."""
    if model not in PROMPT_PRICES_PER_1K:
        message = f"Unknown model: {model}."
        raise ValueError(message)
    prices_for_model = PROMPT_PRICES_PER_1K[model]
    if isinstance(prices_for_model, dict):
        if provider not in prices_for_model:
            message = f"Unknown provider: {provider} for model: {model}."
            raise ValueError(message)
        return prices_for_model[provider]
    return prices_for_model


def get_completion_price(model: Model, provider: Provider) -> float:
    """Get the completion price for a model and provider."""
    if model not in COMPLETION_PRICES_PER_1K:
        message = f"Unknown model: {model}."
        raise ValueError(message)
    prices_for_model = COMPLETION_PRICES_PER_1K[model]
    if isinstance(prices_for_model, dict):
        if provider not in prices_for_model:
            message = f"Unknown provider: {provider} for model: {model}."
            raise ValueError(message)
        return prices_for_model[provider]
    return prices_for_model
