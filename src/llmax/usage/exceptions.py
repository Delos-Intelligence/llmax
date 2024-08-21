"""Prices exceptions."""

from llmax.models.models import Model
from llmax.models.providers import Provider


class PriceNotFoundError(Exception):
    """Raised when the price is not found for a model and provider."""

    def __init__(self, model: Model, provider: Provider | None = None) -> None:
        """Initialize the exception."""
        if provider:
            message = f"Price not found for provider: {provider} and model: {model}."
        else:
            message = f"Price not found for model: {model}."
        super().__init__(message)
