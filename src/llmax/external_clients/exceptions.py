"""External clients exceptions."""

from llmax.models.deployment import Deployment


class ProviderNotFoundError(Exception):
    """Raised when an invalid provider is provided."""

    def __init__(self, deployment: Deployment) -> None:
        """Initialize the exception."""
        message = f"Provider '{deployment.provider}' not avilable for model '{deployment.model}'. Please provide a valid provider."
        super().__init__(message)


class ClientNotFoundError(Exception):
    """Raised when no client is available for the given deployment."""

    def __init__(self, deployment: Deployment) -> None:
        """Initialize the exception."""
        message = f"No client available for '{deployment.model}' model. Please provide a valid model."
        super().__init__(message)
