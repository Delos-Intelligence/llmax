"""Module to represent a deployment of a model."""

from dataclasses import dataclass

from llmax.models.providers import Provider

from .models import Model


@dataclass
class Deployment:
    """Dataclass to represent a deployment of a model."""

    model: Model
    api_key: str
    provider: Provider
    deployment_name: str = ""
    endpoint: str = ""
    api_version: str = "2023-05-15"

    def validate(self) -> None:
        """Validate the deployment."""
        if self.provider == "azure" and not self.endpoint:
            message = "Please provide an endpoint for Azure deployments."
            raise ValueError(message)
        if self.provider == "azure" and not self.deployment_name:
            message = "Please provide a deployment name for Azure deployments."
            raise ValueError(message)
