"""Module to represent a deployment of a model."""

from dataclasses import dataclass

from google.oauth2.service_account import Credentials

from llmax.models.providers import Provider

from .models import OPENAI_MODELS, Model


@dataclass
class Deployment:
    """Dataclass to represent a deployment of a model."""

    model: Model
    api_key: str
    provider: Provider
    deployment_name: str = ""
    endpoint: str = ""
    api_version: str = "2023-05-15"
    project_id: str = ""
    region: str = ""
    creds: Credentials | None = None

    def validate(self) -> None:
        """Validate the deployment."""
        if self.provider == "azure" and not self.endpoint:
            message = "Please provide an endpoint for Azure deployments."
            raise ValueError(message)
        if (
            self.provider == "azure"
            and self.model in OPENAI_MODELS
            and not self.deployment_name
        ):
            message = "Please provide a deployment name for Azure OpenAI deployments."
            raise ValueError(message)
