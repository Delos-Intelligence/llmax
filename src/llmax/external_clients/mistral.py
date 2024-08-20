"""Mistral client for interacting with the Mistral API.

Note: we use the OpenAI client to interact with the Mistral models.

Only azure deployments are supported for now.
"""

from typing import Any

from openai import AsyncOpenAI, OpenAI

from llmax.external_clients.exceptions import ProviderNotFoundError
from llmax.models import Deployment

Client = Any


def get_client(deployment: Deployment) -> Client:
    """Get a Mistral client for the given deployment."""
    match deployment.provider:
        case "azure":
            return OpenAI(
                api_key=deployment.api_key,
                base_url=deployment.endpoint,
            )
        case _:
            raise ProviderNotFoundError(deployment)


def get_aclient(deployment: Deployment) -> Client:
    """Get an async Mistral client for the given deployment."""
    match deployment.provider:
        case "azure":
            return AsyncOpenAI(
                api_key=deployment.api_key,
                base_url=deployment.endpoint,
            )
        case _:
            raise ProviderNotFoundError(deployment)
