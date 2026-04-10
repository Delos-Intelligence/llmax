"""Scaleway client for interacting with Scaleway's Generative APIs.

Note: we use the OpenAI client as Scaleway exposes an OpenAI-compatible API.
"""

from typing import Any

from openai import AsyncOpenAI, OpenAI

from llmax.external_clients.exceptions import ProviderNotFoundError
from llmax.models import Deployment

Client = Any


def get_client(deployment: Deployment) -> Client:
    """Get a Scaleway client for the given deployment."""
    match deployment.provider:
        case "scaleway":
            return OpenAI(
                api_key=deployment.api_key,
                base_url=deployment.endpoint,
            )
        case _:
            raise ProviderNotFoundError(deployment)


def get_aclient(deployment: Deployment) -> Client:
    """Get an async Scaleway client for the given deployment."""
    match deployment.provider:
        case "scaleway":
            return AsyncOpenAI(
                api_key=deployment.api_key,
                base_url=deployment.endpoint,
            )
        case _:
            raise ProviderNotFoundError(deployment)
