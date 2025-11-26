"""Scaleway client for interacting with the Scaleway AI API.

Note: Scaleway uses an OpenAI-compatible API, so we use the OpenAI client
to interact with Scaleway models.

The base URL is: https://api.scaleway.ai/v1
"""

from typing import Any

import httpx
from openai import AsyncOpenAI, OpenAI

from llmax.external_clients.exceptions import ProviderNotFoundError
from llmax.models import Deployment

Client = Any


def get_client(
    deployment: Deployment,
    http_client: httpx.Client | None = None,
) -> Client:
    """Get a Scaleway client for the given deployment.

    Args:
        deployment: The deployment configuration
        http_client: Optional httpx client to prevent OpenAI SDK memory leaks

    Returns:
        The Scaleway client for the deployment
    """
    match deployment.provider:
        case "scaleway":
            if http_client:
                return OpenAI(
                    api_key=deployment.api_key,
                    base_url=deployment.endpoint,
                    http_client=http_client,
                )
            return OpenAI(
                api_key=deployment.api_key,
                base_url=deployment.endpoint,
            )
        case _:
            raise ProviderNotFoundError(deployment)


def get_aclient(
    deployment: Deployment,
    http_client: httpx.AsyncClient | None = None,
) -> Client:
    """Get an async Scaleway client for the given deployment.

    Args:
        deployment: The deployment configuration
        http_client: Optional async httpx client to prevent OpenAI SDK memory leaks

    Returns:
        The async Scaleway client for the deployment
    """
    match deployment.provider:
        case "scaleway":
            if http_client:
                return AsyncOpenAI(
                    api_key=deployment.api_key,
                    base_url=deployment.endpoint,
                    http_client=http_client,
                )
            return AsyncOpenAI(
                api_key=deployment.api_key,
                base_url=deployment.endpoint,
            )
        case _:
            raise ProviderNotFoundError(deployment)
