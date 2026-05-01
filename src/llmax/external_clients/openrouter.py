"""OpenRouter client for interacting with OpenRouter-hosted models.

Note: we use the OpenAI client to interact with OpenRouter models
because OpenRouter exposes an OpenAI-compatible API.

The base URL is: https://openrouter.ai/api/v1
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
    """Get an OpenRouter client for the given deployment.

    Args:
        deployment: The deployment configuration
        http_client: Optional httpx client to prevent OpenAI SDK memory leaks

    Returns:
        The OpenRouter client for the deployment
    """
    match deployment.provider:
        case "openrouter":
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
    """Get an async OpenRouter client for the given deployment.

    Args:
        deployment: The deployment configuration
        http_client: Optional async httpx client

    Returns:
        The async OpenRouter client
    """
    match deployment.provider:
        case "openrouter":
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
