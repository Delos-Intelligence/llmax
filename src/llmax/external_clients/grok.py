"""Grok client for interacting with CometAPI / xAI models.

Note: we use the OpenAI client to interact with Grok models
because these providers expose an OpenAI-compatible API.
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
    """Get a Grok client for the given deployment.

    Args:
        deployment: The deployment configuration
        http_client: Optional httpx client to prevent OpenAI SDK memory leaks

    Returns:
        The Grok client for the deployment
    """
    match deployment.provider:
        case "grok":
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
    """Get an async Grok client for the given deployment.

    Args:
        deployment: The deployment configuration
        http_client: Optional async httpx client

    Returns:
        The async Grok client
    """
    match deployment.provider:
        case "grok":
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
