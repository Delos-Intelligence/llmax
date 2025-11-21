"""Gemini client for interacting with the Cohere API.

Note: we use the OpenAI client to interact with the Cohere models.
"""

from typing import Any

import httpx
from google.auth.transport.requests import Request
from openai import AsyncOpenAI, OpenAI

from llmax.external_clients.exceptions import ProviderNotFoundError
from llmax.models import Deployment

Client = Any


def get_client(
    deployment: Deployment,
    http_client: httpx.Client | None = None,
) -> Client:
    """Get a Gemini client for the given deployment.

    Args:
        deployment: The deployment configuration
        http_client: Optional httpx client to prevent OpenAI SDK memory leaks

    Returns:
        The Gemini client for the deployment
    """
    match deployment.provider:
        case "openai":
            if deployment.creds:
                auth_req = Request()
                deployment.creds.refresh(auth_req)
                if http_client:
                    return OpenAI(
                        api_key=deployment.creds.token,
                        base_url=deployment.endpoint,
                        http_client=http_client,
                    )
                return OpenAI(
                    api_key=deployment.creds.token,
                    base_url=deployment.endpoint,
                )
            raise ProviderNotFoundError(deployment)
        case _:
            raise ProviderNotFoundError(deployment)


def get_aclient(
    deployment: Deployment,
    http_client: httpx.AsyncClient | None = None,
) -> Client:
    """Get an async Gemini client for the given deployment.

    Args:
        deployment: The deployment configuration
        http_client: Optional async httpx client to prevent OpenAI SDK memory leaks

    Returns:
        The async Gemini client for the deployment
    """
    match deployment.provider:
        case "openai":
            if deployment.creds:
                auth_req = Request()
                deployment.creds.refresh(auth_req)
                if http_client:
                    return AsyncOpenAI(
                        api_key=deployment.creds.token,
                        base_url=deployment.endpoint,
                        http_client=http_client,
                    )
                return AsyncOpenAI(
                    api_key=deployment.creds.token,
                    base_url=deployment.endpoint,
                )
            raise ProviderNotFoundError(deployment)
        case _:
            raise ProviderNotFoundError(deployment)
