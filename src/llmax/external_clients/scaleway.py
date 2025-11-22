"""Scaleway client for interacting with the Scaleway AI API.

Note: Scaleway uses an OpenAI-compatible API, so we use the OpenAI client
to interact with Scaleway models.

According to the Scaleway OpenAPI specification, the base URL should be:
https://api.scaleway.ai/v1/{project_id}
"""

from typing import Any

import httpx
from openai import AsyncOpenAI, OpenAI

from llmax.external_clients.exceptions import ProviderNotFoundError
from llmax.models import Deployment

Client = Any

# Base URL for Scaleway Generative APIs
SCALEWAY_BASE_URL = "https://api.scaleway.ai"


def _build_scaleway_url(deployment: Deployment) -> str:
    """Build the Scaleway API URL with project_id.

    Args:
        deployment: The deployment configuration

    Returns:
        The complete base URL for Scaleway API calls
    """
    # If endpoint is explicitly provided, use it (for backward compatibility)
    if deployment.endpoint:
        return deployment.endpoint

    # Otherwise, construct the URL from base URL and project_id
    if not deployment.project_id:
        url_template = f"{SCALEWAY_BASE_URL}/v1/{{project_id}}"
        message = (
            "Scaleway deployments require either an 'endpoint' or a 'project_id'. "
            "If using project_id, the URL will be constructed as: "
            f"{url_template}",
        )
        raise ValueError(message)

    return f"{SCALEWAY_BASE_URL}/v1/{deployment.project_id}"


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
            base_url = _build_scaleway_url(deployment)
            if http_client:
                return OpenAI(
                    api_key=deployment.api_key,
                    base_url=base_url,
                    http_client=http_client,
                )
            return OpenAI(
                api_key=deployment.api_key,
                base_url=base_url,
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
            base_url = _build_scaleway_url(deployment)
            if http_client:
                return AsyncOpenAI(
                    api_key=deployment.api_key,
                    base_url=base_url,
                    http_client=http_client,
                )
            return AsyncOpenAI(
                api_key=deployment.api_key,
                base_url=base_url,
            )
        case _:
            raise ProviderNotFoundError(deployment)
