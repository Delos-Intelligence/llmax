"""OpenAI clients."""

from typing import Any

import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from llmax.external_clients.exceptions import ProviderNotFoundError
from llmax.models import Deployment

Client = Any


def get_client(
    deployment: Deployment,
    http_client: httpx.Client | None = None,
) -> Client:
    """Get a client for the given deployment.

    Args:
        deployment: The deployment configuration
        http_client: Optional httpx client to use. If not provided, OpenAI SDK will create its own
                     (which has a known memory leak bug).

    Returns:
        The OpenAI client for the deployment
    """
    match deployment.provider:
        case "openai":
            if http_client:
                return OpenAI(api_key=deployment.api_key, http_client=http_client)
            return OpenAI(api_key=deployment.api_key)
        case "azure":
            if http_client:
                return AzureOpenAI(
                    api_key=deployment.api_key,
                    api_version=deployment.api_version,
                    azure_endpoint=deployment.endpoint,
                    http_client=http_client,
                )
            return AzureOpenAI(
                api_key=deployment.api_key,
                api_version=deployment.api_version,
                azure_endpoint=deployment.endpoint,
            )
        case _:
            raise ProviderNotFoundError(deployment)


def get_aclient(
    deployment: Deployment,
    http_client: httpx.AsyncClient | None = None,
) -> Client:
    """Get an async client for the given deployment.

    Args:
        deployment: The deployment configuration
        http_client: Optional async httpx client to use. If not provided, OpenAI SDK will create its own
                     (which has a known memory leak bug).

    Returns:
        The async OpenAI client for the deployment
    """
    match deployment.provider:
        case "openai":
            if http_client:
                return AsyncOpenAI(api_key=deployment.api_key, http_client=http_client)
            return AsyncOpenAI(api_key=deployment.api_key)
        case "azure":
            if http_client:
                return AsyncAzureOpenAI(
                    api_key=deployment.api_key,
                    api_version=deployment.api_version,
                    azure_endpoint=deployment.endpoint,
                    http_client=http_client,
                )
            return AsyncAzureOpenAI(
                api_key=deployment.api_key,
                api_version=deployment.api_version,
                azure_endpoint=deployment.endpoint,
            )
        case _:
            raise ProviderNotFoundError(deployment)
