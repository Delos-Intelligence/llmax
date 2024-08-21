"""OpenAI clients."""

from typing import Any

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from llmax.external_clients.exceptions import ProviderNotFoundError
from llmax.models import Deployment

Client = Any


def get_client(deployment: Deployment) -> Client:
    """Get a client for the given deployment."""
    match deployment.provider:
        case "openai":
            return OpenAI(
                api_key=deployment.api_key,
            )
        case "azure":
            return AzureOpenAI(
                api_key=deployment.api_key,
                api_version=deployment.api_version,
                azure_endpoint=deployment.endpoint,
            )
        case _:
            raise ProviderNotFoundError(deployment)


def get_aclient(deployment: Deployment) -> Client:
    """Get an async client for the given deployment."""
    match deployment.provider:
        case "openai":
            return AsyncOpenAI(
                api_key=deployment.api_key,
            )
        case "azure":
            return AsyncAzureOpenAI(
                api_key=deployment.api_key,
                api_version=deployment.api_version,
                azure_endpoint=deployment.endpoint,
            )
        case _:
            raise ProviderNotFoundError(deployment)
