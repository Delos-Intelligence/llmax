from typing import Any

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from llmax import models
from llmax.models import Deployment

Client = Any


def get_client(deployment: Deployment) -> Client:
    """Get a client for the given deployment."""
    if deployment.model not in models.OPENAI_MODELS:
        message = f"Unknown model: {deployment.model}. Please provide a valid OpenAI model name."
        raise ValueError(message)

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
            message = f"Invalid provider for OpenAI model: {deployment.provider}. Please provide a valid provider."
            raise ValueError(message)


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
            message = f"Invalid provider for OpenAI model: {deployment.provider}. Please provide a valid provider."
            raise ValueError(message)
