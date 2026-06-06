"""This module provides functions to get a client for a given deployment."""

from typing import Any

import httpx

from llmax.external_clients.exceptions import ClientNotFoundError
from llmax.models import Deployment
from llmax.models.models import (
    ANTHROPIC_MODELS,
    ELEVEN_LABS_MODELS,
    GEMINI_MODELS,
    META_MODELS,
    MISTRAL_MODELS,
    OPENAI_MODELS,
    OPENROUTER_MODELS,
    SCALEWAY_MODELS,
)

from . import (
    anthropic,
    eleven_labs,
    gemini,
    meta,
    mistral,
    openai,
    openrouter,
    scaleway,
)

Client = Any


def get_client(  # noqa: PLR0911
    deployment: Deployment,
    http_client: httpx.Client | None = None,
) -> Client:
    """Get a client for the given deployment.

    Args:
        deployment: The deployment configuration
        http_client: Optional httpx client for OpenAI/Azure providers

    Returns:
        The client for the deployment
    """
    deployment.validate()
    # OpenRouter exposes an OpenAI-compatible passthrough, so it can serve many
    # models that also have a native provider (e.g. Scaleway/Meta/Mistral). Route
    # by provider first so a model can be deployed on OpenRouter as a fallback.
    if deployment.provider == "openrouter":
        return openrouter.get_client(deployment, http_client=http_client)
    match deployment.model:
        case model if model in OPENAI_MODELS:
            return openai.get_client(deployment, http_client=http_client)
        case model if model in MISTRAL_MODELS:
            return mistral.get_client(deployment, http_client=http_client)
        case model if model in META_MODELS:
            return meta.get_client(deployment, http_client=http_client)
        case model if model in ANTHROPIC_MODELS:
            return anthropic.get_client(deployment, http_client=http_client)
        case model if model in GEMINI_MODELS:
            return gemini.get_client(deployment, http_client=http_client)
        case model if model in SCALEWAY_MODELS:
            return scaleway.get_client(deployment, http_client=http_client)
        case model if model in OPENROUTER_MODELS:
            return openrouter.get_client(deployment, http_client=http_client)
        case _:
            raise ClientNotFoundError(deployment)


def get_aclient(  # noqa: PLR0911
    deployment: Deployment,
    http_client: httpx.AsyncClient | None = None,
) -> Client:
    """Get an async client for the given deployment.

    Args:
        deployment: The deployment configuration
        http_client: Optional async httpx client for OpenAI/Azure providers

    Returns:
        The async client for the deployment
    """
    deployment.validate()
    # OpenRouter exposes an OpenAI-compatible passthrough, so it can serve many
    # models that also have a native provider (e.g. Scaleway/Meta/Mistral). Route
    # by provider first so a model can be deployed on OpenRouter as a fallback.
    if deployment.provider == "openrouter":
        return openrouter.get_aclient(deployment, http_client=http_client)
    match deployment.model:
        case model if model in OPENAI_MODELS:
            return openai.get_aclient(deployment, http_client=http_client)
        case model if model in MISTRAL_MODELS:
            return mistral.get_aclient(deployment, http_client=http_client)
        case model if model in META_MODELS:
            return meta.get_aclient(deployment, http_client=http_client)
        case model if model in ANTHROPIC_MODELS:
            return anthropic.get_aclient(deployment, http_client=http_client)
        case model if model in GEMINI_MODELS:
            return gemini.get_aclient(deployment, http_client=http_client)
        case model if model in SCALEWAY_MODELS:
            return scaleway.get_aclient(deployment, http_client=http_client)
        case model if model in OPENROUTER_MODELS:
            return openrouter.get_aclient(deployment, http_client=http_client)
        case model if model in ELEVEN_LABS_MODELS:
            return eleven_labs.get_aclient(deployment, http_client=http_client)
        case _:
            raise ClientNotFoundError(deployment)
