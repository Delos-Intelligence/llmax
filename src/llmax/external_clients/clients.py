"""This module provides functions to get a client for a given deployment."""

from typing import Any

from llmax.external_clients.exceptions import ClientNotFoundError
from llmax.models import Deployment
from llmax.models.models import (
    ANTHROPIC_MODELS,
    COHERE_MODELS,
    GEMINI_MODELS,
    META_MODELS,
    MISTRAL_MODELS,
    OPENAI_MODELS,
)

from . import anthropic, cohere, gemini, meta, mistral, openai

Client = Any


def get_client(deployment: Deployment) -> Client:
    """Get a client for the given deployment."""
    deployment.validate()
    match deployment.model:
        case model if model in OPENAI_MODELS:
            return openai.get_client(deployment)
        case model if model in MISTRAL_MODELS:
            return mistral.get_client(deployment)
        case model if model in COHERE_MODELS:
            return cohere.get_client(deployment)
        case model if model in META_MODELS:
            return meta.get_client(deployment)
        case model if model in ANTHROPIC_MODELS:
            return anthropic.get_client(deployment)
        case model if model in GEMINI_MODELS:
            return gemini.get_client(deployment)
        case _:
            raise ClientNotFoundError(deployment)


def get_aclient(deployment: Deployment) -> Client:
    """Get an async client for the given deployment."""
    deployment.validate()
    match deployment.model:
        case model if model in OPENAI_MODELS:
            return openai.get_aclient(deployment)
        case model if model in MISTRAL_MODELS:
            return mistral.get_aclient(deployment)
        case model if model in COHERE_MODELS:
            return cohere.get_aclient(deployment)
        case model if model in META_MODELS:
            return meta.get_aclient(deployment)
        case _:
            raise ClientNotFoundError(deployment)
