from typing import Any

from llmax.models import Deployment
from llmax.models.models import MISTRAL_MODELS, OPENAI_MODELS

from . import mistral, openai

Client = Any


def get_client(deployment: Deployment) -> Client:
    """Get a client for the given deployment."""
    if deployment.model in OPENAI_MODELS:
        return openai.get_client(deployment)
    if deployment.model in MISTRAL_MODELS:
        return mistral.get_client(deployment)
    message = f"Unknown model: {deployment.model}. Please provide a valid model name."
    raise ValueError(message)


def get_aclient(deployment: Deployment) -> Client:
    """Get an async client for the given deployment."""
    if deployment.model in OPENAI_MODELS:
        return openai.get_aclient(deployment)
    if deployment.model in MISTRAL_MODELS:
        return mistral.get_aclient(deployment)
    message = f"Unknown model: {deployment.model}. Please provide a valid model name."
    raise ValueError(message)
