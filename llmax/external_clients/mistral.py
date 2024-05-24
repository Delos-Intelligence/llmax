from typing import Any

from openai import OpenAI

from llmax import models
from llmax.models import Deployment
from llmax.utils import logger

Client = Any


def get_client(deployment: Deployment) -> Client:
    """Get a Mistral client for the given deployment."""
    if deployment.model not in models.MISTRAL_MODELS:
        message = f"Unknown model: {deployment.model}. Please provide a valid Mistral model name."
        raise ValueError(message)

    match deployment.provider:
        case "azure":
            return OpenAI(
                api_key=deployment.api_key,
                base_url=deployment.endpoint,
            )
        case _:
            message = f"Invalid provider for Mistral model: {deployment.provider}. Please provide a valid provider."
            raise ValueError(message)


def get_aclient(deployment: Deployment) -> Client:
    """Get an async Mistral client for the given deployment."""
    # TODO: Correct this implementation
    message = "Not implemented yet: Mistral async client. Returning a sync client."
    logger.warning(message)
    return get_client(deployment)
