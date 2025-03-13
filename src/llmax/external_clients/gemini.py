"""Gemini client for interacting with the Cohere API.

Note: we use the OpenAI client to interact with the Cohere models.
"""

from typing import Any

from google.auth.transport.requests import Request
from openai import AsyncOpenAI, OpenAI

from llmax.external_clients.exceptions import ProviderNotFoundError
from llmax.models import Deployment

Client = Any


def get_client(deployment: Deployment) -> Client:
    """Get a Gemini client for the given deployment."""
    match deployment.provider:
        case "openai":
            if deployment.creds:
                auth_req = Request()
                deployment.creds.refresh(auth_req)
                return OpenAI(
                    api_key=deployment.creds.token,
                    base_url=deployment.endpoint,
                )
            raise ProviderNotFoundError(deployment)
        case _:
            raise ProviderNotFoundError(deployment)


def get_aclient(deployment: Deployment) -> Client:
    """Get an async Gemini client for the given deployment."""
    match deployment.provider:
        case "openai":
            if deployment.creds:
                auth_req = Request()
                deployment.creds.refresh(auth_req)
                return AsyncOpenAI(
                    api_key=deployment.creds.token,
                    base_url=deployment.endpoint,
                )
            raise ProviderNotFoundError(deployment)
        case _:
            raise ProviderNotFoundError(deployment)
