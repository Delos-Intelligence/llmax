"""Anthropic clients."""

from typing import Any

from .universal_client.client import UniversalClient

from llmax.external_clients.exceptions import ProviderNotFoundError
from llmax.models import Deployment

Client = Any


class BedrockAnthropicClient:
    pass


def get_client(deployment: Deployment) -> Client:
    """Get a client for the given deployment."""
    match deployment.provider:
        case "aws-bedrock":
            return UniversalClient(
                api_key=deployment.api_key,
                api_version=deployment.api_version,
            )
        case _:
            raise ProviderNotFoundError(deployment)


# def get_aclient(deployment: Deployment) -> Client:
#     """Get an async client for the given deployment."""
#     match deployment.provider:
#         case "aws-bedrock":
#             return AsyncAzureOpenAI(
#                 api_key=deployment.api_key,
#                 api_version=deployment.api_version,
#                 azure_endpoint=deployment.endpoint,
#             )
#         case _:
#             raise ProviderNotFoundError(deployment)
