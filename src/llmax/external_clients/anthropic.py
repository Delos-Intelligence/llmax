"""Anthropic clients."""

import boto3
from typing import Any

from llmax.external_clients.exceptions import ProviderNotFoundError
from llmax.models import Deployment
from llmax.external_clients.universal_client.chat_completion import ChatCompletion
from llmax.external_clients.universal_client.client import UniversalClient

Client = Any


def client_creation_anthropic(aws_key: str, aws_secret_key: str, region_name: str) -> Any:
    return boto3.Session(
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region_name
    ).client('bedrock-runtime')

def completion_call_anthropic() -> ChatCompletion:
    return ChatCompletion()


def get_client(deployment: Deployment) -> Client:
    """Get a client for the given deployment."""
    match deployment.provider:
        case "aws-bedrock":
            return UniversalClient(
                client_creation=client_creation_anthropic,
                completion_call=completion_call_anthropic
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
