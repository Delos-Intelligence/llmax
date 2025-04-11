"""Anthropic clients."""

import json
from collections.abc import Generator
from typing import Any, AsyncGenerator, Dict, List, Optional

import aioboto3  # type: ignore
import boto3
from dateutil import parser
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from llmax.external_clients.exceptions import ProviderNotFoundError
from llmax.external_clients.universal_client.chat_completion_message import (
    ChatCompletionAssistantMessage,
    ChatCompletionSystemMessage,
    ChatCompletionUserMessage,
)
from llmax.external_clients.universal_client.client import (
    AsyncUniversalClient,
    UniversalClient,
)
from llmax.messages import Messages
from llmax.models import Deployment, Model
from llmax.utils import logger

Client = Any


MAPPING_FINISH_REASON = {
    "end_turn": "stop",
    "max_tokens": "stop",
}


def client_creation_anthropic(
    aws_key: str,
    aws_secret_key: str,
    region_name: str,
) -> Any:
    """Create the antropic client."""
    return boto3.Session(
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region_name,
    ).client("bedrock-runtime")


def anthropic_parsing(response: Dict[str, Any]) -> Optional[ChatCompletion]:
    """The parsing from anthropic to openAi."""
    try:
        data = json.loads(response["body"].read())
        input_tokens = response["ResponseMetadata"]["HTTPHeaders"][
            "x-amzn-bedrock-input-token-count"
        ]
        completion_tokens = response["ResponseMetadata"]["HTTPHeaders"][
            "x-amzn-bedrock-output-token-count"
        ]
        chat_completion = {
            "id": response["ResponseMetadata"]["RequestId"],
            "created": int(
                parser.parse(
                    response["ResponseMetadata"]["HTTPHeaders"]["date"],
                ).timestamp(),
            ),
            "model": data["model"],
            "usage": {
                "completion_tokens": completion_tokens,
                "prompt_tokens": input_tokens,
                "total_tokens": completion_tokens + input_tokens,
            },
            "choices": [
                {
                    "finish_reason": MAPPING_FINISH_REASON[data["stop_reason"]],
                    "index": i,
                    "message": {"content": e["text"], "role": data["role"]},
                }
                for i, e in enumerate(data["content"])
            ],
            "object": "chat.completion",
        }
        return ChatCompletion.model_validate(chat_completion)
    except Exception:
        return None


def anthropic_parsing_stream(
    response: Dict[str, Any],
) -> Optional[Generator[ChatCompletionChunk, None, None]]:
    """The parsing from anthropic to openAi in streaming mode."""
    try:
        request_id = response["ResponseMetadata"]["RequestId"]
        created = int(
            parser.parse(
                response["ResponseMetadata"]["HTTPHeaders"]["date"],
            ).timestamp(),
        )
    except Exception:
        return None

    _object = "chat.completion.chunk"
    content = None
    model = None
    role = None
    prompt_tokens = None
    completion_tokens = None
    stop_reason = None
    to_yield = True

    for event in response["body"]:
        chunk = json.loads(event["chunk"]["bytes"])
        if chunk["type"] == "message_start":
            model = chunk["message"]["model"]
            role = chunk["message"]["role"]
            prompt_tokens = chunk["message"]["usage"]["input_tokens"]
            completion_tokens = chunk["message"]["usage"]["output_tokens"]
            content = ""
            to_yield = True
        elif chunk["type"] == "content_block_delta":
            content = chunk["delta"]["text"]
            to_yield = True
        elif chunk["type"] == "content_block_stop":
            to_yield = False
        elif chunk["type"] == "message_delta":
            stop_reason = MAPPING_FINISH_REASON[chunk["delta"]["stop_reason"]]
            content = ""
            to_yield = True
        elif chunk["type"] == "message_stop":
            prompt_tokens = chunk["amazon-bedrock-invocationMetrics"]["inputTokenCount"]
            completion_tokens = chunk["amazon-bedrock-invocationMetrics"][
                "outputTokenCount"
            ]
            content = ""
            to_yield = True

        if to_yield:
            try:
                chat_completion_chunk = {
                    "id": request_id,
                    "created": created,
                    "model": model,
                    "usage": {
                        "completion_tokens": completion_tokens,
                        "prompt_tokens": prompt_tokens,
                        "total_tokens": completion_tokens + prompt_tokens
                        if completion_tokens and prompt_tokens
                        else None,
                    },
                    "choices": [
                        {
                            "delta": {
                                "content": content,
                                "role": role,
                            },
                            "index": 0,
                            "finish_reason": stop_reason,
                        },
                    ],
                    "object": _object,
                }
                yield ChatCompletionChunk.model_validate(chat_completion_chunk)
            except Exception:
                return None
    return None


def completion_call_anthropic(
    client: Any,
    messages: Messages,
    model: Model,
    stream: bool = False,
    *args: Any,  # noqa: ARG001
    **kwargs: Any,  # noqa: ARG001
) -> Optional[ChatCompletion] | Optional[Generator[ChatCompletionChunk, None, None]]:
    """Anthropic call to make a completion."""
    system_message: List[str] = []
    to_remove: List[int] = []

    try:
        counter = 0
        for i, message in enumerate(messages):
            try:
                ChatCompletionAssistantMessage.model_validate(message)
                counter += 1
            except Exception:  # noqa: S110
                pass
            try:
                ChatCompletionSystemMessage.model_validate(message)
                system_message.append(str(message["content"]))
                to_remove.append(i)
                counter += 1
            except Exception:  # noqa: S110
                pass
            try:
                ChatCompletionUserMessage.model_validate(message)
                counter += 1
            except Exception:  # noqa: S110
                pass
        if counter < len(messages):
            logger.error("Incorrect message formatting")
            raise  # noqa: PLE0704
    except Exception:
        return None

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10_000,
        "messages": [
            message for i, message in enumerate(messages) if i not in to_remove
        ],
    }
    if len(system_message) > 0:
        body.update({"system": " ".join(system_message)})

    if stream:
        response_stream = client.invoke_model_with_response_stream(
            modelId=model,
            body=json.dumps(body),
        )
        return anthropic_parsing_stream(response_stream)
    response = client.invoke_model(modelId=model, body=json.dumps(body))
    return anthropic_parsing(response)


def get_client(deployment: Deployment) -> Client:
    """Get a client for the given deployment."""
    match deployment.provider:
        case "aws-bedrock":
            return UniversalClient(
                client_creation=client_creation_anthropic,
                completion_call=completion_call_anthropic,
                deployment=deployment,
                aws_key=deployment.project_id,
                aws_secret_key=deployment.api_key,
                region_name=deployment.region,
            )
        case _:
            raise ProviderNotFoundError(deployment)


AsyncClient = Any


async def async_client_creation_anthropic(
    aws_key: str,
    aws_secret_key: str,
    region_name: str,
) -> Any:
    """Create the async anthropic client."""
    session = aioboto3.Session(
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region_name,
    )
    return await session.client("bedrock-runtime").__aenter__()


async def async_anthropic_parsing(response: Dict[str, Any]) -> Optional[ChatCompletion]:
    """The parsing from anthropic to openAi for async response."""
    try:
        body_bytes = await response["body"].read()
        data = json.loads(body_bytes)
        input_tokens = response["ResponseMetadata"]["HTTPHeaders"][
            "x-amzn-bedrock-input-token-count"
        ]
        completion_tokens = response["ResponseMetadata"]["HTTPHeaders"][
            "x-amzn-bedrock-output-token-count"
        ]
        chat_completion = {
            "id": response["ResponseMetadata"]["RequestId"],
            "created": int(
                parser.parse(
                    response["ResponseMetadata"]["HTTPHeaders"]["date"],
                ).timestamp(),
            ),
            "model": data["model"],
            "usage": {
                "completion_tokens": completion_tokens,
                "prompt_tokens": input_tokens,
                "total_tokens": completion_tokens + input_tokens,
            },
            "choices": [
                {
                    "finish_reason": MAPPING_FINISH_REASON[data["stop_reason"]],
                    "index": i,
                    "message": {"content": e["text"], "role": data["role"]},
                }
                for i, e in enumerate(data["content"])
            ],
            "object": "chat.completion",
        }
        return ChatCompletion.model_validate(chat_completion)
    except Exception:
        return None


async def async_anthropic_parsing_stream(
    response: Dict[str, Any],
) -> AsyncGenerator[ChatCompletionChunk, None]:
    """The parsing from anthropic to openAi in streaming mode for async calls."""
    try:
        request_id = response["ResponseMetadata"]["RequestId"]
        created = int(
            parser.parse(
                response["ResponseMetadata"]["HTTPHeaders"]["date"],
            ).timestamp(),
        )
    except Exception:
        return

    _object = "chat.completion.chunk"
    content = None
    model = None
    role = None
    prompt_tokens = None
    completion_tokens = None
    stop_reason = None
    to_yield = True

    async for event in response["body"]:
        chunk = json.loads(event["chunk"]["bytes"])
        if chunk["type"] == "message_start":
            model = chunk["message"]["model"]
            role = chunk["message"]["role"]
            prompt_tokens = chunk["message"]["usage"]["input_tokens"]
            completion_tokens = chunk["message"]["usage"]["output_tokens"]
            content = ""
            to_yield = True
        elif chunk["type"] == "content_block_delta":
            content = chunk["delta"]["text"]
            to_yield = True
        elif chunk["type"] == "content_block_stop":
            to_yield = False
        elif chunk["type"] == "message_delta":
            stop_reason = MAPPING_FINISH_REASON[chunk["delta"]["stop_reason"]]
            content = ""
            to_yield = True
        elif chunk["type"] == "message_stop":
            prompt_tokens = chunk["amazon-bedrock-invocationMetrics"]["inputTokenCount"]
            completion_tokens = chunk["amazon-bedrock-invocationMetrics"][
                "outputTokenCount"
            ]
            content = ""
            to_yield = True

        if to_yield:
            try:
                chat_completion_chunk = {
                    "id": request_id,
                    "created": created,
                    "model": model,
                    "usage": {
                        "completion_tokens": completion_tokens,
                        "prompt_tokens": prompt_tokens,
                        "total_tokens": completion_tokens + prompt_tokens
                        if completion_tokens and prompt_tokens
                        else None,
                    },
                    "choices": [
                        {
                            "delta": {
                                "content": content,
                                "role": role,
                            },
                            "index": 0,
                            "finish_reason": stop_reason,
                        },
                    ],
                    "object": _object,
                }
                yield ChatCompletionChunk.model_validate(chat_completion_chunk)
            except Exception:
                return
    return


async def async_completion_call_anthropic(
    client: Any,
    messages: Messages,
    model: Model,
    stream: bool = False,
    *args: Any,  # noqa: ARG001
    **kwargs: Any,  # noqa: ARG001
) -> Optional[ChatCompletion] | Optional[AsyncGenerator[ChatCompletionChunk, None]]:
    """Async Anthropic call to make a completion."""
    system_message: List[str] = []
    to_remove: List[int] = []

    try:
        counter = 0
        for i, message in enumerate(messages):
            try:
                ChatCompletionAssistantMessage.model_validate(message)
                counter += 1
            except Exception:  # noqa: S110
                pass
            try:
                ChatCompletionSystemMessage.model_validate(message)
                system_message.append(str(message["content"]))
                to_remove.append(i)
                counter += 1
            except Exception:  # noqa: S110
                pass
            try:
                ChatCompletionUserMessage.model_validate(message)
                counter += 1
            except Exception:  # noqa: S110
                pass
        if counter < len(messages):
            logger.error("Incorrect message formatting")
            raise  # noqa: PLE0704
    except Exception:
        return None

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10_000,
        "messages": [
            message for i, message in enumerate(messages) if i not in to_remove
        ],
    }
    if len(system_message) > 0:
        body.update({"system": " ".join(system_message)})

    if stream:
        response_stream = await client.invoke_model_with_response_stream(
            modelId=model,
            body=json.dumps(body),
        )
        return async_anthropic_parsing_stream(response_stream)
    response = await client.invoke_model(modelId=model, body=json.dumps(body))
    return await async_anthropic_parsing(response)


def get_async_client(deployment: Deployment) -> AsyncClient:
    """Get an async client for the given deployment."""
    match deployment.provider:
        case "aws-bedrock":
            return AsyncUniversalClient(
                client_creation=async_client_creation_anthropic,
                completion_call=async_completion_call_anthropic,
                deployment=deployment,
                aws_key=deployment.project_id,
                aws_secret_key=deployment.api_key,
                region_name=deployment.region,
            )
        case _:
            raise ProviderNotFoundError(deployment)
