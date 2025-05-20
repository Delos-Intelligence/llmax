"""Anthropic clients."""

import json
from collections.abc import Generator
from typing import Any, Dict, List, Optional

import boto3  # type: ignore
from dateutil import parser
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from llmax.external_clients.exceptions import ProviderNotFoundError
from llmax.external_clients.universal_client.chat_completion_message import (
    ChatCompletionAssistantMessage,
    ChatCompletionSystemMessage,
    ChatCompletionUserMessage,
)
from llmax.external_clients.universal_client.client import UniversalClient
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
    model = None
    role = None
    prompt_tokens = None
    completion_tokens = None
    stop_reason = None

    # Track active tool calls
    current_tool_calls = {}
    active_tool_id = None

    for event in response["body"]:
        chunk = json.loads(event["chunk"]["bytes"])
        chunk_type = chunk.get("type")

        # Default values for this chunk
        content = None
        tool_calls = None

        if chunk_type == "message_start":
            model = chunk["message"]["model"]
            role = chunk["message"]["role"]
            prompt_tokens = chunk["message"].get("usage", {}).get("input_tokens")
            completion_tokens = chunk["message"].get("usage", {}).get("output_tokens")

        elif chunk_type == "content_block_start":
            block_index = chunk.get("index", 0)
            content_block = chunk.get("content_block", {})
            block_type = content_block.get("type")

            if block_type == "text":
                # Regular text content block
                pass

            elif block_type == "tool_use":
                # Start of a tool call
                tool_id = content_block.get("id")
                tool_name = content_block.get("name")
                active_tool_id = tool_id

                # Initialize the tool call
                current_tool_calls[tool_id] = {
                    "index": block_index,
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": "",  # Will be filled with JSON
                    },
                }

                # Create initial tool call to send
                tool_calls = [current_tool_calls[tool_id]]

        elif chunk_type == "content_block_delta":
            delta = chunk.get("delta", {})
            delta_type = delta.get("type")

            if delta_type == "text_delta":
                # Regular text content
                content = delta.get("text", "")

            elif delta_type == "input_json_delta" and active_tool_id:
                # Tool JSON input update
                json_fragment = delta.get("partial_json", "")

                # Update the arguments in the current tool call
                if active_tool_id in current_tool_calls:
                    current_tool_id = active_tool_id
                    current_tool_calls[current_tool_id]["function"]["arguments"] = (
                        json_fragment
                    )

                    # Create tool call update to send
                    tool_calls = [
                        {
                            "index": current_tool_calls[current_tool_id]["index"],
                            "id": current_tool_id,
                            "function": {"arguments": json_fragment},
                        },
                    ]

        elif chunk_type == "content_block_stop":
            if active_tool_id:
                active_tool_id = None

        elif chunk_type == "message_delta":
            if "delta" in chunk and "stop_reason" in chunk["delta"]:
                stop_reason = MAPPING_FINISH_REASON.get(
                    chunk["delta"]["stop_reason"],
                    "stop",
                )

        elif chunk_type == "message_stop":
            prompt_tokens = chunk.get("amazon-bedrock-invocationMetrics", {}).get(
                "inputTokenCount",
            )
            completion_tokens = chunk.get("amazon-bedrock-invocationMetrics", {}).get(
                "outputTokenCount",
            )

        # Only yield if we have content or tool calls
        if content is not None or tool_calls is not None:
            try:
                # Prepare delta based on what we have
                delta = {}
                if content is not None:
                    delta["content"] = content
                if role is not None and chunk_type == "message_start":
                    delta["role"] = role
                if tool_calls is not None:
                    delta["tool_calls"] = tool_calls

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
                            "delta": delta,
                            "index": 0,
                            "finish_reason": stop_reason,
                        },
                    ],
                    "object": _object,
                }
                yield ChatCompletionChunk.model_validate(chat_completion_chunk)
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                return None
    return None


def completion_call_anthropic(
    client: Any,
    messages: Messages,
    model: Model,
    stream: bool = False,
    *args: Any,  # noqa: ARG001
    **kwargs: Any,
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

    if "tools" in kwargs:
        tools = kwargs["tools"]
        if tools:
            # Convert OpenAI-style tools to Anthropic format
            anthropic_tools = []
            for tool in tools:
                if tool["type"] == "function":
                    anthropic_tool = {
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "input_schema": tool["function"].get("parameters", {}),
                    }
                    anthropic_tools.append(anthropic_tool)

            if anthropic_tools:
                body.update({"tools": anthropic_tools})

    if "tool_choice" in kwargs:
        tool_choice = kwargs["tool_choice"]
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            body.update({"tool_choice": tool_choice["function"]["name"]})
        elif tool_choice == "auto":
            body.update({"tool_choice": "auto"})

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
