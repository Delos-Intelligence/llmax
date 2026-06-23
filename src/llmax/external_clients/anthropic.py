"""Anthropic clients — native SDK for both direct API and AWS Bedrock."""

import json
import time
from collections.abc import AsyncGenerator, Generator
from typing import Any

import anthropic
import httpx
from anthropic.types import (
    InputJSONDelta,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    TextBlock,
    TextDelta,
    ToolUseBlock,
)
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from llmax.external_clients.exceptions import ProviderNotFoundError
from llmax.messages import Messages
from llmax.models import Deployment, Model
from llmax.utils import logger

Client = Any

MAPPING_FINISH_REASON: dict[str, str] = {
    "end_turn": "stop",
    "max_tokens": "stop",
    "tool_use": "tool_calls",
}

DEFAULT_MAX_TOKENS = 10_000

# Anthropic allows at most 4 cache_control breakpoints per request. Two are always
# reserved for the last two messages (see ``_add_message_cache_control``); the
# remaining two are shared between the system and the tools array. Once the system
# uses both, the tools breakpoint is dropped — the first system breakpoint sits
# after the tools array and already caches it.
_MAX_NON_MESSAGE_BREAKPOINTS = 2


def _extract_system(
    messages: Messages,
) -> tuple[list[dict[str, Any]] | anthropic.NotGiven, Messages, int]:
    """Separate system messages from the list.

    Returns ``(system_blocks, remaining, n_breakpoints)``.

    Two modes:

    - **Default** — every system message has plain-string content: they are
      concatenated into a single text block with one trailing ``cache_control``
      breakpoint. Byte-identical to the historical behaviour.
    - **Manual** — a system message carries *list* content (pre-built Anthropic
      text blocks): the blocks are preserved verbatim, honouring any
      ``cache_control`` the caller placed on them. This lets a caller position
      its own breakpoints — e.g. a stable, cross-user cohort prefix followed by
      a per-user tail. If the caller supplied blocks but marked none, a trailing
      ``cache_control`` is added so caching still applies (default-equivalent).

    ``n_breakpoints`` is how many ``cache_control`` markers the system uses, so
    the caller (``anthropic_create``/``acreate``) can keep the request within
    Anthropic's 4-breakpoint budget — see ``_convert_tools(enable_cache=...)``.
    """
    system_parts: list[str] = []
    manual_blocks: list[dict[str, Any]] = []
    has_list_content = False
    remaining: Messages = []
    for msg in messages:
        if msg.get("role") != "system":
            remaining.append(msg)
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            has_list_content = True
            for block in content:
                if isinstance(block, dict):
                    manual_blocks.append(dict(block))
                else:
                    manual_blocks.append({"type": "text", "text": str(block)})
        else:
            system_parts.append(content)
            manual_blocks.append({"type": "text", "text": content})

    if not manual_blocks:
        return anthropic.NOT_GIVEN, remaining, 0

    if not has_list_content:
        # Default mode — single concatenated block, one trailing breakpoint.
        return [
            {
                "type": "text",
                "text": " ".join(system_parts),
                "cache_control": {"type": "ephemeral"},
            },
        ], remaining, 1

    # Manual mode — preserve caller blocks and the cache_control markers they set.
    n_breakpoints = sum(1 for b in manual_blocks if b.get("cache_control"))
    if n_breakpoints == 0:
        manual_blocks[-1]["cache_control"] = {"type": "ephemeral"}
        n_breakpoints = 1
    return manual_blocks, remaining, n_breakpoints


def _add_message_cache_control(messages: Messages) -> Messages:
    """Add cache breakpoints on the last two messages.

    Both will be resent on the next turn as stable history, so caching them
    now means the next call reads the full conversation from cache.
    Uses 2 of the 4 available Anthropic cache slots (system and tools use the other 2).
    """
    if not messages:
        return messages

    messages = list(messages)

    for idx in (-1, -2):
        if len(messages) < abs(idx):
            break
        target = dict(messages[idx])
        content = target.get("content")

        if isinstance(content, str) and content:
            target["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                },
            ]
        elif isinstance(content, list) and content:
            content = list(content)
            last_block = content[-1]
            if isinstance(last_block, str):
                content[-1] = {
                    "type": "text",
                    "text": last_block,
                    "cache_control": {"type": "ephemeral"},
                }
            elif isinstance(last_block, dict):
                content[-1] = {**last_block, "cache_control": {"type": "ephemeral"}}
            target["content"] = content

        messages[idx] = target

    return messages


def _convert_tools(kwargs: dict[str, Any], *, enable_cache: bool = True) -> dict[str, Any]:
    """Convert OpenAI-style tools to Anthropic format, mutating kwargs in place.

    ``enable_cache`` adds a ``cache_control`` breakpoint on the last tool (default).
    Set it ``False`` to drop that breakpoint — used when the system already places
    a breakpoint *after* the tools array (which subsumes it), so the request stays
    within Anthropic's 4-breakpoint budget.
    """
    if "tools" in kwargs:
        tools = kwargs.pop("tools")
        if tools:
            anthropic_tools = [
                {
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "input_schema": tool["function"].get("parameters", {}),
                }
                for tool in tools
                if tool.get("type") == "function"
            ]
            if anthropic_tools:
                if enable_cache:
                    anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}
                kwargs["tools"] = anthropic_tools

    if "tool_choice" in kwargs:
        tool_choice = kwargs.pop("tool_choice")
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            kwargs["tool_choice"] = {
                "type": "tool",
                "name": tool_choice["function"]["name"],
            }
        elif tool_choice == "auto":
            kwargs["tool_choice"] = {"type": "auto"}
        elif tool_choice == "required":
            kwargs["tool_choice"] = {"type": "any"}

    return kwargs


def _to_chat_completion(response: anthropic.types.Message) -> ChatCompletion:
    """Convert an Anthropic Message to an OpenAI ChatCompletion."""
    choices = []
    for i, block in enumerate(response.content):
        if block.type == "text" and isinstance(block, TextBlock):
            choices.append(
                {
                    "finish_reason": MAPPING_FINISH_REASON.get(
                        response.stop_reason or "",
                        "stop",
                    ),
                    "index": i,
                    "message": {"content": block.text, "role": response.role},
                },
            )
        elif block.type == "tool_use" and isinstance(
            block,
            ToolUseBlock,
        ):
            choices.append(
                {
                    "finish_reason": "tool_calls",
                    "index": i,
                    "message": {
                        "content": None,
                        "role": response.role,
                        "tool_calls": [
                            {
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(block.input),
                                },
                            },
                        ],
                    },
                },
            )

    # If no choices extracted (shouldn't happen), fallback
    if not choices:
        choices.append(
            {
                "finish_reason": MAPPING_FINISH_REASON.get(
                    response.stop_reason or "",
                    "stop",
                ),
                "index": 0,
                "message": {"content": "", "role": response.role},
            },
        )

    cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
    cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
    prompt_tokens = (
        response.usage.input_tokens + cache_read + int(cache_creation * 1.25)
    )
    usage_dict: dict[str, Any] = {
        "completion_tokens": response.usage.output_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": prompt_tokens + response.usage.output_tokens,
    }
    if cache_read:
        usage_dict["prompt_tokens_details"] = {"cached_tokens": cache_read}

    return ChatCompletion.model_validate(
        {
            "id": response.id,
            "created": int(time.time()),
            "model": response.model,
            "usage": usage_dict,
            "choices": choices,
            "object": "chat.completion",
        },
    )


"""
        RawMessageStopEvent,
        RawContentBlockStopEvent,
"""


def _stream_to_chunks(  # noqa: C901, PLR0912, PLR0915
    stream: anthropic.Stream[anthropic.types.RawMessageStreamEvent],
) -> Generator[ChatCompletionChunk, None, None]:
    """Convert Anthropic stream events to ChatCompletionChunk generator."""
    request_id = ""
    model = ""
    created = int(time.time())
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cache_read_tokens: int = 0
    active_tool_id: str | None = None
    current_tool_calls: dict[str, dict[str, Any]] = {}

    with stream:
        for event in stream:
            content: str | None = None
            tool_calls: list[dict[str, Any]] | None = None
            finish_reason: str | None = None

            if event.type == "message_start" and isinstance(
                event,
                RawMessageStartEvent,
            ):
                request_id = event.message.id
                model = event.message.model
                cache_read_tokens = (
                    getattr(event.message.usage, "cache_read_input_tokens", 0) or 0
                )
                cache_creation_tokens = (
                    getattr(event.message.usage, "cache_creation_input_tokens", 0) or 0
                )
                prompt_tokens = (
                    event.message.usage.input_tokens
                    + cache_read_tokens
                    + int(cache_creation_tokens * 1.25)
                )
                completion_tokens = event.message.usage.output_tokens

            elif event.type == "content_block_start" and isinstance(
                event,
                RawContentBlockStartEvent,
            ):
                if event.content_block.type == "tool_use" and isinstance(
                    event.content_block,
                    ToolUseBlock,
                ):
                    block = event.content_block
                    active_tool_id = block.id
                    current_tool_calls[block.id] = {
                        "index": event.index,
                        "id": block.id,
                        "type": "function",
                        "function": {"name": block.name, "arguments": ""},
                    }
                    tool_calls = [current_tool_calls[block.id]]

            elif event.type == "content_block_delta" and isinstance(
                event,
                RawContentBlockDeltaEvent,
            ):
                if event.delta.type == "text_delta" and isinstance(
                    event.delta,
                    TextDelta,
                ):
                    content = event.delta.text
                elif (
                    event.delta.type == "input_json_delta"
                    and active_tool_id
                    and isinstance(event.delta, InputJSONDelta)
                ):
                    fragment = event.delta.partial_json
                    if active_tool_id in current_tool_calls:
                        tool_calls = [
                            {
                                "index": current_tool_calls[active_tool_id]["index"],
                                "id": active_tool_id,
                                "function": {"arguments": fragment},
                            },
                        ]

            elif event.type == "content_block_stop":
                active_tool_id = None

            elif event.type == "message_delta" and isinstance(
                event,
                RawMessageDeltaEvent,
            ):
                finish_reason = MAPPING_FINISH_REASON.get(
                    event.delta.stop_reason or "",
                    "stop",
                )
                completion_tokens = event.usage.output_tokens

            elif event.type == "message_stop":
                continue

            # Only yield if there's something to send
            if (
                content is not None
                or tool_calls is not None
                or finish_reason is not None
            ):
                delta: dict[str, Any] = {}
                if content is not None:
                    delta["content"] = content
                if tool_calls is not None:
                    delta["tool_calls"] = tool_calls

                usage: dict[str, int | dict[str, int]] | None = None
                if prompt_tokens is not None and completion_tokens is not None:
                    usage = {
                        "completion_tokens": completion_tokens,
                        "prompt_tokens": prompt_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                    if cache_read_tokens:
                        usage["prompt_tokens_details"] = {
                            "cached_tokens": cache_read_tokens,
                        }

                try:
                    yield ChatCompletionChunk.model_validate(
                        {
                            "id": request_id,
                            "created": created,
                            "model": model,
                            "usage": usage,
                            "choices": [
                                {
                                    "delta": delta,
                                    "index": 0,
                                    "finish_reason": finish_reason,
                                },
                            ],
                            "object": "chat.completion.chunk",
                        },
                    )
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
                    return


async def _astream_to_chunks(  # noqa: C901, PLR0912, PLR0915
    stream: anthropic.AsyncStream[anthropic.types.RawMessageStreamEvent],
) -> AsyncGenerator[ChatCompletionChunk, None]:
    """Convert Anthropic async stream events to ChatCompletionChunk async generator."""
    request_id = ""
    model = ""
    created = int(time.time())
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cache_read_tokens: int = 0
    active_tool_id: str | None = None
    current_tool_calls: dict[str, dict[str, Any]] = {}

    async with stream:
        async for event in stream:
            content: str | None = None
            tool_calls: list[dict[str, Any]] | None = None
            finish_reason: str | None = None

            if event.type == "message_start" and isinstance(
                event,
                RawMessageStartEvent,
            ):
                request_id = event.message.id
                model = event.message.model
                cache_read_tokens = (
                    getattr(event.message.usage, "cache_read_input_tokens", 0) or 0
                )
                cache_creation_tokens = (
                    getattr(event.message.usage, "cache_creation_input_tokens", 0) or 0
                )
                prompt_tokens = (
                    event.message.usage.input_tokens
                    + cache_read_tokens
                    + int(cache_creation_tokens * 1.25)
                )
                completion_tokens = event.message.usage.output_tokens

            elif event.type == "content_block_start" and isinstance(
                event,
                RawContentBlockStartEvent,
            ):
                if event.content_block.type == "tool_use" and isinstance(
                    event.content_block,
                    ToolUseBlock,
                ):
                    block = event.content_block
                    active_tool_id = block.id
                    current_tool_calls[block.id] = {
                        "index": event.index,
                        "id": block.id,
                        "type": "function",
                        "function": {"name": block.name, "arguments": ""},
                    }
                    tool_calls = [current_tool_calls[block.id]]

            elif event.type == "content_block_delta" and isinstance(
                event,
                RawContentBlockDeltaEvent,
            ):
                if event.delta.type == "text_delta" and isinstance(
                    event.delta,
                    TextDelta,
                ):
                    content = event.delta.text
                elif (
                    event.delta.type == "input_json_delta"
                    and active_tool_id
                    and isinstance(event.delta, InputJSONDelta)
                ):
                    fragment = event.delta.partial_json
                    if active_tool_id in current_tool_calls:
                        tool_calls = [
                            {
                                "index": current_tool_calls[active_tool_id]["index"],
                                "id": active_tool_id,
                                "function": {"arguments": fragment},
                            },
                        ]

            elif event.type == "content_block_stop":
                active_tool_id = None

            elif event.type == "message_delta" and isinstance(
                event,
                RawMessageDeltaEvent,
            ):
                finish_reason = MAPPING_FINISH_REASON.get(
                    event.delta.stop_reason or "",
                    "stop",
                )
                completion_tokens = event.usage.output_tokens

            elif event.type == "message_stop":
                continue

            if (
                content is not None
                or tool_calls is not None
                or finish_reason is not None
            ):
                delta_dict: dict[str, Any] = {}
                if content is not None:
                    delta_dict["content"] = content
                if tool_calls is not None:
                    delta_dict["tool_calls"] = tool_calls

                usage: dict[str, int | dict[str, int]] | None = None
                if prompt_tokens is not None and completion_tokens is not None:
                    usage = {
                        "completion_tokens": completion_tokens,
                        "prompt_tokens": prompt_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                    if cache_read_tokens:
                        usage["prompt_tokens_details"] = {
                            "cached_tokens": cache_read_tokens,
                        }

                try:
                    yield ChatCompletionChunk.model_validate(
                        {
                            "id": request_id,
                            "created": created,
                            "model": model,
                            "usage": usage,
                            "choices": [
                                {
                                    "delta": delta_dict,
                                    "index": 0,
                                    "finish_reason": finish_reason,
                                },
                            ],
                            "object": "chat.completion.chunk",
                        },
                    )
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
                    return


def get_client(
    deployment: Deployment,
    http_client: httpx.Client | None = None,
) -> Client:
    """Get a sync Anthropic client for the given deployment."""
    match deployment.provider:
        case "anthropic":
            if http_client:
                return anthropic.Anthropic(
                    api_key=deployment.api_key,
                    http_client=http_client,
                )
            return anthropic.Anthropic(api_key=deployment.api_key)
        case "aws-bedrock":
            return anthropic.AnthropicBedrock(
                aws_access_key=deployment.project_id,
                aws_secret_key=deployment.api_key,
                aws_region=deployment.region or "us-east-1",
            )
        case _:
            raise ProviderNotFoundError(deployment)


def get_aclient(
    deployment: Deployment,
    http_client: httpx.AsyncClient | None = None,
) -> Client:
    """Get an async Anthropic client for the given deployment."""
    match deployment.provider:
        case "anthropic":
            if http_client:
                return anthropic.AsyncAnthropic(
                    api_key=deployment.api_key,
                    http_client=http_client,
                )
            return anthropic.AsyncAnthropic(api_key=deployment.api_key)
        case "aws-bedrock":
            return anthropic.AsyncAnthropicBedrock(
                aws_access_key=deployment.project_id,
                aws_secret_key=deployment.api_key,
                aws_region=deployment.region or "us-east-1",
            )
        case _:
            raise ProviderNotFoundError(deployment)


def anthropic_create(
    client: anthropic.Anthropic | anthropic.AnthropicBedrock,
    messages: Messages,
    model: Model,
    **kwargs: Any,
) -> ChatCompletion | Generator[ChatCompletionChunk, None, None]:
    """Synchronous Anthropic call, returns ChatCompletion or streaming generator."""
    system, remaining, n_sys_breakpoints = _extract_system(messages)
    remaining = _add_message_cache_control(remaining)
    kwargs = _convert_tools(
        kwargs,
        enable_cache=(n_sys_breakpoints < _MAX_NON_MESSAGE_BREAKPOINTS),
    )
    stream = kwargs.pop("stream", False)

    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = DEFAULT_MAX_TOKENS

    if stream:
        raw_stream = client.messages.create(  # type: ignore[call-overload]
            model=model,
            messages=remaining,
            system=system,
            stream=True,
            **kwargs,
        )  # ty:ignore[no-matching-overload]
        return _stream_to_chunks(raw_stream)

    response = client.messages.create(  # type: ignore[call-overload]
        model=model,
        messages=remaining,
        system=system,
        **kwargs,
    )  # ty:ignore[no-matching-overload]
    return _to_chat_completion(response)


async def anthropic_acreate(
    client: anthropic.AsyncAnthropic | anthropic.AsyncAnthropicBedrock,
    messages: Messages,
    model: Model,
    **kwargs: Any,
) -> ChatCompletion | AsyncGenerator[ChatCompletionChunk, None]:
    """Async Anthropic call, returns ChatCompletion or async streaming generator."""
    system, remaining, n_sys_breakpoints = _extract_system(messages)
    remaining = _add_message_cache_control(remaining)
    kwargs = _convert_tools(
        kwargs,
        enable_cache=(n_sys_breakpoints < _MAX_NON_MESSAGE_BREAKPOINTS),
    )
    stream = kwargs.pop("stream", False)

    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = DEFAULT_MAX_TOKENS

    if stream:
        raw_stream = await client.messages.create(  # type: ignore[call-overload]
            model=model,
            messages=remaining,
            system=system,
            stream=True,
            **kwargs,
        )  # ty:ignore[no-matching-overload]
        return _astream_to_chunks(raw_stream)

    response = await client.messages.create(  # type: ignore[call-overload]
        model=model,
        messages=remaining,
        system=system,
        **kwargs,
    )  # ty:ignore[no-matching-overload]
    return _to_chat_completion(response)
