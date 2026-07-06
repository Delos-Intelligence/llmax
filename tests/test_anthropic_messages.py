"""Tests for OpenAI -> Anthropic message conversion.

Covers the translation done by ``_convert_messages`` before every Anthropic
call: assistant ``tool_calls`` -> ``tool_use`` blocks, ``role: "tool"`` ->
merged ``tool_result`` user messages, ``image_url`` -> ``image`` blocks, and
idempotence for messages already in Anthropic shape.
"""

from llmax.external_clients.anthropic import _convert_messages

WEATHER_CALL = {
    "id": "toolu_123",
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
}


# --- assistant tool_calls -> tool_use blocks ---


def test_assistant_tool_calls_become_tool_use_blocks() -> None:
    """An assistant tool_calls message becomes a tool_use content block."""
    converted = _convert_messages(
        [{"role": "assistant", "content": None, "tool_calls": [WEATHER_CALL]}],
    )
    assert converted == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                },
            ],
        },
    ]


def test_assistant_text_and_parallel_tool_calls() -> None:
    """Text content is kept and parallel calls each get a tool_use block."""
    second_call = {
        "id": "toolu_456",
        "type": "function",
        "function": {"name": "get_time", "arguments": "{}"},
    }
    converted = _convert_messages(
        [
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [WEATHER_CALL, second_call],
            },
        ],
    )
    blocks = converted[0]["content"]
    assert blocks[0] == {"type": "text", "text": "Let me check."}
    assert [b["id"] for b in blocks[1:]] == ["toolu_123", "toolu_456"]
    assert all(b["type"] == "tool_use" for b in blocks[1:])


def test_invalid_tool_arguments_fall_back_to_empty_input() -> None:
    """Unparseable tool arguments degrade to an empty input dict."""
    bad_call = {
        "id": "toolu_bad",
        "type": "function",
        "function": {"name": "f", "arguments": "{not json"},
    }
    converted = _convert_messages(
        [{"role": "assistant", "content": None, "tool_calls": [bad_call]}],
    )
    assert converted[0]["content"][0]["input"] == {}


# --- tool messages -> merged tool_result user message ---


def test_tool_message_becomes_tool_result_user_message() -> None:
    """A tool message becomes a user message with a tool_result block."""
    converted = _convert_messages(
        [{"role": "tool", "tool_call_id": "toolu_123", "content": "Sunny, 22C"}],
    )
    assert converted == [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_123",
                    "content": "Sunny, 22C",
                },
            ],
        },
    ]


def test_consecutive_tool_messages_merge_into_one_user_message() -> None:
    """Consecutive tool messages merge into a single user message."""
    converted = _convert_messages(
        [
            {"role": "assistant", "content": None, "tool_calls": [WEATHER_CALL]},
            {"role": "tool", "tool_call_id": "toolu_123", "content": "Sunny"},
            {"role": "tool", "tool_call_id": "toolu_456", "content": "12:00"},
        ],
    )
    assert [m["role"] for m in converted] == ["assistant", "user"]
    results = converted[1]["content"]
    assert [r["tool_use_id"] for r in results] == ["toolu_123", "toolu_456"]


def test_user_message_after_tool_results_merges_into_same_turn() -> None:
    """A user message right after tool results joins the same turn."""
    converted = _convert_messages(
        [
            {"role": "tool", "tool_call_id": "toolu_123", "content": "Sunny"},
            {"role": "user", "content": "And tomorrow?"},
        ],
    )
    assert len(converted) == 1
    assert converted[0]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_123", "content": "Sunny"},
        {"type": "text", "text": "And tomorrow?"},
    ]


def test_tool_message_with_list_content_is_flattened() -> None:
    """List-of-parts tool content is flattened to plain text."""
    converted = _convert_messages(
        [
            {
                "role": "tool",
                "tool_call_id": "toolu_123",
                "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            },
        ],
    )
    assert converted[0]["content"][0]["content"] == "ab"


# --- image_url -> image blocks ---


def test_image_url_block_becomes_anthropic_image_block() -> None:
    """An OpenAI image_url block becomes an Anthropic image block."""
    converted = _convert_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "https://x.test/a.jpg"}},
                ],
            },
        ],
    )
    assert converted[0]["content"] == [
        {"type": "text", "text": "describe"},
        {"type": "image", "source": {"type": "url", "url": "https://x.test/a.jpg"}},
    ]


def test_data_uri_image_becomes_base64_source() -> None:
    """A data: URI image becomes a base64 image source."""
    converted = _convert_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,AAAA"},
                    },
                ],
            },
        ],
    )
    assert converted[0]["content"] == [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"},
        },
    ]


# --- passthrough / idempotence ---


def test_plain_messages_pass_through_unchanged() -> None:
    """Plain text messages are not modified."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert _convert_messages(messages) == messages


def test_anthropic_shaped_messages_pass_through_unchanged() -> None:
    """Messages already in Anthropic shape are not modified."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
            ],
        },
    ]
    assert _convert_messages(messages) == messages


def test_null_content_becomes_empty_string() -> None:
    """Null content is replaced by an empty string."""
    converted = _convert_messages([{"role": "assistant", "content": None}])
    assert converted == [{"role": "assistant", "content": ""}]


def test_conversion_is_idempotent() -> None:
    """Converting already-converted messages is a no-op."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [WEATHER_CALL]},
        {"role": "tool", "tool_call_id": "toolu_123", "content": "Sunny"},
    ]
    once = _convert_messages(messages)
    assert _convert_messages(once) == once
