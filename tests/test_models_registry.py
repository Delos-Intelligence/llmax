"""Verify the model registry is coherent after deprecation cleanup."""

from llmax.external_clients import get_aclient, get_client
from llmax.external_clients.exceptions import ClientNotFoundError
from llmax.models import (
    ANTHROPIC_MODELS,
    GEMINI_MODELS,
    META_MODELS,
    MISTRAL_MODELS,
    MODELS,
    OPENAI_MODELS,
    OPENROUTER_MODELS,
    PROVIDERS,
    SCALEWAY_MODELS,
    Deployment,
)
from llmax.usage.prices import (
    COMPLETION_PRICES_PER_1M,
    PROMPT_PRICES_PER_1M,
)

REMOVED_MODELS = (
    "glm-4.6",
    "grok-4-1-fast",
    "claude-3.5-sonnet",
    "claude-3-haiku",
    "claude-3.7-sonnet",
    "claude-4-sonnet",
    "command-r",
    "command-r-plus",
    "gpt-3.5",
    "gpt-4-turbo",
    "dall-e-3",
    "gemini-3.1-flash-lite-preview",
    "o3-mini",
    "o3-mini-high",
)


def test_removed_models_absent_from_registry() -> None:
    """No removed model should appear in the Model union."""
    for model in REMOVED_MODELS:
        assert model not in MODELS, f"{model} should have been removed from MODELS"


def test_removed_models_absent_from_provider_groups() -> None:
    """No removed model should appear in any provider tuple."""
    groups = {
        "ANTHROPIC_MODELS": ANTHROPIC_MODELS,
        "GEMINI_MODELS": GEMINI_MODELS,
        "META_MODELS": META_MODELS,
        "MISTRAL_MODELS": MISTRAL_MODELS,
        "OPENAI_MODELS": OPENAI_MODELS,
        "OPENROUTER_MODELS": OPENROUTER_MODELS,
        "SCALEWAY_MODELS": SCALEWAY_MODELS,
    }
    for group_name, group in groups.items():
        for model in REMOVED_MODELS:
            assert model not in group, f"{model} still listed in {group_name}"


def test_grok_and_cohere_providers_removed() -> None:
    """Grok and Cohere providers should be fully dropped."""
    assert "grok" not in PROVIDERS
    assert "cohere" not in PROVIDERS


def test_removed_models_absent_from_pricing() -> None:
    """Pricing dicts should not reference removed models."""
    for model in REMOVED_MODELS:
        assert model not in PROMPT_PRICES_PER_1M, (
            f"{model} still priced in PROMPT_PRICES_PER_1M"
        )
        assert model not in COMPLETION_PRICES_PER_1M, (
            f"{model} still priced in COMPLETION_PRICES_PER_1M"
        )


def test_current_chat_models_have_prompt_prices() -> None:
    """Every chat-capable LLM should expose a prompt price."""
    chat_groups = (
        ANTHROPIC_MODELS,
        META_MODELS,
        MISTRAL_MODELS,
        OPENROUTER_MODELS,
        SCALEWAY_MODELS,
    )
    skip = {
        "whisper-large-v3",
        "bge-multilingual-gemma2",
    }
    for group in chat_groups:
        for model in group:
            if model in skip:
                continue
            assert model in PROMPT_PRICES_PER_1M, (
                f"missing prompt price for {model}"
            )


def test_openai_chat_models_have_prompt_prices() -> None:
    """OpenAI chat-capable models should have a prompt price."""
    non_chat = {
        "ada-v2",
        "gpt-image-1",
        "gpt-image-2",
        "gpt-4o-transcribe",
        "text-embedding-3-large",
        "text-embedding-3-small",
        "tts-1",
        "whisper-1",
    }
    for model in OPENAI_MODELS:
        if model in non_chat:
            continue
        assert model in PROMPT_PRICES_PER_1M, f"missing prompt price for {model}"


def test_get_client_rejects_unknown_model() -> None:
    """get_client should raise for a model not in any provider group."""
    deployment = Deployment(
        model="not-a-real-model",  # type: ignore[arg-type]
        api_key="k",
        provider="openai",
        endpoint="http://x",
    )
    try:
        get_client(deployment)
    except ClientNotFoundError:
        pass
    else:
        msg = "expected ClientNotFoundError"
        raise AssertionError(msg)


def test_get_aclient_dispatches_for_known_groups() -> None:
    """Dispatch should succeed for at least one model per surviving group."""
    samples: list[tuple[str, str]] = [
        ("gpt-5", "openai"),
        ("claude-4.7-opus", "anthropic"),
        ("gemini-3-pro-preview", "gemini"),
        ("mistral-large-2411", "azure"),
        ("llama-4-maverick-17b-128e-instruct-fp8", "azure"),
        ("gpt-oss-120b", "scaleway"),
        ("glm-5.1", "openrouter"),
    ]
    for model, provider in samples:
        deployment = Deployment(
            model=model,  # type: ignore[arg-type]
            api_key="k",
            provider=provider,  # type: ignore[arg-type]
            endpoint="http://x",
            deployment_name=model,
        )
        client = get_aclient(deployment)
        assert client is not None, f"no async client for {model}/{provider}"
