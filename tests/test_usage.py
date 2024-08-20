"""Usage tests."""

from openai.types import CompletionUsage
from src.llmax import tokens
from src.llmax.models import Deployment, Model
from src.llmax.usage import ModelUsage

deployments: dict[Model, Deployment] = {
    "gpt-4-turbo": Deployment(
        model="gpt-4-turbo",
        provider="azure",
        deployment_name="gpt-4-1106-preview",
        api_key="LLMAX_AZURE_OPENAI_FRANCE_KEY",
        endpoint="LLMAX_AZURE_OPENAI_FRANCE_ENDPOINT",
    ),
    "gpt-4o": Deployment(
        model="gpt-4o",
        provider="azure",
        deployment_name="gpt-4o-2024-05-13",
        api_key="LLMAX_AZURE_OPENAI_SWEDENCENTRAL_KEY",
        endpoint="LLMAX_AZURE_OPENAI_SWEDENCENTRAL_ENDPOINT",
    ),
    "mistral-large": Deployment(
        model="mistral-large",
        provider="azure",
        deployment_name="mistral-large",
        api_key="LLMAX_AZURE_MISTRAL_LARGE_KEY",
        endpoint="LLMAX_AZURE_MISTRAL_LARGE_ENDPOINT",
    ),
    "command-r-plus": Deployment(
        model="command-r-plus",
        provider="azure",
        deployment_name="command-r-plus",
        api_key="LLMAX_AZURE_COMMAND_R_PLUS_KEY",
        endpoint="LLMAX_AZURE_COMMAND_R_PLUS_ENDPOINT",
    ),
    "llama-3-70b-instruct": Deployment(
        model="llama-3-70b-instruct",
        provider="azure",
        deployment_name="llama-3-70b-instruct",
        api_key="LLMAX_AZURE_LLAMA_3_70B_INSTRUCT_KEY",
        endpoint="LLMAX_AZURE_LLAMA_3_70B_INSTRUCT_ENDPOINT",
    ),
    "whisper-1": Deployment(
        model="whisper-1",
        provider="azure",
        deployment_name="whisper-1",
        api_key="LLMAX_AZURE_OPENAI_SWEDENCENTRAL_KEY",
        endpoint="LLMAX_AZURE_OPENAI_SWEDENCENTRAL_ENDPOINT",
        api_version="2024-02-01",
    ),
}


def test_model_usage_text() -> None:
    """Test the model usage for text."""
    message = "Raconte moi une blague."
    output = "Pourquoi les plongeurs plongent-ils toujours en arrière et jamais en avant ? Parce que sinon ils tombent dans le bateau !"
    final_cost = 0.00055
    message_token = 8
    output_token = 34
    usage = ModelUsage(
        deployments["gpt-4o"],
        lambda _1, _2: True,
        CompletionUsage(
            completion_tokens=tokens.count(output),
            prompt_tokens=tokens.count(message),
            total_tokens=tokens.count(message) + tokens.count(output),
        ),
    )

    assert tokens.count(message) == message_token
    assert tokens.count(output) == output_token
    assert usage.compute_cost() == final_cost


def test_model_usage_audio() -> None:
    """Test the model usage for audio."""
    cost_audio = 0.0789
    usage = ModelUsage(
        deployments["whisper-1"],
        lambda _1, _2: True,
        audio_duration=789,
    )

    assert usage.compute_cost() == cost_audio
