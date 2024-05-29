"""Main."""

import os

from dotenv import load_dotenv

from llmax.clients import MultiAIClient
from llmax.models import Deployment, Model
from llmax.utils import logger

load_dotenv()


def main() -> None:
    """Main."""
    deployments: dict[Model, Deployment] = {
        "gpt-4-turbo": Deployment(
            model="gpt-4-turbo",
            provider="azure",
            deployment_name="gpt-4-1106-preview",
            api_key=os.getenv("LLMAX_AZURE_OPENAI_KEY", ""),
            endpoint=os.getenv("LLMAX_AZURE_OPENAI_ENDPOINT", ""),
        ),
        "mistral-large": Deployment(
            model="mistral-large",
            provider="azure",
            deployment_name="mistral-large",
            api_key=os.getenv("LLMAX_AZURE_MISTRAL_LARGE_KEY", ""),
            endpoint=os.getenv("LLMAX_AZURE_MISTRAL_LARGE_ENDPOINT", ""),
        ),
        "command-r-plus": Deployment(
            model="command-r-plus",
            provider="azure",
            deployment_name="command-r-plus",
            api_key=os.getenv("LLMAX_AZURE_COMMAND_R_PLUS_KEY", ""),
            endpoint=os.getenv("LLMAX_AZURE_COMMAND_R_PLUS_ENDPOINT", ""),
        ),
        "llama-3-70b-instruct": Deployment(
            model="llama-3-70b-instruct",
            provider="azure",
            deployment_name="llama-3-70b-instruct",
            api_key=os.getenv("LLMAX_AZURE_LLAMA_3_70B_INSTRUCT_KEY", ""),
            endpoint=os.getenv("LLMAX_AZURE_LLAMA_3_70B_INSTRUCT_ENDPOINT", ""),
        ),
    }
    client = MultiAIClient(
        deployments=deployments,
    )
    messages = [
        {"role": "user", "content": "Quel est le meilleur restaurant de Paris?"},
    ]
    MODEL = "llama-3-70b-instruct"
    response = client.stream(messages, MODEL)
    logger.info(f"Chatting with {MODEL} model...")
    for chunk in response:
        print(chunk.choices[0].delta.content, end="")


if __name__ == "__main__":
    main()
