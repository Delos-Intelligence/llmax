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
            api_key=os.getenv("LLMAX_AZURE_MISTRAL_KEY", ""),
            endpoint=os.getenv("LLMAX_AZURE_MISTRAL_ENDPOINT", ""),
        ),
    }
    client = MultiAIClient(
        deployments=deployments,
    )
    messages = [
        {"role": "user", "content": "Quel est le meilleur restaurant de Paris?"},
    ]
    response = client.stream(messages, "mistral-large")
    logger.info(response)
    for chunk in response:
        logger.info(chunk)


if __name__ == "__main__":
    main()
