"""Main."""

import os

from dotenv import load_dotenv

from llmax.clients import MultiAIClient
from llmax.models import Deployment
from llmax.utils import logger

load_dotenv()


def main() -> None:
    """Main."""
    api_key = os.getenv("LLMAX_AZURE_OPENAI_KEY", "")
    endpoint = os.getenv("LLMAX_AZURE_OPENAI_ENDPOINT", "")
    logger.info(f"api_key: {api_key}")
    logger.info(f"endpoint: {endpoint}")
    client = MultiAIClient(
        deployments={
            "gpt-4-turbo": Deployment(
                model="gpt-4-turbo",
                provider="azure",
                deployment_name="gpt-4-1106-preview",
                api_key=api_key,
                endpoint=endpoint,
            ),
        },
    )
    messages = [
        {"role": "user", "content": "Quel est le meilleur restaurant de Paris?"},
    ]
    response = client.stream(messages, "gpt-4-turbo")
    logger.info(response)
    for chunk in response:
        logger.info(chunk)


if __name__ == "__main__":
    main()
