"""Main file to launch the module."""

import os

from dotenv import load_dotenv

from llmax.clients import MultiAIClient
from llmax.models import Deployment, Model
from llmax.models.models import AUDIO
from llmax.utils import logger

load_dotenv()


def main(model: Model, question: str) -> None:
    """Main."""
    deployments: dict[Model, Deployment] = {
        "claude-3.5-sonnet": Deployment(
            model="claude-3.5-sonnet",
            provider="aws-bedrock",
            deployment_name="anthropic.claude-3-5-sonnet-20240620-v1:0",
            api_key=os.getenv("LLMAX_AWS_BEDROCK_ANTHROPIC_GERMAN_KEY", ""),
            project_id=os.getenv("LLMAX_AWS_BEDROCK_ANTHROPIC_GERMAN_PROJECT_ID", ""),
            api_version="bedrock-2023-05-31",
        ),
    }
    client = MultiAIClient(
        deployments=deployments,
    )
    messages = [{"role": "user", "content": question}]

    logger.info(f"Chatting with {model} model...")
    logger.info(deployments[model].endpoint)

    response = client.invoke_to_str(messages, model)
    logger.info(response)

    response_stream = client.stream(messages, model)
    logger.info(response_stream)

if __name__ == "__main__":
    main("claude-3.5-sonnet", "Donne moi la recette du boeuf bourguignon")
