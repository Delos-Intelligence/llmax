import argparse
import os

from dotenv import load_dotenv

from llmax.clients import MultiAIClient
from llmax.models import Deployment, Model
from llmax.utils import logger

load_dotenv()


def main(model: Model, question: str) -> None:
    """Main."""
    deployments: dict[Model, Deployment] = {
        "gpt-4-turbo": Deployment(
            model="gpt-4-turbo",
            provider="azure",
            deployment_name="gpt-4-1106-preview",
            api_key=os.getenv("LLMAX_AZURE_OPENAI_FRANCE_KEY", ""),
            endpoint=os.getenv("LLMAX_AZURE_OPENAI_FRANCE_ENDPOINT", ""),
        ),
        "gpt-4o": Deployment(
            model="gpt-4o",
            provider="azure",
            deployment_name="gpt-4o-2024-05-13",
            api_key=os.getenv("LLMAX_AZURE_OPENAI_SWEDENCENTRAL_KEY", ""),
            endpoint=os.getenv("LLMAX_AZURE_OPENAI_SWEDENCENTRAL_ENDPOINT", ""),
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
        "whisper-1": Deployment(
            model="whisper-1",
            provider="azure",
            deployment_name="whisper-1",
            api_key=os.getenv("LLMAX_AZURE_OPENAI_SWEDENCENTRAL_KEY", ""),
            endpoint=os.getenv("LLMAX_AZURE_OPENAI_SWEDENCENTRAL_ENDPOINT", ""),
        ),
    }
    client = MultiAIClient(
        deployments=deployments,
    )
    messages = [
        {"role": "user", "content": question},
    ]
    logger.info(f"Chatting with {model} model...")
    logger.info(deployments[model].endpoint)

    response = client.invoke_to_str(messages, model)
    print(response)

    response = client.stream(messages, model)
    for chunk in response:
        print(chunk.choices[0].delta.content, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Chatbot")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The model to speak with (e.g., 'gpt-4o', 'gpt-4-turbo', etc.)",
    )
    parser.add_argument(
        "--question", type=str, required=True, help="The question to ask the model"
    )

    args = parser.parse_args()
    main(args.model, args.question)
