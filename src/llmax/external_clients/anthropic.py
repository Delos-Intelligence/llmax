"""Anthropic clients."""

import boto3
from dateutil import parser
import json
from typing import Any, Dict, Generator, Optional

from llmax.external_clients.exceptions import ProviderNotFoundError
from llmax.models import Deployment, Model
from llmax.external_clients.universal_client.chat_completion import ChatCompletion
from llmax.external_clients.universal_client.chat_completion_chunk import ChatCompletionChunk
from llmax.external_clients.universal_client.chat_completion_message import ChatCompletionAssistantMessage, ChatCompletionSystemMessage, ChatCompletionUserMessage
from llmax.external_clients.universal_client.client import UniversalClient
from llmax.messages import Messages

Client = Any


MAPPING_FINISH_REASON = {
    "end_turn": "stop"
}


def client_creation_anthropic(aws_key: str, aws_secret_key: str, region_name: str) -> Any:
    return boto3.Session(
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region_name
    ).client('bedrock-runtime')


def anthropic_parsing(response: Dict[str, Any], stream: bool = True) -> Optional[ChatCompletion] | Optional[Generator[ChatCompletionChunk, None, None]]:
    try:
        request_id = response["ResponseMetadata"]["RequestId"]
        created = int(parser.parse(response["ResponseMetadata"]["HTTPHeaders"]["date"]).timestamp())
    except Exception as e:
        print(e)
        return None
    
    if stream:
        content = None
        model = None
        role = None
        prompt_tokens = None
        completion_tokens = None
        stop_reason = None
        to_yield = True

        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"]) # type: ignore
            if chunk["type"] == "message_start":
                model = chunk["message"]["model"]
                role = chunk["message"]["role"]
                content = ""
                to_yield = True
            elif chunk["type"] == "content_block_delta":
                content = chunk["delta"]["text"]
                to_yield = True
            elif chunk["type"] == "content_block_stop":
                to_yield = False
            elif chunk["type"] == "message_delta":
                stop_reason = MAPPING_FINISH_REASON[chunk["delta"]["stop_reason"]]
                content = ""
                to_yield = True
            elif chunk["type"] == "message_stop":
                prompt_tokens = chunk["amazon-bedrock-invocationMetrics"]["inputTokenCount"]
                completion_tokens = chunk["amazon-bedrock-invocationMetrics"]["outputTokenCount"]
                content = ""
                to_yield = True

            if to_yield:
                try:
                    chat_completion_chunk = {
                        "id": request_id,
                        "created": created,
                        "model": model,
                        "usage": {
                            "completion_tokens": completion_tokens,
                            "prompt_tokens": prompt_tokens,
                            "total_tokens": completion_tokens + prompt_tokens if completion_tokens and prompt_tokens else None
                        },
                        "choices": [
                            {
                                "delta": {
                                    "content": content,
                                    "role": role,
                                },
                                "index": 0,
                                "finish_reason": stop_reason,
                            }
                        ]
                    }
                    yield ChatCompletionChunk.model_validate(chat_completion_chunk)
                except Exception as e:
                    print(e)
                    return None
            
    else:
        try:
            data = json.loads(response["body"].read())
            input_tokens = response["ResponseMetadata"]["HTTPHeaders"]["x-amzn-bedrock-input-token-count"]
            completion_tokens = response["ResponseMetadata"]["HTTPHeaders"]["x-amzn-bedrock-output-token-count"]
            chat_completion = {
                "id": response["ResponseMetadata"]["RequestId"],
                "created": int(parser.parse(response["ResponseMetadata"]["HTTPHeaders"]["date"]).timestamp()),
                "model": data["model"],
                "usage": {
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": input_tokens,
                    "total_tokens": completion_tokens + input_tokens
                },
                "choices": [
                    {
                        "finish_reason": MAPPING_FINISH_REASON[data["stop_reason"]],
                        "index": i,
                        "message": {"content": e["text"]},
                    } for i, e in enumerate(data["content"])
                ]
            }
            return ChatCompletion.model_validate(chat_completion)
        except Exception as e:
            print(e)
            return None

def completion_call_anthropic(client: Any, messages: Messages, model: Model, stream: bool = False, *args, **kwargs) -> Optional[ChatCompletion] | Optional[Generator[ChatCompletionChunk, None, None]]:
    try:
        counter = 0
        for message in messages:
            try:
                ChatCompletionAssistantMessage.model_validate(message)
                counter += 1
            except Exception:
                pass
            try:
                ChatCompletionSystemMessage.model_validate(message)
                counter += 1
            except Exception:
                pass
            try:
                ChatCompletionUserMessage.model_validate(message)
                counter += 1
            except Exception:
                pass
        if counter < len(messages):
            raise
    except Exception:
        return None
    
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": messages
    }
    if stream:
        response_stream = client.invoke_model_with_response_stream(modelId=model, body=json.dumps(body))
        return anthropic_parsing(response_stream)
    else:
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
                region_name=deployment.region
            )
        case _:
            raise ProviderNotFoundError(deployment)


# def get_aclient(deployment: Deployment) -> Client:
#     """Get an async client for the given deployment."""
#     match deployment.provider:
#         case "aws-bedrock":
#             return AsyncAzureOpenAI(
#                 api_key=deployment.api_key,
#                 api_version=deployment.api_version,
#                 azure_endpoint=deployment.endpoint,
#             )
#         case _:
#             raise ProviderNotFoundError(deployment)
