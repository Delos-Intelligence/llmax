from mistralai.async_client import MistralAsyncClient
from mistralai.client import MistralClient
from openai import AsyncAzureOpenAI, AzureOpenAI

Client = MistralClient | AzureOpenAI
AsyncClient = MistralAsyncClient | AsyncAzureOpenAI
