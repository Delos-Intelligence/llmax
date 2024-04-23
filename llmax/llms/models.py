from typing import Literal

AzureModel = Literal[
    "azure-gpt-4-0613",
    "azure-gpt-4-1106-preview",
    # "azure-gpt-4-0125-preview",
    # "azure-gpt-4-vision-preview",
    "azure-gpt-4-32k-0613",
    "azure-gpt-35-turbo-0301",
    "azure-gpt-35-turbo-0613",
    "azure-gpt-35-turbo-1106",
    "azure-gpt-35-turbo-0125",
    "azure-gpt-35-turbo-16k-0613",



    "azure-gpt-35-turbo",
    "azure-gpt-35-turbo-16k",
    "azure-gpt-35-turbo-instruct",
    "azure-gpt-4-32k",
    "azure-text-embedding-ada-002",
    "azure-text-embedding-3-large",
    "azure-text-embedding-3-small",
]

OpenAIModel = Literal[""]

MistralModel = Literal[""]

Model = AzureModel | OpenAIModel | MistralModel



gpt-4-0613 	gpt-4-1106-Preview 	gpt-4-0125-Preview 	gpt-4, vision-preview 	gpt-4-32k, 0613 	gpt-35-turbo, 0301 	gpt-35-turbo, 0613 	gpt-35-turbo, 1106 	gpt-35-turbo, 0125 	gpt-35-turbo-16k, 0613 	gpt-35-turbo-instruct, 0914 	text-embedding-ada-002, 1 	text-embedding-ada-002, 2 	text-embedding-3-small, 1 	text-embedding-3-large, 1 	babbage-002, 1 	dall-e-3, 3.0 	davinci-002, 1 	tts, 001 	tts-hd, 001 	whisper, 001
