from llmax.llms.models import Model

PROMPT_PRICES_PER_1K: dict[Model, float] = {
    "azure-gpt-4": 0.01,
    "azure-gpt-3.5": 0.0005,
    "azure-ada-v2": 0.00010,
    "gpt-4": 0.01,
    "gpt-3.5": 0.0005,
    "ada-v2": 0.00010,
    "text-embedding-3-small": 0.00002,
    "text-embedding-3-large": 0.00013,
}

COMPLETION_PRICES_PER_1K: dict[Model, float] = {
    "azure-gpt-4": 0.03,
    "azure-gpt-3.5": 0.0015,
    "gpt-4": 0.03,
    "gpt-3.5": 0.0015,
}
