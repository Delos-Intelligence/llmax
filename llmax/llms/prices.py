from llmax.llms.models import Model

PROMPT_PRICES_PER_1K: dict[Model, float] = {
    "gpt-4": 0.01,
    "gpt-3.5": 0.0005,
    "text-embedding-3-small": 0.00002,
    "text-embedding-3-large": 0.00013,
    "ada-v2": 0.00010,
}

COMPLETION_PRICES_PER_1K: dict[Model, float] = {
    "gpt-4": 0.03,
    "gpt-3.5": 0.0015,
}
