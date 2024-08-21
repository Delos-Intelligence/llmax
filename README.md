# llmax

Python package to manage most external and internal LLM APIs fluently.


# Installation

To install, run the following command:

```bash
python3 -m pip install delos-llmax
```

# How to use

You first have to define a list of `Deployment` as such, where you need to specify the endpoints, key and deployment_name. Then create the client:

```python
from llmax.clients import MultiAIClient
from llmax.models import Deployment, Model

deployments: dict[Model, Deployment] = {
        "gpt-4o": Deployment(
            model="gpt-4o",
            provider="azure",
            deployment_name="gpt-4o-2024-05-13",
            api_key=os.getenv("LLMAX_AZURE_OPENAI_SWEDENCENTRAL_KEY", ""),
            endpoint=os.getenv("LLMAX_AZURE_OPENAI_SWEDENCENTRAL_ENDPOINT", ""),
        ),
        "whisper-1": Deployment(
            model="whisper-1",
            provider="azure",
            deployment_name="whisper-1",
            api_key=os.getenv("LLMAX_AZURE_OPENAI_SWEDENCENTRAL_KEY", ""),
            endpoint=os.getenv("LLMAX_AZURE_OPENAI_SWEDENCENTRAL_ENDPOINT", ""),
            api_version="2024-02-01",
        ),
    }

client = MultiAIClient(
        deployments=deployments,
    )
```

Then you should define your input (that can be a text, image or audio, following the openai documentation for instance).

```python
messages = [
        {"role": "user", "content": "Raconte moi une blague."},
    ]
```

And finally get the response:

```python
response = client.invoke_to_str(messages, model)
print(response)
```

# Specificities

When creating the client, you can also specify two functions, *increment_usage* and *get_usage*.
The first one is **Callable[[float, Model], bool]** while the second is **Callable[[], float]**.
*increment_usage* is a function that is called after a call of the llm. The float is the price and Model, the model used. It can therefore be used to update your database. *get_usage* returns whether a condition is met. For instance, it can be a function that calls your database and returns whether the user is still active.