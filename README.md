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

# Requêter des modèles

Le client `MultiAIClient` offre plusieurs méthodes pour interagir avec les modèles, que ce soit de manière synchrone ou asynchrone.

## Méthodes synchrones

### `invoke_to_str()`
La méthode la plus simple pour obtenir une réponse textuelle directement :

```python
response = client.invoke_to_str(
    messages=messages,
    model="gpt-4o",
    system="Tu es un assistant utile.",  # Optionnel
    delay=0.0,  # Délai entre les tentatives en cas d'erreur
    tries=1,  # Nombre de tentatives en cas de rate limit
)
print(response)  # Affiche directement le texte de la réponse
```

### `invoke()`
Retourne l'objet de réponse complet (déprécié, préférez la version asynchrone) :

```python
response = client.invoke(messages, model="gpt-4o")
print(response.choices[0].message.content)
```

## Méthodes asynchrones

### `ainvoke_to_str()`
Version asynchrone de `invoke_to_str()` :

```python
import asyncio

async def main():
    response = await client.ainvoke_to_str(
        messages=messages,
        model="gpt-4o",
        system="Tu es un assistant utile.",
    )
    print(response)

asyncio.run(main())
```

### `ainvoke()`
Version asynchrone qui retourne l'objet de réponse complet :

```python
response = await client.ainvoke(messages, model="gpt-4o")
print(response.choices[0].message.content)
```

### Streaming avec `astream()`
Pour recevoir les réponses en temps réel au fur et à mesure de leur génération :

```python
async def stream_response():
    async for chunk in client.astream(messages, model="gpt-4o"):
        if chunk.content:
            print(chunk.content, end="", flush=True)

asyncio.run(stream_response())
```

## Paramètres supplémentaires

Toutes les méthodes acceptent des paramètres supplémentaires via `**kwargs` qui sont transmis directement à l'API sous-jacente. Par exemple :

```python
response = await client.ainvoke_to_str(
    messages=messages,
    model="gpt-4o",
    temperature=0.7,  # Contrôle la créativité
    max_tokens=500,  # Limite la longueur de la réponse
    top_p=0.9,  # Contrôle la diversité
)
```

## Modèles Scaleway

Les modèles Scaleway utilisent une API compatible OpenAI, ce qui permet une intégration transparente avec `llmax`. Pour utiliser un modèle Scaleway, vous devez configurer le déploiement avec le provider `"scaleway"` et fournir soit un `endpoint` complet, soit un `project_id` (recommandé).

### Configuration d'un modèle Scaleway

**Option 1 : Utilisation avec `project_id` (recommandé)**

L'URL sera automatiquement construite comme `https://api.scaleway.ai/v1/{project_id}` :

```python
from llmax.clients import MultiAIClient
from llmax.models import Deployment, Model
import os

deployments: dict[Model, Deployment] = {
    "scaleway/llama-3.3-70b-instruct": Deployment(
        model="scaleway/llama-3.3-70b-instruct",
        provider="scaleway",
        deployment_name="llama-3.3-70b-instruct",  # Le nom du déploiement sur Scaleway
        api_key=os.getenv("SCALEWAY_API_KEY", ""),
        project_id=os.getenv("SCALEWAY_PROJECT_ID", ""),  # Recommandé
    ),
    "scaleway/qwen3-235b-a22b-instruct-2507": Deployment(
        model="scaleway/qwen3-235b-a22b-instruct-2507",
        provider="scaleway",
        deployment_name="qwen3-235b-a22b-instruct-2507",
        api_key=os.getenv("SCALEWAY_API_KEY", ""),
        project_id=os.getenv("SCALEWAY_PROJECT_ID", ""),
    ),
}

client = MultiAIClient(deployments=deployments)
```

**Option 2 : Utilisation avec `endpoint` complet (rétrocompatibilité)**

Vous pouvez également fournir un endpoint complet si vous préférez :

```python
deployments: dict[Model, Deployment] = {
    "scaleway/llama-3.3-70b-instruct": Deployment(
        model="scaleway/llama-3.3-70b-instruct",
        provider="scaleway",
        deployment_name="llama-3.3-70b-instruct",
        api_key=os.getenv("SCALEWAY_API_KEY", ""),
        endpoint=os.getenv("SCALEWAY_ENDPOINT", ""),  # Ex: https://api.scaleway.ai/v1/your-project-id
    ),
}
```

**Note** : Vous devez fournir soit `endpoint` soit `project_id`, mais pas nécessairement les deux. Si vous fournissez `project_id`, l'URL sera construite automatiquement selon la spécification OpenAPI Scaleway.

### Utilisation des modèles Scaleway

Une fois configuré, l'utilisation est identique aux autres modèles :

```python
messages = [
    {"role": "user", "content": "Explique-moi le machine learning en quelques phrases."},
]

# Utilisation synchrone
response = client.invoke_to_str(
    messages=messages,
    model="scaleway/llama-3.3-70b-instruct",
)

# Utilisation asynchrone
response = await client.ainvoke_to_str(
    messages=messages,
    model="scaleway/qwen3-235b-a22b-instruct-2507",
    temperature=0.8,
    max_tokens=300,
)
```

### Modèles Scaleway disponibles

Les modèles suivants sont supportés :
- `scaleway/qwen3-235b-a22b-instruct-2507` - Modèle Qwen 3 (235B)
- `scaleway/gpt-oss-120b` - GPT Open Source (120B)
- `scaleway/gemma-3-27b-it` - Gemma 3 (27B)
- `scaleway/whisper-large-v3` - Whisper pour la transcription audio
- `scaleway/voxtral-small-24b-2507` - Voxtral Small (24B)
- `scaleway/mistral-small-3.2-24b-instruct-2506` - Mistral Small 3.2 (24B)
- `scaleway/llama-3.3-70b-instruct` - Llama 3.3 (70B)
- `scaleway/deepseek-r1-distill-llama-70b` - DeepSeek R1 Distill (70B)

### Note spéciale pour le modèle Qwen

Le modèle `scaleway/qwen3-235b-a22b-instruct-2507` nécessite un format spécial pour les réponses JSON. Si vous utilisez `response_format={"type": "json_object"}`, il sera automatiquement transformé en format `json_schema` requis par Scaleway :

```python
response = await client.ainvoke_to_str(
    messages=messages,
    model="scaleway/qwen3-235b-a22b-instruct-2507",
    response_format={"type": "json_object"},  # Transformé automatiquement
)
```

# Specificities

When creating the client, you can also specify two functions, *increment_usage* and *get_usage*.
The first one is **Callable[[float, Model], bool]** while the second is **Callable[[], float]**.
*increment_usage* is a function that is called after a call of the llm. The float is the price and Model, the model used. It can therefore be used to update your database. *get_usage* returns whether a condition is met. For instance, it can be a function that calls your database and returns whether the user is still active.