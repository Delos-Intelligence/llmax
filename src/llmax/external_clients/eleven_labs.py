"""ElevenLabs client."""

from typing import Any

import httpx
from elevenlabs.client import AsyncElevenLabs

from llmax.models import Deployment

Client = Any


def get_aclient(
    deployment: Deployment,
    http_client: httpx.AsyncClient | None = None,
) -> Client:
    """Get an async ElevenLabs client for the given deployment."""
    return AsyncElevenLabs(api_key=deployment.api_key, httpx_client=http_client)
