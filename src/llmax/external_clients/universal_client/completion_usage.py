from typing import Optional

from pydantic import BaseModel

__all__ = ["CompletionUsage"]


class CompletionUsage(BaseModel):
    completion_tokens: Optional[int] = None
    """Number of tokens in the generated completion."""

    prompt_tokens: Optional[int] = None
    """Number of tokens in the prompt."""

    total_tokens: Optional[int] = None
    """Total number of tokens used in the request (prompt + completion)."""
