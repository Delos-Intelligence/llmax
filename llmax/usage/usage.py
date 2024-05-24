"""Defines the ModelUsage class for tracking usage statistics for a model."""

from dataclasses import dataclass, field
from typing import Any, Callable

from openai.types import CompletionUsage

from llmax.models import Deployment, Model
from llmax.utils import logger

from . import prices, tokens


@dataclass
class ModelUsage:
    """Represents usage statistics for a model including token counts and costs.

    Attributes:
        model: An instance of the Model class representing the model in use.
        increment_usage: A callable that increments usage based on cost.
        tokens_usage: A CompletionUsage instance tracking token usage.
    """

    deployment: Deployment
    increment_usage: Callable[[float, Model], bool]
    tokens_usage: CompletionUsage = field(
        default_factory=lambda: CompletionUsage(
            completion_tokens=0,
            prompt_tokens=0,
            total_tokens=0,
        ),
    )

    def __repr__(self) -> str:
        """Generates a string representation of model usage statistics.

        Returns:
            A formatted string displaying prompt, completion, and total tokens and cost.
        """
        return (
            f"\tPrompt Tokens: {self.tokens_usage.prompt_tokens}\n"
            f"\tCompletion Tokens: {self.tokens_usage.completion_tokens}\n"
            f"\tTotal Tokens: {self.tokens_usage.total_tokens}\n"
            f"Total Cost (USD): ${self.compute_cost():.6f}"
        )

    def add_tokens(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Adds token counts to the usage statistics.

        Args:
            prompt_tokens: The number of tokens used in the prompt.
            completion_tokens: The number of tokens generated as completion.
        """
        self.tokens_usage.prompt_tokens += prompt_tokens
        self.tokens_usage.completion_tokens += completion_tokens
        self.tokens_usage.total_tokens += prompt_tokens + completion_tokens

    def add_messages(self, messages: Any) -> None:
        """Counts tokens in messages and adds them to the prompt token count.

        Args:
            messages: The messages to count tokens from.
        """
        prompt_tokens = tokens.count(repr(messages))
        self.add_tokens(prompt_tokens=prompt_tokens)

    def compute_cost(self) -> float:
        """Calculates the total cost based on token usage.

        Returns:
            The total cost for the prompt and completion tokens.
        """
        cost = 0

        if prompt_tokens := self.tokens_usage.prompt_tokens:
            cost += prices.get_prompt_price(
                self.deployment.model,
                self.deployment.provider,
            ) * (prompt_tokens / 1000)

        if completion_tokens := self.tokens_usage.completion_tokens:
            cost += prices.get_completion_price(
                self.deployment.model,
                self.deployment.provider,
            ) * (completion_tokens / 1000)

        return cost

    def apply(self) -> None:
        """Applies the token usage, updating the usage statistics and logging the action."""
        message = (
            f"Tokens: {self.tokens_usage.total_tokens} "
            f"({self.tokens_usage.prompt_tokens} + {self.tokens_usage.completion_tokens}) "
            f"Cost: ${self.compute_cost():.6f}"
        )
        logger.debug(f"Applying usage for model '{self.deployment.model}'. {message}")
        _ = self.increment_usage(self.compute_cost(), self.deployment.model)
