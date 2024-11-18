"""Defines the ModelUsage class for tracking usage statistics for a model."""

import math
from dataclasses import dataclass, field
from typing import Callable

from openai.types import CompletionUsage

from llmax.messages.messages import Messages
from llmax.models import Deployment, Model
from llmax.models.models import AUDIO
from llmax.utils import logger

from . import prices, tokens


@dataclass
class ModelUsage:
    """Represents usage statistics for a model including token counts and costs.

    Attributes:
        model: An instance of the Model class representing the model in use.
        increment_usage: A callable that increments usage based on cost.
        tokens_usage: A CompletionUsage instance tracking token usage.
        audio_duration: The total processed audio duration (in seconds).
    """

    deployment: Deployment
    increment_usage: Callable[[float, Model, str], bool]
    tokens_usage: CompletionUsage = field(
        default_factory=lambda: CompletionUsage(
            completion_tokens=0,
            prompt_tokens=0,
            total_tokens=0,
        ),
    )
    audio_duration: float = 0.0

    def __repr__(self) -> str:
        """Generates a string representation of model usage statistics.

        Returns:
            A formatted string displaying prompt, completion, and total tokens and cost.
        """
        if self.deployment.model in AUDIO:
            return (
                f"\tAudio Duration: {self.audio_duration} seconds\n"
                f"Total Cost (USD): ${self.compute_cost():.6f}"
            )

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

    def add_messages(self, messages: Messages) -> None:
        """Counts tokens in messages and adds them to the prompt token count.

        Args:
            messages: The messages to count tokens from.
        """
        prompt_tokens = tokens.count(repr(messages))
        self.add_tokens(prompt_tokens=prompt_tokens)

    def add_audio_duration(self, duration: float) -> None:
        """Adds audio duration to the usage statistics for audio models.

        Args:
            duration: The duration of the audio in seconds.
        """
        self.audio_duration += duration

    def compute_cost(self) -> float:
        """Calculates the total cost based on token usage.

        Returns:
            The total cost for the prompt and completion tokens.
        """
        cost: float = 0
        dep = self.deployment

        if self.audio_duration:
            price = prices.get_stt_price(dep.model, dep.provider)
            cost += price * math.ceil(self.audio_duration) / 60

        if prompt_tokens := self.tokens_usage.prompt_tokens:
            price = prices.get_prompt_price(dep.model, dep.provider)
            cost += price * prompt_tokens / 1000

        if completion_tokens := self.tokens_usage.completion_tokens:
            price = prices.get_completion_price(dep.model, dep.provider)
            cost += price * completion_tokens / 1000

        return cost

    def apply(self, operation: str = "") -> None:
        """Applies the token usage, updating the usage statistics and logging the action."""
        if self.deployment.model in AUDIO:
            message = (
                f"Audio Duration: {self.audio_duration} seconds "
                f"Cost: ${self.compute_cost():.6f}"
            )
        else:
            message = (
                f"Tokens: {self.tokens_usage.total_tokens} "
                f"({self.tokens_usage.prompt_tokens} + {self.tokens_usage.completion_tokens}) "
                f"Cost: ${self.compute_cost():.6f}"
            )
        logger.debug(
            f"[bold purple][LLMAX][/bold purple] Applying usage for model '{self.deployment.model}'. {message}",
        )
        self.increment_usage(self.compute_cost(), self.deployment.model, operation)
