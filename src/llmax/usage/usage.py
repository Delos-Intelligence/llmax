"""Defines the ModelUsage class for tracking usage statistics for a model."""

import math
from dataclasses import dataclass, field
from typing import Callable, Literal

from openai.types import CompletionUsage

from llmax.messages.messages import Messages
from llmax.models import Deployment, Model
from llmax.models.models import AUDIO, IMAGE
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
    image_information: float = 0.0
    tts_information: float = 0.0

    def __repr__(self) -> str:
        """Generates a string representation of model usage statistics.

        Returns:
            A formatted string displaying prompt, completion, and total tokens and cost.
        """
        cost = self.compute_cost()
        cost_message = f"Total Cost (USD): ${cost:.6f}"

        if self.deployment.model in AUDIO:
            return f"\tAudio Duration: {self.audio_duration} seconds\n {cost_message}"

        if self.deployment.model in IMAGE:
            return f"\tImage generation : ~{self.image_information} images\n {cost_message}"

        return (
            f"\tPrompt Tokens: {self.tokens_usage.prompt_tokens}\n"
            f"\tCompletion Tokens: {self.tokens_usage.completion_tokens}\n"
            f"\tTotal Tokens: {self.tokens_usage.total_tokens}\n"
            f"{cost_message}"
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

    def add_image(
        self,
        quality: Literal["standard", "hd"],
        size: Literal["1024x1024", "1024x1792", "1792x1024"],
        n: int = 1,
    ) -> None:
        """Adds image pricing to the usage statistics for image generation models.

        Args:
            quality: The quality of the generated images.
            size: The size of the generated images.
            n: The number of generated images.
        """
        count = 1
        if quality == "hd":
            count += 1
        if size != "1024x1024":
            count += 1
        self.image_information += count * n

    def add_tts(self, text: str) -> None:
        """Records the usage of TTS.

        Args:
            text: The text to be TTS'ed.
        """
        self.tts_information += len(text)

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

        if self.image_information:
            price = prices.get_tti_price(dep.model, dep.provider)
            cost += price * self.image_information

        if self.tts_information:
            price = prices.get_tts_price(dep.model, dep.provider)
            cost += price * self.tts_information

        return cost

    def apply(self, operation: str = "") -> float:
        """Applies the token usage, updating the usage statistics and logging the action."""
        cost = self.compute_cost()
        cost_message = f"Total Cost (USD): ${cost:.6f}"

        if self.deployment.model in AUDIO:
            message = f"Audio Duration: {self.audio_duration} seconds {cost_message}"
        elif self.deployment.model in IMAGE:
            message = (
                f"Image generation : ~{self.image_information} images "
                f"{cost_message}"
            )

        else:
            message = (
                f"Tokens: {self.tokens_usage.total_tokens} "
                f"({self.tokens_usage.prompt_tokens} + {self.tokens_usage.completion_tokens}) "
                f"{cost_message}"
            )

        logger.debug(
            f"[bold purple][LLMAX][/bold purple] Applying usage for model '{self.deployment.model}'. {message}",
        )
        self.increment_usage(cost, self.deployment.model, operation)
        return cost
