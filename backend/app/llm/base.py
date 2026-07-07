"""Provider-agnostic LLM interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    provider: str
    model: str
    cached: bool = False


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def available(self) -> bool:
        """True when the provider has a usable API key/config."""

    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    def generate(
        self, system: str, user: str, max_tokens: int = 700, temperature: float = 0.3
    ) -> LLMResult: ...
