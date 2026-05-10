from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return a list of float vectors."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier used for cost tracking."""
        ...

    @property
    @abstractmethod
    def cost_per_1k_tokens(self) -> float:
        """Cost in USD per 1,000 tokens. Zero for local models."""
        ...

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
