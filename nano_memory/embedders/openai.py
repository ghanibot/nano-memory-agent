from __future__ import annotations

from .base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    PRICING: dict[str, float] = {
        "text-embedding-3-small": 0.02 / 1_000_000,
        "text-embedding-3-large": 0.13 / 1_000_000,
        "text-embedding-ada-002": 0.10 / 1_000_000,
    }

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "openai package is required for OpenAI embeddings. "
                "Install it with: pip install 'nano-memory[openai]'"
            ) from e
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(input=texts, model=self._model)
        return [d.embedding for d in resp.data]

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def cost_per_1k_tokens(self) -> float:
        return self.PRICING.get(self._model, 0.0) * 1000
