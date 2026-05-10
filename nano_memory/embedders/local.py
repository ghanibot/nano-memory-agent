from __future__ import annotations

from .base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install it with: pip install sentence-transformers"
            ) from e
        self._model = SentenceTransformer(model)
        self._model_name = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def cost_per_1k_tokens(self) -> float:
        return 0.0
