from __future__ import annotations

import os

from ..config.schema import EmbedderConfig
from .base import BaseEmbedder


def get_embedder(cfg: EmbedderConfig) -> BaseEmbedder:
    if cfg.provider == "local":
        from .local import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(model=cfg.model)

    if cfg.provider == "openai":
        from .openai import OpenAIEmbedder
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"OpenAI API key not found. Set the {cfg.api_key_env!r} environment variable."
            )
        return OpenAIEmbedder(model=cfg.model, api_key=api_key)

    if cfg.provider == "anthropic":
        raise NotImplementedError(
            "Anthropic does not expose a dedicated embeddings endpoint. "
            "Use provider='local' or provider='openai' instead."
        )

    raise ValueError(f"Unknown embedder provider: {cfg.provider!r}")
