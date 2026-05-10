from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EmbedderConfig(BaseModel):
    provider: Literal["local", "openai", "anthropic"] = "local"
    model: str = "all-MiniLM-L6-v2"
    api_key_env: str = "OPENAI_API_KEY"


class BudgetConfig(BaseModel):
    max_cost_usd: float = 10.0
    alert_at_percent: float = 0.8
    kill_on_exceed: bool = False


class StoreConfig(BaseModel):
    path: str = "~/.nano-memory"


class MemoryConfig(BaseModel):
    namespace: str = "default"
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64
