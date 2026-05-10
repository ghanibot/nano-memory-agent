from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ..config.schema import MemoryConfig
from ..cost.tracker import EmbedCostTracker
from ..embedders.base import BaseEmbedder
from ..embedders.factory import get_embedder
from ..store.schema import MemoryRecord, MemoryType
from ..store.sqlite_store import SQLiteStore


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return max(1, len(text) // 4)


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping character-level chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += chunk_size - chunk_overlap
    return chunks


class Memory:
    """Main API for nano-memory. Save, search, recall, and manage memories."""

    def __init__(self, config: MemoryConfig | str | None = None) -> None:
        if config is None:
            self._cfg = MemoryConfig()
        elif isinstance(config, str):
            self._cfg = _load_yaml_config(config)
        else:
            self._cfg = config

        self._store = SQLiteStore(self._cfg.store.path)
        self._embedder: BaseEmbedder = get_embedder(self._cfg.embedder)
        self._tracker = EmbedCostTracker(
            store_path=self._cfg.store.path,
            budget_usd=self._cfg.budget.max_cost_usd,
        )

    @property
    def namespace(self) -> str:
        return self._cfg.namespace

    def switch_namespace(self, namespace: str) -> None:
        self._cfg = self._cfg.model_copy(update={"namespace": namespace})

    def save(
        self,
        text: str,
        type: MemoryType = "fact",
        metadata: dict[str, Any] | None = None,
    ) -> str | list[str]:
        """Embed and persist text. Returns record id(s). Context type is chunked."""
        if metadata is None:
            metadata = {}

        self._tracker.check_budget(
            kill_on_exceed=self._cfg.budget.kill_on_exceed,
            alert_at_percent=self._cfg.budget.alert_at_percent,
        )

        if type == "context":
            return self._save_chunks(text, metadata)

        embedding = self._embedder.embed_one(text)
        self._track_embedding(text)

        record = MemoryRecord(
            text=text,
            namespace=self._cfg.namespace,
            type=type,
            embedding=embedding,
            metadata=metadata,
        )
        self._store.save(record)
        return record.id

    def _save_chunks(self, text: str, metadata: dict[str, Any]) -> list[str]:
        chunks = _chunk_text(text, self._cfg.chunk_size, self._cfg.chunk_overlap)
        embeddings = self._embedder.embed(chunks)
        self._track_embedding(text)

        ids: list[str] = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_meta = {**metadata, "chunk_index": i, "total_chunks": len(chunks)}
            record = MemoryRecord(
                text=chunk,
                namespace=self._cfg.namespace,
                type="context",
                embedding=embedding,
                metadata=chunk_meta,
            )
            self._store.save(record)
            ids.append(record.id)
        return ids

    def search(
        self,
        query: str,
        top_k: int | None = None,
        type_filter: MemoryType | None = None,
        cross_namespace: bool = False,
    ) -> list[MemoryRecord]:
        """Semantic search. Returns records sorted by cosine similarity descending."""
        k = top_k if top_k is not None else self._cfg.top_k
        query_embedding = self._embedder.embed_one(query)
        self._track_embedding(query)
        return self._store.search(
            query_embedding=query_embedding,
            namespace=self._cfg.namespace,
            top_k=k,
            cross_namespace=cross_namespace,
            type_filter=type_filter,
        )

    def recall(self, query: str, top_k: int | None = None) -> str:
        """Search and return a formatted string ready for LLM context injection."""
        records = self.search(query, top_k=top_k)
        if not records:
            return "[No relevant memories found]"
        lines = [f"[Memory {i + 1} | type={r.type} | score={r.score:.3f}]:\n{r.text}"
                 for i, r in enumerate(records)]
        return "\n\n".join(lines)

    def forget(self, record_id: str) -> bool:
        """Delete a memory by id. Returns True if it existed."""
        return self._store.delete(record_id)

    def list(self, type_filter: MemoryType | None = None) -> list[MemoryRecord]:
        """List all memories in the current namespace."""
        return self._store.list_namespace(self._cfg.namespace, type_filter=type_filter)

    def clear(self) -> int:
        """Delete all memories in the current namespace. Returns count deleted."""
        return self._store.clear_namespace(self._cfg.namespace)

    def export(self, path: str) -> None:
        """Export all memories in the current namespace to a JSON file."""
        records = self._store.list_namespace(self._cfg.namespace)
        data = [
            {
                "id": r.id,
                "namespace": r.namespace,
                "type": r.type,
                "text": r.text,
                "metadata": r.metadata,
                "created_at": r.created_at,
            }
            for r in records
        ]
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def cost_report(self) -> dict:
        return self._tracker.report()

    def _track_embedding(self, text: str) -> None:
        tokens = _estimate_tokens(text)
        self._tracker.record(
            model=self._embedder.model_name,
            tokens_in=tokens,
            cost_per_1k=self._embedder.cost_per_1k_tokens,
        )


def _load_yaml_config(path: str) -> MemoryConfig:
    raw = Path(path).expanduser().read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    return MemoryConfig.model_validate(data)


def load_config(path: str) -> MemoryConfig:
    return _load_yaml_config(path)
