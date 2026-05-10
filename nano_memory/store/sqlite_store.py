from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .schema import MemoryRecord, MemoryType

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    namespace   TEXT NOT NULL,
    type        TEXT NOT NULL,
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_namespace ON memories (namespace);
"""


def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a single query vector and a row matrix."""
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    normed = matrix / norms
    return normed @ query_norm


class SQLiteStore:
    def __init__(self, store_path: str) -> None:
        db_dir = Path(store_path).expanduser()
        db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = db_dir / "memories.db"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self._db_path), check_same_thread=False, isolation_level=None
        )
        self._conn.executescript(_CREATE_TABLE)

    def save(self, record: MemoryRecord) -> None:
        embedding_bytes = np.array(record.embedding, dtype=np.float32).tobytes()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO memories "
                "(id, namespace, type, text, embedding, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.namespace,
                    record.type,
                    record.text,
                    embedding_bytes,
                    json.dumps(record.metadata),
                    record.created_at,
                ),
            )

    def get_by_id(self, record_id: str) -> MemoryRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, namespace, type, text, embedding, metadata, created_at "
                "FROM memories WHERE id = ?",
                (record_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def delete(self, record_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (record_id,))
        return cursor.rowcount > 0

    def list_namespace(
        self, namespace: str, type_filter: MemoryType | None = None
    ) -> list[MemoryRecord]:
        if type_filter is not None:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT id, namespace, type, text, embedding, metadata, created_at "
                    "FROM memories WHERE namespace = ? AND type = ? ORDER BY created_at",
                    (namespace, type_filter),
                ).fetchall()
        else:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT id, namespace, type, text, embedding, metadata, created_at "
                    "FROM memories WHERE namespace = ? ORDER BY created_at",
                    (namespace,),
                ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def clear_namespace(self, namespace: str) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM memories WHERE namespace = ?", (namespace,)
            )
        return cursor.rowcount

    def search(
        self,
        query_embedding: list[float],
        namespace: str,
        top_k: int,
        cross_namespace: bool = False,
        type_filter: MemoryType | None = None,
    ) -> list[MemoryRecord]:
        query_vec = np.array(query_embedding, dtype=np.float32)

        conditions: list[str] = []
        params: list[Any] = []

        if not cross_namespace:
            conditions.append("namespace = ?")
            params.append(namespace)

        if type_filter is not None:
            conditions.append("type = ?")
            params.append(type_filter)

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, namespace, type, text, embedding, metadata, created_at "
                f"FROM memories {where_clause}",
                params,
            ).fetchall()

        if not rows:
            return []

        records = [self._row_to_record(r) for r in rows]
        matrix = np.stack(
            [np.array(r.embedding, dtype=np.float32) for r in records], axis=0
        )
        scores = _cosine_similarity(query_vec, matrix)

        for record, score in zip(records, scores):
            record.score = float(score)

        records.sort(key=lambda r: r.score, reverse=True)
        return records[:top_k]

    @staticmethod
    def _row_to_record(row: tuple) -> MemoryRecord:
        record_id, namespace, rtype, text, embedding_blob, metadata_json, created_at = row
        embedding = np.frombuffer(embedding_blob, dtype=np.float32).tolist()
        metadata = json.loads(metadata_json)
        return MemoryRecord(
            id=record_id,
            namespace=namespace,
            type=rtype,
            text=text,
            embedding=embedding,
            metadata=metadata,
            created_at=created_at,
        )

    def close(self) -> None:
        self._conn.close()
