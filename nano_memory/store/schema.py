from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

MemoryType = Literal["fact", "episode", "preference", "context"]


@dataclass
class MemoryRecord:
    text: str
    namespace: str
    type: MemoryType = "fact"
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    score: float = 0.0
