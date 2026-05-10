from __future__ import annotations

import json
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..store.atomic_writer import atomic_write


@dataclass
class _ModelStats:
    calls: int = 0
    tokens_in: int = 0
    cost_usd: float = 0.0


class EmbedCostTracker:
    """Tracks embedding API calls, token usage, and USD cost.

    Stats are persisted to a JSON file in the store directory so they
    survive across process restarts (global cost accumulates forever;
    session cost resets each instantiation).
    """

    def __init__(self, store_path: str, budget_usd: float = 10.0) -> None:
        self._lock = threading.Lock()
        self._session: dict[str, _ModelStats] = defaultdict(_ModelStats)
        self._budget_usd = budget_usd
        self._stats_path = Path(store_path).expanduser() / "cost_stats.json"
        self._global: dict[str, _ModelStats] = defaultdict(_ModelStats)
        self._load_global()

    def record(self, model: str, tokens_in: int, cost_per_1k: float) -> None:
        cost = (tokens_in / 1000) * cost_per_1k
        with self._lock:
            s = self._session[model]
            s.calls += 1
            s.tokens_in += tokens_in
            s.cost_usd += cost

            g = self._global[model]
            g.calls += 1
            g.tokens_in += tokens_in
            g.cost_usd += cost

            self._persist()

    def total_usd(self, scope: str = "session") -> float:
        stats = self._session if scope == "session" else self._global
        with self._lock:
            return sum(s.cost_usd for s in stats.values())

    def report(self) -> dict:
        with self._lock:
            session_rows = {
                model: {"calls": s.calls, "tokens_in": s.tokens_in, "cost_usd": round(s.cost_usd, 6)}
                for model, s in self._session.items()
            }
            global_rows = {
                model: {"calls": g.calls, "tokens_in": g.tokens_in, "cost_usd": round(g.cost_usd, 6)}
                for model, g in self._global.items()
            }
        return {
            "session": session_rows,
            "global": global_rows,
            "session_total_usd": round(self.total_usd("session"), 6),
            "global_total_usd": round(self.total_usd("global"), 6),
            "budget_usd": self._budget_usd,
            "budget_remaining_usd": round(self._budget_usd - self.total_usd("global"), 6),
        }

    def check_budget(self, kill_on_exceed: bool = False, alert_at_percent: float = 0.8) -> None:
        total = self.total_usd("global")
        alert_threshold = self._budget_usd * alert_at_percent
        if total >= self._budget_usd:
            msg = (
                f"Budget exceeded: ${total:.4f} spent of ${self._budget_usd:.2f} limit."
            )
            if kill_on_exceed:
                raise RuntimeError(msg)
        elif total >= alert_threshold:
            import warnings
            warnings.warn(
                f"Budget alert: ${total:.4f} spent ({total / self._budget_usd:.0%} of "
                f"${self._budget_usd:.2f} budget).",
                stacklevel=3,
            )

    def _persist(self) -> None:
        data = {
            model: {"calls": g.calls, "tokens_in": g.tokens_in, "cost_usd": g.cost_usd}
            for model, g in self._global.items()
        }
        atomic_write(self._stats_path, json.dumps(data, indent=2).encode())

    def _load_global(self) -> None:
        if not self._stats_path.exists():
            return
        try:
            data = json.loads(self._stats_path.read_text())
            for model, row in data.items():
                s = self._global[model]
                s.calls = row.get("calls", 0)
                s.tokens_in = row.get("tokens_in", 0)
                s.cost_usd = row.get("cost_usd", 0.0)
        except (json.JSONDecodeError, KeyError):
            pass
