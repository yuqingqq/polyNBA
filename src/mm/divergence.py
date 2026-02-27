"""Divergence tracking and EMA computation for the market making engine.

Tracks per-token divergence between fair value and market mid, computes
exponential moving averages, and counts threshold breaches.  Thread-safe
via its own lock.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict


def compute_ema(previous: float, new: float, alpha: float) -> float:
    """Compute an exponential moving average update.

    Args:
        previous: Previous EMA value.
        new: New observation.
        alpha: Weight on the new observation (0 = ignore new, 1 = ignore previous).

    Returns:
        Updated EMA value.
    """
    return alpha * new + (1.0 - alpha) * previous


@dataclass
class DivergenceStats:
    """Per-token divergence statistics."""

    token_id: str
    current_bps: float = 0.0
    ema_bps: float = 0.0
    min_bps: float = float("inf")
    max_bps: float = 0.0
    count: int = 0
    max_breach_count: int = 0


class DivergenceTracker:
    """Records per-token divergence observations and computes statistics.

    Usage::

        tracker = DivergenceTracker(max_bps=1500)
        tracker.record("tok_a", current_bps=200.0, ema_bps=180.0)
        print(tracker.format_summary())

    Parameters:
        max_bps: The ``divergence_max_bps`` threshold used for breach counting.
    """

    def __init__(self, max_bps: float = 1500.0) -> None:
        self._max_bps = max_bps
        self._lock = threading.Lock()
        self._stats: Dict[str, DivergenceStats] = {}

    def record(self, token_id: str, current_bps: float, ema_bps: float) -> None:
        """Record a divergence observation.

        Args:
            token_id: Token identifier.
            current_bps: Instantaneous divergence in basis points.
            ema_bps: Current EMA divergence in basis points.
        """
        with self._lock:
            stats = self._stats.get(token_id)
            if stats is None:
                stats = DivergenceStats(token_id=token_id)
                self._stats[token_id] = stats

            stats.current_bps = current_bps
            stats.ema_bps = ema_bps
            stats.count += 1
            stats.min_bps = min(stats.min_bps, current_bps)
            stats.max_bps = max(stats.max_bps, current_bps)
            if current_bps >= self._max_bps:
                stats.max_breach_count += 1

    def get(self, token_id: str) -> DivergenceStats | None:
        """Return a copy of the stats for *token_id*, or ``None``."""
        with self._lock:
            stats = self._stats.get(token_id)
            if stats is None:
                return None
            # Return a copy so callers don't hold a reference to mutable state
            return DivergenceStats(
                token_id=stats.token_id,
                current_bps=stats.current_bps,
                ema_bps=stats.ema_bps,
                min_bps=stats.min_bps,
                max_bps=stats.max_bps,
                count=stats.count,
                max_breach_count=stats.max_breach_count,
            )

    def snapshots(self) -> list[DivergenceStats]:
        """Return copies of all per-token stats."""
        with self._lock:
            return [
                DivergenceStats(
                    token_id=s.token_id,
                    current_bps=s.current_bps,
                    ema_bps=s.ema_bps,
                    min_bps=s.min_bps,
                    max_bps=s.max_bps,
                    count=s.count,
                    max_breach_count=s.max_breach_count,
                )
                for s in self._stats.values()
            ]

    def format_summary(self) -> str:
        """Human-readable one-liner for logging.

        Example::

            Divergence: tok_a: cur=100 ema=85 min=50 max=200 breaches=0/12
        """
        parts: list[str] = []
        for s in self.snapshots():
            parts.append(
                f"{s.token_id}: cur={s.current_bps:.0f} ema={s.ema_bps:.0f} "
                f"min={s.min_bps:.0f} max={s.max_bps:.0f} "
                f"breaches={s.max_breach_count}/{s.count}"
            )
        if not parts:
            return "Divergence: (no data)"
        return "Divergence: " + "; ".join(parts)
