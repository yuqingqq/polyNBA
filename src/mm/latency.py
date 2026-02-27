"""Latency tracking for the market making engine.

Records named latency measurements using a rolling window and computes
percentile statistics.  Thread-safe via its own lock so it can be called
from any thread without contending with the engine lock.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class LatencySnapshot:
    """Point-in-time percentile summary for a single named measurement."""

    name: str
    count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    last_ms: float


class LatencyTracker:
    """Rolling-window latency recorder with percentile reporting.

    Usage::

        tracker = LatencyTracker()
        t0 = time.monotonic()
        do_work()
        tracker.record("tick", time.monotonic() - t0)
        print(tracker.format_summary())

    Parameters:
        window: Maximum number of samples to keep per name (FIFO).
    """

    def __init__(self, window: int = 1000) -> None:
        self._window = window
        self._lock = threading.Lock()
        self._data: Dict[str, List[float]] = defaultdict(list)

    def record(self, name: str, elapsed_seconds: float) -> None:
        """Record a latency observation.

        Args:
            name: Measurement name (e.g. ``"on_fill"``, ``"tick"``).
            elapsed_seconds: Duration in seconds (from ``time.monotonic()`` delta).
        """
        with self._lock:
            buf = self._data[name]
            buf.append(elapsed_seconds)
            if len(buf) > self._window:
                # Trim oldest entries — keep the most recent *window* samples
                del buf[: len(buf) - self._window]

    def snapshot(self, name: str) -> LatencySnapshot | None:
        """Return a percentile snapshot for *name*, or ``None`` if no data."""
        with self._lock:
            buf = self._data.get(name)
            if not buf:
                return None
            samples = list(buf)  # copy under lock

        return self._compute(name, samples)

    def snapshots(self) -> list[LatencySnapshot]:
        """Return snapshots for all recorded names."""
        with self._lock:
            items = {k: list(v) for k, v in self._data.items()}
        return [self._compute(k, v) for k, v in sorted(items.items()) if v]

    def format_summary(self) -> str:
        """Human-readable one-liner for logging.

        Example::

            Latency: on_fill: p50=1.2ms p95=3.4ms p99=5.6ms max=7.8ms n=42; tick: ...
        """
        parts: list[str] = []
        for snap in self.snapshots():
            parts.append(
                f"{snap.name}: p50={snap.p50_ms:.1f}ms p95={snap.p95_ms:.1f}ms "
                f"p99={snap.p99_ms:.1f}ms max={snap.max_ms:.1f}ms n={snap.count}"
            )
        if not parts:
            return "Latency: (no data)"
        return "Latency: " + "; ".join(parts)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _compute(name: str, samples: list[float]) -> LatencySnapshot:
        n = len(samples)
        s = sorted(samples)
        to_ms = 1000.0
        return LatencySnapshot(
            name=name,
            count=n,
            p50_ms=s[_pct_idx(n, 50)] * to_ms,
            p95_ms=s[_pct_idx(n, 95)] * to_ms,
            p99_ms=s[_pct_idx(n, 99)] * to_ms,
            max_ms=s[-1] * to_ms,
            last_ms=samples[-1] * to_ms,
        )


def _pct_idx(n: int, percentile: int) -> int:
    """Return the index for the *percentile*-th value in a sorted list of *n*."""
    idx = int(n * percentile / 100)
    return min(idx, n - 1)
