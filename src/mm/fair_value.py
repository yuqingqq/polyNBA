"""Fair value engine — maintains reference prices for each token."""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class FairValueEngine:
    """Thread-safe fair value store.

    Accepts reference probabilities from an external source and exposes
    them as fair values for the quoting engine.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # token_id -> (probability, timestamp)
        self._values: Dict[str, tuple[float, float]] = {}

    def update(self, token_id: str, reference_probability: float, timestamp: float | None = None) -> None:
        """Update the reference probability for a token.

        Args:
            token_id: The Polymarket token/outcome identifier.
            reference_probability: Fair probability in [0, 1].
            timestamp: Unix timestamp of the observation. Defaults to now.
        """
        if timestamp is None:
            timestamp = time.time()

        if not 0.0 <= reference_probability <= 1.0:
            logger.warning(
                "Reference probability %.4f out of [0,1] for %s, clamping",
                reference_probability,
                token_id,
            )
            reference_probability = max(0.0, min(1.0, reference_probability))

        with self._lock:
            self._values[token_id] = (reference_probability, timestamp)

        logger.debug(
            "Fair value updated: token=%s prob=%.4f ts=%.1f",
            token_id,
            reference_probability,
            timestamp,
        )

    def get_fair_value(self, token_id: str) -> Optional[float]:
        """Return the current fair value for a token, or None if unknown.

        Args:
            token_id: The token to query.

        Returns:
            The reference probability, or None.
        """
        with self._lock:
            entry = self._values.get(token_id)
        if entry is None:
            return None
        return entry[0]

    def touch_all(self) -> None:
        """Refresh timestamps for all tracked tokens without changing values.

        Called after a successful reference poll to signal that the data
        source is still alive, even if the odds haven't changed.
        """
        now = time.time()
        with self._lock:
            for token_id in self._values:
                prob, _ = self._values[token_id]
                self._values[token_id] = (prob, now)

    def is_stale(self, token_id: str, timeout_seconds: int) -> bool:
        """Check whether the fair value data is stale.

        Args:
            token_id: The token to check.
            timeout_seconds: Number of seconds after which data is considered stale.

        Returns:
            True if data is missing or older than timeout_seconds.
        """
        with self._lock:
            entry = self._values.get(token_id)
        if entry is None:
            return True
        _, ts = entry
        return (time.time() - ts) > timeout_seconds

    def get_all_fair_values(self) -> Dict[str, float]:
        """Return a snapshot of all current fair values.

        Returns:
            Dict mapping token_id to fair probability.
        """
        with self._lock:
            return {tid: prob for tid, (prob, _) in self._values.items()}
