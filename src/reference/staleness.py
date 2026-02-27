"""Staleness detection for reference prices.

Detects when reference prices are too old to be useful for trading,
which can happen when external odds sources stop updating or when
there is a connectivity issue.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from .models import ReferencePrice

logger = logging.getLogger(__name__)

# Default maximum age before a reference price is considered stale
DEFAULT_MAX_AGE_SECONDS = 300  # 5 minutes


class StalenessChecker:
    """Check whether reference prices are stale (too old to trade on).

    Usage:
        checker = StalenessChecker()
        if checker.is_stale(ref_price, max_age_seconds=300):
            print("Price is stale, skipping")

        fresh = checker.filter_stale(prices, max_age_seconds=300)
        print(f"{len(fresh)} fresh prices out of {len(prices)}")
    """

    def __init__(self, clock: Optional[callable] = None) -> None:
        """Initialize the StalenessChecker.

        Args:
            clock: Optional callable that returns the current UTC datetime.
                   Defaults to datetime.utcnow. Useful for testing.
        """
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def is_stale(
        self,
        reference: ReferencePrice,
        max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    ) -> bool:
        """Check if a single reference price is stale.

        A reference price is stale if its timestamp is older than
        max_age_seconds from the current time.

        Args:
            reference: The reference price to check.
            max_age_seconds: Maximum acceptable age in seconds.

        Returns:
            True if the reference price is stale (too old), False if fresh.
        """
        now = self._clock()
        age = (now - reference.timestamp).total_seconds()

        if age > max_age_seconds:
            logger.debug(
                "Stale reference price for token %s: age=%.1fs, max=%ds",
                reference.token_id,
                age,
                max_age_seconds,
            )
            return True

        return False

    def filter_stale(
        self,
        references: list[ReferencePrice],
        max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    ) -> list[ReferencePrice]:
        """Filter out stale reference prices, keeping only fresh ones.

        Args:
            references: List of reference prices to filter.
            max_age_seconds: Maximum acceptable age in seconds.

        Returns:
            List of reference prices that are NOT stale.
        """
        fresh = [
            ref for ref in references
            if not self.is_stale(ref, max_age_seconds)
        ]

        stale_count = len(references) - len(fresh)
        if stale_count > 0:
            logger.info(
                "Filtered out %d stale reference prices (max_age=%ds), %d remaining",
                stale_count,
                max_age_seconds,
                len(fresh),
            )

        return fresh

    def get_staleness_report(
        self,
        references: list[ReferencePrice],
        max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    ) -> dict:
        """Generate a summary report on staleness across reference prices.

        Args:
            references: List of reference prices to analyze.
            max_age_seconds: Maximum acceptable age in seconds.

        Returns:
            Dict with keys: total, fresh_count, stale_count,
            oldest_age_seconds, newest_age_seconds.
        """
        if not references:
            return {
                "total": 0,
                "fresh_count": 0,
                "stale_count": 0,
                "oldest_age_seconds": None,
                "newest_age_seconds": None,
            }

        now = self._clock()
        ages = [(now - ref.timestamp).total_seconds() for ref in references]
        fresh_count = sum(1 for a in ages if a <= max_age_seconds)

        return {
            "total": len(references),
            "fresh_count": fresh_count,
            "stale_count": len(references) - fresh_count,
            "oldest_age_seconds": max(ages),
            "newest_age_seconds": min(ages),
        }
