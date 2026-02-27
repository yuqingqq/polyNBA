"""Scan Polymarket for active NBA markets via the Gamma API.

Discovers all active NBA binary contracts with their token IDs,
condition IDs, questions, outcomes, prices, and volumes.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from .models import PolymarketContract

logger = logging.getLogger(__name__)

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"

# Default query parameters for NBA markets
DEFAULT_PARAMS = {
    "tag": "nba",
    "active": "true",
    "closed": "false",
}


class PolymarketScannerError(Exception):
    """Raised when the Polymarket Gamma API returns an error."""
    pass


class PolymarketScanner:
    """Scans Polymarket for active NBA markets using the Gamma API.

    Usage:
        scanner = PolymarketScanner()
        contracts = scanner.get_all_nba_contracts()
        for c in contracts:
            print(c.question, c.outcome, c.current_price)
    """

    def __init__(self, timeout: int = 30) -> None:
        """Initialize the scanner.

        Args:
            timeout: HTTP request timeout in seconds.
        """
        self.timeout = timeout
        self.session = requests.Session()

    def get_nba_events(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """Fetch NBA events from the Gamma API.

        Args:
            limit: Maximum number of events to return per page.
            offset: Pagination offset.

        Returns:
            List of raw event dicts from the API.
        """
        params = {
            **DEFAULT_PARAMS,
            "limit": str(limit),
            "offset": str(offset),
        }

        try:
            response = self.session.get(
                GAMMA_EVENTS_URL, params=params, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            logger.info("Fetched %d NBA events (offset=%d)", len(data), offset)
            return data if isinstance(data, list) else []
        except requests.RequestException as e:
            raise PolymarketScannerError(f"Failed to fetch NBA events: {e}") from e

    def get_nba_markets(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """Fetch NBA markets from the Gamma API.

        Args:
            limit: Maximum number of markets to return per page.
            offset: Pagination offset.

        Returns:
            List of raw market dicts from the API.
        """
        params = {
            **DEFAULT_PARAMS,
            "limit": str(limit),
            "offset": str(offset),
        }

        try:
            response = self.session.get(
                GAMMA_MARKETS_URL, params=params, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            logger.info("Fetched %d NBA markets (offset=%d)", len(data), offset)
            return data if isinstance(data, list) else []
        except requests.RequestException as e:
            raise PolymarketScannerError(f"Failed to fetch NBA markets: {e}") from e

    def get_nba_contracts_via_events(self, page_size: int = 100) -> list[PolymarketContract]:
        """Fetch NBA contracts by filtering events client-side.

        The Gamma API ``tag=nba`` filter is unreliable server-side, so
        this method fetches all events, filters for NBA by title, extracts
        the nested ``markets`` array from each matching event, and parses
        them into :class:`PolymarketContract` models.

        Args:
            page_size: Number of events per API page.

        Returns:
            Deduplicated list of PolymarketContract models for NBA markets.
        """
        all_events = self.get_all_nba_events_with_contracts(page_size=page_size)

        seen_token_ids: set[str] = set()
        contracts: list[PolymarketContract] = []

        for event in all_events:
            if not _is_nba_event(event):
                continue

            nested_markets = event.get("markets", [])
            if not isinstance(nested_markets, list):
                continue

            for market in nested_markets:
                parsed = self._parse_market(market)
                for contract in parsed:
                    if contract.token_id not in seen_token_ids:
                        seen_token_ids.add(contract.token_id)
                        contracts.append(contract)

        # Filter out ended contracts (past end_date or resolved price)
        before_filter = len(contracts)
        contracts = [c for c in contracts if not _is_ended_contract(c)]
        filtered = before_filter - len(contracts)
        if filtered:
            logger.info("Filtered out %d ended contracts", filtered)

        logger.info(
            "Events-based scan: %d NBA contracts from %d total events (%d ended filtered)",
            len(contracts),
            len(all_events),
            filtered,
        )
        return contracts

    def get_all_nba_contracts(self, page_size: int = 100) -> list[PolymarketContract]:
        """Fetch all active NBA contracts, handling pagination.

        Tries the events-based approach first (client-side NBA title
        filtering) which avoids the broken server-side ``tag=nba`` filter.
        Falls back to the old markets-based approach if events returns
        no results.

        Args:
            page_size: Number of items per page.

        Returns:
            List of PolymarketContract models for all active NBA markets.
        """
        # Prefer events-based approach (reliable client-side filtering)
        contracts = self.get_nba_contracts_via_events(page_size=page_size)
        if contracts:
            logger.info("Total NBA contracts found (via events): %d", len(contracts))
            return contracts

        # Fallback to old markets endpoint approach
        logger.warning(
            "Events-based scan returned 0 contracts — "
            "falling back to markets endpoint (tag filter may be unreliable)"
        )
        return self._get_all_nba_contracts_via_markets(page_size=page_size)

    def _get_all_nba_contracts_via_markets(
        self, page_size: int = 100
    ) -> list[PolymarketContract]:
        """Fetch NBA contracts via the markets endpoint (legacy fallback).

        Args:
            page_size: Number of markets per page.

        Returns:
            List of PolymarketContract models.
        """
        all_contracts: list[PolymarketContract] = []
        seen_token_ids: set[str] = set()
        offset = 0

        while True:
            markets = self.get_nba_markets(limit=page_size, offset=offset)
            if not markets:
                break

            for market in markets:
                parsed = self._parse_market(market)
                # R13 fix: Deduplicate by token_id (mirrors events-based path)
                for contract in parsed:
                    if contract.token_id not in seen_token_ids:
                        seen_token_ids.add(contract.token_id)
                        all_contracts.append(contract)

            # If we got fewer results than the page size, we've reached the end
            if len(markets) < page_size:
                break

            offset += page_size

        # Filter out ended contracts (same as events-based path)
        before_filter = len(all_contracts)
        all_contracts = [c for c in all_contracts if not _is_ended_contract(c)]
        filtered = before_filter - len(all_contracts)
        if filtered:
            logger.info("Filtered out %d ended contracts", filtered)

        logger.info("Total NBA contracts found (via markets): %d", len(all_contracts))
        return all_contracts

    def get_all_nba_events_with_contracts(
        self, page_size: int = 100
    ) -> list[dict]:
        """Fetch all NBA events and extract contracts from nested markets.

        Events contain nested markets, each of which may have multiple
        binary contracts (e.g., a championship event with 30 team contracts).

        Args:
            page_size: Number of events per page.

        Returns:
            List of raw event dicts with their nested markets/contracts.
        """
        all_events: list[dict] = []
        offset = 0

        while True:
            events = self.get_nba_events(limit=page_size, offset=offset)
            if not events:
                break

            all_events.extend(events)

            if len(events) < page_size:
                break

            offset += page_size

        logger.info("Total NBA events found: %d", len(all_events))
        return all_events

    def _parse_market(self, market: dict) -> list[PolymarketContract]:
        """Parse a raw Gamma API market dict into PolymarketContract(s).

        A single market may have multiple tokens (e.g., Yes/No outcomes).
        The Gamma API structure varies, so we handle multiple formats.

        Args:
            market: Raw market dict from the Gamma API.

        Returns:
            List of PolymarketContract models.
        """
        contracts: list[PolymarketContract] = []

        question = market.get("question", "")
        condition_id = market.get("conditionId", market.get("condition_id", ""))
        slug = market.get("slug", "")
        event_slug = market.get("eventSlug", market.get("event_slug", ""))
        end_date = market.get("endDate", market.get("end_date"))
        volume = _safe_float(market.get("volume", market.get("volumeNum")))

        # Handle CLOB token IDs — may be stored as clobTokenIds (JSON array string)
        # or as individual fields
        clob_token_ids = market.get("clobTokenIds")
        if isinstance(clob_token_ids, str):
            # Sometimes stored as a JSON string like '["token1","token2"]'
            import json
            try:
                clob_token_ids = json.loads(clob_token_ids)
            except (json.JSONDecodeError, TypeError):
                clob_token_ids = []
        elif not isinstance(clob_token_ids, list):
            clob_token_ids = []

        # Outcomes — may be stored as outcomes (JSON array string) or list
        outcomes = market.get("outcomes")
        if isinstance(outcomes, str):
            import json
            try:
                outcomes = json.loads(outcomes)
            except (json.JSONDecodeError, TypeError):
                outcomes = ["Yes", "No"]
        elif not isinstance(outcomes, list):
            outcomes = ["Yes", "No"]

        # Outcome prices
        outcome_prices = market.get("outcomePrices", market.get("outcome_prices"))
        if isinstance(outcome_prices, str):
            import json
            try:
                outcome_prices = json.loads(outcome_prices)
            except (json.JSONDecodeError, TypeError):
                outcome_prices = []
        elif not isinstance(outcome_prices, list):
            outcome_prices = []

        # C-3: Extract minimum_tick_size from Gamma API market data
        raw_tick_size = market.get("minimumTickSize", market.get("minimum_tick_size"))
        minimum_tick_size = _safe_float(raw_tick_size) if raw_tick_size is not None else 0.01
        if minimum_tick_size is None or minimum_tick_size <= 0:
            minimum_tick_size = 0.01

        # C-2: Extract neg_risk from Gamma API market data
        neg_risk = market.get("negRisk", market.get("neg_risk", False))
        if not isinstance(neg_risk, bool):
            neg_risk = str(neg_risk).lower() in ("true", "1", "yes")

        # C-4: Extract minimum order size (if available)
        raw_min_order_size = market.get("minimumOrderSize", market.get("minimum_order_size"))
        min_order_size = _safe_float(raw_min_order_size) if raw_min_order_size is not None else 5.0
        if min_order_size is None or min_order_size <= 0:
            min_order_size = 5.0

        # Create a contract for each token/outcome pair
        for i, token_id in enumerate(clob_token_ids):
            outcome = outcomes[i] if i < len(outcomes) else f"Outcome_{i}"
            price = _safe_float(outcome_prices[i]) if i < len(outcome_prices) else None

            contract = PolymarketContract(
                token_id=str(token_id),
                condition_id=str(condition_id),
                slug=slug,
                question=question,
                outcome=outcome,
                current_price=price,
                volume=volume,
                event_slug=event_slug,
                end_date=end_date,
                minimum_tick_size=minimum_tick_size,
                neg_risk=neg_risk,
                min_order_size=min_order_size,
            )
            contracts.append(contract)

        return contracts


def _is_ended_contract(contract: PolymarketContract) -> bool:
    """Check if a contract has already ended (game played, awaiting resolution).

    Filters on two signals:
    1. end_date more than 4 hours in the past — the event has concluded.
       A 4-hour buffer is used because Polymarket often sets end_date before
       the actual game start time (e.g., end_date=03:00 but tipoff=03:10).
    2. Price at 0 or 1 — outcome is known, market effectively resolved.

    Either signal alone is sufficient to classify as ended.
    """
    # Check end_date (with 4-hour buffer for games that start after listed end_date)
    if contract.end_date:
        try:
            ed = datetime.fromisoformat(contract.end_date.replace("Z", "+00:00"))
            if ed + timedelta(hours=4) < datetime.now(timezone.utc):
                return True
        except (ValueError, TypeError):
            pass

    # Check resolved price (outcome known)
    if contract.current_price is not None:
        if contract.current_price <= 0.005 or contract.current_price >= 0.995:
            return True

    return False


def _is_nba_event(event: dict) -> bool:
    """Check if an event is NBA-related by title or tags."""
    title = event.get("title", "").lower()
    if "nba" in title:
        return True
    # Game events often have titles like "Rockets vs. Hornets" without "nba"
    # but carry an NBA tag
    tags = event.get("tags", [])
    if isinstance(tags, list):
        for tag in tags:
            slug = tag.get("slug", "") if isinstance(tag, dict) else ""
            if slug == "nba":
                return True
    return False


def _safe_float(value: Optional[object]) -> Optional[float]:
    """Safely convert a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
