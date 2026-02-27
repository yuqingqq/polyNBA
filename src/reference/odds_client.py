"""Client for The Odds API (https://the-odds-api.com/).

Fetches NBA odds from multiple bookmakers, covering championship futures,
game moneylines, spreads, and totals.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import requests

from .models import ExternalOdds
from .odds_models import (
    OddsApiEvent,
    OddsApiResponse,
    OddsApiSport,
)
from .vig_removal import american_to_probability

logger = logging.getLogger(__name__)

API_BASE_URL = "https://api.the-odds-api.com/v4/sports"

# Sport keys for NBA markets
SPORT_NBA = "basketball_nba"
SPORT_NBA_CHAMPIONSHIP = "basketball_nba_championship"


class OddsClientError(Exception):
    """Raised when the Odds API returns an error."""
    pass


class RateLimitExceeded(OddsClientError):
    """Raised when API rate limit is hit."""
    pass


class OddsClient:
    """Client for fetching NBA odds from The Odds API.

    Usage:
        client = OddsClient()  # reads API key from ODDS_API_KEY env var
        events = client.get_nba_game_odds()
        for event in events:
            print(event.home_team, event.away_team)
    """

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30) -> None:
        """Initialize the OddsClient.

        Args:
            api_key: The Odds API key. If not provided, reads from
                     ODDS_API_KEY, ODDS_API_KEY1, ODDS_API_KEY2 env vars.
                     Automatically rotates to next key on quota exhaustion.
            timeout: HTTP request timeout in seconds.

        Raises:
            ValueError: If no API key is found.
        """
        self._api_keys: list[str] = []
        if api_key:
            self._api_keys.append(api_key)
        else:
            # Primary: comma-separated list in ODDS_API_KEYS
            keys_csv = os.environ.get("ODDS_API_KEYS", "")
            if keys_csv:
                self._api_keys = [k.strip() for k in keys_csv.split(",") if k.strip()]
            else:
                # Fallback: single ODDS_API_KEY
                val = os.environ.get("ODDS_API_KEY")
                if val:
                    self._api_keys.append(val)
        if not self._api_keys:
            raise ValueError(
                "No API key provided. Set ODDS_API_KEYS (comma-separated) "
                "or ODDS_API_KEY environment variable."
            )
        self._key_index = 0
        self.api_key = self._api_keys[0]
        self.timeout = timeout
        self.session = requests.Session()
        self.requests_remaining: Optional[int] = None
        self.requests_used: Optional[int] = None
        logger.info("Loaded %d Odds API key(s)", len(self._api_keys))

    def _rotate_key(self) -> bool:
        """Rotate to the next API key, wrapping around to the first key.

        I-8: Uses modular arithmetic so keys cycle. Returns False only when
        there is a single key (nowhere to rotate to).
        """
        next_index = (self._key_index + 1) % len(self._api_keys)
        if next_index == self._key_index:
            return False  # only one key — nowhere to rotate
        self._key_index = next_index
        self.api_key = self._api_keys[next_index]
        logger.warning(
            "Rotated to Odds API key %d/%d",
            next_index + 1,
            len(self._api_keys),
        )
        return True

    def _make_request(self, url: str, params: Optional[dict] = None, _tried_keys: Optional[set] = None) -> OddsApiResponse:
        """Make an authenticated GET request to The Odds API.

        Automatically rotates to next API key on 429 or quota exhaustion.

        Args:
            url: Full API URL.
            params: Query parameters (api key is added automatically).
            _tried_keys: Internal — set of key indices already tried (prevents infinite loops).

        Returns:
            OddsApiResponse with parsed events and rate limit info.

        Raises:
            RateLimitExceeded: If rate limit is hit and no more keys available.
            OddsClientError: For other API errors.
        """
        params = dict(params) if params else {}
        if _tried_keys is None:
            _tried_keys = set()
        _tried_keys.add(self._key_index)
        params["apiKey"] = self.api_key

        logger.debug("GET %s params=%s", url, {k: v for k, v in params.items() if k != "apiKey"})

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as e:
            raise OddsClientError(f"HTTP request failed: {e}") from e

        # Parse rate limit headers
        self.requests_remaining = _parse_int_header(
            response.headers.get("x-requests-remaining")
        )
        self.requests_used = _parse_int_header(
            response.headers.get("x-requests-used")
        )

        if self.requests_remaining is not None:
            logger.info(
                "Odds API quota: %d remaining, %d used (key %d/%d)",
                self.requests_remaining,
                self.requests_used or 0,
                self._key_index + 1,
                len(self._api_keys),
            )

        if response.status_code in (429, 401):
            # I-8: Only rotate if we haven't already tried all keys
            if self._rotate_key() and self._key_index not in _tried_keys:
                params["apiKey"] = self.api_key
                return self._make_request(url, params, _tried_keys=_tried_keys)
            if response.status_code == 429:
                raise RateLimitExceeded(
                    f"Rate limit exceeded on all {len(self._api_keys)} keys"
                )
            raise OddsClientError(
                f"Invalid API key on all {len(self._api_keys)} keys (HTTP 401)"
            )

        if response.status_code != 200:
            raise OddsClientError(
                f"API returned HTTP {response.status_code}: {response.text[:500]}"
            )

        # Auto-rotate when current key is nearly exhausted
        if self.requests_remaining is not None and self.requests_remaining <= 1:
            if self._rotate_key():
                logger.info("Pre-emptively rotated — previous key nearly exhausted")

        # R15-I2 fix: Handle malformed JSON on HTTP 200 (e.g., CDN error page).
        try:
            data = response.json()
        except Exception as e:
            raise OddsClientError(f"Invalid JSON in API response: {e}") from e

        # The API returns a list of events for bulk endpoints, or a single
        # event dict for per-event endpoints (e.g., /events/{id}/odds).
        # R-4 fix: Handle both formats so get_nba_event_odds works correctly.
        events = []
        if isinstance(data, list):
            for item in data:
                try:
                    event = OddsApiEvent(**item)
                    events.append(event)
                except Exception as e:
                    logger.warning("Failed to parse event: %s — %s", e, item.get("id", "unknown"))
        elif isinstance(data, dict) and "id" in data:
            try:
                event = OddsApiEvent(**data)
                events.append(event)
            except Exception as e:
                logger.warning("Failed to parse single event: %s — %s", e, data.get("id", "unknown"))

        return OddsApiResponse(
            events=events,
            requests_remaining=self.requests_remaining,
            requests_used=self.requests_used,
        )

    def get_nba_game_odds(
        self,
        sport_key: str = SPORT_NBA,
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        odds_format: str = "american",
    ) -> list[OddsApiEvent]:
        """Fetch upcoming NBA game odds (moneylines, spreads, totals).

        Args:
            sport_key: Sport key, defaults to 'basketball_nba'.
            regions: Comma-separated regions (e.g., 'us', 'us,eu').
            markets: Comma-separated market types (e.g., 'h2h,spreads,totals').
            odds_format: Odds format ('american' or 'decimal').

        Returns:
            List of OddsApiEvent with bookmaker odds.
        """
        url = f"{API_BASE_URL}/{sport_key}/odds/"
        params = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
        }

        response = self._make_request(url, params)
        logger.info("Fetched %d NBA game events", len(response.events))
        return response.events

    def get_nba_championship_odds(
        self,
        regions: str = "us",
        odds_format: str = "american",
    ) -> list[OddsApiEvent]:
        """Fetch NBA championship / futures odds.

        Args:
            regions: Comma-separated regions.
            odds_format: Odds format.

        Returns:
            List of OddsApiEvent (typically one event with outright odds).
        """
        url = f"{API_BASE_URL}/{SPORT_NBA_CHAMPIONSHIP}/odds/"
        params = {
            "regions": regions,
            "markets": "outrights",
            "oddsFormat": odds_format,
        }

        response = self._make_request(url, params)
        logger.info("Fetched %d championship events", len(response.events))
        return response.events

    def get_nba_event_odds(
        self,
        event_id: str,
        markets: str = "h2h,spreads,totals",
        regions: str = "us",
        odds_format: str = "american",
    ) -> Optional[OddsApiEvent]:
        """Fetch odds for a specific NBA event/game.

        Args:
            event_id: The Odds API event ID.
            markets: Comma-separated market types.
            regions: Comma-separated regions.
            odds_format: Odds format.

        Returns:
            OddsApiEvent if found, None otherwise.
        """
        url = f"{API_BASE_URL}/{SPORT_NBA}/events/{event_id}/odds"
        params = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
        }

        try:
            response = self._make_request(url, params)
            if response.events:
                return response.events[0]
            return None
        except OddsClientError as e:
            logger.error("Failed to fetch event %s: %s", event_id, e)
            return None

    def get_available_sports(self) -> list[OddsApiSport]:
        """List all available sports on The Odds API.

        Returns:
            List of available sports.
        """
        url = f"{API_BASE_URL}/"
        params = {"apiKey": self.api_key}

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return [OddsApiSport(**item) for item in data]
        except Exception as e:
            logger.error("Failed to fetch sports: %s", e)
            return []


def parse_event_to_external_odds(
    event: OddsApiEvent,
    bookmaker_filter: Optional[list[str]] = None,
) -> list[ExternalOdds]:
    """Convert an OddsApiEvent into a flat list of ExternalOdds.

    Args:
        event: Parsed event from The Odds API.
        bookmaker_filter: Optional list of bookmaker keys to include.
            If None, all bookmakers are included.

    Returns:
        List of ExternalOdds, one per outcome per bookmaker per market.
    """
    results: list[ExternalOdds] = []

    for bookmaker in event.bookmakers:
        if bookmaker_filter and bookmaker.key not in bookmaker_filter:
            continue

        for market in bookmaker.markets:
            for outcome in market.outcomes:
                price = outcome.price

                # I-3: Since all API requests specify oddsFormat=american,
                # always treat the price as American odds. The previous
                # heuristic (checking if price looks like an integer >= 100)
                # could misclassify edge cases like decimal odds of 100.0.
                american = round(price)
                implied_prob = american_to_probability(american)
                decimal_odds = None

                odds = ExternalOdds(
                    team=outcome.name,
                    american_odds=american,
                    decimal_odds=decimal_odds,
                    implied_probability=implied_prob,
                    bookmaker=bookmaker.key,
                    # R23-I1: Use timezone-aware UTC datetime for consistency.
                    # Pydantic parses bookmaker.last_update from ISO 8601 as
                    # timezone-aware (tzinfo=UTC), but datetime.utcnow() returns
                    # naive datetime. Mixing aware/naive crashes on comparison.
                    timestamp=bookmaker.last_update or datetime.now(timezone.utc),
                    point=outcome.point,
                    description=outcome.description,
                    market_key=market.key,
                )
                results.append(odds)

    return results


def _parse_int_header(value: Optional[str]) -> Optional[int]:
    """Safely parse an integer from an HTTP response header."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None
