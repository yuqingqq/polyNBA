"""Client for DraftKings Sportsbook public API.

Fetches NBA moneyline odds directly from the DraftKings eventgroups
endpoint (no authentication required) and converts them to OddsApiEvent
format for seamless integration with the composite reference fetcher.

DraftKings is considered the sharpest US book for NBA and provides
odds with 1-5s latency, making it the highest-priority source.

The endpoint may return HTTP 403 behind Akamai CDN when accessed
from certain IPs or without a browser-like User-Agent. All errors
degrade gracefully to an empty list, letting the bot fall through
to Betfair/Kalshi/Odds API.
"""

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

import requests

from .market_mapper import normalize_team_name
from .odds_models import (
    OddsApiBookmaker,
    OddsApiEvent,
    OddsApiMarket,
    OddsApiOutcome,
)

logger = logging.getLogger(__name__)

# DraftKings NBA eventgroup ID
_NBA_EVENTGROUP_URL = (
    "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/42648"
)

# "Away at Home" pattern for determining home/away
_AT_PATTERN = re.compile(r"^(.+?)\s+at\s+(.+?)$", re.IGNORECASE)

# Browser-like User-Agent to avoid Akamai blocks
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


class DraftKingsClient:
    """Client for fetching NBA moneyline odds from DraftKings.

    Usage:
        client = DraftKingsClient()
        events = client.get_nba_game_events()
        for event in events:
            print(event.home_team, event.away_team, event.bookmakers)
    """

    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
        })

        # Configure proxy if provided (residential proxy needed for datacenter IPs)
        proxy_url = os.environ.get("DRAFTKINGS_PROXY_URL", "")
        if proxy_url:
            self.session.proxies = {
                "https": proxy_url,
                "http": proxy_url,
            }
            logger.info("DraftKings using proxy: %s", proxy_url.split("@")[-1])

    def get_nba_game_events(self) -> list[OddsApiEvent]:
        """Fetch NBA moneyline odds and convert to OddsApiEvent format.

        Returns:
            List of OddsApiEvent, one per NBA game with moneyline odds.
            Returns empty list on any error (HTTP, parse, etc.).
        """
        try:
            response = self.session.get(
                _NBA_EVENTGROUP_URL,
                params={"format": "json"},
                timeout=self.timeout,
            )
        except requests.RequestException:
            logger.warning("DraftKings HTTP request failed", exc_info=True)
            return []

        if response.status_code != 200:
            logger.warning(
                "DraftKings API returned HTTP %d", response.status_code,
            )
            return []

        try:
            data = response.json()
        except Exception:
            logger.warning("DraftKings returned invalid JSON", exc_info=True)
            return []

        return self._parse_response(data)

    def _parse_response(self, data: Any) -> list[OddsApiEvent]:
        """Parse the DraftKings eventgroup response into OddsApiEvents.

        Args:
            data: Parsed JSON from the eventgroup endpoint.

        Returns:
            List of OddsApiEvent with moneyline h2h markets.
        """
        event_group = data.get("eventGroup") if isinstance(data, dict) else None
        if not event_group:
            logger.warning("DraftKings response missing eventGroup")
            return []

        # Build event metadata lookup: eventId -> event dict
        events_by_id: dict[int, dict] = {}
        for event in event_group.get("events", []):
            eid = event.get("eventId")
            if eid is not None:
                events_by_id[eid] = event

        # Extract moneyline offers from offer categories
        moneyline_offers = self._extract_moneyline_offers(event_group)
        if not moneyline_offers:
            logger.debug("DraftKings: no moneyline offers found")
            return []

        results: list[OddsApiEvent] = []
        for offer in moneyline_offers:
            event = self._convert_offer(offer, events_by_id)
            if event is not None:
                results.append(event)

        logger.info("DraftKings: %d NBA game events parsed", len(results))
        return results

    def _extract_moneyline_offers(
        self, event_group: dict,
    ) -> list[dict]:
        """Extract moneyline offer dicts from the nested offer structure.

        DraftKings nests offers as:
        eventGroup.offerCategories[].offerSubcategoryDescriptors[]
            .offerSubcategory.offers[][]

        We filter to moneyline offers only (label contains "Moneyline"
        or outcomes lack a "line" field, with exactly 2 outcomes).

        Args:
            event_group: The eventGroup dict from the API response.

        Returns:
            List of offer dicts that represent moneyline markets.
        """
        moneyline_offers: list[dict] = []

        for category in event_group.get("offerCategories", []):
            for descriptor in category.get("offerSubcategoryDescriptors", []):
                subcategory = descriptor.get("offerSubcategory")
                if not subcategory:
                    continue

                for offer_group in subcategory.get("offers", []):
                    if not isinstance(offer_group, list):
                        continue
                    for offer in offer_group:
                        if self._is_moneyline_offer(offer):
                            moneyline_offers.append(offer)

        return moneyline_offers

    def _is_moneyline_offer(self, offer: dict) -> bool:
        """Check if an offer dict represents a moneyline market.

        Moneyline offers either have "Moneyline" in their label, or
        have exactly 2 outcomes without a "line" value (distinguishing
        them from spreads/totals).

        Args:
            offer: A single offer dict.

        Returns:
            True if this is a moneyline offer.
        """
        label = offer.get("label", "")
        if "moneyline" in label.lower():
            return True

        outcomes = offer.get("outcomes", [])
        if len(outcomes) != 2:
            return False

        # Spreads/totals have a numeric "line" field on outcomes
        for outcome in outcomes:
            if outcome.get("line") is not None:
                return False

        return True

    def _convert_offer(
        self, offer: dict, events_by_id: dict[int, dict],
    ) -> Optional[OddsApiEvent]:
        """Convert a DraftKings moneyline offer to an OddsApiEvent.

        Args:
            offer: A moneyline offer dict with outcomes.
            events_by_id: Lookup from eventId to event metadata.

        Returns:
            OddsApiEvent or None if conversion fails.
        """
        outcomes_data = offer.get("outcomes", [])
        if len(outcomes_data) != 2:
            return None

        # Parse odds for each outcome
        parsed_outcomes: list[tuple[str, int]] = []
        for outcome in outcomes_data:
            participant = outcome.get("participant") or outcome.get("label", "")
            if not participant:
                continue

            american = self._parse_american_odds(outcome)
            if american is None:
                continue

            canonical = normalize_team_name(participant)
            if not canonical:
                logger.debug(
                    "DraftKings: unrecognizable team '%s', skipping offer",
                    participant,
                )
                return None

            parsed_outcomes.append((canonical, american))

        if len(parsed_outcomes) != 2:
            return None

        # Determine home/away from event metadata
        event_id = offer.get("eventId")
        event_meta = events_by_id.get(event_id, {}) if event_id else {}

        home_team, away_team = self._assign_home_away(
            parsed_outcomes[0][0],
            parsed_outcomes[1][0],
            event_meta,
        )

        # Build OddsApiEvent
        now = datetime.now(timezone.utc)
        commence_time = self._parse_start_date(event_meta) or now

        api_outcomes = [
            OddsApiOutcome(name=home_team, price=self._get_odds(parsed_outcomes, home_team)),
            OddsApiOutcome(name=away_team, price=self._get_odds(parsed_outcomes, away_team)),
        ]

        bookmaker = OddsApiBookmaker(
            key="draftkings",
            title="DraftKings",
            last_update=now,
            markets=[
                OddsApiMarket(
                    key="h2h",
                    last_update=now,
                    outcomes=api_outcomes,
                ),
            ],
        )

        return OddsApiEvent(
            id=f"dk_{event_id or 'unknown'}",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=commence_time,
            home_team=home_team,
            away_team=away_team,
            bookmakers=[bookmaker],
        )

    @staticmethod
    def _get_odds(
        parsed_outcomes: list[tuple[str, int]], team: str,
    ) -> int:
        """Look up American odds for a team from parsed outcomes."""
        for name, odds in parsed_outcomes:
            if name == team:
                return odds
        return 100  # fallback (should not happen)

    @staticmethod
    def _parse_american_odds(outcome: dict) -> Optional[int]:
        """Parse American odds from a DraftKings outcome dict.

        Tries oddsAmerican string first (e.g., "+150", "-200"),
        then falls back to oddsDecimal conversion.

        Args:
            outcome: A single outcome dict from the offer.

        Returns:
            American odds as int, or None if unparseable.
        """
        # Primary: oddsAmerican string
        odds_str = outcome.get("oddsAmerican")
        if odds_str:
            try:
                return int(odds_str.replace("+", ""))
            except (ValueError, AttributeError):
                pass

        # Fallback: convert decimal odds to American
        odds_decimal = outcome.get("oddsDecimal")
        if odds_decimal is not None:
            try:
                dec = float(odds_decimal)
                if dec >= 2.0:
                    return round((dec - 1) * 100)
                elif dec > 1.0:
                    return round(-100 / (dec - 1))
            except (ValueError, TypeError, ZeroDivisionError):
                pass

        return None

    @staticmethod
    def _assign_home_away(
        team1: str, team2: str, event_meta: dict,
    ) -> tuple[str, str]:
        """Determine home and away teams from event metadata.

        DraftKings event names use "Away at Home" format.
        Falls back to: second team = home (DraftKings default ordering).

        Args:
            team1: First canonical team name.
            team2: Second canonical team name.
            event_meta: Event metadata dict (may have "name" field).

        Returns:
            Tuple of (home_team, away_team).
        """
        event_name = event_meta.get("name", "")
        if event_name:
            match = _AT_PATTERN.match(event_name)
            if match:
                raw_away = match.group(1).strip()
                raw_home = match.group(2).strip()

                away_canonical = normalize_team_name(raw_away)
                home_canonical = normalize_team_name(raw_home)

                if away_canonical and home_canonical:
                    if away_canonical in (team1, team2) and home_canonical in (team1, team2):
                        return (home_canonical, away_canonical)

        # Default: second team is home (DraftKings typical ordering)
        return (team2, team1)

    @staticmethod
    def _parse_start_date(event_meta: dict) -> Optional[datetime]:
        """Parse start date from event metadata.

        Args:
            event_meta: Event dict with possible "startDate" field.

        Returns:
            Parsed datetime or None.
        """
        start_str = event_meta.get("startDate")
        if not start_str:
            return None

        try:
            # DraftKings uses ISO 8601 format
            return datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None
