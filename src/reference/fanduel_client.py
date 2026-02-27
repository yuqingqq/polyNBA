"""Client for FanDuel Sportsbook public API.

Fetches NBA moneyline odds from FanDuel's content-managed-page endpoint
and converts them to OddsApiEvent format for the composite reference fetcher.

FanDuel is one of the most liquid US sportsbooks. The sbapi endpoint
does not require authentication and is not blocked by Akamai (unlike
DraftKings), making it accessible from any IP with curl_cffi.

Uses curl_cffi for browser TLS fingerprint impersonation. Falls back
to plain requests if curl_cffi is not installed.
"""

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

import requests

try:
    from curl_cffi.requests import Session as CffiSession
    _HAS_CFFI = True
except ImportError:
    _HAS_CFFI = False

from .market_mapper import normalize_team_name
from .odds_models import (
    OddsApiBookmaker,
    OddsApiEvent,
    OddsApiMarket,
    OddsApiOutcome,
)

logger = logging.getLogger(__name__)

# FanDuel NBA page endpoint (NY jurisdiction — works globally)
_FANDUEL_NBA_URL = (
    "https://sbapi.ny.sportsbook.fanduel.com/api/content-managed-page"
)

# "Away @ Home" pattern for determining home/away
_AT_PATTERN = re.compile(r"^(.+?)\s+@\s+(.+?)$", re.IGNORECASE)

# Browser impersonation target for curl_cffi
_IMPERSONATE = "chrome131"


class FanDuelClient:
    """Client for fetching NBA moneyline odds from FanDuel.

    Usage:
        client = FanDuelClient()
        events = client.get_nba_game_events()
        for event in events:
            print(event.home_team, event.away_team, event.bookmakers)
    """

    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout

        # Configure proxy if provided
        proxy_url = os.environ.get("FANDUEL_PROXY_URL", "")
        proxies = None
        if proxy_url:
            proxies = {"https": proxy_url, "http": proxy_url}
            logger.info("FanDuel using proxy: %s", proxy_url.split("@")[-1])

        # Use curl_cffi for browser TLS fingerprinting
        self._use_cffi = _HAS_CFFI
        if _HAS_CFFI:
            self.session = CffiSession(impersonate=_IMPERSONATE)
            if proxies:
                self.session.proxies = proxies
        else:
            self.session = requests.Session()
            self.session.headers.update({
                "Accept": "application/json",
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
                ),
            })
            if proxies:
                self.session.proxies = proxies

    def get_nba_game_events(self) -> list[OddsApiEvent]:
        """Fetch NBA moneyline odds and convert to OddsApiEvent format.

        Returns:
            List of OddsApiEvent, one per NBA game with moneyline odds.
            Returns empty list on any error.
        """
        try:
            response = self.session.get(
                _FANDUEL_NBA_URL,
                params={
                    "page": "CUSTOM",
                    "customPageId": "nba",
                    "_ak": "FhMFpcPWXMeyZxOx",
                },
                timeout=self.timeout,
            )
        except Exception:
            logger.warning("FanDuel HTTP request failed", exc_info=True)
            return []

        if response.status_code != 200:
            logger.warning(
                "FanDuel API returned HTTP %d", response.status_code,
            )
            return []

        try:
            data = response.json()
        except Exception:
            logger.warning("FanDuel returned invalid JSON", exc_info=True)
            return []

        return self._parse_response(data)

    def _parse_response(self, data: Any) -> list[OddsApiEvent]:
        """Parse the FanDuel content-managed-page response.

        Args:
            data: Parsed JSON from the FanDuel endpoint.

        Returns:
            List of OddsApiEvent with moneyline h2h markets.
        """
        attachments = data.get("attachments") if isinstance(data, dict) else None
        if not attachments:
            logger.warning("FanDuel response missing attachments")
            return []

        events = attachments.get("events", {})
        markets = attachments.get("markets", {})

        if not events or not markets:
            return []

        # Index game events by ID (events with "@" in name)
        game_events: dict[str, dict] = {}
        for eid, ev in events.items():
            name = ev.get("name", "")
            if "@" in name:
                game_events[str(eid)] = ev

        # Find moneyline markets and match to events
        results: list[OddsApiEvent] = []
        for mid, mkt in markets.items():
            if mkt.get("marketType") != "MONEY_LINE":
                continue

            event_id = str(mkt.get("eventId", ""))
            event_meta = game_events.get(event_id, {})
            if not event_meta:
                continue

            event = self._convert_market(mkt, event_meta)
            if event is not None:
                results.append(event)

        logger.info("FanDuel: %d NBA game events parsed", len(results))
        return results

    def _convert_market(
        self, market: dict, event_meta: dict,
    ) -> Optional[OddsApiEvent]:
        """Convert a FanDuel moneyline market to an OddsApiEvent.

        Args:
            market: A MONEY_LINE market dict with runners.
            event_meta: Event metadata with name and openDate.

        Returns:
            OddsApiEvent or None if conversion fails.
        """
        runners = market.get("runners", [])
        if len(runners) != 2:
            return None

        # Parse odds for each runner
        parsed: list[tuple[str, int]] = []
        for runner in runners:
            name = runner.get("runnerName", "")
            if not name:
                continue

            american = self._parse_american_odds(runner)
            if american is None:
                continue

            canonical = normalize_team_name(name)
            if not canonical:
                logger.debug(
                    "FanDuel: unrecognizable team '%s', skipping", name,
                )
                return None

            parsed.append((canonical, american))

        if len(parsed) != 2:
            return None

        # Determine home/away from event name ("Away @ Home")
        home_team, away_team = self._assign_home_away(
            parsed[0][0], parsed[1][0], event_meta,
        )

        now = datetime.now(timezone.utc)
        commence_time = self._parse_start_date(event_meta) or now

        outcomes = [
            OddsApiOutcome(name=home_team, price=self._get_odds(parsed, home_team)),
            OddsApiOutcome(name=away_team, price=self._get_odds(parsed, away_team)),
        ]

        bookmaker = OddsApiBookmaker(
            key="fanduel",
            title="FanDuel",
            last_update=now,
            markets=[
                OddsApiMarket(key="h2h", last_update=now, outcomes=outcomes),
            ],
        )

        event_id = event_meta.get("eventId", market.get("eventId", "unknown"))
        return OddsApiEvent(
            id=f"fd_{event_id}",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=commence_time,
            home_team=home_team,
            away_team=away_team,
            bookmakers=[bookmaker],
        )

    @staticmethod
    def _get_odds(parsed: list[tuple[str, int]], team: str) -> int:
        """Look up American odds for a team from parsed outcomes."""
        for name, odds in parsed:
            if name == team:
                return odds
        return 100

    @staticmethod
    def _parse_american_odds(runner: dict) -> Optional[int]:
        """Parse American odds from a FanDuel runner dict.

        FanDuel stores odds in runner.winRunnerOdds.americanDisplayOdds.americanOdds.
        Falls back to decimal odds conversion.

        Args:
            runner: A single runner dict from the market.

        Returns:
            American odds as int, or None.
        """
        win_odds = runner.get("winRunnerOdds", {})

        # Primary: American odds string
        american_display = win_odds.get("americanDisplayOdds", {})
        odds_str = american_display.get("americanOdds")
        if odds_str is not None:
            try:
                return int(str(odds_str).replace("+", ""))
            except (ValueError, TypeError):
                pass

        # Fallback: decimal odds
        true_odds = win_odds.get("trueOdds", {})
        decimal_odds = true_odds.get("decimalOdds", {})
        dec = decimal_odds.get("decimalOdds")
        if dec is not None:
            try:
                dec = float(dec)
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

        FanDuel event names use "Away @ Home" format.
        Falls back to: second team = home.

        Args:
            team1: First canonical team name.
            team2: Second canonical team name.
            event_meta: Event metadata dict with "name" field.

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

        # Default: second team is home
        return (team2, team1)

    @staticmethod
    def _parse_start_date(event_meta: dict) -> Optional[datetime]:
        """Parse start date from event metadata.

        Args:
            event_meta: Event dict with possible "openDate" field.

        Returns:
            Parsed datetime or None.
        """
        start_str = event_meta.get("openDate")
        if not start_str:
            return None

        try:
            return datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None
