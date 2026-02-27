"""Client for Kalshi prediction market exchange.

Fetches NBA game markets from the Kalshi public REST API and converts
them to OddsApiEvent format for seamless integration with the existing
market mapper and price adapter pipeline.

Kalshi game markets use series ticker KXNBAGAME.  Each game has two
markets (one per team) grouped under the same event ticker, with titles
like "New Orleans at Utah Winner?".  Each market's YES side represents
that team winning.

Kalshi prices are in cents (0-100), representing probabilities directly.
Mid-prices are vig-free since they come from a limit order book exchange.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

import requests

from .kalshi_models import KalshiMarket, KalshiMarketsResponse
from .market_mapper import normalize_team_name
from .odds_models import (
    OddsApiBookmaker,
    OddsApiEvent,
    OddsApiMarket,
    OddsApiOutcome,
)

logger = logging.getLogger(__name__)

API_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Kalshi NBA game series ticker (individual games, not championship)
NBA_GAME_SERIES_TICKER = "KXNBAGAME"

# Pattern for "Away at Home Winner?" title format
_AT_PATTERN = re.compile(
    r"(.+?)\s+at\s+(.+?)\s+Winner",
    re.IGNORECASE,
)

# Fallback: "Team1 vs Team2" format
_VS_PATTERN = re.compile(
    r"(.+?)\s+(?:vs\.?|v\.?)\s+(.+?)(?:\s*[-:?]|$)",
    re.IGNORECASE,
)


class KalshiClientError(Exception):
    """Raised when a Kalshi API call fails."""
    pass


class KalshiClient:
    """Client for fetching NBA game markets from Kalshi's public API.

    Usage:
        client = KalshiClient()
        events = client.get_nba_game_events()
        for event in events:
            print(event.home_team, event.away_team)
    """

    def __init__(self, timeout: int = 15) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
        })

    def get_nba_markets(
        self, series_ticker: str = NBA_GAME_SERIES_TICKER, status: str = "open",
    ) -> list[KalshiMarket]:
        """Fetch NBA markets from Kalshi for a given series.

        Args:
            series_ticker: Kalshi series ticker (default: KXNBAGAME for games).
            status: Filter by market status ('open', 'closed', 'settled').

        Returns:
            List of KalshiMarket objects.

        Raises:
            KalshiClientError: On API or network errors.
        """
        all_markets: list[KalshiMarket] = []
        cursor: Optional[str] = None

        while True:
            params: dict = {
                "series_ticker": series_ticker,
                "status": status,
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor

            try:
                response = self.session.get(
                    f"{API_BASE_URL}/markets",
                    params=params,
                    timeout=self.timeout,
                )
            except requests.RequestException as e:
                raise KalshiClientError(f"HTTP request failed: {e}") from e

            if response.status_code != 200:
                raise KalshiClientError(
                    f"Kalshi API returned HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )

            try:
                data = response.json()
            except Exception as e:
                raise KalshiClientError(f"Invalid JSON: {e}") from e

            parsed = KalshiMarketsResponse(**data)
            all_markets.extend(parsed.markets)

            if not parsed.cursor:
                break
            cursor = parsed.cursor

        logger.info("Fetched %d Kalshi NBA markets (series=%s)", len(all_markets), series_ticker)
        return all_markets

    def get_nba_game_events(self) -> list[OddsApiEvent]:
        """Fetch NBA game markets and convert to OddsApiEvent format.

        Kalshi game markets have two contracts per game (one per team),
        grouped by event_ticker.  Each contract's YES price represents
        the probability of that team winning.

        Returns:
            List of OddsApiEvent, one per NBA game found on Kalshi.
        """
        try:
            markets = self.get_nba_markets(
                series_ticker=NBA_GAME_SERIES_TICKER, status="open",
            )
        except KalshiClientError:
            logger.warning("Failed to fetch Kalshi NBA game markets", exc_info=True)
            return []

        if not markets:
            return []

        # Group markets by event_ticker (each game has 2 markets)
        by_event: dict[str, list[KalshiMarket]] = {}
        for m in markets:
            by_event.setdefault(m.event_ticker, []).append(m)

        events: list[OddsApiEvent] = []
        for event_ticker, event_markets in by_event.items():
            event = self._convert_event(event_ticker, event_markets)
            if event is not None:
                events.append(event)

        logger.info("Converted %d Kalshi game events to OddsApiEvent format", len(events))
        return events

    def _convert_event(
        self, event_ticker: str, markets: list[KalshiMarket]
    ) -> Optional[OddsApiEvent]:
        """Convert a pair of Kalshi team markets into an OddsApiEvent.

        Each game event has 2 markets: one for each team. The title is
        typically "Away at Home Winner?" and the ticker suffix is the team
        abbreviation (e.g., KXNBAGAME-26FEB28LALGSW-LAL for Lakers).

        Args:
            event_ticker: The Kalshi event ticker (e.g., KXNBAGAME-26FEB28LALGSW).
            markets: The 2 markets belonging to this game.

        Returns:
            OddsApiEvent if teams can be identified, None otherwise.
        """
        # Parse away/home from title ("Away at Home Winner?")
        away_team = None
        home_team = None

        for m in markets:
            parsed = self._parse_at_title(m.title)
            if parsed:
                away_team, home_team = parsed
                break

        # Fallback: try "vs" pattern
        if not away_team or not home_team:
            for m in markets:
                parsed = self._parse_vs_title(m.title)
                if parsed:
                    away_team, home_team = parsed
                    break

        if not away_team or not home_team:
            logger.debug(
                "Could not parse teams from Kalshi event %s (titles: %s)",
                event_ticker,
                [m.title for m in markets],
            )
            return None

        # Build outcomes: each market's YES = that team's win probability
        outcomes = self._build_outcomes_from_pair(markets, home_team, away_team)
        if not outcomes:
            return None

        now = datetime.now(timezone.utc)
        bookmaker = OddsApiBookmaker(
            key="kalshi",
            title="Kalshi",
            last_update=now,
            markets=[
                OddsApiMarket(
                    key="h2h",
                    last_update=now,
                    outcomes=outcomes,
                ),
            ],
        )

        return OddsApiEvent(
            id=f"kalshi_{event_ticker}",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=now,
            home_team=home_team,
            away_team=away_team,
            bookmakers=[bookmaker],
        )

    def _build_outcomes_from_pair(
        self,
        markets: list[KalshiMarket],
        home_team: str,
        away_team: str,
    ) -> list[OddsApiOutcome]:
        """Build h2h outcomes from a pair of Kalshi team markets.

        Each market's YES side is that team's win probability.  We match
        each market to a team using the ticker suffix (team abbreviation).

        Args:
            markets: The 2 Kalshi markets for this game.
            home_team: Canonical home team name.
            away_team: Canonical away team name.

        Returns:
            List of 2 OddsApiOutcome or empty list.
        """
        team_probs: dict[str, float] = {}

        for m in markets:
            # Match market to team: try ticker suffix, then title
            team = self._identify_market_team(m, home_team, away_team)
            if not team:
                continue

            mid = self._compute_mid_probability(m)
            if mid is not None:
                team_probs[team] = mid

        if home_team not in team_probs or away_team not in team_probs:
            # Fallback: if we have only one team, derive the other
            if len(team_probs) == 1:
                team, prob = next(iter(team_probs.items()))
                other = away_team if team == home_team else home_team
                team_probs[other] = 1.0 - prob

        if home_team not in team_probs or away_team not in team_probs:
            return []

        home_prob = team_probs[home_team]
        away_prob = team_probs[away_team]

        # Normalize to sum=1.0
        total = home_prob + away_prob
        if total > 0:
            home_prob /= total
            away_prob /= total

        return [
            OddsApiOutcome(name=home_team, price=_prob_to_american(home_prob)),
            OddsApiOutcome(name=away_team, price=_prob_to_american(away_prob)),
        ]

    def _identify_market_team(
        self, market: KalshiMarket, home_team: str, away_team: str,
    ) -> Optional[str]:
        """Identify which team a market represents.

        Uses ticker suffix (e.g., "-LAL" → Lakers) and title parsing.

        Args:
            market: A Kalshi market.
            home_team: Canonical home team.
            away_team: Canonical away team.

        Returns:
            Canonical team name, or None.
        """
        # Try ticker suffix (e.g., KXNBAGAME-26FEB28LALGSW-LAL → "LAL")
        suffix = market.ticker.rsplit("-", 1)[-1] if "-" in market.ticker else ""
        if suffix:
            team_from_suffix = normalize_team_name(suffix)
            if team_from_suffix in (home_team, away_team):
                return team_from_suffix

        # Fallback: try to find team name in the title that's unique
        title_team = normalize_team_name(market.title)
        if title_team in (home_team, away_team):
            return title_team

        return None

    def _compute_mid_probability(self, market: KalshiMarket) -> Optional[float]:
        """Compute mid-price probability from Kalshi bid/ask (in cents).

        Args:
            market: A KalshiMarket with yes_bid and/or yes_ask.

        Returns:
            Probability (0-1) or None if no valid prices.
        """
        if market.yes_bid is not None and market.yes_ask is not None:
            if market.yes_bid <= 0 and market.yes_ask <= 0:
                return None
            mid_cents = (market.yes_bid + market.yes_ask) / 2.0
            return mid_cents / 100.0
        elif market.yes_bid is not None and market.yes_bid > 0:
            return market.yes_bid / 100.0
        elif market.yes_ask is not None and market.yes_ask > 0:
            return market.yes_ask / 100.0
        elif market.last_price is not None and market.last_price > 0:
            return market.last_price / 100.0
        return None

    @staticmethod
    def _parse_at_title(title: str) -> Optional[tuple[str, str]]:
        """Parse "Away at Home Winner?" title format.

        Args:
            title: Market title (e.g., "New Orleans at Utah Winner?")

        Returns:
            Tuple of (away_canonical, home_canonical) or None.
        """
        match = _AT_PATTERN.search(title)
        if not match:
            return None

        raw_away = match.group(1).strip()
        raw_home = match.group(2).strip()

        away = normalize_team_name(raw_away)
        home = normalize_team_name(raw_home)

        if away and home and away != home:
            return (away, home)
        return None

    @staticmethod
    def _parse_vs_title(title: str) -> Optional[tuple[str, str]]:
        """Parse "Team1 vs Team2" title format (fallback).

        Returns:
            Tuple of (team1_canonical, team2_canonical) or None.
        """
        match = _VS_PATTERN.search(title)
        if not match:
            return None

        raw1 = match.group(1).strip()
        raw2 = match.group(2).strip()

        team1 = normalize_team_name(raw1)
        team2 = normalize_team_name(raw2)

        if team1 and team2 and team1 != team2:
            return (team1, team2)
        return None


def _prob_to_american(prob: float) -> int:
    """Convert probability (0-1) to American odds."""
    prob = max(0.01, min(0.99, prob))
    if prob >= 0.5:
        return round(-100 * prob / (1 - prob))
    else:
        return round(100 * (1 - prob) / prob)
