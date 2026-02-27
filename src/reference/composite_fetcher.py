"""Composite reference price fetcher with per-game fallback.

Orchestrates DraftKings (primary) → Betfair → Kalshi → The Odds API (fallback)
to get the best available reference prices for each NBA game. Each game uses
the highest-priority source that covers it, with lower-priority sources filling
gaps for uncovered games.

DraftKings provides the sharpest US bookmaker odds with 1-5s latency.
Exchange sources (Kalshi, Betfair) provide vig-free mid-prices, while
The Odds API provides aggregated bookmaker odds that require vig removal.
"""

import logging
from typing import Optional

from .betfair_client import BetfairClient
from .draftkings_client import DraftKingsClient
from .fanduel_client import FanDuelClient
from .kalshi_client import KalshiClient
from .market_mapper import MarketMapper, normalize_team_name
from .models import MappedMarket, PolymarketContract, ReferencePrice
from .odds_client import OddsClient, OddsClientError
from .odds_models import OddsApiEvent
from .price_adapter import PriceAdapter

logger = logging.getLogger(__name__)


def _game_key(home_team: Optional[str], away_team: Optional[str]) -> Optional[str]:
    """Create a canonical game key for deduplication.

    Args:
        home_team: Home team name (raw or canonical).
        away_team: Away team name (raw or canonical).

    Returns:
        Canonical key like "Boston Celtics vs Los Angeles Lakers", or None.
    """
    home = normalize_team_name(home_team or "")
    away = normalize_team_name(away_team or "")
    if not home or not away:
        return None
    # Sort to make key order-independent
    teams = sorted([home, away])
    return f"{teams[0]} vs {teams[1]}"


class CompositeReferenceFetcher:
    """Fetches reference prices from multiple sources with per-game fallback.

    Priority order:
      0. FanDuel direct API — liquid US book, accessible globally
      1. DraftKings direct API — sharpest US book (requires US residential IP)
      2. Betfair Exchange — vig-free exchange prices (requires VPN)
      3. Kalshi — US-legal prediction market, vig-free mid-prices
      4. The Odds API — aggregated bookmaker odds (requires vig removal)

    Each game is served by the highest-priority source that covers it.

    Usage:
        fetcher = CompositeReferenceFetcher(
            poly_contracts=contracts,
            mapper=mapper,
            adapter=adapter,
            fanduel_client=fd,
            draftkings_client=dk,
            kalshi_client=kalshi,
            betfair_client=betfair,
            odds_client=odds,
        )
        mapped_markets = fetcher.fetch_mapped_markets()
        # Then adapt to reference prices:
        for mm in mapped_markets:
            ref_prices.extend(adapter.adapt(mm))
    """

    def __init__(
        self,
        poly_contracts: list[PolymarketContract],
        mapper: MarketMapper,
        adapter: PriceAdapter,
        fanduel_client: Optional[FanDuelClient] = None,
        draftkings_client: Optional[DraftKingsClient] = None,
        kalshi_client: Optional[KalshiClient] = None,
        betfair_client: Optional[BetfairClient] = None,
        odds_client: Optional[OddsClient] = None,
    ) -> None:
        self.poly_contracts = poly_contracts
        self.mapper = mapper
        self.adapter = adapter
        self.fanduel_client = fanduel_client
        self.draftkings_client = draftkings_client
        self.kalshi_client = kalshi_client
        self.betfair_client = betfair_client
        self.odds_client = odds_client

    def fetch(self) -> list[ReferencePrice]:
        """Fetch reference prices from all sources with per-game fallback.

        Returns:
            List of ReferencePrice from the best available source per game.
        """
        all_mapped = self.fetch_mapped_markets()

        ref_prices: list[ReferencePrice] = []
        for mm in all_mapped:
            ref_prices.extend(self.adapter.adapt(mm))

        logger.info("Composite fetch: %d reference prices total", len(ref_prices))
        return ref_prices

    def fetch_mapped_markets(self) -> list[MappedMarket]:
        """Fetch and map markets from all sources with per-game fallback.

        Returns:
            List of MappedMarket from the best available source per game.
        """
        covered_games: set[str] = set()
        all_mapped: list[MappedMarket] = []

        # Tier 0: FanDuel direct API (liquid, accessible globally)
        fd_mapped = self._fetch_fanduel(covered_games)
        all_mapped.extend(fd_mapped)

        # Tier 1: DraftKings direct API (sharpest, requires US residential IP)
        dk_mapped = self._fetch_draftkings(covered_games)
        all_mapped.extend(dk_mapped)

        # Tier 2: Betfair (only uncovered games)
        betfair_mapped = self._fetch_betfair(covered_games)
        all_mapped.extend(betfair_mapped)

        # Tier 3: Kalshi (only uncovered games)
        kalshi_mapped = self._fetch_kalshi(covered_games)
        all_mapped.extend(kalshi_mapped)

        # Tier 4: Odds API (only still-uncovered games)
        odds_mapped = self._fetch_odds_api(covered_games)
        all_mapped.extend(odds_mapped)

        logger.info(
            "Composite fetch: %d mapped markets "
            "(fanduel=%d, draftkings=%d, betfair=%d, kalshi=%d, odds_api=%d, "
            "covered_games=%d)",
            len(all_mapped),
            len(fd_mapped),
            len(dk_mapped),
            len(betfair_mapped),
            len(kalshi_mapped),
            len(odds_mapped),
            len(covered_games),
        )
        return all_mapped

    def _fetch_fanduel(self, covered_games: set[str]) -> list[MappedMarket]:
        """Fetch and map FanDuel NBA markets for uncovered games only.

        Args:
            covered_games: Set of game keys already covered (mutated in-place).

        Returns:
            List of MappedMarket from FanDuel (excluding already-covered games).
        """
        if self.fanduel_client is None:
            return []

        try:
            events = self.fanduel_client.get_nba_game_events()
        except Exception:
            logger.warning("FanDuel fetch failed", exc_info=True)
            return []

        if not events:
            return []

        uncovered_events = self._filter_uncovered(events, covered_games)
        if not uncovered_events:
            logger.debug("FanDuel: all games already covered")
            return []

        mapped = self.mapper.map_all_games(
            uncovered_events, self.poly_contracts, skip_in_progress=False,
        )

        for event in uncovered_events:
            key = _game_key(event.home_team, event.away_team)
            if key:
                covered_games.add(key)

        logger.info(
            "FanDuel: %d events (%d uncovered), %d mapped markets",
            len(events), len(uncovered_events), len(mapped),
        )
        return mapped

    def _fetch_draftkings(self, covered_games: set[str]) -> list[MappedMarket]:
        """Fetch and map DraftKings NBA markets for uncovered games only.

        Args:
            covered_games: Set of game keys already covered (mutated in-place).

        Returns:
            List of MappedMarket from DraftKings (excluding already-covered games).
        """
        if self.draftkings_client is None:
            return []

        try:
            events = self.draftkings_client.get_nba_game_events()
        except Exception:
            logger.warning("DraftKings fetch failed", exc_info=True)
            return []

        if not events:
            return []

        # Filter out already-covered games
        uncovered_events = self._filter_uncovered(events, covered_games)
        if not uncovered_events:
            logger.debug("DraftKings: all games already covered")
            return []

        mapped = self.mapper.map_all_games(
            uncovered_events, self.poly_contracts, skip_in_progress=False,
        )

        # Track newly covered games
        for event in uncovered_events:
            key = _game_key(event.home_team, event.away_team)
            if key:
                covered_games.add(key)

        logger.info(
            "DraftKings: %d events (%d uncovered), %d mapped markets",
            len(events), len(uncovered_events), len(mapped),
        )
        return mapped

    def _fetch_kalshi(self, covered_games: set[str]) -> list[MappedMarket]:
        """Fetch and map Kalshi NBA markets for uncovered games only.

        Args:
            covered_games: Set of game keys already covered (mutated in-place).

        Returns:
            List of MappedMarket from Kalshi (excluding already-covered games).
        """
        if self.kalshi_client is None:
            return []

        try:
            events = self.kalshi_client.get_nba_game_events()
        except Exception:
            logger.warning("Kalshi fetch failed", exc_info=True)
            return []

        if not events:
            return []

        # Filter out already-covered games
        uncovered_events = self._filter_uncovered(events, covered_games)
        if not uncovered_events:
            logger.debug("Kalshi: all games already covered by Betfair")
            return []

        mapped = self.mapper.map_all_games(
            uncovered_events, self.poly_contracts, skip_in_progress=False,
        )

        # Track newly covered games
        for event in uncovered_events:
            key = _game_key(event.home_team, event.away_team)
            if key:
                covered_games.add(key)

        logger.info(
            "Kalshi: %d events (%d uncovered), %d mapped markets",
            len(events), len(uncovered_events), len(mapped),
        )
        return mapped

    def _fetch_betfair(self, covered_games: set[str]) -> list[MappedMarket]:
        """Fetch and map Betfair NBA markets for uncovered games only.

        Args:
            covered_games: Set of game keys already covered (mutated in-place).

        Returns:
            List of MappedMarket from Betfair (excluding already-covered games).
        """
        if self.betfair_client is None or not self.betfair_client.available:
            return []

        try:
            events = self.betfair_client.get_nba_game_events()
        except Exception:
            logger.warning("Betfair fetch failed", exc_info=True)
            return []

        if not events:
            return []

        # Filter out already-covered games
        uncovered_events = self._filter_uncovered(events, covered_games)
        if not uncovered_events:
            logger.debug("Betfair: all games already covered by Kalshi")
            return []

        mapped = self.mapper.map_all_games(
            uncovered_events, self.poly_contracts, skip_in_progress=False,
        )

        # Track newly covered games
        for event in uncovered_events:
            key = _game_key(event.home_team, event.away_team)
            if key:
                covered_games.add(key)

        logger.info(
            "Betfair: %d events (%d uncovered), %d mapped markets",
            len(events), len(uncovered_events), len(mapped),
        )
        return mapped

    def _fetch_odds_api(self, covered_games: set[str]) -> list[MappedMarket]:
        """Fetch and map Odds API NBA markets for uncovered games only.

        Args:
            covered_games: Set of game keys already covered (mutated in-place).

        Returns:
            List of MappedMarket from The Odds API (excluding already-covered games).
        """
        if self.odds_client is None:
            return []

        try:
            events = self.odds_client.get_nba_game_odds(regions="us,eu")
        except Exception:
            logger.warning("Odds API fetch failed", exc_info=True)
            return []

        if not events:
            return []

        # Filter out already-covered games
        uncovered_events = self._filter_uncovered(events, covered_games)
        if not uncovered_events:
            logger.debug("Odds API: all games already covered by Kalshi/Betfair")
            return []

        mapped = self.mapper.map_all_games(
            uncovered_events, self.poly_contracts, skip_in_progress=False,
        )

        # Track newly covered games
        for event in uncovered_events:
            key = _game_key(event.home_team, event.away_team)
            if key:
                covered_games.add(key)

        logger.info(
            "Odds API: %d events (%d uncovered), %d mapped markets",
            len(events), len(uncovered_events), len(mapped),
        )
        return mapped

    @staticmethod
    def _filter_uncovered(
        events: list[OddsApiEvent], covered_games: set[str],
    ) -> list[OddsApiEvent]:
        """Filter out events whose games are already covered.

        Args:
            events: Events to filter.
            covered_games: Set of canonical game keys already covered.

        Returns:
            Events not already covered.
        """
        uncovered = []
        for event in events:
            key = _game_key(event.home_team, event.away_team)
            if key is None or key not in covered_games:
                uncovered.append(event)
        return uncovered
