"""Tests for the composite reference price fetcher.

Tests per-game fallback logic, deduplication, mixed coverage scenarios,
vig passthrough for exchange sources, and error handling.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.reference.composite_fetcher import (
    CompositeReferenceFetcher,
    _game_key,
)
from src.reference.models import (
    ExternalOdds,
    MappedMarket,
    MarketType,
    PolymarketContract,
    ReferencePrice,
)
from src.reference.odds_models import (
    OddsApiBookmaker,
    OddsApiEvent,
    OddsApiMarket,
    OddsApiOutcome,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _make_event(
    event_id: str,
    home: str,
    away: str,
    bookmaker_key: str = "pinnacle",
    home_odds: int = -150,
    away_odds: int = 130,
) -> OddsApiEvent:
    """Create a test OddsApiEvent."""
    now = datetime.now(timezone.utc)
    return OddsApiEvent(
        id=event_id,
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=now,
        home_team=home,
        away_team=away,
        bookmakers=[
            OddsApiBookmaker(
                key=bookmaker_key,
                title=bookmaker_key.title(),
                last_update=now,
                markets=[
                    OddsApiMarket(
                        key="h2h",
                        last_update=now,
                        outcomes=[
                            OddsApiOutcome(name=home, price=home_odds),
                            OddsApiOutcome(name=away, price=away_odds),
                        ],
                    ),
                ],
            ),
        ],
    )


def _make_contract(token_id: str, question: str, outcome: str) -> PolymarketContract:
    return PolymarketContract(
        token_id=token_id,
        condition_id=f"cond_{token_id}",
        question=question,
        outcome=outcome,
    )


def _make_mapped(event_name: str, token_ids: list[str]) -> MappedMarket:
    return MappedMarket(
        market_type=MarketType.GAME_ML,
        event_name=event_name,
        polymarket_contracts=[
            _make_contract(tid, event_name, "Yes") for tid in token_ids
        ],
        external_odds=[],
    )


# -------------------------------------------------------------------
# Tests: _game_key
# -------------------------------------------------------------------

class TestGameKey:
    def test_canonical_key(self) -> None:
        key = _game_key("Los Angeles Lakers", "Boston Celtics")
        assert key == "Boston Celtics vs Los Angeles Lakers"

    def test_order_independent(self) -> None:
        """Same game key regardless of home/away order."""
        key1 = _game_key("Los Angeles Lakers", "Boston Celtics")
        key2 = _game_key("Boston Celtics", "Los Angeles Lakers")
        assert key1 == key2

    def test_aliases_resolve(self) -> None:
        """Short names should resolve to canonical."""
        key = _game_key("Lakers", "Celtics")
        assert key == "Boston Celtics vs Los Angeles Lakers"

    def test_none_home(self) -> None:
        assert _game_key(None, "Lakers") is None

    def test_none_away(self) -> None:
        assert _game_key("Lakers", None) is None

    def test_unrecognizable(self) -> None:
        assert _game_key("Unknown A", "Unknown B") is None

    def test_empty_strings(self) -> None:
        assert _game_key("", "") is None


# -------------------------------------------------------------------
# Tests: CompositeReferenceFetcher — Kalshi only
# -------------------------------------------------------------------

class TestKalshiOnly:
    def test_kalshi_provides_all_games(self) -> None:
        """When Kalshi covers all games, Betfair and Odds API are not called."""
        kalshi = MagicMock()
        kalshi.get_nba_game_events.return_value = [
            _make_event("k1", "Los Angeles Lakers", "Boston Celtics", "kalshi"),
        ]

        betfair = MagicMock()
        betfair.available = True
        betfair.get_nba_game_events.return_value = []

        odds = MagicMock()
        odds.get_nba_game_odds.return_value = []

        mapper = MagicMock()
        mapper.map_all_games.return_value = [_make_mapped("Lakers @ Celtics", ["t1"])]

        adapter = MagicMock()
        adapter.adapt.return_value = [MagicMock(spec=ReferencePrice)]

        fetcher = CompositeReferenceFetcher(
            poly_contracts=[],
            mapper=mapper,
            adapter=adapter,
            kalshi_client=kalshi,
            betfair_client=betfair,
            odds_client=odds,
        )

        prices = fetcher.fetch()
        assert len(prices) == 1
        kalshi.get_nba_game_events.assert_called_once()

    def test_kalshi_failure_falls_through(self) -> None:
        """If Kalshi fails, should fall through to Betfair/Odds API."""
        kalshi = MagicMock()
        kalshi.get_nba_game_events.side_effect = Exception("Kalshi down")

        betfair = MagicMock()
        betfair.available = True
        betfair.get_nba_game_events.return_value = [
            _make_event("b1", "Los Angeles Lakers", "Boston Celtics", "betfair"),
        ]

        mapper = MagicMock()
        mapper.map_all_games.return_value = [_make_mapped("Game", ["t1"])]

        adapter = MagicMock()
        adapter.adapt.return_value = [MagicMock(spec=ReferencePrice)]

        fetcher = CompositeReferenceFetcher(
            poly_contracts=[],
            mapper=mapper,
            adapter=adapter,
            kalshi_client=kalshi,
            betfair_client=betfair,
            odds_client=None,
        )

        prices = fetcher.fetch()
        assert len(prices) == 1


# -------------------------------------------------------------------
# Tests: Betfair only
# -------------------------------------------------------------------

class TestBetfairOnly:
    def test_betfair_without_kalshi(self) -> None:
        """Should work when Kalshi client is None."""
        betfair = MagicMock()
        betfair.available = True
        betfair.get_nba_game_events.return_value = [
            _make_event("b1", "Los Angeles Lakers", "Boston Celtics", "betfair"),
        ]

        mapper = MagicMock()
        mapper.map_all_games.return_value = [_make_mapped("Game", ["t1"])]

        adapter = MagicMock()
        adapter.adapt.return_value = [MagicMock(spec=ReferencePrice)]

        fetcher = CompositeReferenceFetcher(
            poly_contracts=[],
            mapper=mapper,
            adapter=adapter,
            kalshi_client=None,
            betfair_client=betfair,
            odds_client=None,
        )

        prices = fetcher.fetch()
        assert len(prices) == 1
        betfair.get_nba_game_events.assert_called_once()

    def test_betfair_unavailable_skipped(self) -> None:
        """Should skip Betfair if not available."""
        betfair = MagicMock()
        betfair.available = False

        odds = MagicMock()
        odds.get_nba_game_odds.return_value = [
            _make_event("o1", "Los Angeles Lakers", "Boston Celtics"),
        ]

        mapper = MagicMock()
        mapper.map_all_games.return_value = [_make_mapped("Game", ["t1"])]

        adapter = MagicMock()
        adapter.adapt.return_value = [MagicMock(spec=ReferencePrice)]

        fetcher = CompositeReferenceFetcher(
            poly_contracts=[],
            mapper=mapper,
            adapter=adapter,
            kalshi_client=None,
            betfair_client=betfair,
            odds_client=odds,
        )

        prices = fetcher.fetch()
        assert len(prices) == 1
        betfair.get_nba_game_events.assert_not_called()


# -------------------------------------------------------------------
# Tests: Odds API only
# -------------------------------------------------------------------

class TestOddsApiOnly:
    def test_odds_api_as_final_fallback(self) -> None:
        """When both exchange clients are None, Odds API serves all."""
        odds = MagicMock()
        odds.get_nba_game_odds.return_value = [
            _make_event("o1", "Los Angeles Lakers", "Boston Celtics"),
        ]

        mapper = MagicMock()
        mapper.map_all_games.return_value = [_make_mapped("Game", ["t1"])]

        adapter = MagicMock()
        adapter.adapt.return_value = [MagicMock(spec=ReferencePrice)]

        fetcher = CompositeReferenceFetcher(
            poly_contracts=[],
            mapper=mapper,
            adapter=adapter,
            kalshi_client=None,
            betfair_client=None,
            odds_client=odds,
        )

        prices = fetcher.fetch()
        assert len(prices) == 1
        odds.get_nba_game_odds.assert_called_once_with(regions="us,eu")


# -------------------------------------------------------------------
# Tests: per-game fallback (mixed coverage)
# -------------------------------------------------------------------

class TestPerGameFallback:
    def test_kalshi_covers_game_a_odds_covers_game_b(self) -> None:
        """Kalshi covers game A, Odds API covers game B — no duplicates."""
        # Game A: Lakers vs Celtics — covered by Kalshi
        kalshi = MagicMock()
        kalshi.get_nba_game_events.return_value = [
            _make_event("k1", "Los Angeles Lakers", "Boston Celtics", "kalshi"),
        ]

        # Game B: Warriors vs Rockets — not on Kalshi
        odds = MagicMock()
        odds.get_nba_game_odds.return_value = [
            _make_event("o1", "Los Angeles Lakers", "Boston Celtics", "pinnacle"),
            _make_event("o2", "Golden State Warriors", "Houston Rockets", "pinnacle"),
        ]

        # map_all_games will be called twice (Kalshi tier, Odds API tier)
        mapper = MagicMock()
        mapper.map_all_games.side_effect = [
            [_make_mapped("Lakers @ Celtics", ["t1"])],  # Kalshi
            [_make_mapped("Warriors @ Rockets", ["t2"])],  # Odds API (filtered)
        ]

        adapter = MagicMock()
        adapter.adapt.return_value = [MagicMock(spec=ReferencePrice)]

        fetcher = CompositeReferenceFetcher(
            poly_contracts=[],
            mapper=mapper,
            adapter=adapter,
            kalshi_client=kalshi,
            betfair_client=None,
            odds_client=odds,
        )

        mapped = fetcher.fetch_mapped_markets()
        assert len(mapped) == 2

        # Verify Odds API was called with filtered events (only uncovered)
        odds_call_events = mapper.map_all_games.call_args_list[1][0][0]
        # Should only have the Warriors game (Lakers already covered)
        assert len(odds_call_events) == 1
        assert odds_call_events[0].home_team == "Golden State Warriors"

    def test_no_duplicate_coverage(self) -> None:
        """A game covered by Betfair should NOT also appear from Kalshi."""
        betfair = MagicMock()
        betfair.available = True
        betfair.get_nba_game_events.return_value = [
            _make_event("b1", "Los Angeles Lakers", "Boston Celtics", "betfair"),
        ]

        kalshi = MagicMock()
        kalshi.get_nba_game_events.return_value = [
            _make_event("k1", "Los Angeles Lakers", "Boston Celtics", "kalshi"),
        ]

        mapper = MagicMock()
        mapper.map_all_games.return_value = [_make_mapped("Lakers @ Celtics", ["t1"])]

        adapter = MagicMock()
        adapter.adapt.return_value = [MagicMock(spec=ReferencePrice)]

        fetcher = CompositeReferenceFetcher(
            poly_contracts=[],
            mapper=mapper,
            adapter=adapter,
            kalshi_client=kalshi,
            betfair_client=betfair,
            odds_client=None,
        )

        mapped = fetcher.fetch_mapped_markets()
        # Only 1 mapped market (from Betfair) — Kalshi game was filtered out
        # because _filter_uncovered removes it before map_all_games is called
        assert len(mapped) == 1

        # map_all_games should only be called once (for Betfair)
        # Kalshi events are fully filtered out, so map_all_games is not called
        assert mapper.map_all_games.call_count == 1


# -------------------------------------------------------------------
# Tests: all sources fail
# -------------------------------------------------------------------

class TestAllSourcesFail:
    def test_all_fail_returns_empty(self) -> None:
        kalshi = MagicMock()
        kalshi.get_nba_game_events.side_effect = Exception("down")

        betfair = MagicMock()
        betfair.available = True
        betfair.get_nba_game_events.side_effect = Exception("down")

        odds = MagicMock()
        odds.get_nba_game_odds.side_effect = Exception("down")

        mapper = MagicMock()
        adapter = MagicMock()

        fetcher = CompositeReferenceFetcher(
            poly_contracts=[],
            mapper=mapper,
            adapter=adapter,
            kalshi_client=kalshi,
            betfair_client=betfair,
            odds_client=odds,
        )

        prices = fetcher.fetch()
        assert prices == []

    def test_no_clients_returns_empty(self) -> None:
        mapper = MagicMock()
        adapter = MagicMock()

        fetcher = CompositeReferenceFetcher(
            poly_contracts=[],
            mapper=mapper,
            adapter=adapter,
            kalshi_client=None,
            betfair_client=None,
            odds_client=None,
        )

        prices = fetcher.fetch()
        assert prices == []


# -------------------------------------------------------------------
# Tests: filter_uncovered
# -------------------------------------------------------------------

class TestFilterUncovered:
    def test_filters_covered_games(self) -> None:
        events = [
            _make_event("1", "Los Angeles Lakers", "Boston Celtics"),
            _make_event("2", "Golden State Warriors", "Houston Rockets"),
        ]
        covered = {"Boston Celtics vs Los Angeles Lakers"}

        result = CompositeReferenceFetcher._filter_uncovered(events, covered)
        assert len(result) == 1
        assert result[0].home_team == "Golden State Warriors"

    def test_empty_covered(self) -> None:
        events = [_make_event("1", "Los Angeles Lakers", "Boston Celtics")]
        result = CompositeReferenceFetcher._filter_uncovered(events, set())
        assert len(result) == 1

    def test_all_covered(self) -> None:
        events = [_make_event("1", "Los Angeles Lakers", "Boston Celtics")]
        covered = {"Boston Celtics vs Los Angeles Lakers"}
        result = CompositeReferenceFetcher._filter_uncovered(events, covered)
        assert len(result) == 0

    def test_unparseable_teams_included(self) -> None:
        """Events with unparseable teams should pass through (game_key=None)."""
        events = [_make_event("1", "Unknown A", "Unknown B")]
        covered = set()
        result = CompositeReferenceFetcher._filter_uncovered(events, covered)
        assert len(result) == 1


# -------------------------------------------------------------------
# Tests: vig passthrough integration
# -------------------------------------------------------------------

class TestVigPassthrough:
    def test_kalshi_uses_exchange_passthrough(self) -> None:
        """Reference prices from Kalshi should use exchange_passthrough vig method."""
        from src.reference.price_adapter import PriceAdapter
        from src.reference.models import MappedMarket, MarketType, ExternalOdds, PolymarketContract

        adapter = PriceAdapter(vig_free_bookmakers={"kalshi", "betfair"})

        # Create a mapped market with kalshi bookmaker
        mapped = MappedMarket(
            market_type=MarketType.GAME_ML,
            event_name="Test Game",
            external_odds=[
                ExternalOdds(
                    team="Los Angeles Lakers",
                    implied_probability=0.6,
                    bookmaker="kalshi",
                    market_key="h2h",
                    american_odds=-150,
                ),
                ExternalOdds(
                    team="Boston Celtics",
                    implied_probability=0.4,
                    bookmaker="kalshi",
                    market_key="h2h",
                    american_odds=150,
                ),
            ],
            polymarket_contracts=[
                PolymarketContract(
                    token_id="t1",
                    condition_id="c1",
                    question="Lakers vs. Celtics",
                    outcome="Los Angeles Lakers",
                ),
                PolymarketContract(
                    token_id="t2",
                    condition_id="c2",
                    question="Lakers vs. Celtics",
                    outcome="Boston Celtics",
                ),
            ],
        )

        prices = adapter.adapt(mapped)
        assert len(prices) == 2

        for rp in prices:
            assert rp.vig_removal_method == "exchange_passthrough"
            assert rp.bookmaker == "kalshi"

    def test_odds_api_uses_proportional(self) -> None:
        """Reference prices from Odds API should use proportional vig removal."""
        from src.reference.price_adapter import PriceAdapter
        from src.reference.models import MappedMarket, MarketType, ExternalOdds, PolymarketContract

        adapter = PriceAdapter(vig_free_bookmakers={"kalshi", "betfair"})

        mapped = MappedMarket(
            market_type=MarketType.GAME_ML,
            event_name="Test Game",
            external_odds=[
                ExternalOdds(
                    team="Los Angeles Lakers",
                    implied_probability=0.55,
                    bookmaker="pinnacle",
                    market_key="h2h",
                    american_odds=-122,
                ),
                ExternalOdds(
                    team="Boston Celtics",
                    implied_probability=0.50,
                    bookmaker="pinnacle",
                    market_key="h2h",
                    american_odds=100,
                ),
            ],
            polymarket_contracts=[
                PolymarketContract(
                    token_id="t1",
                    condition_id="c1",
                    question="Lakers vs. Celtics",
                    outcome="Los Angeles Lakers",
                ),
                PolymarketContract(
                    token_id="t2",
                    condition_id="c2",
                    question="Lakers vs. Celtics",
                    outcome="Boston Celtics",
                ),
            ],
        )

        prices = adapter.adapt(mapped)
        assert len(prices) == 2

        for rp in prices:
            assert rp.vig_removal_method == "proportional"
            assert rp.bookmaker == "pinnacle"

    def test_exchange_passthrough_normalizes(self) -> None:
        """Exchange passthrough should normalize probabilities to sum=1.0."""
        from src.reference.price_adapter import PriceAdapter
        from src.reference.models import MappedMarket, MarketType, ExternalOdds, PolymarketContract

        adapter = PriceAdapter(vig_free_bookmakers={"kalshi"})

        # Probabilities don't perfectly sum to 1 (rounding artifacts)
        mapped = MappedMarket(
            market_type=MarketType.GAME_ML,
            event_name="Test Game",
            external_odds=[
                ExternalOdds(
                    team="Los Angeles Lakers",
                    implied_probability=0.62,
                    bookmaker="kalshi",
                    market_key="h2h",
                    american_odds=-163,
                ),
                ExternalOdds(
                    team="Boston Celtics",
                    implied_probability=0.40,
                    bookmaker="kalshi",
                    market_key="h2h",
                    american_odds=150,
                ),
            ],
            polymarket_contracts=[
                PolymarketContract(
                    token_id="t1",
                    condition_id="c1",
                    question="Lakers vs. Celtics",
                    outcome="Los Angeles Lakers",
                ),
                PolymarketContract(
                    token_id="t2",
                    condition_id="c2",
                    question="Lakers vs. Celtics",
                    outcome="Boston Celtics",
                ),
            ],
        )

        prices = adapter.adapt(mapped)
        total_prob = sum(rp.fair_probability for rp in prices)
        assert total_prob == pytest.approx(1.0, abs=0.02)


# -------------------------------------------------------------------
# Tests: DraftKings as Tier 0
# -------------------------------------------------------------------

class TestDraftKingsTier0:
    def test_draftkings_covers_games_before_betfair(self) -> None:
        """DraftKings events should prevent Betfair/Kalshi from covering same game."""
        dk = MagicMock()
        dk.get_nba_game_events.return_value = [
            _make_event("dk1", "Los Angeles Lakers", "Boston Celtics", "draftkings"),
        ]

        betfair = MagicMock()
        betfair.available = True
        betfair.get_nba_game_events.return_value = [
            _make_event("b1", "Los Angeles Lakers", "Boston Celtics", "betfair"),
        ]

        mapper = MagicMock()
        mapper.map_all_games.return_value = [_make_mapped("Lakers @ Celtics", ["t1"])]

        adapter = MagicMock()
        adapter.adapt.return_value = [MagicMock(spec=ReferencePrice)]

        fetcher = CompositeReferenceFetcher(
            poly_contracts=[],
            mapper=mapper,
            adapter=adapter,
            draftkings_client=dk,
            betfair_client=betfair,
            odds_client=None,
        )

        mapped = fetcher.fetch_mapped_markets()
        # Only DraftKings mapped — Betfair game filtered as duplicate
        assert len(mapped) == 1
        assert mapper.map_all_games.call_count == 1
        dk.get_nba_game_events.assert_called_once()

    def test_draftkings_failure_falls_through(self) -> None:
        """If DraftKings fails, Betfair/Kalshi/OddsAPI should still work."""
        dk = MagicMock()
        dk.get_nba_game_events.side_effect = Exception("DK down")

        kalshi = MagicMock()
        kalshi.get_nba_game_events.return_value = [
            _make_event("k1", "Los Angeles Lakers", "Boston Celtics", "kalshi"),
        ]

        mapper = MagicMock()
        mapper.map_all_games.return_value = [_make_mapped("Game", ["t1"])]

        adapter = MagicMock()
        adapter.adapt.return_value = [MagicMock(spec=ReferencePrice)]

        fetcher = CompositeReferenceFetcher(
            poly_contracts=[],
            mapper=mapper,
            adapter=adapter,
            draftkings_client=dk,
            kalshi_client=kalshi,
            betfair_client=None,
            odds_client=None,
        )

        prices = fetcher.fetch()
        assert len(prices) == 1
        kalshi.get_nba_game_events.assert_called_once()

    def test_draftkings_none_client_skipped(self) -> None:
        """When draftkings_client is None, should skip to next tier."""
        odds = MagicMock()
        odds.get_nba_game_odds.return_value = [
            _make_event("o1", "Los Angeles Lakers", "Boston Celtics"),
        ]

        mapper = MagicMock()
        mapper.map_all_games.return_value = [_make_mapped("Game", ["t1"])]

        adapter = MagicMock()
        adapter.adapt.return_value = [MagicMock(spec=ReferencePrice)]

        fetcher = CompositeReferenceFetcher(
            poly_contracts=[],
            mapper=mapper,
            adapter=adapter,
            draftkings_client=None,
            kalshi_client=None,
            betfair_client=None,
            odds_client=odds,
        )

        prices = fetcher.fetch()
        assert len(prices) == 1
