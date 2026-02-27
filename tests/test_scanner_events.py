"""Tests for the events-based NBA contract scanning approach.

Verifies that get_nba_contracts_via_events() correctly filters events
by title, extracts nested markets, deduplicates by token_id, and
falls back to the markets endpoint when events returns empty.
"""

from unittest.mock import patch

import pytest

from src.reference.models import PolymarketContract
from src.reference.polymarket_scanner import PolymarketScanner


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _make_market(
    question: str,
    token_ids: list[str],
    outcomes: list[str] | None = None,
    prices: list[str] | None = None,
    volume: str = "100000",
    condition_id: str = "cond_1",
) -> dict:
    """Build a raw Gamma API market dict for testing."""
    if outcomes is None:
        outcomes = ["Yes", "No"][: len(token_ids)]
    if prices is None:
        prices = ["0.50"] * len(token_ids)
    return {
        "question": question,
        "conditionId": condition_id,
        "slug": "test-slug",
        "eventSlug": "test-event",
        "clobTokenIds": token_ids,
        "outcomes": outcomes,
        "outcomePrices": prices,
        "volume": volume,
    }


def _make_event(title: str, markets: list[dict] | None = None) -> dict:
    """Build a raw Gamma API event dict with nested markets."""
    return {
        "title": title,
        "markets": markets or [],
    }


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@pytest.fixture()
def scanner() -> PolymarketScanner:
    return PolymarketScanner(timeout=5)


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


class TestGetNbaContractsViaEvents:
    """Tests for get_nba_contracts_via_events()."""

    def test_filters_nba_events_only(self, scanner: PolymarketScanner) -> None:
        """Only events with 'nba' in the title should produce contracts."""
        nba_market = _make_market("Will the Lakers win?", ["tok_lakers"])
        non_nba_market = _make_market("Will Trump be impeached?", ["tok_trump"])

        events = [
            _make_event("2026 NBA Champion", [nba_market]),
            _make_event("How many people will Trump deport", [non_nba_market]),
        ]

        with patch.object(scanner, "get_all_nba_events_with_contracts", return_value=events):
            contracts = scanner.get_nba_contracts_via_events()

        assert len(contracts) == 1
        assert contracts[0].token_id == "tok_lakers"

    def test_nested_markets_parsed_into_contracts(self, scanner: PolymarketScanner) -> None:
        """Nested markets within NBA events should be parsed into PolymarketContract."""
        market = _make_market(
            "Will the Celtics win the NBA Championship?",
            ["tok_yes", "tok_no"],
            outcomes=["Yes", "No"],
            prices=["0.35", "0.65"],
        )
        events = [_make_event("2026 NBA Champion", [market])]

        with patch.object(scanner, "get_all_nba_events_with_contracts", return_value=events):
            contracts = scanner.get_nba_contracts_via_events()

        assert len(contracts) == 2
        assert all(isinstance(c, PolymarketContract) for c in contracts)
        assert contracts[0].question == "Will the Celtics win the NBA Championship?"
        assert contracts[0].outcome == "Yes"
        assert contracts[0].current_price == 0.35
        assert contracts[1].outcome == "No"
        assert contracts[1].current_price == 0.65

    def test_empty_events_returns_empty(self, scanner: PolymarketScanner) -> None:
        """An empty events list should return an empty contracts list."""
        with patch.object(scanner, "get_all_nba_events_with_contracts", return_value=[]):
            contracts = scanner.get_nba_contracts_via_events()

        assert contracts == []

    def test_title_filter_is_case_insensitive(self, scanner: PolymarketScanner) -> None:
        """Title filtering should match 'NBA', 'nba', 'Nba', etc."""
        market_upper = _make_market("Q1", ["tok_upper"])
        market_lower = _make_market("Q2", ["tok_lower"])
        market_mixed = _make_market("Q3", ["tok_mixed"])

        events = [
            _make_event("2026 NBA Champion", [market_upper]),
            _make_event("nba mvp award 2026", [market_lower]),
            _make_event("Nba Eastern Conference", [market_mixed]),
        ]

        with patch.object(scanner, "get_all_nba_events_with_contracts", return_value=events):
            contracts = scanner.get_nba_contracts_via_events()

        assert len(contracts) == 3
        token_ids = {c.token_id for c in contracts}
        assert token_ids == {"tok_upper", "tok_lower", "tok_mixed"}

    def test_deduplication_by_token_id(self, scanner: PolymarketScanner) -> None:
        """Duplicate token IDs across events should be deduplicated."""
        market1 = _make_market("Lakers championship?", ["tok_dup", "tok_unique1"])
        market2 = _make_market("Lakers futures?", ["tok_dup", "tok_unique2"])

        events = [
            _make_event("NBA Champion 2026", [market1]),
            _make_event("NBA Futures 2026", [market2]),
        ]

        with patch.object(scanner, "get_all_nba_events_with_contracts", return_value=events):
            contracts = scanner.get_nba_contracts_via_events()

        token_ids = [c.token_id for c in contracts]
        assert token_ids.count("tok_dup") == 1
        assert "tok_unique1" in token_ids
        assert "tok_unique2" in token_ids
        assert len(contracts) == 3

    def test_non_nba_events_filtered_out(self, scanner: PolymarketScanner) -> None:
        """Events with titles like politics, crypto, etc. should be excluded."""
        events = [
            _make_event(
                "How many people will Trump deport",
                [_make_market("Deportation question", ["tok_politics"])],
            ),
            _make_event(
                "Will Bitcoin hit $100K?",
                [_make_market("BTC price question", ["tok_crypto"])],
            ),
            _make_event(
                "2026 NFL MVP",
                [_make_market("NFL MVP question", ["tok_nfl"])],
            ),
        ]

        with patch.object(scanner, "get_all_nba_events_with_contracts", return_value=events):
            contracts = scanner.get_nba_contracts_via_events()

        assert contracts == []

    def test_event_with_no_markets_handled_gracefully(
        self, scanner: PolymarketScanner
    ) -> None:
        """Events with missing or non-list markets should be skipped."""
        events = [
            _make_event("NBA Championship 2026", None),  # markets=[]
            {"title": "NBA MVP 2026"},  # no 'markets' key at all
            {"title": "NBA ROTY 2026", "markets": "not_a_list"},
        ]

        with patch.object(scanner, "get_all_nba_events_with_contracts", return_value=events):
            contracts = scanner.get_nba_contracts_via_events()

        assert contracts == []


class TestGetAllNbaContractsFallback:
    """Tests for get_all_nba_contracts() fallback behavior."""

    def test_uses_events_when_available(self, scanner: PolymarketScanner) -> None:
        """Should use events-based approach when it returns results."""
        events_contracts = [
            PolymarketContract(
                token_id="tok_events",
                condition_id="cond",
                question="NBA test?",
                outcome="Yes",
            )
        ]

        with patch.object(
            scanner, "get_nba_contracts_via_events", return_value=events_contracts
        ) as mock_events, patch.object(
            scanner, "_get_all_nba_contracts_via_markets"
        ) as mock_markets:
            result = scanner.get_all_nba_contracts()

        mock_events.assert_called_once()
        mock_markets.assert_not_called()
        assert len(result) == 1
        assert result[0].token_id == "tok_events"

    def test_falls_back_to_markets_when_events_empty(
        self, scanner: PolymarketScanner
    ) -> None:
        """Should fall back to markets endpoint when events returns empty."""
        markets_contracts = [
            PolymarketContract(
                token_id="tok_markets",
                condition_id="cond",
                question="NBA fallback?",
                outcome="Yes",
            )
        ]

        with patch.object(
            scanner, "get_nba_contracts_via_events", return_value=[]
        ) as mock_events, patch.object(
            scanner, "_get_all_nba_contracts_via_markets", return_value=markets_contracts
        ) as mock_markets:
            result = scanner.get_all_nba_contracts()

        mock_events.assert_called_once()
        mock_markets.assert_called_once()
        assert len(result) == 1
        assert result[0].token_id == "tok_markets"
