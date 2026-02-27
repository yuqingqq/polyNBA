"""Tests for DraftKings direct API client.

Tests cover initialization, JSON parsing, moneyline filtering,
American odds parsing, home/away assignment, team normalization,
and graceful error handling.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.reference.draftkings_client import DraftKingsClient, _USER_AGENT


# -------------------------------------------------------------------
# Helpers — realistic DraftKings API response structures
# -------------------------------------------------------------------

def _make_outcome(
    participant: str,
    odds_american: str = "-150",
    odds_decimal: float = 1.67,
    line: float | None = None,
    label: str | None = None,
) -> dict:
    """Build a DraftKings outcome dict."""
    out: dict = {
        "participant": participant,
        "oddsAmerican": odds_american,
        "oddsDecimal": odds_decimal,
    }
    if line is not None:
        out["line"] = line
    if label is not None:
        out["label"] = label
    return out


def _make_offer(
    event_id: int,
    outcomes: list[dict],
    label: str = "Moneyline",
) -> dict:
    """Build a DraftKings offer dict."""
    return {
        "eventId": event_id,
        "label": label,
        "outcomes": outcomes,
    }


def _make_event_meta(
    event_id: int,
    name: str = "Los Angeles Lakers at Boston Celtics",
    start_date: str = "2026-03-01T00:00:00Z",
) -> dict:
    """Build a DraftKings event metadata dict."""
    return {
        "eventId": event_id,
        "name": name,
        "startDate": start_date,
    }


def _make_api_response(
    offers: list[dict] | None = None,
    events: list[dict] | None = None,
    category_label: str = "Game Lines",
) -> dict:
    """Build a full DraftKings eventgroup API response."""
    if offers is None:
        offers = []
    if events is None:
        events = []

    return {
        "eventGroup": {
            "events": events,
            "offerCategories": [
                {
                    "name": category_label,
                    "offerSubcategoryDescriptors": [
                        {
                            "offerSubcategory": {
                                "offers": [offers],
                            },
                        },
                    ],
                },
            ],
        },
    }


def _make_single_game_response(
    event_id: int = 12345,
    home: str = "Boston Celtics",
    away: str = "Los Angeles Lakers",
    home_odds: str = "-150",
    away_odds: str = "+130",
    event_name: str | None = None,
) -> dict:
    """Shortcut: build a response with exactly one moneyline game."""
    if event_name is None:
        event_name = f"{away} at {home}"

    offer = _make_offer(
        event_id=event_id,
        outcomes=[
            _make_outcome(away, away_odds),
            _make_outcome(home, home_odds),
        ],
    )
    event_meta = _make_event_meta(event_id, name=event_name)
    return _make_api_response(offers=[offer], events=[event_meta])


# -------------------------------------------------------------------
# Tests: Initialization
# -------------------------------------------------------------------

class TestInit:
    def test_session_created(self) -> None:
        client = DraftKingsClient()
        assert client.session is not None

    def test_custom_timeout(self) -> None:
        client = DraftKingsClient(timeout=5)
        assert client.timeout == 5

    def test_user_agent_header_set(self) -> None:
        client = DraftKingsClient()
        assert client.session.headers.get("User-Agent") == _USER_AGENT

    def test_accept_header_set(self) -> None:
        client = DraftKingsClient()
        assert client.session.headers.get("Accept") == "application/json"

    @patch.dict("os.environ", {"DRAFTKINGS_PROXY_URL": "http://user:pass@proxy.example.com:8080"})
    def test_proxy_configured_from_env(self) -> None:
        client = DraftKingsClient()
        assert client.session.proxies["https"] == "http://user:pass@proxy.example.com:8080"
        assert client.session.proxies["http"] == "http://user:pass@proxy.example.com:8080"

    def test_no_proxy_by_default(self) -> None:
        client = DraftKingsClient()
        assert not client.session.proxies


# -------------------------------------------------------------------
# Tests: Parsing — single game
# -------------------------------------------------------------------

class TestParseSingleGame:
    def test_single_game_returns_one_event(self) -> None:
        client = DraftKingsClient()
        data = _make_single_game_response()
        events = client._parse_response(data)
        assert len(events) == 1

    def test_home_team_assigned(self) -> None:
        client = DraftKingsClient()
        data = _make_single_game_response(
            home="Boston Celtics", away="Los Angeles Lakers",
            event_name="Los Angeles Lakers at Boston Celtics",
        )
        events = client._parse_response(data)
        assert events[0].home_team == "Boston Celtics"

    def test_away_team_assigned(self) -> None:
        client = DraftKingsClient()
        data = _make_single_game_response(
            home="Boston Celtics", away="Los Angeles Lakers",
            event_name="Los Angeles Lakers at Boston Celtics",
        )
        events = client._parse_response(data)
        assert events[0].away_team == "Los Angeles Lakers"

    def test_h2h_market_structure(self) -> None:
        client = DraftKingsClient()
        data = _make_single_game_response()
        events = client._parse_response(data)
        assert len(events[0].bookmakers) == 1
        assert events[0].bookmakers[0].key == "draftkings"
        assert len(events[0].bookmakers[0].markets) == 1
        assert events[0].bookmakers[0].markets[0].key == "h2h"

    def test_event_id_prefix(self) -> None:
        client = DraftKingsClient()
        data = _make_single_game_response(event_id=99999)
        events = client._parse_response(data)
        assert events[0].id == "dk_99999"

    def test_commence_time_parsed(self) -> None:
        client = DraftKingsClient()
        data = _make_single_game_response()
        events = client._parse_response(data)
        assert events[0].commence_time is not None

    def test_sport_key(self) -> None:
        client = DraftKingsClient()
        data = _make_single_game_response()
        events = client._parse_response(data)
        assert events[0].sport_key == "basketball_nba"


# -------------------------------------------------------------------
# Tests: Parsing — multiple games
# -------------------------------------------------------------------

class TestParseMultipleGames:
    def test_two_games(self) -> None:
        client = DraftKingsClient()
        offer1 = _make_offer(
            event_id=1,
            outcomes=[
                _make_outcome("Los Angeles Lakers", "-150"),
                _make_outcome("Boston Celtics", "+130"),
            ],
        )
        offer2 = _make_offer(
            event_id=2,
            outcomes=[
                _make_outcome("Golden State Warriors", "+110"),
                _make_outcome("Houston Rockets", "-130"),
            ],
        )
        events_meta = [
            _make_event_meta(1, "Los Angeles Lakers at Boston Celtics"),
            _make_event_meta(2, "Golden State Warriors at Houston Rockets"),
        ]
        data = _make_api_response(offers=[offer1, offer2], events=events_meta)
        events = client._parse_response(data)
        assert len(events) == 2


# -------------------------------------------------------------------
# Tests: Error handling — HTTP
# -------------------------------------------------------------------

class TestHTTPErrors:
    def test_http_403_returns_empty(self) -> None:
        client = DraftKingsClient()
        mock_response = MagicMock()
        mock_response.status_code = 403
        client.session.get = MagicMock(return_value=mock_response)

        events = client.get_nba_game_events()
        assert events == []

    def test_http_500_returns_empty(self) -> None:
        client = DraftKingsClient()
        mock_response = MagicMock()
        mock_response.status_code = 500
        client.session.get = MagicMock(return_value=mock_response)

        events = client.get_nba_game_events()
        assert events == []

    def test_timeout_returns_empty(self) -> None:
        client = DraftKingsClient()
        client.session.get = MagicMock(
            side_effect=requests.exceptions.Timeout("timeout"),
        )

        events = client.get_nba_game_events()
        assert events == []

    def test_connection_error_returns_empty(self) -> None:
        client = DraftKingsClient()
        client.session.get = MagicMock(
            side_effect=requests.exceptions.ConnectionError("refused"),
        )

        events = client.get_nba_game_events()
        assert events == []

    def test_invalid_json_returns_empty(self) -> None:
        client = DraftKingsClient()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("bad json")
        client.session.get = MagicMock(return_value=mock_response)

        events = client.get_nba_game_events()
        assert events == []


# -------------------------------------------------------------------
# Tests: Error handling — malformed response
# -------------------------------------------------------------------

class TestMalformedResponse:
    def test_missing_event_group(self) -> None:
        client = DraftKingsClient()
        events = client._parse_response({"other": "data"})
        assert events == []

    def test_empty_offers(self) -> None:
        client = DraftKingsClient()
        data = _make_api_response(offers=[], events=[])
        events = client._parse_response(data)
        assert events == []

    def test_non_dict_response(self) -> None:
        client = DraftKingsClient()
        events = client._parse_response([1, 2, 3])
        assert events == []

    def test_none_response(self) -> None:
        client = DraftKingsClient()
        events = client._parse_response(None)
        assert events == []

    def test_missing_offer_categories(self) -> None:
        client = DraftKingsClient()
        data = {"eventGroup": {"events": []}}
        events = client._parse_response(data)
        assert events == []


# -------------------------------------------------------------------
# Tests: Moneyline filtering
# -------------------------------------------------------------------

class TestMoneylineFiltering:
    def test_spread_offers_excluded(self) -> None:
        """Offers with 'line' on outcomes should be excluded."""
        client = DraftKingsClient()
        spread_offer = _make_offer(
            event_id=1,
            label="Spread",
            outcomes=[
                _make_outcome("Los Angeles Lakers", "-110", line=-4.5),
                _make_outcome("Boston Celtics", "-110", line=4.5),
            ],
        )
        assert not client._is_moneyline_offer(spread_offer)

    def test_total_offers_excluded(self) -> None:
        """Over/Under offers with 'line' should be excluded."""
        client = DraftKingsClient()
        total_offer = {
            "label": "Total Points",
            "outcomes": [
                {"participant": "Over", "line": 215.5, "oddsAmerican": "-110"},
                {"participant": "Under", "line": 215.5, "oddsAmerican": "-110"},
            ],
        }
        assert not client._is_moneyline_offer(total_offer)

    def test_moneyline_label_accepted(self) -> None:
        client = DraftKingsClient()
        ml_offer = _make_offer(
            event_id=1,
            label="Moneyline",
            outcomes=[
                _make_outcome("Los Angeles Lakers", "-150"),
                _make_outcome("Boston Celtics", "+130"),
            ],
        )
        assert client._is_moneyline_offer(ml_offer)

    def test_game_moneyline_label_accepted(self) -> None:
        """Labels like 'Game: Moneyline' should also match."""
        client = DraftKingsClient()
        offer = {"label": "Game: Moneyline", "outcomes": []}
        assert client._is_moneyline_offer(offer)

    def test_two_outcomes_no_line_accepted(self) -> None:
        """2 outcomes without 'line' = moneyline even without label."""
        client = DraftKingsClient()
        offer = {
            "label": "",
            "outcomes": [
                _make_outcome("Los Angeles Lakers", "-150"),
                _make_outcome("Boston Celtics", "+130"),
            ],
        }
        assert client._is_moneyline_offer(offer)

    def test_three_outcomes_no_label_rejected(self) -> None:
        """3 outcomes without moneyline label = not moneyline."""
        client = DraftKingsClient()
        offer = {
            "label": "",
            "outcomes": [
                _make_outcome("Team A", "-150"),
                _make_outcome("Team B", "+130"),
                _make_outcome("Draw", "+300"),
            ],
        }
        assert not client._is_moneyline_offer(offer)


# -------------------------------------------------------------------
# Tests: American odds parsing
# -------------------------------------------------------------------

class TestOddsParsing:
    def test_positive_american(self) -> None:
        result = DraftKingsClient._parse_american_odds({"oddsAmerican": "+150"})
        assert result == 150

    def test_negative_american(self) -> None:
        result = DraftKingsClient._parse_american_odds({"oddsAmerican": "-200"})
        assert result == -200

    def test_even_money(self) -> None:
        result = DraftKingsClient._parse_american_odds({"oddsAmerican": "+100"})
        assert result == 100

    def test_even_money_negative(self) -> None:
        result = DraftKingsClient._parse_american_odds({"oddsAmerican": "-100"})
        assert result == -100

    def test_fallback_to_decimal_favorite(self) -> None:
        """Decimal 1.5 → American -200."""
        result = DraftKingsClient._parse_american_odds(
            {"oddsDecimal": 1.5},
        )
        assert result == -200

    def test_fallback_to_decimal_underdog(self) -> None:
        """Decimal 2.5 → American +150."""
        result = DraftKingsClient._parse_american_odds(
            {"oddsDecimal": 2.5},
        )
        assert result == 150

    def test_missing_all_odds(self) -> None:
        result = DraftKingsClient._parse_american_odds({})
        assert result is None

    def test_invalid_odds_string(self) -> None:
        result = DraftKingsClient._parse_american_odds(
            {"oddsAmerican": "EVEN"},
        )
        # Falls back to decimal (not present) → None
        assert result is None


# -------------------------------------------------------------------
# Tests: Home/away assignment
# -------------------------------------------------------------------

class TestHomeAwayAssignment:
    def test_at_format(self) -> None:
        """'Away at Home' format should assign correctly."""
        home, away = DraftKingsClient._assign_home_away(
            "Los Angeles Lakers", "Boston Celtics",
            {"name": "Los Angeles Lakers at Boston Celtics"},
        )
        assert home == "Boston Celtics"
        assert away == "Los Angeles Lakers"

    def test_no_metadata_fallback(self) -> None:
        """Without event metadata, second team = home."""
        home, away = DraftKingsClient._assign_home_away(
            "Los Angeles Lakers", "Boston Celtics", {},
        )
        assert home == "Boston Celtics"
        assert away == "Los Angeles Lakers"

    def test_empty_event_name(self) -> None:
        """Empty event name falls back to default ordering."""
        home, away = DraftKingsClient._assign_home_away(
            "Los Angeles Lakers", "Boston Celtics",
            {"name": ""},
        )
        assert home == "Boston Celtics"
        assert away == "Los Angeles Lakers"

    def test_unrecognizable_event_name(self) -> None:
        """Non-matching event name falls back to default."""
        home, away = DraftKingsClient._assign_home_away(
            "Los Angeles Lakers", "Boston Celtics",
            {"name": "Some Other Format"},
        )
        assert home == "Boston Celtics"
        assert away == "Los Angeles Lakers"


# -------------------------------------------------------------------
# Tests: Team normalization
# -------------------------------------------------------------------

class TestTeamNormalization:
    def test_full_names_recognized(self) -> None:
        """Full team names should normalize correctly."""
        client = DraftKingsClient()
        data = _make_single_game_response(
            home="Boston Celtics", away="Los Angeles Lakers",
        )
        events = client._parse_response(data)
        assert len(events) == 1
        teams = {events[0].home_team, events[0].away_team}
        assert "Boston Celtics" in teams
        assert "Los Angeles Lakers" in teams

    def test_unrecognizable_teams_skipped(self) -> None:
        """Offers with unknown team names should be skipped."""
        client = DraftKingsClient()
        offer = _make_offer(
            event_id=1,
            outcomes=[
                _make_outcome("Unknown Team Alpha", "-150"),
                _make_outcome("Unknown Team Beta", "+130"),
            ],
        )
        data = _make_api_response(offers=[offer], events=[])
        events = client._parse_response(data)
        assert events == []


# -------------------------------------------------------------------
# Tests: Start date parsing
# -------------------------------------------------------------------

class TestStartDateParsing:
    def test_iso_format(self) -> None:
        result = DraftKingsClient._parse_start_date(
            {"startDate": "2026-03-01T00:00:00Z"},
        )
        assert result is not None
        assert result.year == 2026
        assert result.month == 3

    def test_missing_start_date(self) -> None:
        result = DraftKingsClient._parse_start_date({})
        assert result is None

    def test_invalid_start_date(self) -> None:
        result = DraftKingsClient._parse_start_date(
            {"startDate": "not-a-date"},
        )
        assert result is None

    def test_iso_with_offset(self) -> None:
        result = DraftKingsClient._parse_start_date(
            {"startDate": "2026-03-01T19:30:00-05:00"},
        )
        assert result is not None


# -------------------------------------------------------------------
# Tests: Full end-to-end parsing
# -------------------------------------------------------------------

class TestEndToEnd:
    def test_full_response_parsing(self) -> None:
        """Parse a realistic multi-game DraftKings response."""
        client = DraftKingsClient()

        offer1 = _make_offer(
            event_id=100,
            outcomes=[
                _make_outcome("Golden State Warriors", "+110", 2.10),
                _make_outcome("Dallas Mavericks", "-130", 1.77),
            ],
        )
        offer2 = _make_offer(
            event_id=200,
            outcomes=[
                _make_outcome("Miami Heat", "-200", 1.50),
                _make_outcome("Chicago Bulls", "+170", 2.70),
            ],
        )
        events_meta = [
            _make_event_meta(100, "Golden State Warriors at Dallas Mavericks"),
            _make_event_meta(200, "Miami Heat at Chicago Bulls"),
        ]
        data = _make_api_response(offers=[offer1, offer2], events=events_meta)
        events = client._parse_response(data)

        assert len(events) == 2

        # Game 1: Warriors at Mavericks
        game1 = events[0]
        assert game1.home_team == "Dallas Mavericks"
        assert game1.away_team == "Golden State Warriors"
        assert game1.bookmakers[0].key == "draftkings"

        # Game 2: Heat at Bulls
        game2 = events[1]
        assert game2.home_team == "Chicago Bulls"
        assert game2.away_team == "Miami Heat"

    def test_get_nba_game_events_success(self) -> None:
        """Full HTTP mock: get_nba_game_events returns parsed events."""
        client = DraftKingsClient()
        data = _make_single_game_response()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = data
        client.session.get = MagicMock(return_value=mock_response)

        events = client.get_nba_game_events()
        assert len(events) == 1
        assert events[0].bookmakers[0].key == "draftkings"

    def test_odds_values_preserved(self) -> None:
        """Verify that parsed American odds values are correct."""
        client = DraftKingsClient()
        data = _make_single_game_response(
            home_odds="-150", away_odds="+130",
        )
        events = client._parse_response(data)
        assert len(events) == 1

        outcomes = events[0].bookmakers[0].markets[0].outcomes
        odds_by_team = {o.name: o.price for o in outcomes}
        assert odds_by_team["Boston Celtics"] == -150
        assert odds_by_team["Los Angeles Lakers"] == 130


import requests  # noqa: E402 — already imported in module, needed for test patches
