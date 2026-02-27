"""Tests for FanDuel direct API client.

Tests cover initialization, JSON parsing, moneyline filtering,
American odds parsing, home/away assignment, team normalization,
and graceful error handling.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.reference.fanduel_client import FanDuelClient


# -------------------------------------------------------------------
# Helpers — realistic FanDuel API response structures
# -------------------------------------------------------------------

def _make_runner(
    name: str,
    american_odds: int = -150,
    decimal_odds: float = 1.67,
) -> dict:
    """Build a FanDuel runner dict."""
    return {
        "runnerName": name,
        "winRunnerOdds": {
            "americanDisplayOdds": {"americanOdds": str(american_odds)},
            "trueOdds": {
                "decimalOdds": {"decimalOdds": decimal_odds},
            },
        },
    }


def _make_market(
    event_id: int,
    runners: list[dict],
    market_type: str = "MONEY_LINE",
) -> dict:
    """Build a FanDuel market dict."""
    return {
        "eventId": event_id,
        "marketType": market_type,
        "marketName": "Moneyline" if market_type == "MONEY_LINE" else "Other",
        "runners": runners,
    }


def _make_event(
    event_id: int,
    name: str = "Los Angeles Lakers @ Boston Celtics",
    open_date: str = "2026-03-01T00:00:00.000Z",
) -> dict:
    """Build a FanDuel event metadata dict."""
    return {
        "eventId": event_id,
        "name": name,
        "openDate": open_date,
    }


def _make_api_response(
    events: dict | None = None,
    markets: dict | None = None,
) -> dict:
    """Build a full FanDuel content-managed-page API response."""
    return {
        "layout": {},
        "attachments": {
            "events": events or {},
            "markets": markets or {},
        },
    }


def _make_single_game_response(
    event_id: int = 12345,
    home: str = "Boston Celtics",
    away: str = "Los Angeles Lakers",
    home_odds: int = -150,
    away_odds: int = 130,
    event_name: str | None = None,
) -> dict:
    """Shortcut: build a response with exactly one moneyline game."""
    if event_name is None:
        event_name = f"{away} @ {home}"

    events = {str(event_id): _make_event(event_id, name=event_name)}
    markets = {
        "m1": _make_market(
            event_id=event_id,
            runners=[
                _make_runner(away, away_odds),
                _make_runner(home, home_odds),
            ],
        ),
    }
    return _make_api_response(events=events, markets=markets)


# -------------------------------------------------------------------
# Tests: Initialization
# -------------------------------------------------------------------

class TestInit:
    def test_session_created(self) -> None:
        client = FanDuelClient()
        assert client.session is not None

    def test_custom_timeout(self) -> None:
        client = FanDuelClient(timeout=5)
        assert client.timeout == 5

    @patch.dict("os.environ", {"FANDUEL_PROXY_URL": "http://user:pass@proxy.example.com:8080"})
    def test_proxy_configured_from_env(self) -> None:
        client = FanDuelClient()
        assert client.session.proxies["https"] == "http://user:pass@proxy.example.com:8080"

    def test_no_proxy_by_default(self) -> None:
        client = FanDuelClient()
        assert not client.session.proxies


# -------------------------------------------------------------------
# Tests: Parsing — single game
# -------------------------------------------------------------------

class TestParseSingleGame:
    def test_single_game_returns_one_event(self) -> None:
        client = FanDuelClient()
        data = _make_single_game_response()
        events = client._parse_response(data)
        assert len(events) == 1

    def test_home_team_assigned(self) -> None:
        client = FanDuelClient()
        data = _make_single_game_response()
        events = client._parse_response(data)
        assert events[0].home_team == "Boston Celtics"

    def test_away_team_assigned(self) -> None:
        client = FanDuelClient()
        data = _make_single_game_response()
        events = client._parse_response(data)
        assert events[0].away_team == "Los Angeles Lakers"

    def test_h2h_market_structure(self) -> None:
        client = FanDuelClient()
        data = _make_single_game_response()
        events = client._parse_response(data)
        assert events[0].bookmakers[0].key == "fanduel"
        assert events[0].bookmakers[0].markets[0].key == "h2h"

    def test_event_id_prefix(self) -> None:
        client = FanDuelClient()
        data = _make_single_game_response(event_id=99999)
        events = client._parse_response(data)
        assert events[0].id == "fd_99999"

    def test_commence_time_parsed(self) -> None:
        client = FanDuelClient()
        data = _make_single_game_response()
        events = client._parse_response(data)
        assert events[0].commence_time is not None

    def test_sport_key(self) -> None:
        client = FanDuelClient()
        data = _make_single_game_response()
        events = client._parse_response(data)
        assert events[0].sport_key == "basketball_nba"


# -------------------------------------------------------------------
# Tests: Parsing — multiple games
# -------------------------------------------------------------------

class TestParseMultipleGames:
    def test_two_games(self) -> None:
        client = FanDuelClient()
        events_meta = {
            "1": _make_event(1, "Los Angeles Lakers @ Boston Celtics"),
            "2": _make_event(2, "Golden State Warriors @ Houston Rockets"),
        }
        markets_data = {
            "m1": _make_market(1, [
                _make_runner("Los Angeles Lakers", 130),
                _make_runner("Boston Celtics", -150),
            ]),
            "m2": _make_market(2, [
                _make_runner("Golden State Warriors", 110),
                _make_runner("Houston Rockets", -130),
            ]),
        }
        data = _make_api_response(events=events_meta, markets=markets_data)
        events = client._parse_response(data)
        assert len(events) == 2


# -------------------------------------------------------------------
# Tests: Error handling — HTTP
# -------------------------------------------------------------------

class TestHTTPErrors:
    def test_http_403_returns_empty(self) -> None:
        client = FanDuelClient()
        mock_response = MagicMock()
        mock_response.status_code = 403
        client.session.get = MagicMock(return_value=mock_response)
        assert client.get_nba_game_events() == []

    def test_http_500_returns_empty(self) -> None:
        client = FanDuelClient()
        mock_response = MagicMock()
        mock_response.status_code = 500
        client.session.get = MagicMock(return_value=mock_response)
        assert client.get_nba_game_events() == []

    def test_timeout_returns_empty(self) -> None:
        client = FanDuelClient()
        client.session.get = MagicMock(side_effect=Exception("timeout"))
        assert client.get_nba_game_events() == []

    def test_invalid_json_returns_empty(self) -> None:
        client = FanDuelClient()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("bad json")
        client.session.get = MagicMock(return_value=mock_response)
        assert client.get_nba_game_events() == []


# -------------------------------------------------------------------
# Tests: Error handling — malformed response
# -------------------------------------------------------------------

class TestMalformedResponse:
    def test_missing_attachments(self) -> None:
        client = FanDuelClient()
        assert client._parse_response({"layout": {}}) == []

    def test_empty_events(self) -> None:
        client = FanDuelClient()
        data = _make_api_response(events={}, markets={})
        assert client._parse_response(data) == []

    def test_non_dict_response(self) -> None:
        client = FanDuelClient()
        assert client._parse_response([1, 2, 3]) == []

    def test_none_response(self) -> None:
        client = FanDuelClient()
        assert client._parse_response(None) == []


# -------------------------------------------------------------------
# Tests: Market filtering
# -------------------------------------------------------------------

class TestMarketFiltering:
    def test_non_moneyline_markets_excluded(self) -> None:
        """Spread and total markets should be excluded."""
        client = FanDuelClient()
        events_meta = {"1": _make_event(1, "Lakers @ Celtics")}
        markets_data = {
            "m1": _make_market(1, [
                _make_runner("Los Angeles Lakers", -110),
                _make_runner("Boston Celtics", -110),
            ], market_type="MATCH_HANDICAP_(2-WAY)"),
            "m2": _make_market(1, [
                _make_runner("Over", -110),
                _make_runner("Under", -110),
            ], market_type="TOTAL_POINTS_(OVER/UNDER)"),
        }
        data = _make_api_response(events=events_meta, markets=markets_data)
        assert client._parse_response(data) == []

    def test_futures_events_excluded(self) -> None:
        """Events without '@' in name (futures) should be skipped."""
        client = FanDuelClient()
        events_meta = {"1": _make_event(1, "NBA Futures")}
        markets_data = {
            "m1": _make_market(1, [
                _make_runner("Los Angeles Lakers", 500),
                _make_runner("Boston Celtics", 300),
            ]),
        }
        data = _make_api_response(events=events_meta, markets=markets_data)
        assert client._parse_response(data) == []


# -------------------------------------------------------------------
# Tests: American odds parsing
# -------------------------------------------------------------------

class TestOddsParsing:
    def test_positive_american(self) -> None:
        runner = _make_runner("Team", american_odds=150)
        assert FanDuelClient._parse_american_odds(runner) == 150

    def test_negative_american(self) -> None:
        runner = _make_runner("Team", american_odds=-200)
        assert FanDuelClient._parse_american_odds(runner) == -200

    def test_even_money(self) -> None:
        runner = _make_runner("Team", american_odds=100)
        assert FanDuelClient._parse_american_odds(runner) == 100

    def test_fallback_to_decimal(self) -> None:
        """If American odds missing, convert from decimal."""
        runner = {
            "runnerName": "Team",
            "winRunnerOdds": {
                "trueOdds": {
                    "decimalOdds": {"decimalOdds": 2.5},
                },
            },
        }
        assert FanDuelClient._parse_american_odds(runner) == 150

    def test_missing_all_odds(self) -> None:
        runner = {"runnerName": "Team", "winRunnerOdds": {}}
        assert FanDuelClient._parse_american_odds(runner) is None


# -------------------------------------------------------------------
# Tests: Home/away assignment
# -------------------------------------------------------------------

class TestHomeAwayAssignment:
    def test_at_format(self) -> None:
        home, away = FanDuelClient._assign_home_away(
            "Los Angeles Lakers", "Boston Celtics",
            {"name": "Los Angeles Lakers @ Boston Celtics"},
        )
        assert home == "Boston Celtics"
        assert away == "Los Angeles Lakers"

    def test_no_metadata_fallback(self) -> None:
        home, away = FanDuelClient._assign_home_away(
            "Los Angeles Lakers", "Boston Celtics", {},
        )
        assert home == "Boston Celtics"
        assert away == "Los Angeles Lakers"

    def test_empty_event_name(self) -> None:
        home, away = FanDuelClient._assign_home_away(
            "Los Angeles Lakers", "Boston Celtics", {"name": ""},
        )
        assert home == "Boston Celtics"
        assert away == "Los Angeles Lakers"


# -------------------------------------------------------------------
# Tests: Team normalization
# -------------------------------------------------------------------

class TestTeamNormalization:
    def test_full_names_recognized(self) -> None:
        client = FanDuelClient()
        data = _make_single_game_response()
        events = client._parse_response(data)
        assert len(events) == 1
        teams = {events[0].home_team, events[0].away_team}
        assert "Boston Celtics" in teams
        assert "Los Angeles Lakers" in teams

    def test_unrecognizable_teams_skipped(self) -> None:
        client = FanDuelClient()
        events_meta = {"1": _make_event(1, "Unknown A @ Unknown B")}
        markets_data = {
            "m1": _make_market(1, [
                _make_runner("Unknown Team Alpha", -150),
                _make_runner("Unknown Team Beta", 130),
            ]),
        }
        data = _make_api_response(events=events_meta, markets=markets_data)
        assert client._parse_response(data) == []


# -------------------------------------------------------------------
# Tests: Start date parsing
# -------------------------------------------------------------------

class TestStartDateParsing:
    def test_iso_format(self) -> None:
        result = FanDuelClient._parse_start_date(
            {"openDate": "2026-03-01T00:00:00.000Z"},
        )
        assert result is not None
        assert result.year == 2026

    def test_missing_date(self) -> None:
        assert FanDuelClient._parse_start_date({}) is None

    def test_invalid_date(self) -> None:
        assert FanDuelClient._parse_start_date({"openDate": "bad"}) is None


# -------------------------------------------------------------------
# Tests: End-to-end
# -------------------------------------------------------------------

class TestEndToEnd:
    def test_full_response_parsing(self) -> None:
        client = FanDuelClient()
        events_meta = {
            "100": _make_event(100, "Golden State Warriors @ Dallas Mavericks"),
            "200": _make_event(200, "Miami Heat @ Chicago Bulls"),
        }
        markets_data = {
            "m1": _make_market(100, [
                _make_runner("Golden State Warriors", 110),
                _make_runner("Dallas Mavericks", -130),
            ]),
            "m2": _make_market(200, [
                _make_runner("Miami Heat", -200),
                _make_runner("Chicago Bulls", 170),
            ]),
        }
        data = _make_api_response(events=events_meta, markets=markets_data)
        events = client._parse_response(data)

        assert len(events) == 2
        home_teams = {e.home_team for e in events}
        assert "Dallas Mavericks" in home_teams
        assert "Chicago Bulls" in home_teams

    def test_odds_values_preserved(self) -> None:
        client = FanDuelClient()
        data = _make_single_game_response(home_odds=-150, away_odds=130)
        events = client._parse_response(data)

        outcomes = events[0].bookmakers[0].markets[0].outcomes
        odds_by_team = {o.name: o.price for o in outcomes}
        assert odds_by_team["Boston Celtics"] == -150
        assert odds_by_team["Los Angeles Lakers"] == 130

    def test_get_nba_game_events_success(self) -> None:
        client = FanDuelClient()
        data = _make_single_game_response()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = data
        client.session.get = MagicMock(return_value=mock_response)

        events = client.get_nba_game_events()
        assert len(events) == 1
        assert events[0].bookmakers[0].key == "fanduel"
