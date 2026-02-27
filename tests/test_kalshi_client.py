"""Tests for the Kalshi client module.

Tests market fetching, parsing, conversion to OddsApiEvent format,
team name extraction from "Away at Home Winner?" titles, mid-price
computation from paired team markets, and error handling.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.reference.kalshi_client import (
    KalshiClient,
    KalshiClientError,
    _AT_PATTERN,
    _VS_PATTERN,
    _prob_to_american,
)
from src.reference.kalshi_models import (
    KalshiMarket,
    KalshiMarketsResponse,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _make_market(
    ticker: str = "KXNBAGAME-26FEB28LALGSW-LAL",
    event_ticker: str = "KXNBAGAME-26FEB28LALGSW",
    title: str = "Los Angeles L at Golden State Winner?",
    yes_bid: int = 55,
    yes_ask: int = 60,
    status: str = "open",
    **kwargs,
) -> dict:
    """Create a raw market dict as returned by the Kalshi API."""
    return {
        "ticker": ticker,
        "event_ticker": event_ticker,
        "title": title,
        "subtitle": "",
        "status": status,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": 100 - yes_ask if yes_ask else None,
        "no_ask": 100 - yes_bid if yes_bid else None,
        "last_price": (yes_bid + yes_ask) // 2 if yes_bid and yes_ask else None,
        "volume": 1000,
        "open_interest": 500,
        "result": None,
        **kwargs,
    }


def _mock_response(markets: list[dict], status_code: int = 200, cursor: str = None):
    """Create a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {
        "markets": markets,
        "cursor": cursor,
    }
    resp.text = str({"markets": markets})
    return resp


# -------------------------------------------------------------------
# Tests: KalshiClient initialization
# -------------------------------------------------------------------

class TestKalshiClientInit:
    def test_creates_session(self) -> None:
        client = KalshiClient()
        assert client.session is not None
        assert client.timeout == 15

    def test_custom_timeout(self) -> None:
        client = KalshiClient(timeout=5)
        assert client.timeout == 5


# -------------------------------------------------------------------
# Tests: get_nba_markets
# -------------------------------------------------------------------

class TestGetNbaMarkets:
    def test_fetches_markets_successfully(self) -> None:
        client = KalshiClient()
        market_data = [_make_market()]
        client.session.get = MagicMock(return_value=_mock_response(market_data))

        markets = client.get_nba_markets()
        assert len(markets) == 1
        assert markets[0].ticker == "KXNBAGAME-26FEB28LALGSW-LAL"

    def test_uses_game_series_ticker(self) -> None:
        """Default should use KXNBAGAME series ticker."""
        client = KalshiClient()
        client.session.get = MagicMock(return_value=_mock_response([]))

        client.get_nba_markets()
        call_kwargs = client.session.get.call_args
        assert call_kwargs[1]["params"]["series_ticker"] == "KXNBAGAME"

    def test_pagination(self) -> None:
        client = KalshiClient()
        page1 = _mock_response([_make_market(ticker="M1")], cursor="page2")
        page2 = _mock_response([_make_market(ticker="M2")], cursor=None)
        client.session.get = MagicMock(side_effect=[page1, page2])

        markets = client.get_nba_markets()
        assert len(markets) == 2
        assert client.session.get.call_count == 2

    def test_http_error_raises(self) -> None:
        client = KalshiClient()
        error_resp = MagicMock()
        error_resp.status_code = 500
        error_resp.text = "Internal Server Error"
        client.session.get = MagicMock(return_value=error_resp)

        with pytest.raises(KalshiClientError, match="HTTP 500"):
            client.get_nba_markets()

    def test_network_error_raises(self) -> None:
        import requests as req
        client = KalshiClient()
        client.session.get = MagicMock(side_effect=req.ConnectionError("timeout"))

        with pytest.raises(KalshiClientError, match="HTTP request failed"):
            client.get_nba_markets()

    def test_invalid_json_raises(self) -> None:
        client = KalshiClient()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("not json")
        client.session.get = MagicMock(return_value=resp)

        with pytest.raises(KalshiClientError, match="Invalid JSON"):
            client.get_nba_markets()

    def test_empty_response(self) -> None:
        client = KalshiClient()
        client.session.get = MagicMock(return_value=_mock_response([]))
        assert client.get_nba_markets() == []


# -------------------------------------------------------------------
# Tests: get_nba_game_events — paired team market format
# -------------------------------------------------------------------

class TestGetNbaGameEvents:
    def _make_game_pair(self, event_ticker, away_abbr, home_abbr,
                        away_name, home_name, away_bid, away_ask,
                        home_bid, home_ask):
        """Create a pair of market dicts for one game."""
        title = f"{away_name} at {home_name} Winner?"
        return [
            _make_market(
                ticker=f"{event_ticker}-{away_abbr}",
                event_ticker=event_ticker,
                title=title,
                yes_bid=away_bid, yes_ask=away_ask,
            ),
            _make_market(
                ticker=f"{event_ticker}-{home_abbr}",
                event_ticker=event_ticker,
                title=title,
                yes_bid=home_bid, yes_ask=home_ask,
            ),
        ]

    def test_converts_game_pair(self) -> None:
        """Two paired markets should produce one OddsApiEvent."""
        client = KalshiClient()
        markets = self._make_game_pair(
            "KXNBAGAME-26FEB28LALGSW", "LAL", "GSW",
            "Los Angeles L", "Golden State",
            49, 62,  # LAL: bid=49, ask=62
            38, 42,  # GSW: bid=38, ask=42
        )
        client.session.get = MagicMock(return_value=_mock_response(markets))

        events = client.get_nba_game_events()
        assert len(events) == 1
        event = events[0]
        assert event.sport_key == "basketball_nba"
        assert event.bookmakers[0].key == "kalshi"
        assert len(event.bookmakers[0].markets[0].outcomes) == 2

    def test_home_away_correct(self) -> None:
        """Home/away should be parsed from 'Away at Home' title."""
        client = KalshiClient()
        markets = self._make_game_pair(
            "EVT1", "HOU", "MIA",
            "Houston", "Miami",
            57, 60, 39, 43,
        )
        client.session.get = MagicMock(return_value=_mock_response(markets))

        events = client.get_nba_game_events()
        assert len(events) == 1
        assert events[0].away_team == "Houston Rockets"
        assert events[0].home_team == "Miami Heat"

    def test_multiple_games(self) -> None:
        client = KalshiClient()
        markets = (
            self._make_game_pair(
                "EVT1", "LAL", "GSW", "Los Angeles L", "Golden State",
                49, 62, 38, 42,
            ) + self._make_game_pair(
                "EVT2", "HOU", "MIA", "Houston", "Miami",
                57, 60, 39, 43,
            )
        )
        client.session.get = MagicMock(return_value=_mock_response(markets))

        events = client.get_nba_game_events()
        assert len(events) == 2

    def test_unparseable_teams_skipped(self) -> None:
        client = KalshiClient()
        markets = [_make_market(
            event_ticker="EVT1",
            ticker="EVT1-XXX",
            title="Some Random Event Title",
        )]
        client.session.get = MagicMock(return_value=_mock_response(markets))

        events = client.get_nba_game_events()
        assert len(events) == 0

    def test_api_failure_returns_empty(self) -> None:
        import requests as req
        client = KalshiClient()
        client.session.get = MagicMock(side_effect=req.ConnectionError())

        events = client.get_nba_game_events()
        assert events == []

    def test_probabilities_from_both_markets(self) -> None:
        """Each team's mid-price should come from its own market."""
        client = KalshiClient()
        # LAL bid=60 ask=70 → mid=65%, GSW bid=30 ask=40 → mid=35%
        markets = self._make_game_pair(
            "EVT1", "LAL", "GSW", "Los Angeles L", "Golden State",
            60, 70, 30, 40,
        )
        client.session.get = MagicMock(return_value=_mock_response(markets))

        events = client.get_nba_game_events()
        assert len(events) == 1
        outcomes = events[0].bookmakers[0].markets[0].outcomes

        # Find which outcome is which
        from src.reference.vig_removal import american_to_probability
        probs = {o.name: american_to_probability(round(o.price)) for o in outcomes}
        # After normalization: LAL=65/100, GSW=35/100 → LAL=0.65, GSW=0.35
        assert probs["Los Angeles Lakers"] == pytest.approx(0.65, abs=0.05)
        assert probs["Golden State Warriors"] == pytest.approx(0.35, abs=0.05)

    def test_event_id_prefix(self) -> None:
        client = KalshiClient()
        markets = self._make_game_pair(
            "EVT1", "LAL", "GSW", "Los Angeles L", "Golden State",
            50, 55, 45, 50,
        )
        client.session.get = MagicMock(return_value=_mock_response(markets))

        events = client.get_nba_game_events()
        assert events[0].id.startswith("kalshi_")

    def test_single_market_derives_complement(self) -> None:
        """If only one team market has prices, derive the other."""
        client = KalshiClient()
        markets = [
            _make_market(
                ticker="EVT1-LAL",
                event_ticker="EVT1",
                title="Los Angeles L at Golden State Winner?",
                yes_bid=60, yes_ask=65,
            ),
            _make_market(
                ticker="EVT1-GSW",
                event_ticker="EVT1",
                title="Los Angeles L at Golden State Winner?",
                yes_bid=0, yes_ask=0,  # no liquidity
            ),
        ]
        client.session.get = MagicMock(return_value=_mock_response(markets))

        events = client.get_nba_game_events()
        assert len(events) == 1
        assert len(events[0].bookmakers[0].markets[0].outcomes) == 2


# -------------------------------------------------------------------
# Tests: mid-price computation
# -------------------------------------------------------------------

class TestMidPriceComputation:
    def test_bid_ask_midpoint(self) -> None:
        client = KalshiClient()
        market = KalshiMarket(ticker="T1", yes_bid=55, yes_ask=60)
        assert client._compute_mid_probability(market) == pytest.approx(0.575)

    def test_bid_only(self) -> None:
        client = KalshiClient()
        market = KalshiMarket(ticker="T1", yes_bid=60, yes_ask=None)
        assert client._compute_mid_probability(market) == pytest.approx(0.60)

    def test_ask_only(self) -> None:
        client = KalshiClient()
        market = KalshiMarket(ticker="T1", yes_bid=None, yes_ask=70)
        assert client._compute_mid_probability(market) == pytest.approx(0.70)

    def test_last_price_fallback(self) -> None:
        client = KalshiClient()
        market = KalshiMarket(ticker="T1", yes_bid=None, yes_ask=None, last_price=45)
        assert client._compute_mid_probability(market) == pytest.approx(0.45)

    def test_no_prices_returns_none(self) -> None:
        client = KalshiClient()
        market = KalshiMarket(ticker="T1", yes_bid=None, yes_ask=None, last_price=None)
        assert client._compute_mid_probability(market) is None

    def test_zero_prices_returns_none(self) -> None:
        client = KalshiClient()
        market = KalshiMarket(ticker="T1", yes_bid=0, yes_ask=0)
        assert client._compute_mid_probability(market) is None


# -------------------------------------------------------------------
# Tests: title parsing
# -------------------------------------------------------------------

class TestTitleParsing:
    def test_at_format(self) -> None:
        result = KalshiClient._parse_at_title("New Orleans at Utah Winner?")
        assert result is not None
        away, home = result
        assert away == "New Orleans Pelicans"
        assert home == "Utah Jazz"

    def test_at_format_abbreviated(self) -> None:
        result = KalshiClient._parse_at_title("Los Angeles L at Golden State Winner?")
        assert result is not None
        away, home = result
        assert away == "Los Angeles Lakers"
        assert home == "Golden State Warriors"

    def test_at_format_full_names(self) -> None:
        result = KalshiClient._parse_at_title("Houston at Miami Winner?")
        assert result is not None
        assert result[0] == "Houston Rockets"
        assert result[1] == "Miami Heat"

    def test_at_unrecognizable(self) -> None:
        assert KalshiClient._parse_at_title("Unknown at Mystery Winner?") is None

    def test_vs_fallback(self) -> None:
        result = KalshiClient._parse_vs_title("Lakers vs Celtics")
        assert result is not None
        assert "Los Angeles Lakers" in result
        assert "Boston Celtics" in result

    def test_no_pattern(self) -> None:
        assert KalshiClient._parse_at_title("Will the Lakers win?") is None
        assert KalshiClient._parse_vs_title("Will the Lakers win?") is None


# -------------------------------------------------------------------
# Tests: team identification from ticker suffix
# -------------------------------------------------------------------

class TestTeamIdentification:
    def test_ticker_suffix_lal(self) -> None:
        client = KalshiClient()
        m = KalshiMarket(ticker="KXNBAGAME-26FEB28LALGSW-LAL")
        result = client._identify_market_team(
            m, "Golden State Warriors", "Los Angeles Lakers",
        )
        assert result == "Los Angeles Lakers"

    def test_ticker_suffix_gsw(self) -> None:
        client = KalshiClient()
        m = KalshiMarket(ticker="KXNBAGAME-26FEB28LALGSW-GSW")
        result = client._identify_market_team(
            m, "Golden State Warriors", "Los Angeles Lakers",
        )
        assert result == "Golden State Warriors"

    def test_unknown_suffix_uses_title(self) -> None:
        client = KalshiClient()
        m = KalshiMarket(
            ticker="KXNBAGAME-EVT1-XXX",
            title="Houston at Miami Winner?",
        )
        result = client._identify_market_team(
            m, "Miami Heat", "Houston Rockets",
        )
        # Title contains both teams — normalize_team_name picks longest match
        assert result in ("Miami Heat", "Houston Rockets")


# -------------------------------------------------------------------
# Tests: probability to American odds conversion
# -------------------------------------------------------------------

class TestProbToAmerican:
    def test_favorite(self) -> None:
        assert _prob_to_american(0.6) < 0

    def test_underdog(self) -> None:
        assert _prob_to_american(0.3) > 0

    def test_extreme_clamp(self) -> None:
        assert _prob_to_american(0.0) > 0
        assert _prob_to_american(1.0) < 0


# -------------------------------------------------------------------
# Tests: regex patterns
# -------------------------------------------------------------------

class TestPatterns:
    def test_at_pattern(self) -> None:
        match = _AT_PATTERN.search("Houston at Miami Winner?")
        assert match is not None
        assert match.group(1).strip() == "Houston"
        assert match.group(2).strip() == "Miami"

    def test_vs_pattern(self) -> None:
        match = _VS_PATTERN.search("Lakers vs. Celtics")
        assert match is not None
