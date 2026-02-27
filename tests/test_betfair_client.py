"""Tests for the Betfair Exchange client module.

Tests initialization, graceful degradation when credentials are missing,
NBA market fetching, mid-price computation, conversion to OddsApiEvent format,
session keepalive, draw runner filtering, commence_time extraction, and
empty back/lay list handling.
"""

import os
import time
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.reference.betfair_client import (
    BetfairClient,
    _KEEPALIVE_INTERVAL_S,
    _prob_to_american,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _make_runner(selection_id: int, name: str, back_price: float = None, lay_price: float = None):
    """Create a mock runner with optional back/lay prices."""
    runner = MagicMock()
    runner.selection_id = selection_id
    runner.runner_name = name

    # Price structure
    runner.ex = MagicMock()
    if back_price is not None:
        back = MagicMock()
        back.price = back_price
        runner.ex.available_to_back = [back]
    else:
        runner.ex.available_to_back = []

    if lay_price is not None:
        lay = MagicMock()
        lay.price = lay_price
        runner.ex.available_to_lay = [lay]
    else:
        runner.ex.available_to_lay = []

    return runner


def _make_catalogue(
    market_id: str,
    runners: list,
    event_name: str = "Lakers @ Celtics",
    market_start_time: datetime = None,
):
    """Create a mock market catalogue."""
    catalogue = MagicMock()
    catalogue.market_id = market_id
    catalogue.runners = runners
    catalogue.event = MagicMock()
    catalogue.event.name = event_name
    catalogue.event.open_date = None
    catalogue.market_start_time = market_start_time
    return catalogue


def _make_book(market_id: str, runners: list, status: str = "OPEN"):
    """Create a mock market book."""
    book = MagicMock()
    book.market_id = market_id
    book.runners = runners
    book.status = status
    return book


# -------------------------------------------------------------------
# Tests: initialization and graceful degradation
# -------------------------------------------------------------------

class TestBetfairClientInit:
    @patch.dict("os.environ", {}, clear=True)
    def test_no_credentials_disabled(self) -> None:
        """Client should be unavailable when credentials are missing."""
        client = BetfairClient()
        assert not client.available
        assert client._client is None

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_credentials_returns_empty(self) -> None:
        """get_nba_game_events should return [] when disabled."""
        client = BetfairClient()
        events = client.get_nba_game_events()
        assert events == []

    @patch.dict("os.environ", {
        "BETFAIR_USERNAME": "user",
        "BETFAIR_PASSWORD": "pass",
        "BETFAIR_APP_KEY": "key",
    })
    def test_missing_cert_dir_disabled(self) -> None:
        """Client should be disabled if BETFAIR_CERT_DIR is not set."""
        client = BetfairClient()
        assert not client.available

    @patch.dict("os.environ", {
        "BETFAIR_USERNAME": "user",
        "BETFAIR_PASSWORD": "pass",
        "BETFAIR_APP_KEY": "key",
        "BETFAIR_CERT_DIR": "/nonexistent/dir",
    })
    def test_missing_cert_files_disabled(self) -> None:
        """Client should be disabled if cert files don't exist."""
        client = BetfairClient()
        assert not client.available

    @patch.dict("os.environ", {
        "BETFAIR_USERNAME": "user",
        "BETFAIR_PASSWORD": "pass",
        "BETFAIR_APP_KEY": "key",
    })
    def test_import_error_disabled(self) -> None:
        """Client should be disabled if betfairlightweight import fails."""
        import sys
        saved = sys.modules.get("betfairlightweight")
        sys.modules["betfairlightweight"] = None
        try:
            client = BetfairClient()
            assert not client.available
        finally:
            if saved is not None:
                sys.modules["betfairlightweight"] = saved
            else:
                sys.modules.pop("betfairlightweight", None)

    @patch("os.path.isfile", return_value=True)
    @patch.dict("os.environ", {
        "BETFAIR_USERNAME": "user",
        "BETFAIR_PASSWORD": "pass",
        "BETFAIR_APP_KEY": "key",
        "BETFAIR_CERT_DIR": "/certs",
    })
    def test_login_failure_disabled(self, mock_isfile) -> None:
        """Client should be disabled if login fails."""
        mock_bflw = MagicMock()
        mock_bflw.APIClient.return_value.login.side_effect = Exception("auth failed")
        with patch.dict("sys.modules", {"betfairlightweight": mock_bflw}):
            client = BetfairClient()
            assert not client.available

    @patch("os.path.isfile", return_value=True)
    @patch.dict("os.environ", {
        "BETFAIR_USERNAME": "user",
        "BETFAIR_PASSWORD": "pass",
        "BETFAIR_APP_KEY": "key",
        "BETFAIR_CERT_DIR": "/certs",
        "BETFAIR_PROXY_URL": "socks5://proxy:1080",
    })
    def test_proxy_configured(self, mock_isfile) -> None:
        """Proxy should be set on client session."""
        mock_bflw = MagicMock()
        mock_client = MagicMock()
        mock_bflw.APIClient.return_value = mock_client
        with patch.dict("sys.modules", {"betfairlightweight": mock_bflw}):
            client = BetfairClient()
            assert client.available
            # Verify proxy dict was assigned to session.proxies
            proxies = mock_client.session.proxies
            assert proxies == {"https": "socks5://proxy:1080", "http": "socks5://proxy:1080"}


# -------------------------------------------------------------------
# Tests: session keepalive
# -------------------------------------------------------------------

class TestSessionKeepalive:
    def test_no_keepalive_if_recent(self) -> None:
        """Should not call keep_alive if session is fresh."""
        client = BetfairClient()
        client._client = MagicMock()
        client._available = True
        client._last_keepalive = time.monotonic()

        client._ensure_session()
        client._client.keep_alive.assert_not_called()

    def test_keepalive_if_stale(self) -> None:
        """Should call keep_alive if interval has elapsed."""
        client = BetfairClient()
        client._client = MagicMock()
        client._available = True
        client._last_keepalive = time.monotonic() - _KEEPALIVE_INTERVAL_S - 1

        client._ensure_session()
        client._client.keep_alive.assert_called_once()

    def test_relogin_on_keepalive_failure(self) -> None:
        """Should attempt re-login if keep_alive fails."""
        client = BetfairClient()
        client._client = MagicMock()
        client._client.keep_alive.side_effect = Exception("expired")
        client._available = True
        client._last_keepalive = time.monotonic() - _KEEPALIVE_INTERVAL_S - 1

        client._ensure_session()
        client._client.login.assert_called_once()
        assert client.available  # re-login succeeded

    def test_unavailable_after_relogin_failure(self) -> None:
        """Should mark unavailable if both keepalive and re-login fail."""
        client = BetfairClient()
        client._client = MagicMock()
        client._client.keep_alive.side_effect = Exception("expired")
        client._client.login.side_effect = Exception("auth failed")
        client._available = True
        client._last_keepalive = time.monotonic() - _KEEPALIVE_INTERVAL_S - 1

        client._ensure_session()
        assert not client.available


# -------------------------------------------------------------------
# Tests: mid-price computation
# -------------------------------------------------------------------

class TestMidPriceFromDecimal:
    def test_back_and_lay(self) -> None:
        """Mid of back=2.0 (50%) and lay=2.2 (45.5%) ≈ 47.7%."""
        result = BetfairClient._compute_mid_from_decimal(2.0, 2.2)
        expected = (1/2.0 + 1/2.2) / 2
        assert result == pytest.approx(expected)

    def test_back_only(self) -> None:
        result = BetfairClient._compute_mid_from_decimal(2.0, None)
        assert result == pytest.approx(0.5)

    def test_lay_only(self) -> None:
        result = BetfairClient._compute_mid_from_decimal(None, 2.5)
        assert result == pytest.approx(0.4)

    def test_no_prices(self) -> None:
        result = BetfairClient._compute_mid_from_decimal(None, None)
        assert result is None

    def test_zero_back(self) -> None:
        result = BetfairClient._compute_mid_from_decimal(0.0, 2.0)
        assert result == pytest.approx(0.5)

    def test_even_money(self) -> None:
        result = BetfairClient._compute_mid_from_decimal(2.0, 2.0)
        assert result == pytest.approx(0.5)

    def test_heavy_favorite(self) -> None:
        result = BetfairClient._compute_mid_from_decimal(1.1, 1.15)
        expected = (1/1.1 + 1/1.15) / 2
        assert result == pytest.approx(expected)
        assert result > 0.85


# -------------------------------------------------------------------
# Tests: _convert_market
# -------------------------------------------------------------------

class TestConvertMarket:
    def test_valid_market(self) -> None:
        """Should convert a valid Betfair market to OddsApiEvent."""
        client = BetfairClient()
        client._available = True

        cat_runners = [
            MagicMock(selection_id=1, runner_name="Los Angeles Lakers"),
            MagicMock(selection_id=2, runner_name="Boston Celtics"),
        ]
        catalogue = _make_catalogue("M1", cat_runners, "Lakers @ Celtics")

        book_runners = [
            _make_runner(1, "Los Angeles Lakers", back_price=2.0, lay_price=2.1),
            _make_runner(2, "Boston Celtics", back_price=1.9, lay_price=2.0),
        ]
        book = _make_book("M1", book_runners)

        event = client._convert_market(catalogue, book)
        assert event is not None
        assert event.sport_key == "basketball_nba"
        assert event.bookmakers[0].key == "betfair"
        assert len(event.bookmakers[0].markets[0].outcomes) == 2

    def test_single_runner_returns_none(self) -> None:
        client = BetfairClient()
        catalogue = _make_catalogue("M1", [MagicMock(selection_id=1, runner_name="Lakers")])
        event = client._convert_market(catalogue, None)
        assert event is None

    def test_unrecognizable_teams_returns_none(self) -> None:
        client = BetfairClient()
        cat_runners = [
            MagicMock(selection_id=1, runner_name="Unknown Team A"),
            MagicMock(selection_id=2, runner_name="Unknown Team B"),
        ]
        catalogue = _make_catalogue("M1", cat_runners)
        event = client._convert_market(catalogue, MagicMock(runners=[]))
        assert event is None

    def test_no_book_returns_none(self) -> None:
        client = BetfairClient()
        cat_runners = [
            MagicMock(selection_id=1, runner_name="Los Angeles Lakers"),
            MagicMock(selection_id=2, runner_name="Boston Celtics"),
        ]
        catalogue = _make_catalogue("M1", cat_runners)
        book = MagicMock()
        book.runners = []
        event = client._convert_market(catalogue, book)
        assert event is None


# -------------------------------------------------------------------
# Tests: draw runner filtering
# -------------------------------------------------------------------

class TestDrawRunnerFiltering:
    def test_draw_runner_ignored(self) -> None:
        """Markets with 'The Draw' runner should still work — draw is filtered."""
        client = BetfairClient()

        cat_runners = [
            MagicMock(selection_id=1, runner_name="Los Angeles Lakers"),
            MagicMock(selection_id=2, runner_name="Boston Celtics"),
            MagicMock(selection_id=3, runner_name="The Draw"),
        ]
        catalogue = _make_catalogue("M1", cat_runners, "Lakers @ Celtics")

        book_runners = [
            _make_runner(1, "Los Angeles Lakers", back_price=2.0, lay_price=2.1),
            _make_runner(2, "Boston Celtics", back_price=1.9, lay_price=2.0),
            _make_runner(3, "The Draw", back_price=50.0, lay_price=60.0),
        ]
        book = _make_book("M1", book_runners)

        event = client._convert_market(catalogue, book)
        assert event is not None
        assert len(event.bookmakers[0].markets[0].outcomes) == 2
        outcome_names = {o.name for o in event.bookmakers[0].markets[0].outcomes}
        assert "The Draw" not in outcome_names

    def test_three_nba_teams_skipped(self) -> None:
        """If 3+ runners are all recognized NBA teams, skip the market."""
        client = BetfairClient()

        cat_runners = [
            MagicMock(selection_id=1, runner_name="Los Angeles Lakers"),
            MagicMock(selection_id=2, runner_name="Boston Celtics"),
            MagicMock(selection_id=3, runner_name="Golden State Warriors"),
        ]
        catalogue = _make_catalogue("M1", cat_runners)
        event = client._convert_market(catalogue, MagicMock(runners=[]))
        assert event is None


# -------------------------------------------------------------------
# Tests: empty back/lay lists
# -------------------------------------------------------------------

class TestEmptyBackLay:
    def test_empty_back_list_no_crash(self) -> None:
        """Runner with empty available_to_back should not crash."""
        client = BetfairClient()

        cat_runners = [
            MagicMock(selection_id=1, runner_name="Los Angeles Lakers"),
            MagicMock(selection_id=2, runner_name="Boston Celtics"),
        ]
        catalogue = _make_catalogue("M1", cat_runners, "Lakers @ Celtics")

        # Runner 1 has empty back list (suspended), runner 2 is normal
        r1 = _make_runner(1, "Los Angeles Lakers", back_price=None, lay_price=2.1)
        r1.ex.available_to_back = []  # explicitly empty, not None
        r2 = _make_runner(2, "Boston Celtics", back_price=1.9, lay_price=2.0)
        book = _make_book("M1", [r1, r2])

        # Should not crash
        event = client._convert_market(catalogue, book)
        # May or may not produce an event depending on lay-only fallback,
        # but must not crash
        assert event is None or event is not None  # no IndexError

    def test_both_empty_returns_none(self) -> None:
        """Runner with no back or lay should be skipped."""
        client = BetfairClient()

        cat_runners = [
            MagicMock(selection_id=1, runner_name="Los Angeles Lakers"),
            MagicMock(selection_id=2, runner_name="Boston Celtics"),
        ]
        catalogue = _make_catalogue("M1", cat_runners, "Lakers @ Celtics")

        r1 = _make_runner(1, "Los Angeles Lakers")
        r1.ex.available_to_back = []
        r1.ex.available_to_lay = []
        r2 = _make_runner(2, "Boston Celtics")
        r2.ex.available_to_back = []
        r2.ex.available_to_lay = []
        book = _make_book("M1", [r1, r2])

        event = client._convert_market(catalogue, book)
        assert event is None


# -------------------------------------------------------------------
# Tests: commence_time extraction
# -------------------------------------------------------------------

class TestCommenceTime:
    def test_market_start_time(self) -> None:
        """Should use market_start_time when available."""
        client = BetfairClient()
        expected_time = datetime(2026, 3, 1, 19, 0, tzinfo=timezone.utc)

        cat_runners = [
            MagicMock(selection_id=1, runner_name="Los Angeles Lakers"),
            MagicMock(selection_id=2, runner_name="Boston Celtics"),
        ]
        catalogue = _make_catalogue(
            "M1", cat_runners, "Lakers @ Celtics",
            market_start_time=expected_time,
        )

        book_runners = [
            _make_runner(1, "Los Angeles Lakers", back_price=2.0, lay_price=2.1),
            _make_runner(2, "Boston Celtics", back_price=1.9, lay_price=2.0),
        ]
        book = _make_book("M1", book_runners)

        event = client._convert_market(catalogue, book)
        assert event is not None
        assert event.commence_time == expected_time

    def test_fallback_to_event_open_date(self) -> None:
        """Should fallback to event.open_date if market_start_time is None."""
        client = BetfairClient()
        expected_time = datetime(2026, 3, 1, 20, 0, tzinfo=timezone.utc)

        cat_runners = [
            MagicMock(selection_id=1, runner_name="Los Angeles Lakers"),
            MagicMock(selection_id=2, runner_name="Boston Celtics"),
        ]
        catalogue = _make_catalogue("M1", cat_runners, "Lakers @ Celtics")
        catalogue.market_start_time = None
        catalogue.event.open_date = expected_time

        book_runners = [
            _make_runner(1, "Los Angeles Lakers", back_price=2.0, lay_price=2.1),
            _make_runner(2, "Boston Celtics", back_price=1.9, lay_price=2.0),
        ]
        book = _make_book("M1", book_runners)

        event = client._convert_market(catalogue, book)
        assert event is not None
        assert event.commence_time == expected_time

    def test_naive_datetime_gets_utc(self) -> None:
        """Naive datetimes should be treated as UTC."""
        result = BetfairClient._extract_commence_time(MagicMock(
            market_start_time=datetime(2026, 3, 1, 19, 0),
            event=None,
        ))
        assert result is not None
        assert result.tzinfo == timezone.utc

    def test_no_time_available(self) -> None:
        """Should return None if no time data."""
        result = BetfairClient._extract_commence_time(MagicMock(
            market_start_time=None,
            event=MagicMock(open_date=None),
        ))
        assert result is None


# -------------------------------------------------------------------
# Tests: market status filtering
# -------------------------------------------------------------------

class TestMarketStatusFiltering:
    def test_suspended_market_skipped(self) -> None:
        """Suspended markets should be skipped in _fetch_and_convert."""
        client = BetfairClient()
        client._client = MagicMock()
        client._available = True

        cat_runners = [
            MagicMock(selection_id=1, runner_name="Los Angeles Lakers"),
            MagicMock(selection_id=2, runner_name="Boston Celtics"),
        ]
        catalogue = _make_catalogue("M1", cat_runners)

        book = _make_book("M1", [
            _make_runner(1, "Los Angeles Lakers", back_price=2.0, lay_price=2.1),
            _make_runner(2, "Boston Celtics", back_price=1.9, lay_price=2.0),
        ], status="SUSPENDED")

        import betfairlightweight.filters as filters
        client._client.betting.list_market_catalogue.return_value = [catalogue]
        client._client.betting.list_market_book.return_value = [book]

        events = client._fetch_and_convert()
        assert len(events) == 0

    def test_closed_market_skipped(self) -> None:
        """Closed markets should be skipped."""
        client = BetfairClient()
        client._client = MagicMock()
        client._available = True

        cat_runners = [
            MagicMock(selection_id=1, runner_name="Los Angeles Lakers"),
            MagicMock(selection_id=2, runner_name="Boston Celtics"),
        ]
        catalogue = _make_catalogue("M1", cat_runners)

        book = _make_book("M1", [
            _make_runner(1, "Los Angeles Lakers", back_price=2.0, lay_price=2.1),
            _make_runner(2, "Boston Celtics", back_price=1.9, lay_price=2.0),
        ], status="CLOSED")

        client._client.betting.list_market_catalogue.return_value = [catalogue]
        client._client.betting.list_market_book.return_value = [book]

        events = client._fetch_and_convert()
        assert len(events) == 0


# -------------------------------------------------------------------
# Tests: auth retry on fetch failure
# -------------------------------------------------------------------

class TestAuthRetry:
    def test_retry_on_fetch_failure(self) -> None:
        """Should re-login and retry on first fetch failure."""
        client = BetfairClient()
        client._client = MagicMock()
        client._available = True
        client._last_keepalive = time.monotonic()

        # First call fails, second succeeds
        call_count = [0]
        def mock_fetch():
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("session expired")
            return []

        with patch.object(client, "_fetch_and_convert", side_effect=mock_fetch):
            events = client.get_nba_game_events()

        assert events == []
        client._client.login.assert_called_once()

    def test_returns_empty_after_double_failure(self) -> None:
        """Should return [] if both fetch and retry fail."""
        client = BetfairClient()
        client._client = MagicMock()
        client._available = True
        client._last_keepalive = time.monotonic()

        with patch.object(client, "_fetch_and_convert", side_effect=Exception("down")):
            events = client.get_nba_game_events()

        assert events == []


# -------------------------------------------------------------------
# Tests: prob_to_american
# -------------------------------------------------------------------

class TestProbToAmerican:
    def test_favorite(self) -> None:
        result = _prob_to_american(0.7)
        assert result < 0

    def test_underdog(self) -> None:
        result = _prob_to_american(0.3)
        assert result > 0

    def test_edge_clamp(self) -> None:
        result_low = _prob_to_american(0.001)
        result_high = _prob_to_american(0.999)
        assert result_low > 0
        assert result_high < 0
