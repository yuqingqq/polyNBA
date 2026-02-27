"""Tests for the trade feed processor."""

import time

import pytest

from src.data.models import Side, TradeEvent
from src.data.trade_feed import TradeFeed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trade(
    asset_id: str = "asset_1",
    price: float = 0.50,
    size: float = 100.0,
    side: Side = Side.BUY,
    timestamp_ms: float | None = None,
) -> TradeEvent:
    """Create a TradeEvent with a millisecond-epoch timestamp."""
    if timestamp_ms is None:
        timestamp_ms = time.time() * 1000.0
    return TradeEvent(
        asset_id=asset_id,
        market="0xmarket1",
        price=price,
        size=size,
        side=side,
        timestamp=str(int(timestamp_ms)),
        fee_rate_bps="0",
    )


# ---------------------------------------------------------------------------
# Tests: recording trades
# ---------------------------------------------------------------------------


class TestRecordTrades:
    def test_record_single_trade(self) -> None:
        feed = TradeFeed()
        t = _make_trade()
        feed.on_trade(t)
        assert feed.last_price("asset_1") == 0.50

    def test_last_price_returns_most_recent(self) -> None:
        feed = TradeFeed()
        feed.on_trade(_make_trade(price=0.40))
        feed.on_trade(_make_trade(price=0.45))
        feed.on_trade(_make_trade(price=0.50))
        assert feed.last_price("asset_1") == 0.50

    def test_last_price_missing_asset(self) -> None:
        feed = TradeFeed()
        assert feed.last_price("nonexistent") is None

    def test_max_trades_respected(self) -> None:
        feed = TradeFeed(max_trades=5)
        for i in range(10):
            feed.on_trade(_make_trade(price=float(i)))
        trades = feed.recent_trades("asset_1", n=100)
        assert len(trades) == 5
        # Most recent trades should remain
        assert trades[-1].price == 9.0

    def test_recent_trades(self) -> None:
        feed = TradeFeed()
        for i in range(20):
            feed.on_trade(_make_trade(price=float(i)))
        recent = feed.recent_trades("asset_1", n=3)
        assert len(recent) == 3
        assert recent[0].price == 17.0
        assert recent[2].price == 19.0

    def test_recent_trades_empty(self) -> None:
        feed = TradeFeed()
        assert feed.recent_trades("no_trades") == []


# ---------------------------------------------------------------------------
# Tests: VWAP
# ---------------------------------------------------------------------------


class TestVWAP:
    def test_vwap_simple(self) -> None:
        """Two trades at the same time, different prices/sizes."""
        feed = TradeFeed()
        now_ms = time.time() * 1000.0
        feed.on_trade(_make_trade(price=0.50, size=100.0, timestamp_ms=now_ms))
        feed.on_trade(_make_trade(price=0.60, size=100.0, timestamp_ms=now_ms))

        vwap = feed.vwap("asset_1", window_seconds=60)
        assert vwap is not None
        assert vwap == pytest.approx(0.55)  # (50+60)/200 = 0.55

    def test_vwap_weighted(self) -> None:
        feed = TradeFeed()
        now_ms = time.time() * 1000.0
        feed.on_trade(_make_trade(price=0.40, size=300.0, timestamp_ms=now_ms))
        feed.on_trade(_make_trade(price=0.60, size=100.0, timestamp_ms=now_ms))

        vwap = feed.vwap("asset_1", window_seconds=60)
        assert vwap is not None
        # (0.40*300 + 0.60*100) / 400 = (120+60)/400 = 0.45
        assert vwap == pytest.approx(0.45)

    def test_vwap_excludes_old_trades(self) -> None:
        feed = TradeFeed()
        now_ms = time.time() * 1000.0
        old_ms = now_ms - 120_000  # 2 minutes ago

        feed.on_trade(_make_trade(price=0.30, size=1000.0, timestamp_ms=old_ms))
        feed.on_trade(_make_trade(price=0.50, size=100.0, timestamp_ms=now_ms))

        vwap = feed.vwap("asset_1", window_seconds=60)
        assert vwap is not None
        assert vwap == pytest.approx(0.50)

    def test_vwap_no_trades_in_window(self) -> None:
        feed = TradeFeed()
        old_ms = (time.time() - 300) * 1000.0
        feed.on_trade(_make_trade(price=0.50, size=100.0, timestamp_ms=old_ms))
        # Window of 60s should exclude the trade
        assert feed.vwap("asset_1", window_seconds=60) is None

    def test_vwap_missing_asset(self) -> None:
        feed = TradeFeed()
        assert feed.vwap("nonexistent", window_seconds=60) is None


# ---------------------------------------------------------------------------
# Tests: trade flow
# ---------------------------------------------------------------------------


class TestTradeFlow:
    def test_trade_flow_basic(self) -> None:
        feed = TradeFeed()
        now_ms = time.time() * 1000.0
        feed.on_trade(_make_trade(side=Side.BUY, size=100.0, timestamp_ms=now_ms))
        feed.on_trade(_make_trade(side=Side.SELL, size=60.0, timestamp_ms=now_ms))
        feed.on_trade(_make_trade(side=Side.BUY, size=40.0, timestamp_ms=now_ms))

        flow = feed.trade_flow("asset_1", window_seconds=60)
        assert flow["buy_volume"] == pytest.approx(140.0)
        assert flow["sell_volume"] == pytest.approx(60.0)
        assert flow["net_flow"] == pytest.approx(80.0)

    def test_trade_flow_excludes_old(self) -> None:
        feed = TradeFeed()
        now_ms = time.time() * 1000.0
        old_ms = now_ms - 120_000

        feed.on_trade(_make_trade(side=Side.BUY, size=500.0, timestamp_ms=old_ms))
        feed.on_trade(_make_trade(side=Side.SELL, size=10.0, timestamp_ms=now_ms))

        flow = feed.trade_flow("asset_1", window_seconds=60)
        assert flow["buy_volume"] == pytest.approx(0.0)
        assert flow["sell_volume"] == pytest.approx(10.0)
        assert flow["net_flow"] == pytest.approx(-10.0)

    def test_trade_flow_missing_asset(self) -> None:
        feed = TradeFeed()
        flow = feed.trade_flow("nonexistent", window_seconds=60)
        assert flow["buy_volume"] == 0.0
        assert flow["sell_volume"] == 0.0
        assert flow["net_flow"] == 0.0
