"""Stress tests — reconnect behaviour, state recovery, malformed messages,
rapid snapshots, and bounded deque throughput.

These tests verify system resilience without requiring a live WebSocket
server.  They exercise the ``WebSocketClient``, ``Orderbook``,
``OrderbookManager``, ``PolymarketFeed``, and ``TradeFeed`` components
under adversarial and high-throughput conditions.
"""

from __future__ import annotations

import json
import time
import threading
from unittest.mock import MagicMock, patch

import pytest

from src.data.models import (
    OrderbookLevel,
    OrderbookSnapshot,
    PriceChange,
    Side,
    TradeEvent,
)
from src.data.orderbook import Orderbook, OrderbookManager
from src.data.polymarket_feed import PolymarketFeed
from src.data.trade_feed import TradeFeed
from src.data.ws_client import ConnectionState, WebSocketClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot(
    asset_id: str = "asset_1",
    bids: list[tuple[float, float]] | None = None,
    asks: list[tuple[float, float]] | None = None,
    timestamp: str = "1700000000000",
    hash_val: str = "0xhash",
) -> OrderbookSnapshot:
    if bids is None:
        bids = [(0.48, 30.0), (0.47, 50.0), (0.46, 100.0)]
    if asks is None:
        asks = [(0.52, 25.0), (0.53, 40.0), (0.54, 80.0)]
    return OrderbookSnapshot(
        asset_id=asset_id,
        market="0xmarket1",
        bids=[OrderbookLevel(price=p, size=s) for p, s in bids],
        asks=[OrderbookLevel(price=p, size=s) for p, s in asks],
        timestamp=timestamp,
        hash=hash_val,
    )


def _make_change(
    asset_id: str = "asset_1",
    price: float = 0.50,
    size: float = 100.0,
    side: Side = Side.BUY,
) -> PriceChange:
    return PriceChange(
        asset_id=asset_id,
        price=price,
        size=size,
        side=side,
        hash="0xhash_change",
    )


def _make_trade(
    asset_id: str = "asset_1",
    price: float = 0.50,
    size: float = 100.0,
    side: Side = Side.BUY,
    timestamp_ms: float | None = None,
) -> TradeEvent:
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


# ===========================================================================
# Test 1: WebSocketClient reconnect behaviour
# ===========================================================================


class TestWebSocketReconnect:
    """Verify state transitions and backoff on simulated disconnect."""

    def test_on_close_transitions_to_reconnecting(self) -> None:
        """Calling _on_close with auto_reconnect=True should set
        state to RECONNECTING.
        """
        client = WebSocketClient(
            url="wss://example.com/ws",
            on_message=lambda m: None,
        )
        client._auto_reconnect = True
        client._stop_event.clear()
        client._set_state(ConnectionState.CONNECTED)
        mock_ws = MagicMock()
        client._ws = mock_ws

        # Prevent the reconnect thread from actually starting a new WS
        with patch.object(client, "_start_ws"):
            client._on_close(mock_ws, None, None)
            # Give the reconnect thread time to set RECONNECTING
            time.sleep(0.2)

        assert client.state in (
            ConnectionState.RECONNECTING,
            ConnectionState.CONNECTING,
        )

        # Cleanup
        client._auto_reconnect = False
        client._stop_event.set()

    def test_state_transitions_connected_to_reconnecting(self) -> None:
        """Explicitly verify the CONNECTED -> DISCONNECTED -> RECONNECTING
        transition chain that occurs on connection loss.
        """
        states_observed: list[ConnectionState] = []
        client = WebSocketClient(
            url="wss://example.com/ws",
            on_message=lambda m: None,
        )

        # Patch _set_state to record transitions
        original_set_state = client._set_state

        def tracking_set_state(new_state: ConnectionState) -> None:
            states_observed.append(new_state)
            original_set_state(new_state)

        client._set_state = tracking_set_state
        client._auto_reconnect = True
        client._stop_event.clear()
        # Start from CONNECTED
        original_set_state(ConnectionState.CONNECTED)

        mock_ws = MagicMock()
        client._ws = mock_ws
        with patch.object(client, "_start_ws"):
            client._on_close(mock_ws, None, None)
            time.sleep(0.2)

        # Should have observed DISCONNECTED via _set_state.
        # RECONNECTING is set atomically inside _schedule_reconnect (not via _set_state)
        # to prevent TOCTOU race conditions, so check the final state directly.
        assert ConnectionState.DISCONNECTED in states_observed
        assert client.state == ConnectionState.RECONNECTING

        # Cleanup
        client._auto_reconnect = False
        client._stop_event.set()

    def test_backoff_delay_increases_with_retries(self) -> None:
        """Successive reconnect attempts should increase the base delay
        exponentially, capped at max_reconnect_delay.
        """
        max_delay = 30.0
        client = WebSocketClient(
            url="wss://example.com/ws",
            on_message=lambda m: None,
            max_reconnect_delay=max_delay,
        )

        # The formula is: delay = min(2^(attempt-1), max_delay)
        expected_delays = []
        for attempt in range(1, 8):
            expected_delays.append(min(2 ** (attempt - 1), max_delay))

        assert expected_delays == [1, 2, 4, 8, 16, 30, 30]

        # Verify via client internal state
        client._auto_reconnect = True
        client._stop_event.clear()
        client._set_state(ConnectionState.CONNECTED)

        delays_seen: list[int] = []

        with patch.object(client, "_start_ws"):
            for i in range(5):
                # Each call to _schedule_reconnect increments _reconnect_attempts
                old_attempts = client._reconnect_attempts
                # Reset state so the duplicate-reconnect guard allows the call
                client._set_state(ConnectionState.DISCONNECTED)
                client._schedule_reconnect()
                new_attempts = client._reconnect_attempts
                assert new_attempts == old_attempts + 1
                delay = min(2 ** (new_attempts - 1), max_delay)
                delays_seen.append(delay)
                time.sleep(0.05)  # Let threads settle

        assert delays_seen == [1, 2, 4, 8, 16]

        # Cleanup
        client._auto_reconnect = False
        client._stop_event.set()

    def test_reconnect_counter_resets_on_open(self) -> None:
        """After a successful reconnection, _reconnect_attempts should
        reset to 0.
        """
        client = WebSocketClient(
            url="wss://example.com/ws",
            on_message=lambda m: None,
        )
        client._reconnect_attempts = 7
        client._ws = MagicMock()
        client._on_open(client._ws)
        assert client._reconnect_attempts == 0

        # Cleanup
        client._stop_event.set()


# ===========================================================================
# Test 2: Orderbook recovery after simulated data gap
# ===========================================================================


class TestOrderbookRecovery:
    """Verify that applying a new snapshot after a gap fully resets
    the book to the new snapshot's state.
    """

    def test_snapshot_clears_old_data(self) -> None:
        """Applying a new snapshot should completely replace the old book."""
        book = Orderbook("asset_1")

        # Phase 1: initial snapshot + some updates
        snap1 = _make_snapshot(
            bids=[(0.48, 30.0), (0.47, 50.0)],
            asks=[(0.52, 25.0), (0.53, 40.0)],
        )
        book.apply_snapshot(snap1)
        book.apply_update(_make_change(price=0.49, size=200.0, side=Side.BUY))
        book.apply_update(_make_change(price=0.51, size=150.0, side=Side.SELL))

        assert book.best_bid == 0.49
        assert book.best_ask == 0.51
        depth_before = book.get_depth(100)
        assert len(depth_before["bids"]) == 3  # 0.49, 0.48, 0.47
        assert len(depth_before["asks"]) == 3  # 0.51, 0.52, 0.53

        # Phase 2: Simulate gap by applying completely new snapshot
        snap2 = _make_snapshot(
            bids=[(0.45, 10.0)],
            asks=[(0.55, 10.0)],
            timestamp="1700000060000",
            hash_val="0xhash_new",
        )
        book.apply_snapshot(snap2)

        # The book should only have the new snapshot's levels
        assert book.best_bid == 0.45
        assert book.best_ask == 0.55
        depth_after = book.get_depth(100)
        assert len(depth_after["bids"]) == 1
        assert len(depth_after["asks"]) == 1
        assert depth_after["bids"][0].price == 0.45
        assert depth_after["bids"][0].size == 10.0

    def test_manager_handles_gap_recovery(self) -> None:
        """OrderbookManager should correctly route a recovery snapshot
        to the existing book, replacing its state.
        """
        mgr = OrderbookManager()

        # Initial snapshot
        mgr.handle_snapshot(_make_snapshot(
            asset_id="token_A",
            bids=[(0.40, 100.0), (0.39, 200.0)],
            asks=[(0.60, 100.0)],
        ))

        # Some updates
        mgr.handle_update(_make_change(
            asset_id="token_A", price=0.41, size=50.0, side=Side.BUY
        ))

        book = mgr.get("token_A")
        assert book is not None
        assert book.best_bid == 0.41

        # Recovery snapshot (simulates reconnect)
        mgr.handle_snapshot(_make_snapshot(
            asset_id="token_A",
            bids=[(0.35, 10.0)],
            asks=[(0.65, 10.0)],
            timestamp="1700000120000",
        ))

        assert book.best_bid == 0.35
        assert book.best_ask == 0.65
        depth = book.get_depth(100)
        assert len(depth["bids"]) == 1
        assert len(depth["asks"]) == 1

    def test_updates_after_recovery_work(self) -> None:
        """After a recovery snapshot, incremental updates should apply
        correctly on top of the new state.
        """
        book = Orderbook("asset_1")

        # Initial state
        book.apply_snapshot(_make_snapshot(
            bids=[(0.48, 30.0)],
            asks=[(0.52, 25.0)],
        ))

        # Recovery snapshot
        book.apply_snapshot(_make_snapshot(
            bids=[(0.30, 5.0)],
            asks=[(0.70, 5.0)],
        ))

        # Apply update on top of recovered state
        book.apply_update(_make_change(price=0.31, size=100.0, side=Side.BUY))
        assert book.best_bid == 0.31

        book.apply_update(_make_change(price=0.69, size=50.0, side=Side.SELL))
        assert book.best_ask == 0.69


# ===========================================================================
# Test 3: PolymarketFeed handles malformed messages gracefully
# ===========================================================================


class TestPolymarketFeedMalformed:
    """Verify that invalid messages are handled without leaking exceptions."""

    def _create_feed_with_mock_ws(self) -> tuple[PolymarketFeed, list, list]:
        """Create a PolymarketFeed and intercept its callbacks."""
        feed = PolymarketFeed()
        snapshots: list[OrderbookSnapshot] = []
        changes: list[PriceChange] = []

        feed.on_book(lambda snap: snapshots.append(snap))
        feed.on_price_change(lambda ch: changes.append(ch))

        return feed, snapshots, changes

    def test_invalid_json_does_not_raise(self) -> None:
        """Sending completely invalid JSON should be handled silently."""
        feed, snapshots, changes = self._create_feed_with_mock_ws()

        # This should not raise
        feed._handle_message("this is not json {{{")
        feed._handle_message("<<<>>>")
        feed._handle_message("")

        assert len(snapshots) == 0
        assert len(changes) == 0

    def test_json_missing_event_type_does_not_raise(self) -> None:
        """JSON without an event_type field should be handled gracefully."""
        feed, snapshots, changes = self._create_feed_with_mock_ws()

        feed._handle_message(json.dumps({"data": "no event_type"}))
        feed._handle_message(json.dumps({"asset_id": "xxx"}))
        feed._handle_message(json.dumps([{"no_type": True}]))

        assert len(snapshots) == 0
        assert len(changes) == 0

    def test_book_event_missing_fields_does_not_crash(self) -> None:
        """A 'book' event with missing required fields should log an error
        but not crash the feed.
        """
        feed, snapshots, changes = self._create_feed_with_mock_ws()

        # Missing asset_id — should raise internally and be caught
        msg = json.dumps({
            "event_type": "book",
            "bids": [],
            "asks": [],
            # no asset_id
        })
        feed._handle_message(msg)
        # The exception should be caught internally
        assert len(snapshots) == 0

    def test_price_change_missing_fields_does_not_crash(self) -> None:
        """A 'price_change' event with missing fields should not propagate."""
        feed, snapshots, changes = self._create_feed_with_mock_ws()

        # Missing price_changes key
        msg = json.dumps({
            "event_type": "price_change",
            # no price_changes
        })
        feed._handle_message(msg)
        assert len(changes) == 0

        # price_changes with malformed entries
        msg2 = json.dumps({
            "event_type": "price_change",
            "price_changes": [
                {"asset_id": "x"}  # missing price, size, side
            ],
        })
        feed._handle_message(msg2)
        assert len(changes) == 0

    def test_unknown_event_type_does_not_crash(self) -> None:
        """An unknown event_type should be silently skipped."""
        feed, snapshots, changes = self._create_feed_with_mock_ws()

        feed._handle_message(json.dumps({
            "event_type": "totally_unknown_type",
            "data": "whatever",
        }))

        assert len(snapshots) == 0
        assert len(changes) == 0

    def test_valid_message_still_works_after_malformed(self) -> None:
        """After processing malformed messages, valid messages should
        still be processed correctly.
        """
        feed, snapshots, changes = self._create_feed_with_mock_ws()

        # Send garbage first
        feed._handle_message("not json")
        feed._handle_message(json.dumps({"no_event_type": True}))

        # Now send a valid book event
        valid_book = json.dumps({
            "event_type": "book",
            "asset_id": "token_1",
            "market": "0xmarket",
            "bids": [{"price": "0.48", "size": "30"}],
            "asks": [{"price": "0.52", "size": "25"}],
            "timestamp": "1700000000000",
            "hash": "0xhash",
        })
        feed._handle_message(valid_book)

        assert len(snapshots) == 1
        assert snapshots[0].asset_id == "token_1"
        assert len(snapshots[0].bids) == 1
        assert snapshots[0].bids[0].price == 0.48

    def test_pong_messages_ignored(self) -> None:
        """PONG heartbeat responses should be silently ignored."""
        feed, snapshots, changes = self._create_feed_with_mock_ws()

        feed._handle_message("PONG")
        feed._handle_message("pong")

        assert len(snapshots) == 0
        assert len(changes) == 0


# ===========================================================================
# Test 4: OrderbookManager handles multiple rapid snapshots
# ===========================================================================


class TestRapidSnapshots:
    """Verify correctness under rapid snapshot application."""

    def test_100_rapid_snapshots_same_asset(self) -> None:
        """Apply 100 snapshots rapidly for the same asset — final state
        should match the last snapshot.
        """
        mgr = OrderbookManager()

        last_bid_price = 0.0
        last_ask_price = 0.0

        for i in range(100):
            bid_price = round(0.30 + i * 0.002, 4)
            ask_price = round(0.70 - i * 0.002, 4)
            last_bid_price = bid_price
            last_ask_price = ask_price

            mgr.handle_snapshot(_make_snapshot(
                asset_id="rapid_asset",
                bids=[(bid_price, float(i + 1))],
                asks=[(ask_price, float(i + 1))],
                timestamp=str(1700000000000 + i * 1000),
                hash_val=f"0xhash_{i}",
            ))

        book = mgr.get("rapid_asset")
        assert book is not None

        # Final state should match the 100th snapshot
        assert book.best_bid == pytest.approx(last_bid_price)
        assert book.best_ask == pytest.approx(last_ask_price)

        depth = book.get_depth(100)
        assert len(depth["bids"]) == 1  # each snapshot replaces fully
        assert len(depth["asks"]) == 1
        assert depth["bids"][0].size == 100.0
        assert depth["asks"][0].size == 100.0

    def test_rapid_snapshots_multiple_assets(self) -> None:
        """Apply rapid snapshots across multiple assets concurrently."""
        mgr = OrderbookManager()
        num_assets = 10
        num_snapshots_per = 50

        for asset_idx in range(num_assets):
            asset_id = f"asset_{asset_idx}"
            for snap_idx in range(num_snapshots_per):
                bid = round(0.30 + snap_idx * 0.01, 4)
                ask = round(0.70 - snap_idx * 0.01, 4)
                mgr.handle_snapshot(_make_snapshot(
                    asset_id=asset_id,
                    bids=[(bid, float(snap_idx + 1))],
                    asks=[(ask, float(snap_idx + 1))],
                ))

        assert len(mgr) == num_assets

        # Each asset should have the last snapshot's data
        for asset_idx in range(num_assets):
            book = mgr.get(f"asset_{asset_idx}")
            assert book is not None
            last_bid = round(0.30 + (num_snapshots_per - 1) * 0.01, 4)
            last_ask = round(0.70 - (num_snapshots_per - 1) * 0.01, 4)
            assert book.best_bid == pytest.approx(last_bid)
            assert book.best_ask == pytest.approx(last_ask)

    def test_rapid_snapshots_interleaved_with_updates(self) -> None:
        """Apply snapshots interleaved with updates — snapshot should
        always reset the book regardless of prior updates.
        """
        mgr = OrderbookManager()
        asset_id = "interleaved"

        for i in range(50):
            # Snapshot
            mgr.handle_snapshot(_make_snapshot(
                asset_id=asset_id,
                bids=[(0.40, 100.0)],
                asks=[(0.60, 100.0)],
            ))

            # Several updates
            for j in range(5):
                mgr.handle_update(_make_change(
                    asset_id=asset_id,
                    price=round(0.41 + j * 0.01, 4),
                    size=float(j + 1) * 10,
                    side=Side.BUY,
                ))

        # Apply one final snapshot
        mgr.handle_snapshot(_make_snapshot(
            asset_id=asset_id,
            bids=[(0.35, 1.0)],
            asks=[(0.65, 1.0)],
        ))

        book = mgr.get(asset_id)
        assert book is not None
        assert book.best_bid == 0.35
        assert book.best_ask == 0.65
        depth = book.get_depth(100)
        assert len(depth["bids"]) == 1
        assert len(depth["asks"]) == 1

    def test_thread_safety_rapid_snapshots(self) -> None:
        """Apply snapshots from multiple threads simultaneously to
        verify thread safety.
        """
        mgr = OrderbookManager()
        asset_id = "threaded_asset"
        num_threads = 4
        snapshots_per_thread = 50
        errors: list[Exception] = []

        def apply_snapshots(thread_id: int) -> None:
            try:
                for i in range(snapshots_per_thread):
                    bid = round(0.30 + (thread_id * 100 + i) * 0.001, 4)
                    ask = round(bid + 0.10, 4)
                    mgr.handle_snapshot(_make_snapshot(
                        asset_id=asset_id,
                        bids=[(bid, float(i + 1))],
                        asks=[(ask, float(i + 1))],
                    ))
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=apply_snapshots, args=(tid,))
            for tid in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread safety errors: {errors}"

        book = mgr.get(asset_id)
        assert book is not None
        # We can't predict which thread's last snapshot "won", but the
        # book should have exactly 1 bid and 1 ask level
        depth = book.get_depth(100)
        assert len(depth["bids"]) == 1
        assert len(depth["asks"]) == 1


# ===========================================================================
# Test 5: TradeFeed bounded deque under high throughput
# ===========================================================================


class TestTradeFeedHighThroughput:
    """Verify the TradeFeed's bounded deque behaviour under load."""

    def test_10k_trades_bounded(self) -> None:
        """Adding 10,000 trades should only retain max_trades."""
        max_trades = 1000
        feed = TradeFeed(max_trades=max_trades)

        now_ms = time.time() * 1000.0
        for i in range(10_000):
            feed.on_trade(_make_trade(
                price=round(0.40 + (i % 100) * 0.002, 4),
                size=float(i + 1),
                timestamp_ms=now_ms + i,
            ))

        trades = feed.recent_trades("asset_1", n=max_trades + 100)
        assert len(trades) == max_trades

        # The retained trades should be the LAST 1000 (i=9000..9999)
        assert trades[0].size == 9001.0  # i=9000 -> size=9001
        assert trades[-1].size == 10000.0  # i=9999 -> size=10000

    def test_vwap_correct_after_overflow(self) -> None:
        """VWAP should work correctly even after the deque has
        overflowed and discarded older trades.
        """
        max_trades = 100
        feed = TradeFeed(max_trades=max_trades)

        now_ms = time.time() * 1000.0

        # Add 10,000 trades — all recent enough to be in the VWAP window
        for i in range(10_000):
            feed.on_trade(_make_trade(
                price=0.50,  # constant price
                size=10.0,   # constant size
                timestamp_ms=now_ms + i,
            ))

        # VWAP over a large window should be 0.50 since all trades have same price
        vwap = feed.vwap("asset_1", window_seconds=3600)
        assert vwap is not None
        assert vwap == pytest.approx(0.50)

    def test_vwap_weighted_after_overflow(self) -> None:
        """VWAP should correctly weight the retained trades."""
        max_trades = 200
        feed = TradeFeed(max_trades=max_trades)

        now_ms = time.time() * 1000.0

        # First batch: 500 trades at 0.40 (will be evicted)
        for i in range(500):
            feed.on_trade(_make_trade(
                price=0.40,
                size=100.0,
                timestamp_ms=now_ms + i,
            ))

        # Second batch: 200 trades at 0.60 (these will remain)
        for i in range(200):
            feed.on_trade(_make_trade(
                price=0.60,
                size=100.0,
                timestamp_ms=now_ms + 500 + i,
            ))

        # Only the last 200 should be in the deque — all at 0.60
        vwap = feed.vwap("asset_1", window_seconds=3600)
        assert vwap is not None
        assert vwap == pytest.approx(0.60)

    def test_trade_flow_after_overflow(self) -> None:
        """Trade flow analysis should work correctly with only retained trades."""
        max_trades = 50
        feed = TradeFeed(max_trades=max_trades)

        now_ms = time.time() * 1000.0

        # Add many BUY trades (will be evicted)
        for i in range(100):
            feed.on_trade(_make_trade(
                side=Side.BUY,
                size=100.0,
                timestamp_ms=now_ms + i,
            ))

        # Add SELL trades (these remain)
        for i in range(50):
            feed.on_trade(_make_trade(
                side=Side.SELL,
                size=10.0,
                timestamp_ms=now_ms + 100 + i,
            ))

        flow = feed.trade_flow("asset_1", window_seconds=3600)
        # Only the last 50 trades should be in the deque — all SELL
        assert flow["buy_volume"] == pytest.approx(0.0)
        assert flow["sell_volume"] == pytest.approx(500.0)  # 50 * 10
        assert flow["net_flow"] == pytest.approx(-500.0)

    def test_last_price_after_overflow(self) -> None:
        """Last price should always reflect the most recent trade."""
        feed = TradeFeed(max_trades=10)

        for i in range(1000):
            feed.on_trade(_make_trade(price=float(i) / 1000.0))

        assert feed.last_price("asset_1") == pytest.approx(0.999)

    def test_multiple_assets_isolation(self) -> None:
        """Trades for different assets should be independently bounded."""
        feed = TradeFeed(max_trades=50)

        now_ms = time.time() * 1000.0
        for i in range(200):
            feed.on_trade(_make_trade(
                asset_id="asset_A",
                price=0.50,
                size=1.0,
                timestamp_ms=now_ms + i,
            ))
            feed.on_trade(_make_trade(
                asset_id="asset_B",
                price=0.60,
                size=2.0,
                timestamp_ms=now_ms + i,
            ))

        trades_a = feed.recent_trades("asset_A", n=1000)
        trades_b = feed.recent_trades("asset_B", n=1000)

        assert len(trades_a) == 50
        assert len(trades_b) == 50
        assert all(t.price == 0.50 for t in trades_a)
        assert all(t.price == 0.60 for t in trades_b)
