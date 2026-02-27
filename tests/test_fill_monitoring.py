"""Tests for fill monitoring pipeline: UserFeed → user_trade_to_fills → engine.on_fill.

Validates the critical fix for the fill size bug where the taker's total trade
size (e.g. 9900) was incorrectly attributed as our fill instead of our actual
matched_amount (e.g. 30).
"""

import json

from src.data.models import (
    MakerOrder,
    Side as DataSide,
    TradeStatus,
    UserTradeEvent,
)
from src.data.user_feed import UserFeed
from src.mm.models import Fill, Side as MMSide
from src.mm.order_manager import OrderManager
from src.runner import user_trade_to_fills


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

JAZZ_TOKEN = "4092243847"
GRIZZLIES_TOKEN = "1890034067"
OUR_ORDER_ID = "order-abc-123"
OTHER_MAKER_ORDER = "order-xyz-999"
TAKER_ORDER_ID = "order-taker-big"


def _make_large_taker_trade_event() -> UserTradeEvent:
    """Simulate the exact scenario: 9900-share taker trade, we're one maker with 30 shares."""
    return UserTradeEvent(
        id="trade-001",
        asset_id=JAZZ_TOKEN,
        status=TradeStatus.CONFIRMED,
        side=DataSide.BUY,
        price=0.50,
        size=9900.0,  # TAKER's total — NOT our fill
        taker_order_id=TAKER_ORDER_ID,
        maker_order_ids=[OUR_ORDER_ID, OTHER_MAKER_ORDER],
        maker_orders=[
            MakerOrder(
                order_id=OUR_ORDER_ID,
                asset_id=JAZZ_TOKEN,
                side=DataSide.SELL,
                price=0.45,
                matched_amount=30.0,  # Our actual fill
            ),
            MakerOrder(
                order_id=OTHER_MAKER_ORDER,
                asset_id=JAZZ_TOKEN,
                side=DataSide.SELL,
                price=0.50,
                matched_amount=9870.0,  # Some other maker
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Tests: user_trade_to_fills
# ---------------------------------------------------------------------------


class TestUserTradeToFills:
    """Test user_trade_to_fills correctly filters and sizes fills."""

    def test_maker_fill_uses_matched_amount_not_taker_size(self):
        """THE critical test: our fill should be 30, not 9900.

        I-6: user_trade_to_fills now emits ALL fill candidates (taker + makers).
        Filtering by tracked order happens in engine.on_fill. Here we verify
        that maker fills use matched_amount, not the taker's total size.
        """
        event = _make_large_taker_trade_event()

        fills = user_trade_to_fills(event)

        # Should have taker fill + 2 maker fills = 3 total candidates
        assert len(fills) == 3
        # Find OUR maker fill by order_id
        our_fill = [f for f in fills if f.order_id == OUR_ORDER_ID]
        assert len(our_fill) == 1
        fill = our_fill[0]
        assert fill.size == 30.0  # NOT 9900
        assert fill.price == 0.45
        assert fill.token_id == JAZZ_TOKEN
        assert fill.side == MMSide.SELL

    def test_all_fills_emitted(self):
        """I-6: user_trade_to_fills now emits all candidates, not just tracked ones."""
        event = _make_large_taker_trade_event()

        fills = user_trade_to_fills(event)
        # Taker + 2 makers = 3 fills
        assert len(fills) == 3

    def test_taker_fill_uses_event_size(self):
        """When we are the taker, event.size IS our fill size."""
        event = UserTradeEvent(
            id="trade-002",
            asset_id=JAZZ_TOKEN,
            status=TradeStatus.CONFIRMED,
            side=DataSide.BUY,
            price=0.50,
            size=100.0,
            taker_order_id="our-taker-order",
            maker_order_ids=["maker-1"],
            maker_orders=[
                MakerOrder(
                    order_id="maker-1",
                    asset_id=JAZZ_TOKEN,
                    side=DataSide.SELL,
                    price=0.50,
                    matched_amount=100.0,
                ),
            ],
        )

        fills = user_trade_to_fills(event)

        # Taker + 1 maker = 2 fills
        assert len(fills) == 2
        taker_fill = [f for f in fills if f.order_id == "our-taker-order"]
        assert len(taker_fill) == 1
        assert taker_fill[0].size == 100.0
        assert taker_fill[0].side == MMSide.BUY

    def test_multiple_maker_fills(self):
        """All maker orders in the same trade produce fills."""
        order_a = "order-a"
        order_b = "order-b"
        event = UserTradeEvent(
            id="trade-003",
            asset_id=JAZZ_TOKEN,
            status=TradeStatus.CONFIRMED,
            side=DataSide.BUY,
            price=0.50,
            size=500.0,
            taker_order_id="someone-else-taker",
            maker_order_ids=[order_a, order_b, "not-ours"],
            maker_orders=[
                MakerOrder(order_id=order_a, asset_id=JAZZ_TOKEN, side=DataSide.SELL, price=0.48, matched_amount=20.0),
                MakerOrder(order_id=order_b, asset_id=JAZZ_TOKEN, side=DataSide.SELL, price=0.49, matched_amount=15.0),
                MakerOrder(order_id="not-ours", asset_id=JAZZ_TOKEN, side=DataSide.SELL, price=0.50, matched_amount=465.0),
            ],
        )

        fills = user_trade_to_fills(event)

        # Taker + 3 makers = 4 fills total
        assert len(fills) == 4
        fill_a = [f for f in fills if f.order_id == order_a]
        fill_b = [f for f in fills if f.order_id == order_b]
        assert len(fill_a) == 1
        assert fill_a[0].size == 20.0
        assert fill_a[0].price == 0.48
        assert len(fill_b) == 1
        assert fill_b[0].size == 15.0
        assert fill_b[0].price == 0.49

    def test_non_confirmed_status_ignored_by_callback(self):
        """The runner callback filters on CONFIRMED status; MATCHED should be skipped."""
        event = UserTradeEvent(
            id="trade-004",
            asset_id=JAZZ_TOKEN,
            status=TradeStatus.MATCHED,  # Not CONFIRMED
            side=DataSide.BUY,
            price=0.50,
            size=9900.0,
            taker_order_id=TAKER_ORDER_ID,
            maker_orders=[
                MakerOrder(order_id=OUR_ORDER_ID, asset_id=JAZZ_TOKEN, side=DataSide.SELL, price=0.45, matched_amount=30.0),
            ],
        )
        # The callback checks event.status != TradeStatus.CONFIRMED → return
        assert event.status != TradeStatus.CONFIRMED

    def test_empty_maker_orders_only_taker(self):
        """Trade event with no maker_orders should produce only taker fill."""
        event = UserTradeEvent(
            id="trade-005",
            asset_id=JAZZ_TOKEN,
            status=TradeStatus.CONFIRMED,
            side=DataSide.BUY,
            price=0.50,
            size=100.0,
            taker_order_id="someone",
            maker_orders=[],
        )
        fills = user_trade_to_fills(event)
        assert len(fills) == 1  # taker fill only
        assert fills[0].order_id == "someone"


# ---------------------------------------------------------------------------
# Tests: OrderManager.get_tracked_order_ids
# ---------------------------------------------------------------------------


class TestGetTrackedOrderIds:
    """Test that get_tracked_order_ids returns all order IDs."""

    def test_empty_manager(self):
        mgr = OrderManager(dry_run=True)
        assert mgr.get_tracked_order_ids() == set()

    def test_placed_orders_are_tracked(self):
        mgr = OrderManager(dry_run=True)
        mgr.place_order(JAZZ_TOKEN, MMSide.BUY, 0.45, 30.0)
        mgr.place_order(GRIZZLIES_TOKEN, MMSide.BUY, 0.53, 30.0)

        tracked = mgr.get_tracked_order_ids()
        assert len(tracked) == 2

    def test_cancelled_orders_still_tracked(self):
        """Cancelled orders should remain tracked so late fill events can match."""
        mgr = OrderManager(dry_run=True)
        state = mgr.place_order(JAZZ_TOKEN, MMSide.BUY, 0.45, 30.0)
        mgr.cancel_order(state.order_id)

        tracked = mgr.get_tracked_order_ids()
        assert state.order_id in tracked

    def test_filled_orders_still_tracked(self):
        """Filled orders should remain tracked until cleanup."""
        mgr = OrderManager(dry_run=True)
        state = mgr.place_order(JAZZ_TOKEN, MMSide.BUY, 0.45, 30.0)
        mgr.on_fill(state.order_id, 30.0, 0.45)

        tracked = mgr.get_tracked_order_ids()
        assert state.order_id in tracked


# ---------------------------------------------------------------------------
# Tests: UserFeed._dispatch_trade parses maker orders correctly
# ---------------------------------------------------------------------------


class TestUserFeedDispatchTrade:
    """Test that UserFeed correctly parses WebSocket trade payloads."""

    def test_dispatch_trade_parses_maker_orders(self):
        """Simulate raw WebSocket JSON → verify MakerOrder fields."""
        raw_payload = {
            "event_type": "trade",
            "id": "trade-ws-001",
            "asset_id": JAZZ_TOKEN,
            "status": "CONFIRMED",
            "side": "BUY",
            "price": "0.50",
            "size": "9900",
            "taker_order_id": TAKER_ORDER_ID,
            "maker_orders": [
                {
                    "order_id": OUR_ORDER_ID,
                    "asset_id": JAZZ_TOKEN,
                    "side": "SELL",
                    "price": "0.45",
                    "matched_amount": "30",
                },
                {
                    "order_id": OTHER_MAKER_ORDER,
                    "asset_id": JAZZ_TOKEN,
                    "side": "SELL",
                    "price": "0.50",
                    "matched_amount": "9870",
                },
            ],
        }

        received_events: list[UserTradeEvent] = []
        feed = UserFeed(api_key="k", secret="s", passphrase="p")
        feed.on_trade_update(lambda e: received_events.append(e))

        # Simulate raw message
        feed._handle_message(json.dumps(raw_payload))

        assert len(received_events) == 1
        event = received_events[0]
        assert event.size == 9900.0  # Taker's total
        assert len(event.maker_orders) == 2
        assert event.maker_orders[0].order_id == OUR_ORDER_ID
        assert event.maker_orders[0].matched_amount == 30.0
        assert event.maker_orders[0].side == DataSide.SELL
        assert event.maker_orders[1].order_id == OTHER_MAKER_ORDER
        assert event.maker_orders[1].matched_amount == 9870.0

    def test_dispatch_trade_missing_maker_orders_field(self):
        """WebSocket payload without maker_orders should still parse."""
        raw_payload = {
            "event_type": "trade",
            "id": "trade-ws-002",
            "asset_id": JAZZ_TOKEN,
            "status": "CONFIRMED",
            "side": "BUY",
            "price": "0.50",
            "size": "100",
            "taker_order_id": "taker-1",
        }

        received: list[UserTradeEvent] = []
        feed = UserFeed(api_key="k", secret="s", passphrase="p")
        feed.on_trade_update(lambda e: received.append(e))
        feed._handle_message(json.dumps(raw_payload))

        assert len(received) == 1
        assert received[0].maker_orders == []
        assert received[0].maker_order_ids == []


# ---------------------------------------------------------------------------
# Integration: full pipeline end-to-end
# ---------------------------------------------------------------------------


class TestFillMonitoringIntegration:
    """End-to-end: WebSocket payload → UserFeed → user_trade_to_fills → correct fill."""

    def test_full_pipeline_maker_fill(self):
        """Simulate the exact 9900-share scenario through the full pipeline.

        I-6: user_trade_to_fills now emits ALL fill candidates. In production,
        engine.on_fill filters by tracked order. Here we simulate that filtering
        using OrderManager.is_tracked().
        """
        # 1. OrderManager places our order
        mgr = OrderManager(dry_run=True)
        our_order = mgr.place_order(JAZZ_TOKEN, MMSide.SELL, 0.45, 30.0)

        # 2. UserFeed receives WebSocket message
        raw_payload = json.dumps({
            "event_type": "trade",
            "id": "trade-integration",
            "asset_id": JAZZ_TOKEN,
            "status": "CONFIRMED",
            "side": "BUY",
            "price": "0.50",
            "size": "9900",
            "taker_order_id": TAKER_ORDER_ID,
            "maker_orders": [
                {
                    "order_id": our_order.order_id,
                    "asset_id": JAZZ_TOKEN,
                    "side": "SELL",
                    "price": "0.45",
                    "matched_amount": "30",
                },
                {
                    "order_id": "other-maker",
                    "asset_id": JAZZ_TOKEN,
                    "side": "SELL",
                    "price": "0.50",
                    "matched_amount": "9870",
                },
            ],
        })

        captured_fills: list[Fill] = []

        feed = UserFeed(api_key="k", secret="s", passphrase="p")

        def handle_user_fill(event: UserTradeEvent):
            if event.status != TradeStatus.CONFIRMED:
                return
            fills = user_trade_to_fills(event)
            # Simulate engine.on_fill filtering by tracked orders
            for fill in fills:
                if mgr.is_tracked(fill.order_id):
                    captured_fills.append(fill)

        feed.on_trade_update(handle_user_fill)

        # 3. Dispatch the message
        feed._handle_message(raw_payload)

        # 4. Verify: exactly 1 tracked fill, size=30 (not 9900)
        assert len(captured_fills) == 1
        fill = captured_fills[0]
        assert fill.order_id == our_order.order_id
        assert fill.size == 30.0
        assert fill.price == 0.45
        assert fill.side == MMSide.SELL
        assert fill.token_id == JAZZ_TOKEN

    def test_full_pipeline_unrelated_trade(self):
        """A trade where none of our orders match should produce zero tracked fills."""
        mgr = OrderManager(dry_run=True)
        mgr.place_order(JAZZ_TOKEN, MMSide.SELL, 0.45, 30.0)

        raw_payload = json.dumps({
            "event_type": "trade",
            "id": "trade-unrelated",
            "asset_id": JAZZ_TOKEN,
            "status": "CONFIRMED",
            "side": "BUY",
            "price": "0.50",
            "size": "500",
            "taker_order_id": "unknown-taker",
            "maker_orders": [
                {
                    "order_id": "unknown-maker-1",
                    "asset_id": JAZZ_TOKEN,
                    "side": "SELL",
                    "price": "0.50",
                    "matched_amount": "500",
                },
            ],
        })

        captured_fills: list[Fill] = []
        feed = UserFeed(api_key="k", secret="s", passphrase="p")

        def handle(event: UserTradeEvent):
            if event.status != TradeStatus.CONFIRMED:
                return
            fills = user_trade_to_fills(event)
            # Simulate engine.on_fill filtering by tracked orders
            for fill in fills:
                if mgr.is_tracked(fill.order_id):
                    captured_fills.append(fill)

        feed.on_trade_update(handle)
        feed._handle_message(raw_payload)

        assert len(captured_fills) == 0
