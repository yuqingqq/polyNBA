"""Integration tests for fill-driven requoting and orderbook integration.

Tests that fills trigger immediate requoting, old orders get cancelled,
and the engine correctly integrates with OrderbookManager for market_mid.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.mm.config import MMConfig
from src.mm.engine import MarketMakingEngine
from src.mm.models import Fill, Side


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> MMConfig:
    return MMConfig(
        dry_run=True,
        spread_bps=200,
        order_size=50,
        max_position=500,
        max_total_position=2000,
        update_interval_seconds=2,
        stale_data_timeout_seconds=60,
        max_loss=200,
    )


@pytest.fixture
def engine(config: MMConfig) -> MarketMakingEngine:
    """Engine with one token registered and fair value set."""
    eng = MarketMakingEngine(config=config, client=None)
    eng.add_token("tok_a")
    eng.fair_value_engine.update("tok_a", 0.50)
    return eng


def _make_mock_orderbook_manager(mid_prices: dict[str, float | None]) -> MagicMock:
    """Create a mock OrderbookManager that returns specified mid prices."""
    mgr = MagicMock()

    def mock_get(asset_id: str):
        if asset_id not in mid_prices:
            return None
        book = MagicMock()
        mid = mid_prices[asset_id]
        book.mid_price = mid
        book.best_bid = None
        book.best_ask = None
        book.get_bbo.return_value = (mid, None, None)
        return book

    mgr.get.side_effect = mock_get
    return mgr


# ---------------------------------------------------------------------------
# Tests: Fill-driven requoting
# ---------------------------------------------------------------------------


class TestFillTriggersRequote:
    """Fill events should trigger requoting via the tick loop.

    C-3: on_fill no longer requotes inline (which blocked the tick loop
    during HTTP calls). Instead it adds the token to _pending_requotes,
    and the next _tick() processes it. Tests call _tick() after on_fill
    to simulate this.
    """

    def test_fill_triggers_requote_on_next_tick(self, engine: MarketMakingEngine) -> None:
        """After a fill + tick, new orders should be placed."""
        engine._tick()

        buy_orders = [
            o for o in engine.order_manager.get_open_orders("tok_a")
            if o.side == Side.BUY
        ]
        assert len(buy_orders) > 0
        old_order_id = buy_orders[0].order_id

        # Fill the buy order
        fill = Fill(
            order_id=old_order_id,
            token_id="tok_a",
            side=Side.BUY,
            price=0.49,
            size=50.0,
        )
        engine.on_fill(fill)

        # C-3: on_fill queues requote; _tick processes it
        engine._tick()

        open_orders = engine.order_manager.get_open_orders("tok_a")
        assert len(open_orders) > 0, "Expected orders after fill-driven requote"

        # The old filled order should no longer be in open orders
        old_still_open = [o for o in open_orders if o.order_id == old_order_id]
        assert len(old_still_open) == 0, "Filled order should not be open"

    def test_fill_cancels_old_orders(self, config: MMConfig) -> None:
        """A large fill should cancel old orders and replace them with skewed ones."""
        eng = MarketMakingEngine(config=config, client=None)
        eng.add_token("tok_a")
        eng.fair_value_engine.update("tok_a", 0.50)
        eng._tick()

        orders_before = eng.order_manager.get_open_orders("tok_a")
        order_ids_before = {o.order_id for o in orders_before}

        # Use actual buy order ID from placed orders
        buy_order = next(o for o in orders_before if o.side == Side.BUY)

        # Large fill to create enough skew to cross the requote threshold
        # skew = 0.5 * (300/500) * 0.02 = 0.006 = 60bps > 50bps threshold
        fill = Fill(
            order_id=buy_order.order_id,
            token_id="tok_a",
            side=Side.BUY,
            price=0.49,
            size=300.0,
        )
        eng.on_fill(fill)

        # C-3: Process pending requote via tick
        eng._tick()

        # After requote, old order IDs should be cancelled/replaced
        orders_after = eng.order_manager.get_open_orders("tok_a")
        new_order_ids = {o.order_id for o in orders_after}

        # The old orders should be cancelled and replaced with skewed ones
        assert not new_order_ids.intersection(order_ids_before), \
            "Old orders should be replaced after fill-driven requote with large position change"

    def test_fill_skews_quotes_after_position_change(self, engine: MarketMakingEngine) -> None:
        """After a large fill, quotes should be skewed due to position change."""
        engine._tick()
        orders_before = engine.order_manager.get_open_orders("tok_a")
        buy_order = next(o for o in orders_before if o.side == Side.BUY)
        bid_before = buy_order.price

        # Large buy fill — should skew bid down
        fill = Fill(
            order_id=buy_order.order_id,
            token_id="tok_a",
            side=Side.BUY,
            price=0.49,
            size=300.0,
        )
        engine.on_fill(fill)

        # C-3: Process pending requote via tick
        engine._tick()

        orders_after = engine.order_manager.get_open_orders("tok_a")
        buys_after = [o for o in orders_after if o.side == Side.BUY]
        assert len(buys_after) > 0
        bid_after = buys_after[0].price

        assert bid_after < bid_before, (
            f"Bid should be lower after large buy: {bid_before} -> {bid_after}"
        )

    def test_unregistered_token_fill_no_requote(self, config: MMConfig) -> None:
        """Fill for an unregistered token should not trigger requote."""
        eng = MarketMakingEngine(config=config, client=None)
        eng.add_token("tok_a")
        eng.fair_value_engine.update("tok_a", 0.50)
        eng._tick()

        orders_before = eng.order_manager.get_open_orders()
        count_before = len(orders_before)

        # Fill for a token that isn't registered
        fill = Fill(
            order_id="unregistered-fill",
            token_id="tok_unknown",
            side=Side.BUY,
            price=0.50,
            size=50.0,
        )
        eng.on_fill(fill)

        # C-3: Process pending requotes (there shouldn't be any for unregistered token)
        eng._tick()

        # Orders for tok_a should be unchanged
        orders_after = eng.order_manager.get_open_orders("tok_a")
        assert len(orders_after) == count_before

    def test_max_position_drops_bid_on_fill(self, config: MMConfig) -> None:
        """Fill to max position should drop the bid after next tick."""
        eng = MarketMakingEngine(config=config, client=None)
        eng.add_token("tok_a")
        eng.fair_value_engine.update("tok_a", 0.50)
        eng._tick()

        # Use actual buy order ID from placed orders
        orders = eng.order_manager.get_open_orders("tok_a")
        buy_order = next(o for o in orders if o.side == Side.BUY)

        # Fill to max position
        fill = Fill(
            order_id=buy_order.order_id,
            token_id="tok_a",
            side=Side.BUY,
            price=0.50,
            size=config.max_position,
        )
        eng.on_fill(fill)

        # C-3: Process pending requote via tick
        eng._tick()

        orders = eng.order_manager.get_open_orders("tok_a")
        buys = [o for o in orders if o.side == Side.BUY]
        sells = [o for o in orders if o.side == Side.SELL]

        assert len(buys) == 0, "Bid should be dropped at max position"
        assert len(sells) > 0, "Ask should still be quoted"


# ---------------------------------------------------------------------------
# Tests: OrderbookManager integration
# ---------------------------------------------------------------------------


class TestOrderbookIntegration:
    """Engine with OrderbookManager for market_mid divergence."""

    def test_engine_with_orderbook_manager(self, config: MMConfig) -> None:
        """Engine should accept and use an orderbook_manager."""
        ob_mgr = _make_mock_orderbook_manager({"tok_a": 0.50})
        eng = MarketMakingEngine(config=config, client=None, orderbook_manager=ob_mgr)
        eng.add_token("tok_a")
        eng.fair_value_engine.update("tok_a", 0.50)

        eng._tick()
        orders = eng.order_manager.get_open_orders("tok_a")
        assert len(orders) > 0

    def test_divergent_mid_widens_spread(self, config: MMConfig) -> None:
        """When market_mid diverges from fair_value, spread should widen."""
        # Engine without orderbook
        eng_no_ob = MarketMakingEngine(config=config, client=None)
        eng_no_ob.add_token("tok_a")
        eng_no_ob.fair_value_engine.update("tok_a", 0.50)
        eng_no_ob._tick()

        orders_no_ob = eng_no_ob.order_manager.get_open_orders("tok_a")
        bid_no_ob = next(o.price for o in orders_no_ob if o.side == Side.BUY)
        ask_no_ob = next(o.price for o in orders_no_ob if o.side == Side.SELL)
        spread_no_ob = ask_no_ob - bid_no_ob

        # Engine with divergent orderbook (mid=0.55, fv=0.50 → 500bps)
        ob_mgr = _make_mock_orderbook_manager({"tok_a": 0.55})
        eng_ob = MarketMakingEngine(config=config, client=None, orderbook_manager=ob_mgr)
        eng_ob.add_token("tok_a")
        eng_ob.fair_value_engine.update("tok_a", 0.50)
        eng_ob._tick()

        orders_ob = eng_ob.order_manager.get_open_orders("tok_a")
        bid_ob = next(o.price for o in orders_ob if o.side == Side.BUY)
        ask_ob = next(o.price for o in orders_ob if o.side == Side.SELL)
        spread_ob = ask_ob - bid_ob

        assert spread_ob > spread_no_ob, (
            f"Spread should be wider with divergence: {spread_no_ob} vs {spread_ob}"
        )

    def test_engine_without_orderbook_backward_compat(self, config: MMConfig) -> None:
        """Engine without orderbook_manager should work as before."""
        eng = MarketMakingEngine(config=config, client=None)
        eng.add_token("tok_a")
        eng.fair_value_engine.update("tok_a", 0.50)
        eng._tick()

        orders = eng.order_manager.get_open_orders("tok_a")
        assert len(orders) >= 2  # bid + ask

    def test_orderbook_no_mid_no_divergence(self, config: MMConfig) -> None:
        """If orderbook has no mid_price, no divergence should be applied."""
        ob_mgr = _make_mock_orderbook_manager({"tok_a": None})
        eng = MarketMakingEngine(config=config, client=None, orderbook_manager=ob_mgr)
        eng.add_token("tok_a")
        eng.fair_value_engine.update("tok_a", 0.50)
        eng._tick()

        orders = eng.order_manager.get_open_orders("tok_a")
        assert len(orders) >= 2

    def test_orderbook_unknown_token_no_divergence(self, config: MMConfig) -> None:
        """If orderbook doesn't have the token, no divergence applied."""
        ob_mgr = _make_mock_orderbook_manager({})  # empty
        eng = MarketMakingEngine(config=config, client=None, orderbook_manager=ob_mgr)
        eng.add_token("tok_a")
        eng.fair_value_engine.update("tok_a", 0.50)
        eng._tick()

        orders = eng.order_manager.get_open_orders("tok_a")
        assert len(orders) >= 2


# ---------------------------------------------------------------------------
# Tests: Orderbook with partial data (only bids or only asks)
# ---------------------------------------------------------------------------


def _make_mock_orderbook_with_bba(
    mid: float | None,
    best_bid: float | None,
    best_ask: float | None,
) -> MagicMock:
    """Create a mock OrderbookManager with explicit best_bid/best_ask."""
    mgr = MagicMock()
    book = MagicMock()
    book.mid_price = mid
    book.best_bid = best_bid
    book.best_ask = best_ask
    book.get_bbo.return_value = (mid, best_bid, best_ask)
    mgr.get.return_value = book
    return mgr


class TestOrderbookPartialData:
    """Engine with orderbook that has only bids or only asks."""

    def test_orderbook_only_bids(self, config: MMConfig) -> None:
        """Orderbook with best_bid but no best_ask should still work."""
        ob_mgr = _make_mock_orderbook_with_bba(mid=0.50, best_bid=0.49, best_ask=None)
        eng = MarketMakingEngine(config=config, client=None, orderbook_manager=ob_mgr)
        eng.add_token("tok_a")
        eng.fair_value_engine.update("tok_a", 0.50)
        eng._tick()

        orders = eng.order_manager.get_open_orders("tok_a")
        assert len(orders) >= 2  # Both sides should still quote

    def test_orderbook_only_asks(self, config: MMConfig) -> None:
        """Orderbook with best_ask but no best_bid should still work."""
        ob_mgr = _make_mock_orderbook_with_bba(mid=0.50, best_bid=None, best_ask=0.51)
        eng = MarketMakingEngine(config=config, client=None, orderbook_manager=ob_mgr)
        eng.add_token("tok_a")
        eng.fair_value_engine.update("tok_a", 0.50)
        eng._tick()

        orders = eng.order_manager.get_open_orders("tok_a")
        assert len(orders) >= 2  # Both sides should still quote

    def test_orderbook_no_best_bid_ask(self, config: MMConfig) -> None:
        """Orderbook with mid but no best_bid/best_ask (both None)."""
        ob_mgr = _make_mock_orderbook_with_bba(mid=0.50, best_bid=None, best_ask=None)
        eng = MarketMakingEngine(config=config, client=None, orderbook_manager=ob_mgr)
        eng.add_token("tok_a")
        eng.fair_value_engine.update("tok_a", 0.50)
        eng._tick()

        orders = eng.order_manager.get_open_orders("tok_a")
        assert len(orders) >= 2
