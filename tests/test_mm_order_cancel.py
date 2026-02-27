"""Tests for game-start order cancellation detection & recovery.

Tests that external CANCELLATION events (e.g. Polymarket auto-cancelling
all orders at game start) are properly handled: orders marked cancelled,
quote state cleared, and requotes enqueued for the next tick.
"""

from __future__ import annotations

import pytest

from src.mm.config import MMConfig
from src.mm.engine import MarketMakingEngine
from src.mm.models import OrderStatus, Side


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


@pytest.fixture
def paired_engine(config: MMConfig) -> MarketMakingEngine:
    """Engine with a binary pair registered."""
    eng = MarketMakingEngine(config=config, client=None)
    eng.add_token("tok_yes")
    eng.add_token("tok_no")
    eng.add_token_pair("tok_yes", "tok_no")
    eng.fair_value_engine.update("tok_yes", 0.60)
    eng.fair_value_engine.update("tok_no", 0.40)
    return eng


# ---------------------------------------------------------------------------
# OrderManager.on_cancel tests
# ---------------------------------------------------------------------------


class TestOrderManagerOnCancel:
    """Unit tests for OrderManager.on_cancel()."""

    def test_cancel_marks_live_order(self, engine: MarketMakingEngine):
        """on_cancel marks a tracked LIVE order as CANCELLED and returns token_id."""
        om = engine.order_manager
        state = om.place_order("tok_a", Side.BUY, 0.49, 50.0)
        assert state.status == OrderStatus.LIVE

        result = om.on_cancel(state.order_id)
        assert result == "tok_a"
        # Verify it's now CANCELLED
        orders = om.get_open_orders("tok_a")
        assert all(o.order_id != state.order_id for o in orders)

    def test_cancel_unknown_order_returns_none(self, engine: MarketMakingEngine):
        """on_cancel returns None for an unknown order ID."""
        om = engine.order_manager
        result = om.on_cancel("nonexistent-order-id")
        assert result is None

    def test_cancel_already_cancelled_returns_none(self, engine: MarketMakingEngine):
        """on_cancel returns None for an already-CANCELLED order (idempotent)."""
        om = engine.order_manager
        state = om.place_order("tok_a", Side.BUY, 0.49, 50.0)

        # Cancel once
        result1 = om.on_cancel(state.order_id)
        assert result1 == "tok_a"

        # Cancel again — already terminal
        result2 = om.on_cancel(state.order_id)
        assert result2 is None

    def test_cancel_filled_order_returns_none(self, engine: MarketMakingEngine):
        """on_cancel returns None for a FILLED order — don't corrupt fills."""
        om = engine.order_manager
        state = om.place_order("tok_a", Side.BUY, 0.49, 50.0)

        # Fill the order
        om.on_fill(state.order_id, 50.0, 0.49)

        # Cancel after fill — should be no-op
        result = om.on_cancel(state.order_id)
        assert result is None

    def test_cancel_failed_order_returns_none(self, engine: MarketMakingEngine):
        """on_cancel returns None for a FAILED order."""
        om = engine.order_manager
        # Place a sub-minimum order to get FAILED status
        state = om.place_order("tok_a", Side.BUY, 0.49, 1.0)
        assert state.status == OrderStatus.FAILED

        result = om.on_cancel(state.order_id)
        assert result is None


# ---------------------------------------------------------------------------
# Engine.on_order_cancel tests
# ---------------------------------------------------------------------------


class TestEngineOnOrderCancel:
    """Integration tests for engine.on_order_cancel()."""

    def test_cancel_clears_current_quotes(self, engine: MarketMakingEngine):
        """on_order_cancel clears _current_quotes for the cancelled token."""
        # Place quotes via a tick to populate _current_quotes
        engine._tick()
        assert "tok_a" in engine._current_quotes

        # Get one of the placed order IDs
        orders = engine.order_manager.get_open_orders("tok_a")
        assert len(orders) > 0
        order_id = orders[0].order_id

        engine.on_order_cancel(order_id)
        assert "tok_a" not in engine._current_quotes

    def test_cancel_enqueues_pending_requote(self, engine: MarketMakingEngine):
        """on_order_cancel enqueues the token in _pending_requotes."""
        engine._tick()
        orders = engine.order_manager.get_open_orders("tok_a")
        assert len(orders) > 0

        # Clear any pending requotes from the tick
        engine._pending_requotes.clear()

        engine.on_order_cancel(orders[0].order_id)
        assert "tok_a" in engine._pending_requotes

    def test_cancel_unknown_order_is_noop(self, engine: MarketMakingEngine):
        """on_order_cancel for an unknown order is a no-op."""
        engine._tick()
        assert "tok_a" in engine._current_quotes

        # Clear pending requotes
        engine._pending_requotes.clear()

        engine.on_order_cancel("unknown-order-id")
        # Quotes should be untouched
        assert "tok_a" in engine._current_quotes
        assert "tok_a" not in engine._pending_requotes

    def test_mass_cancellation_all_orders(self, engine: MarketMakingEngine):
        """Mass cancellation (all orders) clears quotes; next tick re-places."""
        engine._tick()
        assert "tok_a" in engine._current_quotes

        orders = engine.order_manager.get_open_orders("tok_a")
        assert len(orders) > 0

        # Cancel all orders (simulating game-start auto-cancel)
        for order in orders:
            engine.on_order_cancel(order.order_id)

        assert "tok_a" not in engine._current_quotes
        assert "tok_a" in engine._pending_requotes

        # Next tick should re-place quotes
        engine._tick()
        assert "tok_a" in engine._current_quotes

    def test_cancel_during_risk_halt_no_requote(self, engine: MarketMakingEngine):
        """Cancel during risk halt marks order cancelled but doesn't enqueue requote."""
        engine._tick()
        orders = engine.order_manager.get_open_orders("tok_a")
        assert len(orders) > 0

        # Simulate risk halt
        engine._risk_halted = True
        engine._pending_requotes.clear()

        engine.on_order_cancel(orders[0].order_id)

        # Order should be cancelled (quote cleared)
        assert "tok_a" not in engine._current_quotes
        # But NOT enqueued for requote during risk halt
        assert "tok_a" not in engine._pending_requotes

    def test_complement_token_enqueued(self, paired_engine: MarketMakingEngine):
        """Cancellation enqueues the complement token for paired markets."""
        eng = paired_engine
        eng._tick()

        orders_yes = eng.order_manager.get_open_orders("tok_yes")
        assert len(orders_yes) > 0

        eng._pending_requotes.clear()

        eng.on_order_cancel(orders_yes[0].order_id)

        # Both the cancelled token and its complement should be enqueued
        assert "tok_yes" in eng._pending_requotes
        assert "tok_no" in eng._pending_requotes
