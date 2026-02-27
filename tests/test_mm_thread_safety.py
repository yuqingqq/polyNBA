"""Thread safety tests for the market making engine.

Tests concurrent fills + ticks, rapid fills without deadlock,
and fill during shutdown.
"""

from __future__ import annotations

import threading

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
        update_interval_seconds=0.1,  # fast ticks for concurrency tests
        stale_data_timeout_seconds=60,
        max_loss=200,
    )


@pytest.fixture
def engine(config: MMConfig) -> MarketMakingEngine:
    eng = MarketMakingEngine(config=config, client=None)
    eng.add_token("tok_a")
    eng.fair_value_engine.update("tok_a", 0.50)
    eng._tick()  # initial quotes
    return eng


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConcurrentFillsAndTicks:
    """Concurrent fills and ticks should not corrupt state."""

    def test_concurrent_fills_and_ticks(self, engine: MarketMakingEngine) -> None:
        """Multiple fills and ticks running concurrently should not raise."""
        errors: list[Exception] = []

        def send_fills():
            try:
                for i in range(20):
                    fill = Fill(
                        order_id=f"conc-fill-{i}",
                        token_id="tok_a",
                        side=Side.BUY if i % 2 == 0 else Side.SELL,
                        price=0.50,
                        size=1.0,
                    )
                    engine.on_fill(fill)
            except Exception as e:
                errors.append(e)

        def run_ticks():
            try:
                for _ in range(20):
                    engine._tick()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=send_fills)
        t2 = threading.Thread(target=run_ticks)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not t1.is_alive(), "Fill thread should have completed"
        assert not t2.is_alive(), "Tick thread should have completed"
        assert len(errors) == 0, f"Errors during concurrent execution: {errors}"


class TestRapidFillsNoDeadlock:
    """Rapid successive fills should not cause deadlock."""

    def test_rapid_fills_no_deadlock(self, engine: MarketMakingEngine) -> None:
        """Sending many fills rapidly from multiple threads should complete without deadlock."""
        errors: list[Exception] = []
        num_threads = 4
        fills_per_thread = 10

        def send_fills(thread_id: int):
            try:
                for i in range(fills_per_thread):
                    fill = Fill(
                        order_id=f"rapid-{thread_id}-{i}",
                        token_id="tok_a",
                        side=Side.BUY if i % 2 == 0 else Side.SELL,
                        price=0.50,
                        size=1.0,
                    )
                    engine.on_fill(fill)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=send_fills, args=(tid,))
            for tid in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        for t in threads:
            assert not t.is_alive(), "Thread should have completed (possible deadlock)"
        assert len(errors) == 0, f"Errors during rapid fills: {errors}"


class TestFillDuringShutdown:
    """Fills arriving during/after shutdown should not crash."""

    def test_fill_after_stop(self, engine: MarketMakingEngine) -> None:
        """A fill arriving after stop() should still be processed without error."""
        # Get an actual order ID from orders placed by the fixture's _tick()
        orders = engine.order_manager.get_open_orders("tok_a")
        buy_order = next(o for o in orders if o.side == Side.BUY)

        engine.stop()

        fill = Fill(
            order_id=buy_order.order_id,
            token_id="tok_a",
            side=Side.BUY,
            price=0.50,
            size=50.0,
        )
        # Should not raise
        engine.on_fill(fill)

        # Inventory should still be updated
        pos = engine.inventory_manager.get_position("tok_a")
        assert pos.size > 0

    def test_concurrent_fill_and_stop(self, engine: MarketMakingEngine) -> None:
        """Concurrent fill and stop should not deadlock or crash."""
        errors: list[Exception] = []

        def send_fills():
            try:
                for i in range(10):
                    fill = Fill(
                        order_id=f"shutdown-fill-{i}",
                        token_id="tok_a",
                        side=Side.BUY,
                        price=0.50,
                        size=1.0,
                    )
                    engine.on_fill(fill)
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=send_fills)
        t.start()
        engine.stop()
        t.join(timeout=5)

        assert not t.is_alive(), "Fill thread should complete"
        assert len(errors) == 0
