"""Integration tests for the market making engine — full pipeline in dry-run mode.

These tests exercise the complete flow from fair value through quoting,
order management, inventory tracking, and risk controls. All tests run
in dry-run mode with no network access required.
"""

from __future__ import annotations

import time

import pytest

from src.mm.config import MMConfig
from src.mm.engine import MarketMakingEngine
from src.mm.models import Fill, OrderStatus, Side


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> MMConfig:
    """Standard integration test config."""
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
    """Engine with three NBA tokens registered and fair values set."""
    eng = MarketMakingEngine(config=config, client=None)
    tokens = {
        "token_okc_thunder": 0.35,
        "token_boston_celtics": 0.22,
        "token_lakers": 0.08,
    }
    for token_id, fv in tokens.items():
        eng.add_token(token_id)
        eng.fair_value_engine.update(token_id, fv)
    return eng


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullCycleDryRun:
    """test_full_cycle_dry_run -- init all components, run one cycle, verify quotes generated."""

    def test_engine_initializes_in_dry_run(self, engine: MarketMakingEngine) -> None:
        """Engine should be in dry-run mode."""
        assert engine.order_manager.dry_run is True

    def test_one_cycle_generates_quotes(self, engine: MarketMakingEngine) -> None:
        """Running one tick should produce open orders for all registered tokens."""
        engine._tick()

        open_orders = engine.order_manager.get_open_orders()
        assert len(open_orders) > 0, "Expected open orders after one cycle"

        # Each token should have at least one order
        tokens_with_orders = {o.token_id for o in open_orders}
        assert "token_okc_thunder" in tokens_with_orders
        assert "token_boston_celtics" in tokens_with_orders
        assert "token_lakers" in tokens_with_orders

    def test_both_sides_quoted(self, engine: MarketMakingEngine) -> None:
        """Each token should have both a BUY and SELL order at zero inventory."""
        engine._tick()

        for token_id in ["token_okc_thunder", "token_boston_celtics", "token_lakers"]:
            orders = engine.order_manager.get_open_orders(token_id)
            sides = {o.side for o in orders}
            assert Side.BUY in sides, f"No BUY order for {token_id}"
            assert Side.SELL in sides, f"No SELL order for {token_id}"

    def test_bid_below_ask(self, engine: MarketMakingEngine) -> None:
        """Bid price should be strictly below ask price for each token."""
        engine._tick()

        for token_id in ["token_okc_thunder", "token_boston_celtics", "token_lakers"]:
            orders = engine.order_manager.get_open_orders(token_id)
            buys = [o for o in orders if o.side == Side.BUY]
            sells = [o for o in orders if o.side == Side.SELL]
            if buys and sells:
                assert buys[0].price < sells[0].price, (
                    f"Inverted spread for {token_id}: "
                    f"bid={buys[0].price} >= ask={sells[0].price}"
                )

    def test_order_size_matches_config(self, engine: MarketMakingEngine, config: MMConfig) -> None:
        """Order sizes should match the configured order_size."""
        engine._tick()

        for order in engine.order_manager.get_open_orders():
            assert order.size == config.order_size, (
                f"Order size {order.size} != config {config.order_size}"
            )

    def test_orders_are_live(self, engine: MarketMakingEngine) -> None:
        """All placed orders should have LIVE status in dry-run mode."""
        engine._tick()

        for order in engine.order_manager.get_open_orders():
            assert order.status == OrderStatus.LIVE

    def test_risk_check_passes_initially(self, engine: MarketMakingEngine) -> None:
        """Risk check should pass with fresh data and no positions."""
        result = engine.risk_manager.check_all(
            engine.inventory_manager, engine.fair_value_engine
        )
        assert result.should_halt is False
        assert result.max_position_breached is False
        assert result.max_loss_breached is False
        assert result.stale_data is False


class TestFillUpdatesInventory:
    """test_fill_updates_inventory -- simulate fill, verify position change."""

    def test_buy_fill_increases_position(self, engine: MarketMakingEngine) -> None:
        """A buy fill should increase the position for that token."""
        engine._tick()

        # Find an existing buy order to fill
        buy_orders = [
            o for o in engine.order_manager.get_open_orders("token_okc_thunder")
            if o.side == Side.BUY
        ]
        assert len(buy_orders) > 0, "No buy orders to fill"

        fill = Fill(
            order_id=buy_orders[0].order_id,
            token_id="token_okc_thunder",
            side=Side.BUY,
            price=0.34,
            size=50.0,
        )
        engine.on_fill(fill)

        pos = engine.inventory_manager.get_position("token_okc_thunder")
        assert pos.size == pytest.approx(50.0)
        assert pos.avg_entry_price == pytest.approx(0.34)

    def test_sell_fill_decreases_position(self, engine: MarketMakingEngine) -> None:
        """A sell fill should decrease (go short) the position for that token."""
        engine._tick()

        sell_orders = [
            o for o in engine.order_manager.get_open_orders("token_boston_celtics")
            if o.side == Side.SELL
        ]
        assert len(sell_orders) > 0, "No sell orders to fill"

        fill = Fill(
            order_id=sell_orders[0].order_id,
            token_id="token_boston_celtics",
            side=Side.SELL,
            price=0.24,
            size=50.0,
        )
        engine.on_fill(fill)

        pos = engine.inventory_manager.get_position("token_boston_celtics")
        assert pos.size == pytest.approx(-50.0)
        assert pos.avg_entry_price == pytest.approx(0.24)

    def test_fill_updates_order_status(self, engine: MarketMakingEngine) -> None:
        """A full fill should mark the order as FILLED."""
        engine._tick()

        buy_orders = [
            o for o in engine.order_manager.get_open_orders("token_okc_thunder")
            if o.side == Side.BUY
        ]
        order_id = buy_orders[0].order_id

        fill = Fill(
            order_id=order_id,
            token_id="token_okc_thunder",
            side=Side.BUY,
            price=0.34,
            size=50.0,
        )
        engine.on_fill(fill)

        # The order should now be FILLED (filled_size == size)
        remaining = [
            o for o in engine.order_manager.get_open_orders("token_okc_thunder")
            if o.order_id == order_id
        ]
        assert len(remaining) == 0, "Filled order should not appear in open orders"

    def test_total_exposure_updates_after_fill(self, engine: MarketMakingEngine) -> None:
        """Total exposure should reflect the new position after a fill."""
        engine._tick()

        assert engine.inventory_manager.get_total_exposure() == pytest.approx(0.0)

        buy_orders = [
            o for o in engine.order_manager.get_open_orders("token_okc_thunder")
            if o.side == Side.BUY
        ]
        fill = Fill(
            order_id=buy_orders[0].order_id,
            token_id="token_okc_thunder",
            side=Side.BUY,
            price=0.34,
            size=50.0,
        )
        engine.on_fill(fill)

        assert engine.inventory_manager.get_total_exposure() == pytest.approx(50.0)


class TestInventorySkewAfterFill:
    """test_inventory_skew_after_fill -- verify quotes skew after accumulating inventory."""

    def test_long_position_skews_bid_down(self, engine: MarketMakingEngine) -> None:
        """After a large buy fill, the bid should be lower than with zero inventory.

        Skew = inventory_skew_factor * (position / max_position) * spread.
        With position=300, skew = 0.5 * (300/500) * 0.02 = 0.006, enough
        to shift the quote by at least one tick (0.01) after rounding.
        """
        # Get quotes at zero inventory
        engine._tick()
        orders_before = engine.order_manager.get_open_orders("token_okc_thunder")
        bid_before = next(o.price for o in orders_before if o.side == Side.BUY)
        buy_order = next(o for o in orders_before if o.side == Side.BUY)

        # Simulate a large buy fill to generate meaningful skew
        fill = Fill(
            order_id=buy_order.order_id,
            token_id="token_okc_thunder",
            side=Side.BUY,
            price=0.34,
            size=300.0,
        )
        engine.on_fill(fill)

        # Run another tick to get skewed quotes
        engine._tick()

        orders_after = engine.order_manager.get_open_orders("token_okc_thunder")
        buys_after = [o for o in orders_after if o.side == Side.BUY]
        assert len(buys_after) > 0, "Expected bid after fill"
        bid_after = buys_after[0].price

        assert bid_after < bid_before, (
            f"Expected bid to decrease with long position: "
            f"before={bid_before:.4f} after={bid_after:.4f}"
        )

    def test_long_position_skews_ask_down(self, engine: MarketMakingEngine) -> None:
        """After a large buy fill, the ask should also be lower (skew shifts both sides).

        Uses a non-tick-aligned fair value (0.355) so that the skew of
        0.006 (position=300) is enough to shift the ceiling-rounded ask
        down by a full tick.
        """
        # Set a non-tick-aligned fair value so skew can move the ask
        engine.fair_value_engine.update("token_okc_thunder", 0.355)
        engine._tick()
        orders_before = engine.order_manager.get_open_orders("token_okc_thunder")
        ask_before = next(o.price for o in orders_before if o.side == Side.SELL)
        buy_order = next(o for o in orders_before if o.side == Side.BUY)

        fill = Fill(
            order_id=buy_order.order_id,
            token_id="token_okc_thunder",
            side=Side.BUY,
            price=0.34,
            size=300.0,
        )
        engine.on_fill(fill)

        engine._tick()

        orders_after = engine.order_manager.get_open_orders("token_okc_thunder")
        asks_after = [o for o in orders_after if o.side == Side.SELL]
        assert len(asks_after) > 0, "Expected ask after fill"
        ask_after = asks_after[0].price

        assert ask_after < ask_before, (
            f"Expected ask to decrease with long position: "
            f"before={ask_before:.4f} after={ask_after:.4f}"
        )

    def test_short_position_skews_quotes_up(self, engine: MarketMakingEngine) -> None:
        """After a large sell fill (short), quotes should shift upward.

        Uses a non-tick-aligned fair value (0.505) so that the skew of
        0.006 is enough to shift the floor-rounded bid up by a full tick.
        """
        # Use a non-tick-aligned fv so the skew can push the bid to the next tick
        engine.fair_value_engine.update("token_okc_thunder", 0.505)
        engine._tick()
        orders_before = engine.order_manager.get_open_orders("token_okc_thunder")
        bid_before = next(o.price for o in orders_before if o.side == Side.BUY)
        ask_before = next(o.price for o in orders_before if o.side == Side.SELL)
        sell_order = next(o for o in orders_before if o.side == Side.SELL)

        fill = Fill(
            order_id=sell_order.order_id,
            token_id="token_okc_thunder",
            side=Side.SELL,
            price=0.505,
            size=300.0,
        )
        engine.on_fill(fill)

        engine._tick()

        orders_after = engine.order_manager.get_open_orders("token_okc_thunder")
        buys_after = [o for o in orders_after if o.side == Side.BUY]
        asks_after = [o for o in orders_after if o.side == Side.SELL]

        assert len(buys_after) > 0, "Expected bid after short fill"
        assert len(asks_after) > 0, "Expected ask after short fill"

        assert buys_after[0].price > bid_before, (
            f"Expected bid to increase with short position: "
            f"before={bid_before:.4f} after={buys_after[0].price:.4f}"
        )
        assert asks_after[0].price > ask_before, (
            f"Expected ask to increase with short position: "
            f"before={ask_before:.4f} after={asks_after[0].price:.4f}"
        )

    def test_large_position_drops_one_side(self, engine: MarketMakingEngine, config: MMConfig) -> None:
        """At max position, one side of the quote should be dropped."""
        engine._tick()

        orders = engine.order_manager.get_open_orders("token_okc_thunder")
        buy_order = next(o for o in orders if o.side == Side.BUY)

        # Fill to max position
        fill = Fill(
            order_id=buy_order.order_id,
            token_id="token_okc_thunder",
            side=Side.BUY,
            price=0.35,
            size=config.max_position,
        )
        engine.on_fill(fill)

        engine._tick()

        orders = engine.order_manager.get_open_orders("token_okc_thunder")
        buys = [o for o in orders if o.side == Side.BUY]
        sells = [o for o in orders if o.side == Side.SELL]

        # At max long, bid should be dropped
        assert len(buys) == 0, "Expected no BUY orders at max long position"
        assert len(sells) > 0, "Expected SELL orders at max long position"


class TestRiskHaltOnStaleData:
    """test_risk_halt_on_stale_data -- verify engine halts when data is stale."""

    def test_stale_data_triggers_halt(self, config: MMConfig) -> None:
        """When fair value data is older than timeout, risk check should detect staleness.

        Staleness is now per-token and does not trigger a global halt.
        The engine skips stale tokens individually instead.
        """
        eng = MarketMakingEngine(config=config, client=None)
        eng.add_token("token_okc_thunder")

        # Set stale timestamp (2 minutes ago, timeout is 60s)
        stale_ts = time.time() - 120
        eng.fair_value_engine.update("token_okc_thunder", 0.35, timestamp=stale_ts)

        result = eng.risk_manager.check_all(eng.inventory_manager, eng.fair_value_engine)
        assert result.stale_data is True
        assert result.should_halt is False  # Staleness no longer triggers halt
        assert "token_okc_thunder" in result.stale_tokens

    def test_stale_data_cancels_orders_on_tick(self, config: MMConfig) -> None:
        """A tick with stale data should cancel all existing orders."""
        eng = MarketMakingEngine(config=config, client=None)
        eng.add_token("token_okc_thunder")

        # First create some orders with fresh data
        eng.fair_value_engine.update("token_okc_thunder", 0.35)
        eng._tick()
        assert len(eng.order_manager.get_open_orders()) > 0

        # Now make data stale
        stale_ts = time.time() - 120
        eng.fair_value_engine.update("token_okc_thunder", 0.35, timestamp=stale_ts)

        eng._tick()
        assert len(eng.order_manager.get_open_orders()) == 0

    def test_fresh_data_does_not_halt(self, engine: MarketMakingEngine) -> None:
        """Fresh data should not trigger a halt."""
        result = engine.risk_manager.check_all(
            engine.inventory_manager, engine.fair_value_engine
        )
        assert result.stale_data is False
        assert result.should_halt is False

    def test_no_data_is_stale(self, config: MMConfig) -> None:
        """An engine with no fair value data at all should be stale.

        Staleness is now per-token and does not trigger a global halt.
        """
        eng = MarketMakingEngine(config=config, client=None)
        eng.add_token("token_okc_thunder")

        result = eng.risk_manager.check_all(eng.inventory_manager, eng.fair_value_engine)
        assert result.stale_data is True
        assert result.should_halt is False  # Staleness no longer triggers halt


class TestKillSwitchCancelsAll:
    """test_kill_switch_cancels_all -- verify kill switch removes all open orders."""

    def test_kill_switch_removes_all_orders(self, engine: MarketMakingEngine) -> None:
        """Kill switch should cancel every open order across all tokens."""
        engine._tick()
        assert len(engine.order_manager.get_open_orders()) > 0

        engine.risk_manager.kill_switch(engine.order_manager)
        assert len(engine.order_manager.get_open_orders()) == 0

    def test_kill_switch_on_max_loss(self, config: MMConfig) -> None:
        """Max loss breach should trigger kill switch via tick."""
        eng = MarketMakingEngine(config=config, client=None)
        eng.add_token("token_okc_thunder")
        eng.fair_value_engine.update("token_okc_thunder", 0.50)

        # Place orders
        eng._tick()
        assert len(eng.order_manager.get_open_orders()) > 0

        # Use actual order IDs from placed orders
        orders = eng.order_manager.get_open_orders("token_okc_thunder")
        buy_order = next(o for o in orders if o.side == Side.BUY)
        sell_order = next(o for o in orders if o.side == Side.SELL)

        # Simulate big loss: buy at 0.60, sell at 0.10
        eng.on_fill(Fill(
            order_id=buy_order.order_id,
            token_id="token_okc_thunder",
            side=Side.BUY,
            price=0.60,
            size=500.0,
        ))
        eng.on_fill(Fill(
            order_id=sell_order.order_id,
            token_id="token_okc_thunder",
            side=Side.SELL,
            price=0.10,
            size=500.0,
        ))

        # PnL = 500 * (0.10 - 0.60) = -250 < -200
        pnl = eng.inventory_manager.get_pnl()
        assert pnl < -config.max_loss

        # Tick should trigger halt and kill switch
        eng._tick()
        assert len(eng.order_manager.get_open_orders()) == 0

    def test_kill_switch_idempotent(self, engine: MarketMakingEngine) -> None:
        """Calling kill switch with no orders should not raise."""
        engine.risk_manager.kill_switch(engine.order_manager)
        assert len(engine.order_manager.get_open_orders()) == 0

    def test_orders_all_cancelled_status(self, engine: MarketMakingEngine) -> None:
        """After kill switch, order statuses should be CANCELLED."""
        engine._tick()
        # Grab order IDs before kill
        order_ids = [o.order_id for o in engine.order_manager.get_open_orders()]
        assert len(order_ids) > 0

        engine.risk_manager.kill_switch(engine.order_manager)

        # All orders should now be CANCELLED
        for oid in order_ids:
            order = engine.order_manager._orders[oid]
            assert order.status == OrderStatus.CANCELLED, (
                f"Order {oid} status is {order.status}, expected CANCELLED"
            )


class TestRequoteOnPriceMove:
    """test_requote_on_price_move -- verify engine re-quotes when fair value moves beyond threshold."""

    def test_large_price_move_triggers_requote(self, config: MMConfig) -> None:
        """A fair value change larger than requote_threshold_bps should produce new quotes."""
        eng = MarketMakingEngine(config=config, client=None)
        eng.add_token("token_okc_thunder")
        eng.fair_value_engine.update("token_okc_thunder", 0.50)

        eng._tick()
        orders_v1 = eng.order_manager.get_open_orders("token_okc_thunder")
        bid_v1 = next(o.price for o in orders_v1 if o.side == Side.BUY)
        ask_v1 = next(o.price for o in orders_v1 if o.side == Side.SELL)

        # Move fair value by 3 cents (300 bps > threshold of 50 bps)
        eng.fair_value_engine.update("token_okc_thunder", 0.53)
        eng._tick()

        orders_v2 = eng.order_manager.get_open_orders("token_okc_thunder")
        bid_v2 = next(o.price for o in orders_v2 if o.side == Side.BUY)
        ask_v2 = next(o.price for o in orders_v2 if o.side == Side.SELL)

        assert bid_v2 > bid_v1, f"Bid should increase: {bid_v1:.4f} -> {bid_v2:.4f}"
        assert ask_v2 > ask_v1, f"Ask should increase: {ask_v1:.4f} -> {ask_v2:.4f}"

    def test_small_price_move_no_requote(self, config: MMConfig) -> None:
        """A fair value change smaller than requote_threshold_bps should NOT requote.

        Uses a mid-tick fair value (0.505) so that a tiny move (0.001)
        does not cross a tick boundary under directional rounding (bids
        floor, asks ceil).
        """
        eng = MarketMakingEngine(config=config, client=None)
        eng.add_token("token_okc_thunder")
        eng.fair_value_engine.update("token_okc_thunder", 0.505)

        eng._tick()
        orders_v1 = eng.order_manager.get_open_orders("token_okc_thunder")
        order_ids_v1 = {o.order_id for o in orders_v1}

        # Move fair value by 0.001 (10 bps < threshold of 50 bps)
        eng.fair_value_engine.update("token_okc_thunder", 0.506)
        eng._tick()

        orders_v2 = eng.order_manager.get_open_orders("token_okc_thunder")
        order_ids_v2 = {o.order_id for o in orders_v2}

        # Same order IDs should still be live (no requote)
        assert order_ids_v1 == order_ids_v2, (
            "Orders should not have changed for a small price move"
        )

    def test_price_move_down_updates_quotes(self, config: MMConfig) -> None:
        """A downward fair value move should shift quotes down."""
        eng = MarketMakingEngine(config=config, client=None)
        eng.add_token("token_lakers")
        eng.fair_value_engine.update("token_lakers", 0.50)

        eng._tick()
        orders_v1 = eng.order_manager.get_open_orders("token_lakers")
        bid_v1 = next(o.price for o in orders_v1 if o.side == Side.BUY)
        ask_v1 = next(o.price for o in orders_v1 if o.side == Side.SELL)

        # Move fair value down by 3 cents
        eng.fair_value_engine.update("token_lakers", 0.47)
        eng._tick()

        orders_v2 = eng.order_manager.get_open_orders("token_lakers")
        bid_v2 = next(o.price for o in orders_v2 if o.side == Side.BUY)
        ask_v2 = next(o.price for o in orders_v2 if o.side == Side.SELL)

        assert bid_v2 < bid_v1, f"Bid should decrease: {bid_v1:.4f} -> {bid_v2:.4f}"
        assert ask_v2 < ask_v1, f"Ask should decrease: {ask_v1:.4f} -> {ask_v2:.4f}"

    def test_multiple_price_updates_tracked(self, config: MMConfig) -> None:
        """Multiple consecutive price moves should each trigger appropriate requoting."""
        eng = MarketMakingEngine(config=config, client=None)
        eng.add_token("token_okc_thunder")

        prices = [0.30, 0.35, 0.40, 0.45]
        bids: list[float] = []
        asks: list[float] = []

        for fv in prices:
            eng.fair_value_engine.update("token_okc_thunder", fv)
            eng._tick()
            orders = eng.order_manager.get_open_orders("token_okc_thunder")
            bids.append(next(o.price for o in orders if o.side == Side.BUY))
            asks.append(next(o.price for o in orders if o.side == Side.SELL))

        # Bids and asks should be monotonically increasing
        for i in range(1, len(bids)):
            assert bids[i] > bids[i - 1], (
                f"Bid should increase: step {i}: {bids[i-1]:.4f} -> {bids[i]:.4f}"
            )
            assert asks[i] > asks[i - 1], (
                f"Ask should increase: step {i}: {asks[i-1]:.4f} -> {asks[i]:.4f}"
            )


class TestIntegrationScript:
    """Run the full integration test script and generate the report."""

    def test_run_integration_script(self) -> None:
        """Execute the integration test script and verify the report is generated."""
        from src.mm.integration_test import run_integration_test
        from pathlib import Path

        report_md = run_integration_test()

        # Verify report has content
        assert len(report_md) > 100, "Report should have meaningful content"
        assert "Integration Test Report" in report_md
        assert "ALL CHECKS PASSED" in report_md

        # Write report to workspace
        workspace_dir = Path(
            "/home/yuqing/polymarket-trading/orchestrator/"
            "PROGRAMS/P-2026-003-market-making/workspace"
        )
        workspace_dir.mkdir(parents=True, exist_ok=True)
        report_path = workspace_dir / "integration-test-report.md"
        report_path.write_text(report_md)
        assert report_path.exists()
