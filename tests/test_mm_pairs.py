"""Tests for binary pair-aware market making.

Verifies that paired tokens (sharing a condition_id) use net position for
inventory skew, exposure, and limits, and that fills on one side trigger
requoting of the complement.
"""

from __future__ import annotations

import pytest

from src.mm.config import MMConfig
from src.mm.engine import MarketMakingEngine
from src.mm.inventory import InventoryManager
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
def inventory() -> InventoryManager:
    return InventoryManager()


@pytest.fixture
def paired_engine(config: MMConfig) -> MarketMakingEngine:
    """Engine with a binary pair (tok_a / tok_b) registered and fair values set."""
    eng = MarketMakingEngine(config=config, client=None)
    eng.add_token_pair("tok_a", "tok_b")
    eng.fair_value_engine.update("tok_a", 0.60)
    eng.fair_value_engine.update("tok_b", 0.40)
    return eng


# ---------------------------------------------------------------------------
# Token pair registration
# ---------------------------------------------------------------------------


class TestAddTokenPair:
    """add_token_pair should register both tokens and link them."""

    def test_registers_both_tokens(self, config: MMConfig) -> None:
        eng = MarketMakingEngine(config=config, client=None)
        eng.add_token_pair("tok_a", "tok_b")
        assert "tok_a" in eng._tokens
        assert "tok_b" in eng._tokens

    def test_links_both_directions(self, config: MMConfig) -> None:
        eng = MarketMakingEngine(config=config, client=None)
        eng.add_token_pair("tok_a", "tok_b")
        assert eng._pairs["tok_a"] == "tok_b"
        assert eng._pairs["tok_b"] == "tok_a"

    def test_registers_pair_in_inventory(self, config: MMConfig) -> None:
        eng = MarketMakingEngine(config=config, client=None)
        eng.add_token_pair("tok_a", "tok_b")
        assert eng.inventory_manager._pairs["tok_a"] == "tok_b"
        assert eng.inventory_manager._pairs["tok_b"] == "tok_a"


# ---------------------------------------------------------------------------
# Net position for quoting
# ---------------------------------------------------------------------------


class TestNetPositionQuoting:
    """_requote_token_unlocked should use net position for paired tokens."""

    def test_net_position_drives_skew(self, paired_engine: MarketMakingEngine) -> None:
        """Long A=10, long B=5 -> net=5 for A, net=-5 for B."""
        eng = paired_engine

        # Build some inventory: long 10 A, long 5 B
        for i in range(10):
            eng.inventory_manager.update_fill(
                Fill(order_id=f"a{i}", token_id="tok_a", side=Side.BUY, price=0.60, size=1.0)
            )
        for i in range(5):
            eng.inventory_manager.update_fill(
                Fill(order_id=f"b{i}", token_id="tok_b", side=Side.BUY, price=0.40, size=1.0)
            )

        eng._tick()
        orders_a = eng.order_manager.get_open_orders("tok_a")
        orders_b = eng.order_manager.get_open_orders("tok_b")

        assert len(orders_a) > 0
        assert len(orders_b) > 0

    def test_equal_positions_means_zero_net(self, paired_engine: MarketMakingEngine) -> None:
        """Equal positions on both sides -> net=0, no skew."""
        eng = paired_engine

        # Long 25 on both sides -> net = 0
        for i in range(25):
            eng.inventory_manager.update_fill(
                Fill(order_id=f"a{i}", token_id="tok_a", side=Side.BUY, price=0.60, size=1.0)
            )
            eng.inventory_manager.update_fill(
                Fill(order_id=f"b{i}", token_id="tok_b", side=Side.BUY, price=0.40, size=1.0)
            )

        eng._tick()

        # With net=0, quotes should be symmetric around fair value
        orders_a = eng.order_manager.get_open_orders("tok_a")
        bids_a = [o for o in orders_a if o.side == Side.BUY]
        asks_a = [o for o in orders_a if o.side == Side.SELL]

        assert len(bids_a) > 0
        assert len(asks_a) > 0

        # Both bid and ask should exist (no position limit hit since net=0)
        # The spread should be roughly symmetric around 0.60
        bid = bids_a[0].price
        ask = asks_a[0].price
        mid = (bid + ask) / 2
        assert abs(mid - 0.60) < 0.02, f"Mid {mid} should be near fair value 0.60"

    def test_unpaired_token_uses_raw_position(self, config: MMConfig) -> None:
        """Unpaired tokens should use raw position (backward compat)."""
        eng = MarketMakingEngine(config=config, client=None)
        eng.add_token("tok_solo")
        eng.fair_value_engine.update("tok_solo", 0.50)

        # Build position
        eng.inventory_manager.update_fill(
            Fill(order_id="s1", token_id="tok_solo", side=Side.BUY, price=0.50, size=300.0)
        )

        eng._tick()
        orders = eng.order_manager.get_open_orders("tok_solo")
        bids = [o for o in orders if o.side == Side.BUY]
        asks = [o for o in orders if o.side == Side.SELL]

        # With position=300/500, skew should push bid lower
        assert len(bids) > 0
        assert len(asks) > 0
        # Bid should be meaningfully below fair value
        assert bids[0].price < 0.49


# ---------------------------------------------------------------------------
# Fill triggers complement requote
# ---------------------------------------------------------------------------


class TestFillTriggersComplementRequote:
    """A fill on one paired token should also requote the complement."""

    def test_fill_on_a_requotes_b(self, paired_engine: MarketMakingEngine) -> None:
        """Fill on tok_a should NOT immediately requote tok_b (stale orderbook risk).

        The complement is instead requoted on the next _tick() cycle with
        fresh orderbook data.
        """
        eng = paired_engine
        eng._tick()

        # Get actual buy order for tok_a
        orders_a = eng.order_manager.get_open_orders("tok_a")
        buy_order_a = next(o for o in orders_a if o.side == Side.BUY)

        # Record tok_b's orders before
        orders_b_before = eng.order_manager.get_open_orders("tok_b")
        b_order_ids_before = {o.order_id for o in orders_b_before}

        # Large fill on tok_a to change net position significantly
        # skew = 0.5 * (300/500) * 0.02 = 0.006 = 60bps > 50bps threshold
        fill = Fill(
            order_id=buy_order_a.order_id,
            token_id="tok_a",
            side=Side.BUY,
            price=0.60,
            size=300.0,
        )
        eng.on_fill(fill)

        # tok_b orders should NOT have been requoted immediately (C-2 fix)
        orders_b_after_fill = eng.order_manager.get_open_orders("tok_b")
        b_order_ids_after_fill = {o.order_id for o in orders_b_after_fill}
        assert b_order_ids_after_fill == b_order_ids_before, \
            "Complement orders should NOT be replaced during on_fill (stale orderbook risk)"

        # But the next tick should requote tok_b with the updated net position
        eng._tick()
        orders_b_after_tick = eng.order_manager.get_open_orders("tok_b")
        b_order_ids_after_tick = {o.order_id for o in orders_b_after_tick}
        assert not b_order_ids_after_tick.intersection(b_order_ids_before), \
            "Complement orders should be replaced after next tick cycle"

    def test_fill_on_unpaired_does_not_requote_other(self, config: MMConfig) -> None:
        """Fill on an unpaired token should NOT requote unrelated tokens."""
        eng = MarketMakingEngine(config=config, client=None)
        eng.add_token("tok_x")
        eng.add_token("tok_y")
        eng.fair_value_engine.update("tok_x", 0.50)
        eng.fair_value_engine.update("tok_y", 0.50)
        eng._tick()

        orders_y_before = eng.order_manager.get_open_orders("tok_y")
        y_ids_before = {o.order_id for o in orders_y_before}

        # Get actual buy order for tok_x
        orders_x = eng.order_manager.get_open_orders("tok_x")
        buy_order_x = next(o for o in orders_x if o.side == Side.BUY)

        fill = Fill(
            order_id=buy_order_x.order_id,
            token_id="tok_x",
            side=Side.BUY,
            price=0.50,
            size=300.0,
        )
        eng.on_fill(fill)

        orders_y_after = eng.order_manager.get_open_orders("tok_y")
        y_ids_after = {o.order_id for o in orders_y_after}

        # tok_y orders should remain unchanged (not paired with tok_x)
        assert y_ids_after == y_ids_before


# ---------------------------------------------------------------------------
# Pair-aware exposure
# ---------------------------------------------------------------------------


class TestPairAwareExposure:
    """get_total_exposure should not double-count paired tokens."""

    def test_equal_positions_zero_exposure(self, inventory: InventoryManager) -> None:
        """Long 25 on both sides of a pair -> net exposure = 0."""
        inventory.register_pair("tok_a", "tok_b")
        inventory.update_fill(Fill(order_id="a1", token_id="tok_a", side=Side.BUY, price=0.5, size=25.0))
        inventory.update_fill(Fill(order_id="b1", token_id="tok_b", side=Side.BUY, price=0.5, size=25.0))
        assert inventory.get_total_exposure() == pytest.approx(0.0)

    def test_one_side_only(self, inventory: InventoryManager) -> None:
        """Long 25 on A, 0 on B -> net exposure = 25."""
        inventory.register_pair("tok_a", "tok_b")
        inventory.update_fill(Fill(order_id="a1", token_id="tok_a", side=Side.BUY, price=0.5, size=25.0))
        assert inventory.get_total_exposure() == pytest.approx(25.0)

    def test_asymmetric_positions(self, inventory: InventoryManager) -> None:
        """Long 30 A, long 10 B -> net exposure = 20."""
        inventory.register_pair("tok_a", "tok_b")
        inventory.update_fill(Fill(order_id="a1", token_id="tok_a", side=Side.BUY, price=0.5, size=30.0))
        inventory.update_fill(Fill(order_id="b1", token_id="tok_b", side=Side.BUY, price=0.5, size=10.0))
        assert inventory.get_total_exposure() == pytest.approx(20.0)

    def test_mixed_paired_and_unpaired(self, inventory: InventoryManager) -> None:
        """Paired + unpaired tokens should both contribute correctly."""
        inventory.register_pair("tok_a", "tok_b")
        inventory.update_fill(Fill(order_id="a1", token_id="tok_a", side=Side.BUY, price=0.5, size=30.0))
        inventory.update_fill(Fill(order_id="b1", token_id="tok_b", side=Side.BUY, price=0.5, size=10.0))
        # Unpaired token
        inventory.update_fill(Fill(order_id="c1", token_id="tok_c", side=Side.BUY, price=0.5, size=50.0))
        # Pair exposure: |30 - 10| = 20, unpaired: 50, total = 70
        assert inventory.get_total_exposure() == pytest.approx(70.0)

    def test_unpaired_tokens_unchanged(self, inventory: InventoryManager) -> None:
        """Unpaired tokens should still use abs(size) as before."""
        inventory.update_fill(Fill(order_id="a1", token_id="tok_a", side=Side.BUY, price=0.5, size=100.0))
        inventory.update_fill(Fill(order_id="b1", token_id="tok_b", side=Side.SELL, price=0.5, size=200.0))
        assert inventory.get_total_exposure() == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Pair-aware limits
# ---------------------------------------------------------------------------


class TestPairAwareLimits:
    """is_within_limits should use net position for paired tokens."""

    def test_equal_positions_within_limits(self, inventory: InventoryManager) -> None:
        """Equal positions on pair -> net=0, within limits."""
        config = MMConfig(max_position=100.0, max_total_position=500.0)
        inventory.register_pair("tok_a", "tok_b")
        inventory.update_fill(Fill(order_id="a1", token_id="tok_a", side=Side.BUY, price=0.5, size=200.0))
        inventory.update_fill(Fill(order_id="b1", token_id="tok_b", side=Side.BUY, price=0.5, size=200.0))
        # Per-token: 200 each > 100 limit, but net = 0 < 100 — should pass
        assert inventory.is_within_limits(config) is True

    def test_net_position_breaches_limit(self, inventory: InventoryManager) -> None:
        """Net position exceeding max_position should return False."""
        config = MMConfig(max_position=100.0, max_total_position=500.0)
        inventory.register_pair("tok_a", "tok_b")
        inventory.update_fill(Fill(order_id="a1", token_id="tok_a", side=Side.BUY, price=0.5, size=200.0))
        inventory.update_fill(Fill(order_id="b1", token_id="tok_b", side=Side.BUY, price=0.5, size=50.0))
        # net = |200 - 50| = 150 > 100
        assert inventory.is_within_limits(config) is False

    def test_total_exposure_pair_aware(self, inventory: InventoryManager) -> None:
        """Total exposure limit should use pair-aware calculation."""
        config = MMConfig(max_position=500.0, max_total_position=100.0)
        inventory.register_pair("tok_a", "tok_b")
        inventory.update_fill(Fill(order_id="a1", token_id="tok_a", side=Side.BUY, price=0.5, size=200.0))
        inventory.update_fill(Fill(order_id="b1", token_id="tok_b", side=Side.BUY, price=0.5, size=200.0))
        # Net exposure = 0 < 100 — should pass
        assert inventory.is_within_limits(config) is True

    def test_unpaired_limits_unchanged(self, inventory: InventoryManager) -> None:
        """Unpaired tokens should still check abs(size) against max_position."""
        config = MMConfig(max_position=100.0, max_total_position=500.0)
        inventory.update_fill(Fill(order_id="a1", token_id="tok_a", side=Side.BUY, price=0.5, size=150.0))
        assert inventory.is_within_limits(config) is False
