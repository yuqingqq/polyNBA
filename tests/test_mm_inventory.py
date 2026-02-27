"""Tests for inventory management — fills, PnL, exposure, limits."""

from __future__ import annotations

import pytest

from src.mm.config import MMConfig
from src.mm.inventory import InventoryManager
from src.mm.models import Fill, Side


@pytest.fixture
def inventory() -> InventoryManager:
    return InventoryManager()


# ------------------------------------------------------------------
# Fill processing
# ------------------------------------------------------------------


class TestFillUpdates:
    """Tests for position updates from fills."""

    def test_buy_increases_position(self, inventory: InventoryManager) -> None:
        """A buy fill should increase the position."""
        fill = Fill(order_id="o1", token_id="t1", side=Side.BUY, price=0.50, size=100.0)
        inventory.update_fill(fill)
        pos = inventory.get_position("t1")
        assert pos.size == 100.0
        assert pos.avg_entry_price == pytest.approx(0.50)

    def test_sell_decreases_position(self, inventory: InventoryManager) -> None:
        """A sell fill should decrease (or go short) the position."""
        fill = Fill(order_id="o1", token_id="t1", side=Side.SELL, price=0.50, size=100.0)
        inventory.update_fill(fill)
        pos = inventory.get_position("t1")
        assert pos.size == -100.0
        assert pos.avg_entry_price == pytest.approx(0.50)

    def test_multiple_buys_average_entry(self, inventory: InventoryManager) -> None:
        """Multiple buys should compute weighted average entry price."""
        inventory.update_fill(Fill(order_id="o1", token_id="t1", side=Side.BUY, price=0.40, size=100.0))
        inventory.update_fill(Fill(order_id="o2", token_id="t1", side=Side.BUY, price=0.60, size=100.0))
        pos = inventory.get_position("t1")
        assert pos.size == 200.0
        assert pos.avg_entry_price == pytest.approx(0.50)

    def test_partial_close_books_realized_pnl(self, inventory: InventoryManager) -> None:
        """Partially closing a long position should book realized PnL."""
        inventory.update_fill(Fill(order_id="o1", token_id="t1", side=Side.BUY, price=0.40, size=100.0))
        inventory.update_fill(Fill(order_id="o2", token_id="t1", side=Side.SELL, price=0.50, size=50.0))
        pos = inventory.get_position("t1")
        assert pos.size == 50.0
        assert pos.avg_entry_price == pytest.approx(0.40)
        # realized PnL: 50 * (0.50 - 0.40) = 5.0
        assert pos.realized_pnl == pytest.approx(5.0)

    def test_full_close_books_realized_pnl(self, inventory: InventoryManager) -> None:
        """Fully closing a position should book all realized PnL."""
        inventory.update_fill(Fill(order_id="o1", token_id="t1", side=Side.BUY, price=0.40, size=100.0))
        inventory.update_fill(Fill(order_id="o2", token_id="t1", side=Side.SELL, price=0.60, size=100.0))
        pos = inventory.get_position("t1")
        assert pos.size == 0.0
        # realized PnL: 100 * (0.60 - 0.40) = 20.0
        assert pos.realized_pnl == pytest.approx(20.0)

    def test_flip_position(self, inventory: InventoryManager) -> None:
        """Selling more than the long position should flip to short."""
        inventory.update_fill(Fill(order_id="o1", token_id="t1", side=Side.BUY, price=0.40, size=100.0))
        inventory.update_fill(Fill(order_id="o2", token_id="t1", side=Side.SELL, price=0.50, size=150.0))
        pos = inventory.get_position("t1")
        assert pos.size == -50.0
        # realized from closing long: 100 * (0.50 - 0.40) = 10.0
        assert pos.realized_pnl == pytest.approx(10.0)
        # new avg entry for short portion
        assert pos.avg_entry_price == pytest.approx(0.50)

    def test_short_close_realized_pnl(self, inventory: InventoryManager) -> None:
        """Closing a short position should book realized PnL correctly."""
        inventory.update_fill(Fill(order_id="o1", token_id="t1", side=Side.SELL, price=0.60, size=100.0))
        inventory.update_fill(Fill(order_id="o2", token_id="t1", side=Side.BUY, price=0.40, size=100.0))
        pos = inventory.get_position("t1")
        assert pos.size == 0.0
        # realized PnL: 100 * (0.60 - 0.40) = 20.0
        assert pos.realized_pnl == pytest.approx(20.0)

    def test_fee_deducted_from_realized_pnl(self, inventory: InventoryManager) -> None:
        """Fees should be subtracted from realized PnL on position reduction."""
        inventory.update_fill(Fill(order_id="o1", token_id="t1", side=Side.BUY, price=0.40, size=100.0))
        inventory.update_fill(Fill(order_id="o2", token_id="t1", side=Side.SELL, price=0.50, size=100.0, fee=2.0))
        pos = inventory.get_position("t1")
        # realized: 100 * (0.50 - 0.40) - 2.0 = 8.0
        assert pos.realized_pnl == pytest.approx(8.0)


# ------------------------------------------------------------------
# PnL (unrealized)
# ------------------------------------------------------------------


class TestPnL:
    """Tests for unrealized PnL via mark prices."""

    def test_unrealized_pnl_long(self, inventory: InventoryManager) -> None:
        """Long position with higher mark should have positive unrealized PnL."""
        inventory.update_fill(Fill(order_id="o1", token_id="t1", side=Side.BUY, price=0.40, size=100.0))
        inventory.update_mark_price("t1", 0.50)
        pos = inventory.get_position("t1")
        # 100 * (0.50 - 0.40) = 10.0
        assert pos.unrealized_pnl == pytest.approx(10.0)

    def test_unrealized_pnl_short(self, inventory: InventoryManager) -> None:
        """Short position with lower mark should have positive unrealized PnL."""
        inventory.update_fill(Fill(order_id="o1", token_id="t1", side=Side.SELL, price=0.60, size=100.0))
        inventory.update_mark_price("t1", 0.50)
        pos = inventory.get_position("t1")
        # 100 * (0.60 - 0.50) = 10.0
        assert pos.unrealized_pnl == pytest.approx(10.0)

    def test_total_pnl(self, inventory: InventoryManager) -> None:
        """get_pnl should sum realized + unrealized across all tokens."""
        inventory.update_fill(Fill(order_id="o1", token_id="t1", side=Side.BUY, price=0.40, size=100.0))
        inventory.update_fill(Fill(order_id="o2", token_id="t1", side=Side.SELL, price=0.45, size=50.0))
        # realized: 50 * 0.05 = 2.5; remaining 50 long at 0.40
        inventory.update_mark_price("t1", 0.48)
        # unrealized: 50 * (0.48 - 0.40) = 4.0
        total = inventory.get_pnl()
        assert total == pytest.approx(6.5)

    def test_zero_position_zero_unrealized(self, inventory: InventoryManager) -> None:
        """Flat position should have zero unrealized PnL."""
        inventory.update_fill(Fill(order_id="o1", token_id="t1", side=Side.BUY, price=0.50, size=100.0))
        inventory.update_fill(Fill(order_id="o2", token_id="t1", side=Side.SELL, price=0.55, size=100.0))
        inventory.update_mark_price("t1", 0.60)
        pos = inventory.get_position("t1")
        assert pos.unrealized_pnl == pytest.approx(0.0)


# ------------------------------------------------------------------
# Total exposure
# ------------------------------------------------------------------


class TestTotalExposure:
    """Tests for total exposure calculation."""

    def test_single_token_long(self, inventory: InventoryManager) -> None:
        """Exposure of a single long position."""
        inventory.update_fill(Fill(order_id="o1", token_id="t1", side=Side.BUY, price=0.5, size=100.0))
        assert inventory.get_total_exposure() == pytest.approx(100.0)

    def test_single_token_short(self, inventory: InventoryManager) -> None:
        """Exposure of a single short position (absolute value)."""
        inventory.update_fill(Fill(order_id="o1", token_id="t1", side=Side.SELL, price=0.5, size=100.0))
        assert inventory.get_total_exposure() == pytest.approx(100.0)

    def test_multi_token(self, inventory: InventoryManager) -> None:
        """Total exposure sums absolute positions across tokens."""
        inventory.update_fill(Fill(order_id="o1", token_id="t1", side=Side.BUY, price=0.5, size=100.0))
        inventory.update_fill(Fill(order_id="o2", token_id="t2", side=Side.SELL, price=0.5, size=200.0))
        assert inventory.get_total_exposure() == pytest.approx(300.0)

    def test_empty_exposure(self, inventory: InventoryManager) -> None:
        """No positions should have zero exposure."""
        assert inventory.get_total_exposure() == pytest.approx(0.0)


# ------------------------------------------------------------------
# Limits
# ------------------------------------------------------------------


class TestLimits:
    """Tests for position limit checks."""

    def test_within_limits(self, inventory: InventoryManager) -> None:
        """All positions within limits should return True."""
        config = MMConfig(max_position=200.0, max_total_position=500.0)
        inventory.update_fill(Fill(order_id="o1", token_id="t1", side=Side.BUY, price=0.5, size=100.0))
        assert inventory.is_within_limits(config) is True

    def test_per_token_breach(self, inventory: InventoryManager) -> None:
        """A single token exceeding max_position should return False."""
        config = MMConfig(max_position=100.0, max_total_position=500.0)
        inventory.update_fill(Fill(order_id="o1", token_id="t1", side=Side.BUY, price=0.5, size=150.0))
        assert inventory.is_within_limits(config) is False

    def test_total_breach(self, inventory: InventoryManager) -> None:
        """Total exposure exceeding max_total_position should return False."""
        config = MMConfig(max_position=200.0, max_total_position=250.0)
        inventory.update_fill(Fill(order_id="o1", token_id="t1", side=Side.BUY, price=0.5, size=150.0))
        inventory.update_fill(Fill(order_id="o2", token_id="t2", side=Side.BUY, price=0.5, size=150.0))
        # total = 300 > 250
        assert inventory.is_within_limits(config) is False

    def test_short_position_breach(self, inventory: InventoryManager) -> None:
        """A short position exceeding max_position (abs) should return False."""
        config = MMConfig(max_position=100.0, max_total_position=500.0)
        inventory.update_fill(Fill(order_id="o1", token_id="t1", side=Side.SELL, price=0.5, size=150.0))
        assert inventory.is_within_limits(config) is False
