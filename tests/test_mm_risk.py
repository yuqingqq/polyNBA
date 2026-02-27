"""Tests for the risk manager — position limits, loss limits, stale data, kill switch."""

from __future__ import annotations

import time

import pytest

from src.mm.config import MMConfig
from src.mm.fair_value import FairValueEngine
from src.mm.inventory import InventoryManager
from src.mm.models import Fill, Side
from src.mm.order_manager import OrderManager
from src.mm.risk import RiskManager


@pytest.fixture
def config() -> MMConfig:
    return MMConfig(
        max_position=100.0,
        max_total_position=300.0,
        max_loss=50.0,
        stale_data_timeout_seconds=2,
    )


@pytest.fixture
def risk(config: MMConfig) -> RiskManager:
    return RiskManager(config)


@pytest.fixture
def inventory() -> InventoryManager:
    return InventoryManager()


@pytest.fixture
def fv_engine() -> FairValueEngine:
    return FairValueEngine()


# ------------------------------------------------------------------
# Max position breach
# ------------------------------------------------------------------


class TestMaxPositionBreach:
    """Tests for per-token position limit checks."""

    def test_within_limit(self, risk: RiskManager, inventory: InventoryManager, fv_engine: FairValueEngine) -> None:
        """No breach when position is within limit."""
        fv_engine.update("t1", 0.5)
        fill = Fill(order_id="o1", token_id="t1", side=Side.BUY, price=0.5, size=50.0)
        inventory.update_fill(fill)
        result = risk.check_all(inventory, fv_engine)
        assert result.max_position_breached is False
        assert result.should_halt is False

    def test_breach(self, risk: RiskManager, inventory: InventoryManager, fv_engine: FairValueEngine) -> None:
        """Breach when position exceeds limit."""
        fv_engine.update("t1", 0.5)
        fill = Fill(order_id="o1", token_id="t1", side=Side.BUY, price=0.5, size=150.0)
        inventory.update_fill(fill)
        result = risk.check_all(inventory, fv_engine)
        assert result.max_position_breached is True
        assert result.should_halt is True
        assert any("position" in r.lower() for r in result.reasons)


# ------------------------------------------------------------------
# Max total position breach
# ------------------------------------------------------------------


class TestMaxTotalPositionBreach:
    """Tests for total exposure limit checks."""

    def test_within_total_limit(self, risk: RiskManager, inventory: InventoryManager, fv_engine: FairValueEngine) -> None:
        """No breach when total exposure is within limit."""
        fv_engine.update("t1", 0.5)
        fv_engine.update("t2", 0.5)
        inventory.update_fill(Fill(order_id="o1", token_id="t1", side=Side.BUY, price=0.5, size=80.0))
        inventory.update_fill(Fill(order_id="o2", token_id="t2", side=Side.BUY, price=0.5, size=80.0))
        result = risk.check_all(inventory, fv_engine)
        assert result.max_total_breached is False

    def test_total_breach(self, risk: RiskManager, inventory: InventoryManager, fv_engine: FairValueEngine) -> None:
        """Breach when total exposure exceeds limit."""
        fv_engine.update("t1", 0.5)
        fv_engine.update("t2", 0.5)
        fv_engine.update("t3", 0.5)
        fv_engine.update("t4", 0.5)
        inventory.update_fill(Fill(order_id="o1", token_id="t1", side=Side.BUY, price=0.5, size=90.0))
        inventory.update_fill(Fill(order_id="o2", token_id="t2", side=Side.BUY, price=0.5, size=90.0))
        inventory.update_fill(Fill(order_id="o3", token_id="t3", side=Side.BUY, price=0.5, size=90.0))
        inventory.update_fill(Fill(order_id="o4", token_id="t4", side=Side.SELL, price=0.5, size=90.0))
        # total = 90+90+90+90 = 360 > 300
        result = risk.check_all(inventory, fv_engine)
        assert result.max_total_breached is True
        assert result.should_halt is True


# ------------------------------------------------------------------
# Max loss breach
# ------------------------------------------------------------------


class TestMaxLossBreach:
    """Tests for PnL loss limit checks."""

    def test_no_loss_breach(self, risk: RiskManager, inventory: InventoryManager, fv_engine: FairValueEngine) -> None:
        """No breach when PnL is above the loss limit."""
        fv_engine.update("t1", 0.5)
        result = risk.check_all(inventory, fv_engine)
        assert result.max_loss_breached is False

    def test_loss_breach(self, risk: RiskManager, inventory: InventoryManager, fv_engine: FairValueEngine) -> None:
        """Breach when realized PnL exceeds the loss limit."""
        fv_engine.update("t1", 0.5)
        # Buy at 0.6, sell at 0.3 -> loss per unit = 0.3, for 200 units = -60
        inventory.update_fill(Fill(order_id="o1", token_id="t1", side=Side.BUY, price=0.6, size=200.0))
        inventory.update_fill(Fill(order_id="o2", token_id="t1", side=Side.SELL, price=0.3, size=200.0))
        result = risk.check_all(inventory, fv_engine)
        assert result.max_loss_breached is True
        assert result.should_halt is True
        assert any("loss" in r.lower() or "pnl" in r.lower() for r in result.reasons)


# ------------------------------------------------------------------
# Stale data
# ------------------------------------------------------------------


class TestStaleData:
    """Tests for stale data detection."""

    def test_no_data_is_stale(self, risk: RiskManager, inventory: InventoryManager, fv_engine: FairValueEngine) -> None:
        """No fair values at all should count as stale.

        Staleness is now per-token and does not trigger a global halt.
        """
        result = risk.check_all(inventory, fv_engine)
        assert result.stale_data is True
        assert result.should_halt is False  # Staleness no longer triggers halt

    def test_fresh_data_is_not_stale(self, risk: RiskManager, inventory: InventoryManager, fv_engine: FairValueEngine) -> None:
        """Recently updated data should not be stale."""
        fv_engine.update("t1", 0.5, timestamp=time.time())
        result = risk.check_all(inventory, fv_engine)
        assert result.stale_data is False

    def test_old_data_is_stale(self, risk: RiskManager, inventory: InventoryManager, fv_engine: FairValueEngine) -> None:
        """Data older than the timeout should be stale.

        Staleness is now per-token: the stale token is recorded in
        stale_tokens but should_halt is False (only position/loss halts).
        """
        fv_engine.update("t1", 0.5, timestamp=time.time() - 10)  # 10s ago, timeout=2s
        result = risk.check_all(inventory, fv_engine)
        assert result.stale_data is True
        assert result.should_halt is False  # Staleness no longer triggers halt
        assert "t1" in result.stale_tokens


# ------------------------------------------------------------------
# Kill switch
# ------------------------------------------------------------------


class TestKillSwitch:
    """Tests for the emergency kill switch."""

    def test_kill_switch_cancels_all(self, risk: RiskManager) -> None:
        """Kill switch should cancel all open orders."""
        om = OrderManager(client=None, dry_run=True)
        # Place some orders
        om.place_order("t1", Side.BUY, 0.49, 50.0)
        om.place_order("t1", Side.SELL, 0.51, 50.0)
        om.place_order("t2", Side.BUY, 0.30, 25.0)
        assert len(om.get_open_orders()) == 3

        risk.kill_switch(om)
        assert len(om.get_open_orders()) == 0

    def test_kill_switch_idempotent(self, risk: RiskManager) -> None:
        """Kill switch on empty book should not raise."""
        om = OrderManager(client=None, dry_run=True)
        risk.kill_switch(om)  # should not raise
        assert len(om.get_open_orders()) == 0
