"""Unit and integration tests for divergence EMA infrastructure.

Covers: compute_ema math, DivergenceTracker record/get/breach/summary,
quoting EMA parameter behavior, and engine EMA integration.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.mm.divergence import compute_ema, DivergenceTracker
from src.mm.config import MMConfig
from src.mm.engine import MarketMakingEngine
from src.mm.quoting import QuotingEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker() -> DivergenceTracker:
    return DivergenceTracker(max_bps=1500.0)


@pytest.fixture
def quoting_config() -> MMConfig:
    """Config for quoting EMA tests."""
    return MMConfig(
        spread_bps=200,
        order_size=100.0,
        max_position=500.0,
        inventory_skew_factor=0.5,
        divergence_widen_bps=500,
        divergence_max_bps=1500,
        divergence_size_reduction=0.5,
        divergence_ema_alpha=0.3,
    )


@pytest.fixture
def engine_config() -> MMConfig:
    """Config for engine integration tests."""
    return MMConfig(
        dry_run=True,
        spread_bps=200,
        order_size=50,
        max_position=500,
        max_total_position=2000,
        update_interval_seconds=2,
        stale_data_timeout_seconds=60,
        max_loss=200,
        divergence_widen_bps=500,
        divergence_max_bps=1500,
        divergence_ema_alpha=0.3,
    )


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


# ===========================================================================
# Part 1: compute_ema tests
# ===========================================================================


class TestComputeEma:
    """Pure function EMA computation."""

    def test_alpha_one_returns_new(self) -> None:
        """alpha=1 should return the new value (ignore previous)."""
        assert compute_ema(100.0, 200.0, alpha=1.0) == pytest.approx(200.0)

    def test_alpha_zero_returns_previous(self) -> None:
        """alpha=0 should return the previous value (ignore new)."""
        assert compute_ema(100.0, 200.0, alpha=0.0) == pytest.approx(100.0)

    def test_partial_alpha(self) -> None:
        """alpha=0.3: result = 0.3*new + 0.7*previous."""
        result = compute_ema(100.0, 200.0, alpha=0.3)
        expected = 0.3 * 200.0 + 0.7 * 100.0  # 130
        assert result == pytest.approx(expected)

    def test_convergence(self) -> None:
        """Repeated application should converge toward the constant input."""
        ema = 0.0
        for _ in range(100):
            ema = compute_ema(ema, 500.0, alpha=0.3)
        assert ema == pytest.approx(500.0, abs=0.1)

    def test_spike_smoothing(self) -> None:
        """Single spike from 0 should be smoothed."""
        ema = compute_ema(0.0, 1500.0, alpha=0.3)
        # 0.3 * 1500 = 450
        assert ema == pytest.approx(450.0)


# ===========================================================================
# Part 2: DivergenceTracker tests
# ===========================================================================


class TestDivergenceTrackerBasic:
    """Record and retrieve stats."""

    def test_record_and_get(self, tracker: DivergenceTracker) -> None:
        """Recording creates stats accessible via get()."""
        tracker.record("tok_a", current_bps=200.0, ema_bps=180.0)
        stats = tracker.get("tok_a")
        assert stats is not None
        assert stats.current_bps == pytest.approx(200.0)
        assert stats.ema_bps == pytest.approx(180.0)
        assert stats.count == 1

    def test_get_unknown_returns_none(self, tracker: DivergenceTracker) -> None:
        """get() for an unknown token returns None."""
        assert tracker.get("unknown") is None

    def test_min_max_tracking(self, tracker: DivergenceTracker) -> None:
        """min and max are tracked across observations."""
        tracker.record("tok_a", current_bps=100.0, ema_bps=100.0)
        tracker.record("tok_a", current_bps=300.0, ema_bps=200.0)
        tracker.record("tok_a", current_bps=50.0, ema_bps=155.0)
        stats = tracker.get("tok_a")
        assert stats is not None
        assert stats.min_bps == pytest.approx(50.0)
        assert stats.max_bps == pytest.approx(300.0)
        assert stats.count == 3


class TestDivergenceTrackerBreach:
    """Breach counting."""

    def test_breach_counted(self, tracker: DivergenceTracker) -> None:
        """Observations at or above max_bps are counted as breaches."""
        tracker.record("tok_a", current_bps=1500.0, ema_bps=500.0)
        tracker.record("tok_a", current_bps=200.0, ema_bps=400.0)
        tracker.record("tok_a", current_bps=2000.0, ema_bps=600.0)
        stats = tracker.get("tok_a")
        assert stats is not None
        assert stats.max_breach_count == 2
        assert stats.count == 3

    def test_no_breach_below_max(self, tracker: DivergenceTracker) -> None:
        """Observations below max_bps should not count as breaches."""
        tracker.record("tok_a", current_bps=1499.0, ema_bps=500.0)
        stats = tracker.get("tok_a")
        assert stats is not None
        assert stats.max_breach_count == 0


class TestDivergenceTrackerSnapshots:
    """Snapshots and formatting."""

    def test_snapshots_returns_all(self, tracker: DivergenceTracker) -> None:
        """snapshots() returns stats for all tokens."""
        tracker.record("tok_a", current_bps=100.0, ema_bps=90.0)
        tracker.record("tok_b", current_bps=200.0, ema_bps=180.0)
        snaps = tracker.snapshots()
        assert len(snaps) == 2
        ids = {s.token_id for s in snaps}
        assert ids == {"tok_a", "tok_b"}

    def test_format_summary_with_data(self, tracker: DivergenceTracker) -> None:
        """format_summary includes token name and stats."""
        tracker.record("tok_a", current_bps=100.0, ema_bps=85.0)
        summary = tracker.format_summary()
        assert "Divergence:" in summary
        assert "tok_a:" in summary
        assert "cur=100" in summary
        assert "ema=85" in summary

    def test_format_summary_empty(self, tracker: DivergenceTracker) -> None:
        """format_summary with no data returns a placeholder."""
        assert "no data" in tracker.format_summary()

    def test_get_returns_copy(self, tracker: DivergenceTracker) -> None:
        """get() returns a copy; mutating it should not affect internal state."""
        tracker.record("tok_a", current_bps=100.0, ema_bps=90.0)
        stats = tracker.get("tok_a")
        assert stats is not None
        stats.current_bps = 999.0
        original = tracker.get("tok_a")
        assert original is not None
        assert original.current_bps == pytest.approx(100.0)


# ===========================================================================
# Part 3: Quoting EMA parameter tests
# ===========================================================================


class TestQuotingEmaParameter:
    """generate_quotes with divergence_ema_bps parameter."""

    def test_instant_above_max_ema_below_still_quotes(self, quoting_config: MMConfig) -> None:
        """Key test: instant divergence above max but EMA below → should still quote."""
        engine = QuotingEngine(quoting_config)
        # fv=0.50, mid=0.66 → instant=1600 bps (> max 1500)
        # But if ema_bps=480 (< max 1500), should still quote
        quote = engine.generate_quotes(
            "tok_a", fair_value=0.50, position=0.0,
            market_mid=0.66, divergence_ema_bps=480.0,
        )
        assert quote is not None, "Should quote when EMA is below max even if instant exceeds max"

    def test_ema_above_max_stops_quoting(self, quoting_config: MMConfig) -> None:
        """When EMA exceeds max_bps, should return None."""
        engine = QuotingEngine(quoting_config)
        # fv=0.50, mid=0.66 → instant=1600 bps
        # ema_bps=1600 (> max 1500) → should stop
        quote = engine.generate_quotes(
            "tok_a", fair_value=0.50, position=0.0,
            market_mid=0.66, divergence_ema_bps=1600.0,
        )
        assert quote is None, "Should stop quoting when EMA exceeds max"

    def test_per_token_override_widens_threshold(self, quoting_config: MMConfig) -> None:
        """Per-token override should allow wider max_bps."""
        cfg = quoting_config.model_copy(update={
            "divergence_overrides": {"tok_a": {"max_bps": 3000}},
        })
        engine = QuotingEngine(cfg)
        # fv=0.50, mid=0.66 → instant=1600 bps
        # ema_bps=2000 → above default max (1500) but below override (3000)
        quote = engine.generate_quotes(
            "tok_a", fair_value=0.50, position=0.0,
            market_mid=0.66, divergence_ema_bps=2000.0,
        )
        assert quote is not None, "Per-token override should allow quoting at 2000 bps EMA"

    def test_per_token_override_widen_bps(self, quoting_config: MMConfig) -> None:
        """Per-token widen_bps override should affect spread widening."""
        # Use tick_size=0.001 to avoid rounding masking the difference
        cfg_default = quoting_config.model_copy(update={"tick_size": 0.001})
        cfg_override = quoting_config.model_copy(update={
            "tick_size": 0.001,
            "divergence_overrides": {"tok_a": {"widen_bps": 1000}},
        })
        engine_default = QuotingEngine(cfg_default)
        engine_override = QuotingEngine(cfg_override)
        # With widen_bps=1000 (vs default 500), at 500bps divergence the ratio=0.5 instead of 1.0
        # So spread_mult = 1.5 instead of 2.0
        q_default = engine_default.generate_quotes(
            "tok_a", fair_value=0.50, position=0.0, market_mid=0.55,
        )
        q_override = engine_override.generate_quotes(
            "tok_a", fair_value=0.50, position=0.0, market_mid=0.55,
        )
        assert q_default is not None and q_override is not None
        spread_default = q_default.ask_price - q_default.bid_price
        spread_override = q_override.ask_price - q_override.bid_price
        assert spread_override < spread_default, "Override widen_bps should reduce spread widening"

    def test_none_ema_backward_compat(self, quoting_config: MMConfig) -> None:
        """When divergence_ema_bps is None, behavior should match current (instantaneous)."""
        engine = QuotingEngine(quoting_config)
        # fv=0.50, mid=0.65 → instant=1500 bps (>= max) → should return None
        quote_no_ema = engine.generate_quotes(
            "tok_a", fair_value=0.50, position=0.0,
            market_mid=0.65,
        )
        assert quote_no_ema is None, "Without EMA, instantaneous at max should stop quoting"

    def test_none_ema_below_max_quotes(self, quoting_config: MMConfig) -> None:
        """When divergence_ema_bps is None and instant below max, should quote normally."""
        engine = QuotingEngine(quoting_config)
        quote = engine.generate_quotes(
            "tok_a", fair_value=0.50, position=0.0,
            market_mid=0.55,
        )
        assert quote is not None


# ===========================================================================
# Part 4: Engine EMA integration tests
# ===========================================================================


class TestEngineEmaComputation:
    """Engine computes and passes EMA to quoting."""

    def test_engine_computes_ema_on_tick(self, engine_config: MMConfig) -> None:
        """After a tick with orderbook, engine should have EMA state."""
        ob_mgr = _make_mock_orderbook_manager({"tok_a": 0.55})
        eng = MarketMakingEngine(config=engine_config, client=None, orderbook_manager=ob_mgr)
        eng.add_token("tok_a")
        eng.fair_value_engine.update("tok_a", 0.50)
        eng._tick()

        assert "tok_a" in eng._divergence_emas
        # First observation: ema seeded with first value
        # divergence = |0.50 - 0.55| = 0.05 = 500 bps
        assert eng._divergence_emas["tok_a"] == pytest.approx(500.0)

    def test_spike_smoothed_by_ema(self, engine_config: MMConfig) -> None:
        """A spike should be smoothed by EMA so quoting continues."""
        ob_mgr = _make_mock_orderbook_manager({"tok_a": 0.50})
        eng = MarketMakingEngine(config=engine_config, client=None, orderbook_manager=ob_mgr)
        eng.add_token("tok_a")
        eng.fair_value_engine.update("tok_a", 0.50)

        # First tick: divergence=0 → ema=0
        eng._tick()
        assert eng._divergence_emas["tok_a"] == pytest.approx(0.0)

        # Spike: mid jumps to 0.66 → instant=1600 bps
        ob_mgr.get.side_effect = None
        book = MagicMock()
        book.mid_price = 0.66
        book.best_bid = None
        book.best_ask = None
        book.get_bbo.return_value = (0.66, None, None)
        ob_mgr.get.return_value = book

        eng._tick()

        # EMA = 0.3 * 1600 + 0.7 * 0 = 480 (below max 1500)
        assert eng._divergence_emas["tok_a"] == pytest.approx(480.0)
        # Should still have quotes (EMA below max)
        orders = eng.order_manager.get_open_orders("tok_a")
        assert len(orders) > 0, "Should still quote when EMA is below max despite instant spike"

    def test_ema_seeded_with_first_observation(self, engine_config: MMConfig) -> None:
        """First EMA observation should be seeded (not blended with zero)."""
        ob_mgr = _make_mock_orderbook_manager({"tok_a": 0.55})
        eng = MarketMakingEngine(config=engine_config, client=None, orderbook_manager=ob_mgr)
        eng.add_token("tok_a")
        eng.fair_value_engine.update("tok_a", 0.50)
        eng._tick()

        # First observation: seeded with the value, not blended with 0
        # divergence = 500 bps → ema should be 500 (seeded)
        assert eng._divergence_emas["tok_a"] == pytest.approx(500.0)

    def test_divergence_tracker_recorded(self, engine_config: MMConfig) -> None:
        """Engine records divergence in the DivergenceTracker."""
        ob_mgr = _make_mock_orderbook_manager({"tok_a": 0.55})
        eng = MarketMakingEngine(config=engine_config, client=None, orderbook_manager=ob_mgr)
        eng.add_token("tok_a")
        eng.fair_value_engine.update("tok_a", 0.50)
        eng._tick()

        stats = eng._divergence_tracker.get("tok_a")
        assert stats is not None
        assert stats.count == 1
        assert stats.current_bps == pytest.approx(500.0)

    def test_no_orderbook_no_ema(self, engine_config: MMConfig) -> None:
        """Without orderbook, no EMA should be computed."""
        eng = MarketMakingEngine(config=engine_config, client=None)
        eng.add_token("tok_a")
        eng.fair_value_engine.update("tok_a", 0.50)
        eng._tick()

        assert "tok_a" not in eng._divergence_emas
