"""Unit tests for divergence-aware quoting.

Tests the spread/size multiplier logic when a market_mid is provided
to generate_quotes(). Covers: no market_mid, zero divergence, moderate
divergence, at threshold, beyond max, negative direction, with skew,
size floor, and market_mid in Quote.
"""

from __future__ import annotations

import pytest

from src.mm.config import MMConfig
from src.mm.quoting import QuotingEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> MMConfig:
    """Standard config for divergence tests."""
    return MMConfig(
        spread_bps=200,  # 2% total spread
        order_size=100.0,
        max_position=500.0,
        inventory_skew_factor=0.5,
        divergence_widen_bps=500,  # 5% = spread doubles
        divergence_max_bps=1500,  # 15% = stop quoting
        divergence_size_reduction=0.5,
    )


@pytest.fixture
def engine(config: MMConfig) -> QuotingEngine:
    return QuotingEngine(config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDivergenceNoMarketMid:
    """When market_mid is None, no divergence adjustment should be applied."""

    def test_no_market_mid_returns_base_spread(self, engine: QuotingEngine, config: MMConfig) -> None:
        """Without market_mid, spread should be the base spread."""
        quote = engine.generate_quotes("tok_a", fair_value=0.50, position=0.0)
        assert quote is not None
        spread = quote.ask_price - quote.bid_price
        expected_spread = config.spread_bps / 10_000.0
        assert spread == pytest.approx(expected_spread, abs=config.tick_size)

    def test_no_market_mid_base_size(self, engine: QuotingEngine, config: MMConfig) -> None:
        """Without market_mid, order size should be the base order_size."""
        quote = engine.generate_quotes("tok_a", fair_value=0.50, position=0.0)
        assert quote is not None
        assert quote.bid_size == pytest.approx(config.order_size)
        assert quote.ask_size == pytest.approx(config.order_size)

    def test_no_market_mid_field_is_none(self, engine: QuotingEngine) -> None:
        """Quote.market_mid should be None when not provided."""
        quote = engine.generate_quotes("tok_a", fair_value=0.50, position=0.0)
        assert quote is not None
        assert quote.market_mid is None


class TestDivergenceZero:
    """When market_mid == fair_value, multipliers should be 1.0."""

    def test_zero_divergence_same_as_no_mid(self, engine: QuotingEngine) -> None:
        """market_mid == fair_value should produce the same quote as no mid."""
        q_none = engine.generate_quotes("tok_a", fair_value=0.50, position=0.0)
        q_zero = engine.generate_quotes("tok_a", fair_value=0.50, position=0.0, market_mid=0.50)
        assert q_none is not None and q_zero is not None
        assert q_none.bid_price == q_zero.bid_price
        assert q_none.ask_price == q_zero.ask_price
        assert q_none.bid_size == q_zero.bid_size

    def test_zero_divergence_market_mid_recorded(self, engine: QuotingEngine) -> None:
        """Quote.market_mid should be recorded even when divergence is zero."""
        quote = engine.generate_quotes("tok_a", fair_value=0.50, position=0.0, market_mid=0.50)
        assert quote is not None
        assert quote.market_mid == 0.50


class TestDivergenceModerate:
    """250 bps divergence (half of widen threshold)."""

    def test_moderate_spread_widens(self, engine: QuotingEngine, config: MMConfig) -> None:
        """At 250 bps divergence (ratio=0.5), spread should be 1.5x base."""
        # fv=0.50, mid=0.525 → divergence=250 bps
        q_base = engine.generate_quotes("tok_a", fair_value=0.50, position=0.0)
        q_div = engine.generate_quotes("tok_a", fair_value=0.50, position=0.0, market_mid=0.525)
        assert q_base is not None and q_div is not None

        base_spread = q_base.ask_price - q_base.bid_price
        div_spread = q_div.ask_price - q_div.bid_price
        # 1.5x base spread, allow one tick rounding
        assert div_spread >= base_spread * 1.4
        assert div_spread <= base_spread * 1.6 + config.tick_size

    def test_moderate_size_reduces(self, engine: QuotingEngine, config: MMConfig) -> None:
        """At 250 bps (ratio=0.5), size should be 0.75x base."""
        # size_mult = max(0.1, 1.0 - 0.5 * (1.0 - 0.5)) = 1.0 - 0.25 = 0.75
        quote = engine.generate_quotes("tok_a", fair_value=0.50, position=0.0, market_mid=0.525)
        assert quote is not None
        expected = config.order_size * 0.75
        assert quote.bid_size == pytest.approx(expected)
        assert quote.ask_size == pytest.approx(expected)


class TestDivergenceAtThreshold:
    """500 bps divergence (at widen threshold — ratio=1.0)."""

    def test_at_threshold_spread_doubles(self, engine: QuotingEngine, config: MMConfig) -> None:
        """At 500 bps divergence, spread should be 2.0x base."""
        q_base = engine.generate_quotes("tok_a", fair_value=0.50, position=0.0)
        q_div = engine.generate_quotes("tok_a", fair_value=0.50, position=0.0, market_mid=0.55)
        assert q_base is not None and q_div is not None

        base_spread = q_base.ask_price - q_base.bid_price
        div_spread = q_div.ask_price - q_div.bid_price
        assert div_spread >= base_spread * 1.9
        assert div_spread <= base_spread * 2.1 + config.tick_size

    def test_at_threshold_size_halves(self, engine: QuotingEngine, config: MMConfig) -> None:
        """At 500 bps divergence, size should be 0.5x base."""
        quote = engine.generate_quotes("tok_a", fair_value=0.50, position=0.0, market_mid=0.55)
        assert quote is not None
        expected = config.order_size * 0.5
        assert quote.bid_size == pytest.approx(expected)
        assert quote.ask_size == pytest.approx(expected)


class TestDivergenceBeyondMax:
    """>=1500 bps divergence — should return None."""

    def test_beyond_max_returns_none(self, engine: QuotingEngine) -> None:
        """At 1500 bps divergence, generate_quotes should return None."""
        # fv=0.50, mid=0.65 → divergence=1500 bps
        quote = engine.generate_quotes("tok_a", fair_value=0.50, position=0.0, market_mid=0.65)
        assert quote is None

    def test_well_beyond_max_returns_none(self, engine: QuotingEngine) -> None:
        """At 3000 bps divergence, generate_quotes should return None."""
        quote = engine.generate_quotes("tok_a", fair_value=0.20, position=0.0, market_mid=0.50)
        assert quote is None


class TestDivergenceNegativeDirection:
    """Divergence should work the same regardless of direction."""

    def test_mid_below_fv_widens_spread(self, engine: QuotingEngine, config: MMConfig) -> None:
        """market_mid < fair_value should also widen the spread."""
        q_base = engine.generate_quotes("tok_a", fair_value=0.50, position=0.0)
        # mid=0.475 → divergence=250 bps
        q_div = engine.generate_quotes("tok_a", fair_value=0.50, position=0.0, market_mid=0.475)
        assert q_base is not None and q_div is not None

        base_spread = q_base.ask_price - q_base.bid_price
        div_spread = q_div.ask_price - q_div.bid_price
        assert div_spread > base_spread


class TestDivergenceWithSkew:
    """Divergence combined with inventory skew."""

    def test_skew_uses_widened_spread(self, engine: QuotingEngine, config: MMConfig) -> None:
        """Inventory skew should apply to the widened spread, not the base."""
        position = 250.0  # half of max_position
        # With divergence, the spread is wider, so the skew effect is larger
        q_base = engine.generate_quotes("tok_a", fair_value=0.50, position=position)
        q_div = engine.generate_quotes("tok_a", fair_value=0.50, position=position, market_mid=0.55)
        assert q_base is not None and q_div is not None

        # Both should have skewed quotes, but the divergence one more so
        # Skew = factor * (pos/max) * spread — wider spread → larger skew
        # The bid should be pushed down more with divergence
        assert q_div.bid_price <= q_base.bid_price


class TestDivergenceSizeFloor:
    """Size multiplier should floor at 0.1x."""

    def test_high_divergence_size_floors(self, engine: QuotingEngine, config: MMConfig) -> None:
        """At 1000 bps (ratio=2.0), size_mult would be negative without floor → floor at 0.1."""
        # fv=0.50, mid=0.60 → divergence=1000 bps, ratio=2.0
        # size_mult = max(0.1, 1.0 - 2.0 * 0.5) = max(0.1, 0.0) = 0.1
        quote = engine.generate_quotes("tok_a", fair_value=0.50, position=0.0, market_mid=0.60)
        assert quote is not None
        expected = config.order_size * 0.1
        assert quote.bid_size == pytest.approx(expected)
        assert quote.ask_size == pytest.approx(expected)


class TestDivergenceWidenBpsZero:
    """When divergence_widen_bps is 0, no divergence adjustment should be applied."""

    def test_zero_widen_bps_no_crash(self) -> None:
        """divergence_widen_bps=0 should not crash with ZeroDivisionError."""
        cfg = MMConfig(
            spread_bps=200,
            order_size=100.0,
            max_position=500.0,
            divergence_widen_bps=0,
            divergence_max_bps=1500,
            divergence_size_reduction=0.5,
        )
        eng = QuotingEngine(cfg)
        # Should not raise ZeroDivisionError
        quote = eng.generate_quotes("tok_a", fair_value=0.50, position=0.0, market_mid=0.525)
        assert quote is not None

    def test_zero_widen_bps_base_spread(self) -> None:
        """With divergence_widen_bps=0, spread and size should be base values (no widening)."""
        cfg = MMConfig(
            spread_bps=200,
            order_size=100.0,
            max_position=500.0,
            divergence_widen_bps=0,
            divergence_max_bps=1500,
            divergence_size_reduction=0.5,
        )
        eng = QuotingEngine(cfg)
        quote = eng.generate_quotes("tok_a", fair_value=0.50, position=0.0, market_mid=0.525)
        assert quote is not None
        # No divergence adjustment, so spread should be base 200 bps = 0.02
        spread = quote.ask_price - quote.bid_price
        assert spread == pytest.approx(0.02, abs=cfg.tick_size)
        # Size should be base order_size
        assert quote.bid_size == pytest.approx(100.0)
        assert quote.ask_size == pytest.approx(100.0)
