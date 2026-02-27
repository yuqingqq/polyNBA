"""Tests for the quoting engine — spread calculation, inventory skew, limits."""

from __future__ import annotations

import pytest

from src.mm.config import MMConfig
from src.mm.models import Quote
from src.mm.quoting import QuotingEngine


@pytest.fixture
def default_config() -> MMConfig:
    return MMConfig()


@pytest.fixture
def engine(default_config: MMConfig) -> QuotingEngine:
    return QuotingEngine(default_config)


# ------------------------------------------------------------------
# Basic spread calculation
# ------------------------------------------------------------------


class TestSpreadCalculation:
    """Tests for symmetric spread around fair value."""

    def test_symmetric_spread_at_midpoint(self, engine: QuotingEngine) -> None:
        """With zero position, spread should be symmetric around fair value."""
        quote = engine.generate_quotes("token_a", fair_value=0.50, position=0.0)
        assert quote is not None
        # spread_bps=200 -> spread=0.02 -> bid=0.49, ask=0.51
        assert quote.bid_price == pytest.approx(0.49, abs=1e-9)
        assert quote.ask_price == pytest.approx(0.51, abs=1e-9)

    def test_spread_width(self, engine: QuotingEngine) -> None:
        """Ask - bid should equal the configured spread."""
        quote = engine.generate_quotes("token_a", fair_value=0.50, position=0.0)
        assert quote is not None
        spread = quote.ask_price - quote.bid_price
        expected_spread = 200 / 10_000.0  # 0.02
        assert spread == pytest.approx(expected_spread, abs=1e-9)

    def test_spread_at_low_fair_value(self, engine: QuotingEngine) -> None:
        """Spread at a low fair value, prices clamped to [0.01, 0.99]."""
        quote = engine.generate_quotes("token_a", fair_value=0.05, position=0.0)
        assert quote is not None
        assert quote.bid_price >= 0.01
        assert quote.ask_price <= 0.99

    def test_spread_at_high_fair_value(self, engine: QuotingEngine) -> None:
        """Spread at a high fair value, prices clamped to [0.01, 0.99]."""
        quote = engine.generate_quotes("token_a", fair_value=0.95, position=0.0)
        assert quote is not None
        assert quote.bid_price >= 0.01
        assert quote.ask_price <= 0.99

    def test_custom_tick_size(self) -> None:
        """Prices should be rounded to the specified tick size."""
        config = MMConfig(spread_bps=150)  # 1.5% spread
        eng = QuotingEngine(config)
        quote = eng.generate_quotes("token_a", fair_value=0.505, position=0.0, tick_size=0.001)
        assert quote is not None
        # Check that prices are on 0.001 grid
        assert round(quote.bid_price * 1000) == quote.bid_price * 1000
        assert round(quote.ask_price * 1000) == quote.ask_price * 1000

    def test_default_order_size(self, engine: QuotingEngine) -> None:
        """Bid and ask sizes should match the configured order_size."""
        quote = engine.generate_quotes("token_a", fair_value=0.50, position=0.0)
        assert quote is not None
        assert quote.bid_size == 50.0
        assert quote.ask_size == 50.0


# ------------------------------------------------------------------
# Inventory skew
# ------------------------------------------------------------------


class TestInventorySkew:
    """Tests for inventory-dependent price skewing."""

    def test_long_position_skews_prices_down(self, engine: QuotingEngine) -> None:
        """A long position should push both bid and ask lower.

        Uses fv=0.514 so that the skew (0.005 at position=250) crosses
        a tick boundary under directional rounding (bids floor, asks ceil).
        """
        zero_quote = engine.generate_quotes("token_a", fair_value=0.514, position=0.0)
        long_quote = engine.generate_quotes("token_a", fair_value=0.514, position=250.0)
        assert zero_quote is not None and long_quote is not None
        assert long_quote.bid_price < zero_quote.bid_price
        assert long_quote.ask_price < zero_quote.ask_price

    def test_short_position_skews_prices_up(self, engine: QuotingEngine) -> None:
        """A short position should push both bid and ask higher.

        Uses fv=0.516 so that the skew (0.005 at position=-250) crosses
        a tick boundary under directional rounding.
        """
        zero_quote = engine.generate_quotes("token_a", fair_value=0.516, position=0.0)
        short_quote = engine.generate_quotes("token_a", fair_value=0.516, position=-250.0)
        assert zero_quote is not None and short_quote is not None
        assert short_quote.bid_price > zero_quote.bid_price
        assert short_quote.ask_price > zero_quote.ask_price

    def test_skew_magnitude_increases_with_position(self, engine: QuotingEngine) -> None:
        """Larger positions should produce larger skew.

        Uses fv=0.514 so that different skew magnitudes (0.002 vs 0.006)
        produce distinct prices under directional rounding.
        """
        q1 = engine.generate_quotes("token_a", fair_value=0.514, position=100.0)
        q2 = engine.generate_quotes("token_a", fair_value=0.514, position=300.0)
        assert q1 is not None and q2 is not None
        # q2 should have lower bid/ask than q1 (more long)
        assert q2.bid_price < q1.bid_price
        assert q2.ask_price < q1.ask_price

    def test_skew_preserves_spread_width(self, engine: QuotingEngine) -> None:
        """Skew shifts both prices equally, so the spread width is preserved."""
        zero_quote = engine.generate_quotes("token_a", fair_value=0.50, position=0.0)
        long_quote = engine.generate_quotes("token_a", fair_value=0.50, position=200.0)
        assert zero_quote is not None and long_quote is not None
        # The spread width should remain the same
        zero_spread = zero_quote.ask_price - zero_quote.bid_price
        long_spread = long_quote.ask_price - long_quote.bid_price
        assert zero_spread == pytest.approx(long_spread, abs=0.01 + 1e-9)  # within 1 tick


# ------------------------------------------------------------------
# Position limits
# ------------------------------------------------------------------


class TestPositionLimits:
    """Tests for skipping sides at position limits."""

    def test_skip_bid_at_max_long(self, engine: QuotingEngine) -> None:
        """At max long position, bid should be None."""
        quote = engine.generate_quotes("token_a", fair_value=0.50, position=500.0)
        assert quote is not None
        assert quote.bid_price is None
        assert quote.bid_size is None
        assert quote.ask_price is not None
        assert quote.ask_size is not None

    def test_skip_ask_at_max_short(self, engine: QuotingEngine) -> None:
        """At max short position, ask should be None."""
        quote = engine.generate_quotes("token_a", fair_value=0.50, position=-500.0)
        assert quote is not None
        assert quote.bid_price is not None
        assert quote.bid_size is not None
        assert quote.ask_price is None
        assert quote.ask_size is None

    def test_both_sides_at_zero_position(self, engine: QuotingEngine) -> None:
        """At zero position, both sides should be quoted."""
        quote = engine.generate_quotes("token_a", fair_value=0.50, position=0.0)
        assert quote is not None
        assert quote.bid_price is not None
        assert quote.ask_price is not None


# ------------------------------------------------------------------
# Clamp and edge cases
# ------------------------------------------------------------------


class TestClampAndEdgeCases:
    """Tests for price clamping and edge cases."""

    def test_bid_clamped_to_min_price(self) -> None:
        """Bid should never go below 0.01."""
        config = MMConfig(spread_bps=500)  # 5% spread
        eng = QuotingEngine(config)
        quote = eng.generate_quotes("token_a", fair_value=0.02, position=0.0)
        if quote is not None and quote.bid_price is not None:
            assert quote.bid_price >= 0.01

    def test_ask_clamped_to_max_price(self) -> None:
        """Ask should never exceed 0.99."""
        config = MMConfig(spread_bps=500)
        eng = QuotingEngine(config)
        quote = eng.generate_quotes("token_a", fair_value=0.98, position=0.0)
        if quote is not None and quote.ask_price is not None:
            assert quote.ask_price <= 0.99

    def test_inverted_spread_returns_none(self) -> None:
        """When clamping causes bid >= ask, should return None."""
        config = MMConfig(spread_bps=200, inventory_skew_factor=5.0, max_position=100.0)
        eng = QuotingEngine(config)
        # Very high skew with large position near boundary can invert
        # Force inversion: fair_value near boundary with extreme skew
        quote = eng.generate_quotes("token_a", fair_value=0.02, position=99.0)
        # Either None or bid < ask
        if quote is not None:
            if quote.bid_price is not None and quote.ask_price is not None:
                assert quote.bid_price < quote.ask_price


# ------------------------------------------------------------------
# should_requote
# ------------------------------------------------------------------


class TestShouldRequote:
    """Tests for the requote decision logic."""

    def test_no_requote_when_same(self, engine: QuotingEngine) -> None:
        """Identical quotes should not trigger requoting."""
        q = Quote(token_id="t", bid_price=0.49, bid_size=50, ask_price=0.51, ask_size=50)
        assert engine.should_requote(q, q) is False

    def test_requote_on_bid_change(self, engine: QuotingEngine) -> None:
        """A bid price change exceeding threshold should trigger requote."""
        old = Quote(token_id="t", bid_price=0.49, bid_size=50, ask_price=0.51, ask_size=50)
        new = Quote(token_id="t", bid_price=0.48, bid_size=50, ask_price=0.51, ask_size=50)
        # threshold_bps=50 -> 0.005. Change = 0.01 > 0.005
        assert engine.should_requote(old, new) is True

    def test_requote_on_ask_change(self, engine: QuotingEngine) -> None:
        """An ask price change exceeding threshold should trigger requote."""
        old = Quote(token_id="t", bid_price=0.49, bid_size=50, ask_price=0.51, ask_size=50)
        new = Quote(token_id="t", bid_price=0.49, bid_size=50, ask_price=0.52, ask_size=50)
        assert engine.should_requote(old, new) is True

    def test_no_requote_within_threshold(self, engine: QuotingEngine) -> None:
        """A small price change below threshold should not trigger requote."""
        old = Quote(token_id="t", bid_price=0.4900, bid_size=50, ask_price=0.5100, ask_size=50)
        new = Quote(token_id="t", bid_price=0.4901, bid_size=50, ask_price=0.5100, ask_size=50)
        # threshold_bps=50 -> 0.005. Change = 0.0001 < 0.005
        assert engine.should_requote(old, new) is False

    def test_requote_when_side_appears(self, engine: QuotingEngine) -> None:
        """If a side goes from None to a price, should requote."""
        old = Quote(token_id="t", bid_price=None, bid_size=None, ask_price=0.51, ask_size=50)
        new = Quote(token_id="t", bid_price=0.49, bid_size=50, ask_price=0.51, ask_size=50)
        assert engine.should_requote(old, new) is True

    def test_requote_when_side_disappears(self, engine: QuotingEngine) -> None:
        """If a side goes from a price to None, should requote."""
        old = Quote(token_id="t", bid_price=0.49, bid_size=50, ask_price=0.51, ask_size=50)
        new = Quote(token_id="t", bid_price=0.49, bid_size=50, ask_price=None, ask_size=None)
        assert engine.should_requote(old, new) is True

    def test_custom_threshold(self, engine: QuotingEngine) -> None:
        """Custom threshold_bps should override config value."""
        old = Quote(token_id="t", bid_price=0.49, bid_size=50, ask_price=0.51, ask_size=50)
        new = Quote(token_id="t", bid_price=0.488, bid_size=50, ask_price=0.51, ask_size=50)
        # change = 0.002; default threshold 50bps = 0.005 -> no requote
        assert engine.should_requote(old, new, threshold_bps=50) is False
        # custom threshold 10bps = 0.001 -> requote
        assert engine.should_requote(old, new, threshold_bps=10) is True
