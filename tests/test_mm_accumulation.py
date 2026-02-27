"""Tests for aggressive initial inventory accumulation (t20).

Covers the new accumulation config fields, the generate_accumulation_quote
method on QuotingEngine, and the _is_accumulating detection on MarketMakingEngine.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.mm.config import MMConfig
from src.mm.engine import MarketMakingEngine
from src.mm.models import Fill, Side
from src.mm.quoting import QuotingEngine


# ------------------------------------------------------------------
# Config defaults
# ------------------------------------------------------------------


class TestAccumulationConfigDefaults:
    """Verify new MMConfig fields have correct defaults."""

    def test_accumulation_config_defaults(self) -> None:
        config = MMConfig()
        assert config.accumulation_enabled is True
        assert config.accumulation_spread_bps == 50
        assert config.accumulation_cross_spread is False
        assert config.target_initial_position == 30.0
        assert config.accumulation_size_multiplier == 1.5
        assert config.accumulation_max_price_cents == 95


# ------------------------------------------------------------------
# generate_accumulation_quote
# ------------------------------------------------------------------


class TestGenerateAccumulationQuote:
    """Tests for QuotingEngine.generate_accumulation_quote."""

    def test_generate_accumulation_quote_basic(self) -> None:
        """Bid-only at fair_value - accum_spread/2, correct size with multiplier."""
        config = MMConfig(
            order_size=50.0,
            accumulation_spread_bps=50,
            accumulation_size_multiplier=1.5,
            accumulation_max_price_cents=95,
        )
        engine = QuotingEngine(config)
        quote = engine.generate_accumulation_quote(
            token_id="token_a",
            fair_value=0.50,
            tick_size=0.01,
        )
        assert quote is not None
        # spread = 50/10000 = 0.005, half = 0.0025
        # raw_bid = 0.50 - 0.0025 = 0.4975
        # floor(0.4975 / 0.01) * 0.01 = floor(49.75) * 0.01 = 49 * 0.01 = 0.49
        assert quote.bid_price == pytest.approx(0.49, abs=1e-9)
        assert quote.bid_size == pytest.approx(75.0, abs=1e-9)  # 50 * 1.5
        # Ask side should be None (bid-only)
        assert quote.ask_price is None
        assert quote.ask_size is None

    def test_generate_accumulation_quote_cross_spread(self) -> None:
        """When cross_spread=True and best_ask given, bid at best_ask."""
        config = MMConfig(
            order_size=50.0,
            accumulation_spread_bps=50,
            accumulation_cross_spread=True,
            accumulation_size_multiplier=1.5,
            accumulation_max_price_cents=95,
        )
        engine = QuotingEngine(config)
        quote = engine.generate_accumulation_quote(
            token_id="token_a",
            fair_value=0.50,
            tick_size=0.01,
            best_ask=0.52,
        )
        assert quote is not None
        # With cross_spread=True, raw_bid = best_ask = 0.52
        # floor(0.52 / 0.01) * 0.01 = 0.52
        assert quote.bid_price == pytest.approx(0.52, abs=1e-9)
        assert quote.bid_size == pytest.approx(75.0, abs=1e-9)

    def test_generate_accumulation_quote_price_cap(self) -> None:
        """When fair_value is very high (0.98), bid capped at max_price_cents/100."""
        config = MMConfig(
            order_size=50.0,
            accumulation_spread_bps=50,
            accumulation_size_multiplier=1.5,
            accumulation_max_price_cents=95,
        )
        engine = QuotingEngine(config)
        quote = engine.generate_accumulation_quote(
            token_id="token_a",
            fair_value=0.98,
            tick_size=0.01,
        )
        # raw_bid = 0.98 - 0.0025 = 0.9775
        # floor(0.9775 / 0.01) * 0.01 = 0.97
        # 0.97 > 0.95 (price cap) => return None
        assert quote is None

    def test_generate_accumulation_quote_no_skew(self) -> None:
        """Different positions don't affect accumulation bid price (no skew)."""
        config = MMConfig(
            order_size=50.0,
            accumulation_spread_bps=50,
            accumulation_size_multiplier=1.5,
            accumulation_max_price_cents=95,
            inventory_skew_factor=0.5,
            max_position=500.0,
        )
        engine = QuotingEngine(config)

        # generate_accumulation_quote does not take a position argument,
        # so position cannot affect the price. Verify identical quotes
        # regardless of what position the inventory has.
        quote1 = engine.generate_accumulation_quote(
            token_id="token_a",
            fair_value=0.50,
            tick_size=0.01,
        )
        quote2 = engine.generate_accumulation_quote(
            token_id="token_a",
            fair_value=0.50,
            tick_size=0.01,
        )
        assert quote1 is not None and quote2 is not None
        assert quote1.bid_price == quote2.bid_price
        assert quote1.bid_size == quote2.bid_size

    def test_generate_accumulation_quote_zero_bid_returns_none(self) -> None:
        """If bid price rounds down to 0, return None."""
        config = MMConfig(
            order_size=50.0,
            accumulation_spread_bps=200,
            accumulation_size_multiplier=1.5,
            accumulation_max_price_cents=95,
        )
        engine = QuotingEngine(config)
        # fair_value very low so bid goes to 0 or below
        quote = engine.generate_accumulation_quote(
            token_id="token_a",
            fair_value=0.005,
            tick_size=0.01,
        )
        assert quote is None

    def test_generate_accumulation_quote_cross_spread_no_best_ask(self) -> None:
        """When cross_spread=True but best_ask is None, fall back to spread-based bid."""
        config = MMConfig(
            order_size=50.0,
            accumulation_spread_bps=50,
            accumulation_cross_spread=True,
            accumulation_size_multiplier=1.5,
            accumulation_max_price_cents=95,
        )
        engine = QuotingEngine(config)
        quote = engine.generate_accumulation_quote(
            token_id="token_a",
            fair_value=0.50,
            tick_size=0.01,
            best_ask=None,  # no orderbook data
        )
        assert quote is not None
        # Should fall back to spread-based bid: 0.50 - 0.0025 = 0.4975 -> 0.49
        assert quote.bid_price == pytest.approx(0.49, abs=1e-9)

    def test_generate_accumulation_quote_tick_aligned_floats(self) -> None:
        """Bid price exactly on a tick boundary should not lose a tick to float error."""
        config = MMConfig(
            order_size=50.0,
            accumulation_spread_bps=0,  # zero spread so raw_bid = fair_value
            accumulation_size_multiplier=1.0,
            accumulation_max_price_cents=95,
        )
        engine = QuotingEngine(config)
        # 0.29 is a known floating-point trouble spot: 0.29/0.01 = 28.9999...
        quote = engine.generate_accumulation_quote(
            token_id="token_a",
            fair_value=0.29,
            tick_size=0.01,
        )
        assert quote is not None
        assert quote.bid_price == pytest.approx(0.29, abs=1e-9)


# ------------------------------------------------------------------
# _is_accumulating
# ------------------------------------------------------------------


class TestIsAccumulating:
    """Tests for MarketMakingEngine._is_accumulating."""

    def _make_engine(self, **config_kwargs) -> MarketMakingEngine:
        """Helper to create a MarketMakingEngine with given config overrides."""
        config = MMConfig(**config_kwargs)
        # When dry_run=False, OrderManager requires a client object
        client = None if config.dry_run else MagicMock()
        engine = MarketMakingEngine(config=config, client=client)
        engine.add_token("token_a")
        engine.fair_value_engine.update("token_a", 0.50)
        return engine

    def test_is_accumulating_below_target(self) -> None:
        """Returns True when position < target."""
        engine = self._make_engine(
            dry_run=False,
            accumulation_enabled=True,
            target_initial_position=30.0,
        )
        # Position is 0 (below target of 30)
        assert engine._is_accumulating("token_a") is True

    def test_is_accumulating_above_target(self) -> None:
        """Returns False when position > target."""
        engine = self._make_engine(
            dry_run=False,
            accumulation_enabled=True,
            target_initial_position=30.0,
        )
        # Simulate fill to get position above target
        fill = Fill(
            order_id="fill-1",
            token_id="token_a",
            side=Side.BUY,
            price=0.50,
            size=35.0,
        )
        engine.inventory_manager.update_fill(fill)
        assert engine._is_accumulating("token_a") is False

    def test_is_accumulating_exactly_at_target(self) -> None:
        """Returns False when position == target (uses strict <, not <=)."""
        engine = self._make_engine(
            dry_run=False,
            accumulation_enabled=True,
            target_initial_position=30.0,
        )
        # Fill exactly to target
        fill = Fill(
            order_id="fill-1",
            token_id="token_a",
            side=Side.BUY,
            price=0.50,
            size=30.0,
        )
        engine.inventory_manager.update_fill(fill)
        assert engine._is_accumulating("token_a") is False

    def test_is_accumulating_dry_run(self) -> None:
        """Returns False in dry_run mode."""
        engine = self._make_engine(
            dry_run=True,
            accumulation_enabled=True,
            target_initial_position=30.0,
        )
        # Position is 0 (below target) but dry_run is True
        assert engine._is_accumulating("token_a") is False

    def test_is_accumulating_disabled(self) -> None:
        """Returns False when accumulation_enabled=False."""
        engine = self._make_engine(
            dry_run=False,
            accumulation_enabled=False,
            target_initial_position=30.0,
        )
        # Position is 0 (below target) but accumulation is disabled
        assert engine._is_accumulating("token_a") is False
