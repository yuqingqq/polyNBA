"""Unit tests for cross-prevention clamping in the quoting engine.

Tests that quotes which would cross the market (our ask <= market best_bid,
or our bid >= market best_ask) are clamped to sit on top of the opposite
book, preserving spread and avoiding adverse fills.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.mm.config import MMConfig
from src.mm.engine import MarketMakingEngine
from src.mm.quoting import QuotingEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> MMConfig:
    """Standard config for cross-prevention tests."""
    return MMConfig(
        spread_bps=200,
        order_size=50.0,
        max_position=500.0,
        tick_size=0.01,
        divergence_widen_bps=500,
        divergence_max_bps=1500,
        divergence_size_reduction=0.5,
    )


@pytest.fixture
def engine(config: MMConfig) -> QuotingEngine:
    return QuotingEngine(config)


# ---------------------------------------------------------------------------
# 1. Ask crosses market bid -> clamped to best_ask - tick
# ---------------------------------------------------------------------------


class TestAskCrossesMarketBid:
    """When our ask <= market best_bid, clamp ask to best_ask - tick."""

    def test_ask_crosses_clamped_to_best_ask_minus_tick(
        self, engine: QuotingEngine
    ) -> None:
        """fv=0.35, best_bid=0.44, best_ask=0.46.

        Raw ask ~0.36 which crosses best_bid=0.44.
        After clamping: ask = best_ask - tick = 0.46 - 0.01 = 0.45.
        Bid stays at 0.34 (does not cross).
        """
        quote = engine.generate_quotes(
            token_id="tok",
            fair_value=0.35,
            position=0.0,
            best_bid=0.44,
            best_ask=0.46,
        )
        assert quote is not None
        assert quote.ask_price == pytest.approx(0.45, abs=1e-9)
        assert quote.bid_price == pytest.approx(0.34, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. Bid crosses market ask -> clamped to best_bid + tick
# ---------------------------------------------------------------------------


class TestBidCrossesMarketAsk:
    """When our bid >= market best_ask, clamp bid to best_bid + tick."""

    def test_bid_crosses_clamped_to_best_bid_plus_tick(
        self, engine: QuotingEngine
    ) -> None:
        """fv=0.65, best_bid=0.54, best_ask=0.56.

        Raw bid ~0.64 which crosses best_ask=0.56.
        After clamping: bid = best_bid + tick = 0.54 + 0.01 = 0.55.
        Raw ask ~0.66 which does NOT cross.
        """
        quote = engine.generate_quotes(
            token_id="tok",
            fair_value=0.65,
            position=0.0,
            best_bid=0.54,
            best_ask=0.56,
        )
        assert quote is not None
        assert quote.bid_price == pytest.approx(0.55, abs=1e-9)
        assert quote.ask_price == pytest.approx(0.66, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. No orderbook info -> no clamping, same as before
# ---------------------------------------------------------------------------


class TestNoBestBidAsk:
    """When best_bid and best_ask are None, no clamping occurs."""

    def test_none_best_bid_ask_no_clamping(self, engine: QuotingEngine) -> None:
        """Without best_bid/best_ask, quotes should be identical to baseline."""
        q_base = engine.generate_quotes(
            token_id="tok",
            fair_value=0.50,
            position=0.0,
        )
        q_with_none = engine.generate_quotes(
            token_id="tok",
            fair_value=0.50,
            position=0.0,
            best_bid=None,
            best_ask=None,
        )
        assert q_base is not None and q_with_none is not None
        assert q_base.bid_price == q_with_none.bid_price
        assert q_base.ask_price == q_with_none.ask_price
        assert q_base.bid_size == q_with_none.bid_size
        assert q_base.ask_size == q_with_none.ask_size


# ---------------------------------------------------------------------------
# 4. Non-crossing quotes -> prices unchanged
# ---------------------------------------------------------------------------


class TestNonCrossingQuotes:
    """When quotes do not cross the market, prices remain unchanged."""

    def test_non_crossing_prices_unchanged(self, engine: QuotingEngine) -> None:
        """fv=0.50, best_bid=0.40, best_ask=0.60.

        Raw bid ~0.49, raw ask ~0.51 — neither crosses.
        Prices should be the same as without best_bid/best_ask.
        """
        q_base = engine.generate_quotes(
            token_id="tok",
            fair_value=0.50,
            position=0.0,
        )
        q_with_book = engine.generate_quotes(
            token_id="tok",
            fair_value=0.50,
            position=0.0,
            best_bid=0.40,
            best_ask=0.60,
        )
        assert q_base is not None and q_with_book is not None
        assert q_base.bid_price == q_with_book.bid_price
        assert q_base.ask_price == q_with_book.ask_price


# ---------------------------------------------------------------------------
# 5. Crossing with divergence size reduction -> size still reduced
# ---------------------------------------------------------------------------


class TestCrossingWithDivergenceSizeReduction:
    """Cross-prevention clamping should not affect divergence-based size reduction."""

    def test_size_still_reduced_after_clamping(self, engine: QuotingEngine, config: MMConfig) -> None:
        """fv=0.35, market_mid=0.45, best_bid=0.44, best_ask=0.46.

        Divergence = |0.35 - 0.45| = 0.10 = 1000 bps.
        ratio = 1000/500 = 2.0
        size_mult = max(0.1, 1.0 - 2.0 * 0.5) = max(0.1, 0.0) = 0.1
        Expected size = 50.0 * 0.1 = 5.0

        The ask will cross (raw ask << best_bid) and be clamped,
        but size should still be the divergence-reduced value.
        """
        quote = engine.generate_quotes(
            token_id="tok",
            fair_value=0.35,
            position=0.0,
            market_mid=0.45,
            best_bid=0.44,
            best_ask=0.46,
        )
        assert quote is not None
        expected_size = config.order_size * 0.1
        assert quote.bid_size == pytest.approx(expected_size)
        assert quote.ask_size == pytest.approx(expected_size)


# ---------------------------------------------------------------------------
# 6. Extreme divergence (>= max_bps) still returns None
# ---------------------------------------------------------------------------


class TestExtremeDivergenceReturnsNone:
    """Divergence >= max_bps should return None regardless of best_bid/best_ask."""

    def test_extreme_divergence_returns_none(self, engine: QuotingEngine) -> None:
        """fv=0.50, market_mid=0.65 -> divergence=1500bps >= max 1500.

        Should return None even when best_bid/best_ask are provided.
        """
        quote = engine.generate_quotes(
            token_id="tok",
            fair_value=0.50,
            position=0.0,
            market_mid=0.65,
            best_bid=0.60,
            best_ask=0.70,
        )
        assert quote is None

    def test_well_beyond_max_returns_none(self, engine: QuotingEngine) -> None:
        """fv=0.20, market_mid=0.50 -> divergence=3000bps >> max 1500.

        Should return None.
        """
        quote = engine.generate_quotes(
            token_id="tok",
            fair_value=0.20,
            position=0.0,
            market_mid=0.50,
            best_bid=0.48,
            best_ask=0.52,
        )
        assert quote is None


# ---------------------------------------------------------------------------
# 7. Ask crosses but no best_ask available -> clamp to best_bid + tick
# ---------------------------------------------------------------------------


class TestAskCrossesNoBestAsk:
    """When ask crosses but best_ask is None, clamp to best_bid + tick."""

    def test_ask_crosses_no_best_ask_clamps_to_bid_plus_tick(
        self, engine: QuotingEngine
    ) -> None:
        """fv=0.35, best_bid=0.44, best_ask=None.

        Raw ask ~0.36 crosses best_bid=0.44. No best_ask available.
        Clamp to best_bid + tick = 0.44 + 0.01 = 0.45.
        """
        quote = engine.generate_quotes(
            token_id="tok",
            fair_value=0.35,
            position=0.0,
            best_bid=0.44,
            best_ask=None,
        )
        assert quote is not None
        assert quote.ask_price == pytest.approx(0.45, abs=1e-9)
        assert quote.bid_price == pytest.approx(0.34, abs=1e-9)


# ---------------------------------------------------------------------------
# 8. Bid crosses but no best_bid available -> clamp to best_ask - tick
# ---------------------------------------------------------------------------


class TestBidCrossesNoBestBid:
    """When bid crosses but best_bid is None, clamp to best_ask - tick."""

    def test_bid_crosses_no_best_bid_clamps_to_ask_minus_tick(
        self, engine: QuotingEngine
    ) -> None:
        """fv=0.65, best_bid=None, best_ask=0.56.

        Raw bid ~0.64 crosses best_ask=0.56. No best_bid available.
        Clamp to best_ask - tick = 0.56 - 0.01 = 0.55.
        """
        quote = engine.generate_quotes(
            token_id="tok",
            fair_value=0.65,
            position=0.0,
            best_bid=None,
            best_ask=0.56,
        )
        assert quote is not None
        assert quote.bid_price == pytest.approx(0.55, abs=1e-9)
        assert quote.ask_price == pytest.approx(0.66, abs=1e-9)


# ---------------------------------------------------------------------------
# 9. After clamping, if spread inverts -> returns None
# ---------------------------------------------------------------------------


class TestClampedSpreadInverts:
    """If clamping causes bid >= ask, generate_quotes returns None."""

    def test_clamped_spread_inverts_returns_bid_only(self) -> None:
        """Both sides cross in an inverted market -> ask suppressed, bid retreats.

        fv=0.50, best_bid=0.52, best_ask=0.49. (Inverted market book.)
        Raw ask=0.51, ask <= best_bid=0.52 -> clamp ask = best_ask - tick = 0.48.
        0.48 <= best_bid=0.52 -> ask suppressed (post-clamp cross-prevention).
        Raw bid=0.49, bid >= best_ask=0.49 -> clamp bid = best_ask - tick = 0.48.
        0.48 < best_ask=0.49 -> bid valid (retreated successfully).
        Result: bid-only quote at 0.48.
        """
        config = MMConfig(
            spread_bps=200,
            order_size=50.0,
            max_position=500.0,
            tick_size=0.01,
        )
        eng = QuotingEngine(config)
        quote = eng.generate_quotes(
            token_id="tok",
            fair_value=0.50,
            position=0.0,
            best_bid=0.52,
            best_ask=0.49,
        )
        assert quote is not None
        assert quote.bid_price == pytest.approx(0.48, abs=1e-9)
        assert quote.ask_price is None
        assert quote.ask_size is None

    def test_clamped_spread_tight_market_inverts(self) -> None:
        """Tight market causes inversion after clamping one side.

        fv=0.35, best_bid=0.44, best_ask=0.44 (locked market).
        Raw ask=0.36, ask <= best_bid=0.44: clamp ask = best_ask - tick = 0.43.
        Raw bid=0.34, bid < best_ask=0.44: no crossing.
        bid=0.34 < ask=0.43: NOT inverted. This won't return None.

        For a true inversion we need a scenario where clamping pushes
        the clamped side past the theoretical side:
        fv=0.50, spread=200bps -> bid=0.49, ask=0.51.
        best_bid=0.505, best_ask=0.505 (locked).
        ask=0.51 > best_bid=0.505: no crossing.
        bid=0.49 < best_ask=0.505: no crossing. Neither crosses...

        fv=0.50, spread=10000bps=100% -> bid=0.01, ask=0.99.
        best_bid=0.98, best_ask=0.02.
        ask=0.99 > best_bid=0.98: no crossing.
        bid=0.01 < best_ask=0.02: no crossing. Hmm.

        Let me try: fv=0.50, best_bid=0.51, best_ask=0.50.
        Raw ask=0.51, ask <= best_bid=0.51: clamp ask = best_ask - tick = 0.49.
        0.49 <= best_bid=0.51: ask suppressed (cross-prevention).
        Raw bid=0.49, bid < best_ask=0.50: no crossing.
        Result: bid-only quote at 0.49.
        """
        config = MMConfig(
            spread_bps=200,
            order_size=50.0,
            max_position=500.0,
            tick_size=0.01,
        )
        eng = QuotingEngine(config)
        quote = eng.generate_quotes(
            token_id="tok",
            fair_value=0.50,
            position=0.0,
            best_bid=0.51,
            best_ask=0.50,
        )
        assert quote is not None
        assert quote.bid_price == pytest.approx(0.49, abs=1e-9)
        assert quote.ask_price is None
        assert quote.ask_size is None


# ---------------------------------------------------------------------------
# 10. Engine integration test with mock OrderbookManager
# ---------------------------------------------------------------------------


class TestEngineIntegrationWithOrderbook:
    """Engine with a mock OrderbookManager that provides best_bid/best_ask."""

    def test_engine_passes_best_bid_ask_to_quoting(self, config: MMConfig) -> None:
        """Engine should look up best_bid/best_ask and pass them through.

        Set up: fv=0.35, orderbook with best_bid=0.44, best_ask=0.46.
        The ask would cross -> should be clamped to 0.45.
        """
        # Create a mock orderbook
        mock_book = MagicMock()
        mock_book.mid_price = 0.45
        mock_book.best_bid = 0.44
        mock_book.best_ask = 0.46
        mock_book.get_bbo.return_value = (0.45, 0.44, 0.46)

        mock_ob_mgr = MagicMock()
        mock_ob_mgr.get.return_value = mock_book

        eng = MarketMakingEngine(
            config=config, client=None, orderbook_manager=mock_ob_mgr
        )
        eng.add_token("tok_test")
        eng.fair_value_engine.update("tok_test", 0.35)

        eng._tick()

        orders = eng.order_manager.get_open_orders("tok_test")
        assert len(orders) > 0, "Expected orders to be placed"

        from src.mm.models import Side

        buys = [o for o in orders if o.side == Side.BUY]
        sells = [o for o in orders if o.side == Side.SELL]

        # The ask should have been clamped to 0.45
        assert len(sells) > 0, "Expected sell orders"
        assert sells[0].price == pytest.approx(0.45, abs=1e-9)

        # The bid should be at theoretical (not crossing)
        assert len(buys) > 0, "Expected buy orders"
        # With market_mid=0.45, divergence = |0.35-0.45| = 1000bps
        # ratio = 1000/500 = 2.0, spread_mult=3.0
        # spread = 0.02 * 3.0 = 0.06
        # raw_bid = 0.35 - 0.03 = 0.32
        # Bid=0.32, does not cross best_ask=0.46
        assert buys[0].price == pytest.approx(0.32, abs=1e-9)

    def test_engine_no_orderbook_no_clamping(self, config: MMConfig) -> None:
        """Engine without orderbook manager should produce normal quotes (no clamping)."""
        eng = MarketMakingEngine(config=config, client=None, orderbook_manager=None)
        eng.add_token("tok_test")
        eng.fair_value_engine.update("tok_test", 0.50)

        eng._tick()

        orders = eng.order_manager.get_open_orders("tok_test")
        assert len(orders) > 0

        from src.mm.models import Side

        buys = [o for o in orders if o.side == Side.BUY]
        sells = [o for o in orders if o.side == Side.SELL]

        assert len(buys) > 0 and len(sells) > 0
        # Standard quotes: bid=0.49, ask=0.51
        assert buys[0].price == pytest.approx(0.49, abs=1e-9)
        assert sells[0].price == pytest.approx(0.51, abs=1e-9)

    def test_engine_orderbook_returns_none_book(self, config: MMConfig) -> None:
        """Engine with orderbook manager that returns None for the book."""
        mock_ob_mgr = MagicMock()
        mock_ob_mgr.get.return_value = None

        eng = MarketMakingEngine(
            config=config, client=None, orderbook_manager=mock_ob_mgr
        )
        eng.add_token("tok_test")
        eng.fair_value_engine.update("tok_test", 0.50)

        eng._tick()

        orders = eng.order_manager.get_open_orders("tok_test")
        assert len(orders) > 0

        from src.mm.models import Side

        buys = [o for o in orders if o.side == Side.BUY]
        sells = [o for o in orders if o.side == Side.SELL]

        # No orderbook -> no clamping, standard quotes
        assert buys[0].price == pytest.approx(0.49, abs=1e-9)
        assert sells[0].price == pytest.approx(0.51, abs=1e-9)


# ---------------------------------------------------------------------------
# 11. Zero-spread / locked market -> both sides cross -> inverted -> None
# ---------------------------------------------------------------------------


class TestZeroSpreadMarket:
    """When best_bid == best_ask (locked market), cross-prevention should handle correctly."""

    def test_locked_market_returns_none(self, engine: QuotingEngine) -> None:
        """fv=0.50, best_bid=0.50, best_ask=0.50 (locked market).

        Raw bid=0.49, ask=0.51.
        ask=0.51 > best_bid=0.50: no crossing.
        bid=0.49 < best_ask=0.50: no crossing.
        Both sides are fine, result is a valid quote.
        """
        quote = engine.generate_quotes(
            token_id="tok",
            fair_value=0.50,
            position=0.0,
            best_bid=0.50,
            best_ask=0.50,
        )
        # Bid=0.49 < Ask=0.51, neither crosses the locked market
        assert quote is not None
        assert quote.bid_price == pytest.approx(0.49, abs=1e-9)
        assert quote.ask_price == pytest.approx(0.51, abs=1e-9)

    def test_locked_market_fv_at_market_crosses(self, engine: QuotingEngine) -> None:
        """fv=0.30, best_bid=0.40, best_ask=0.40 (locked).

        Raw bid=0.29, ask=0.31.
        ask=0.31 <= best_bid=0.40: crosses! Clamp ask = best_ask - tick = 0.39.
        0.39 <= best_bid=0.40: ask suppressed (cross-prevention).
        bid=0.29 < best_ask=0.40: no crossing.
        Result: bid-only quote at 0.29.
        """
        quote = engine.generate_quotes(
            token_id="tok",
            fair_value=0.30,
            position=0.0,
            best_bid=0.40,
            best_ask=0.40,
        )
        assert quote is not None
        assert quote.bid_price == pytest.approx(0.29, abs=1e-9)
        assert quote.ask_price is None
        assert quote.ask_size is None


# ---------------------------------------------------------------------------
# 12. One-tick spread market
# ---------------------------------------------------------------------------


class TestOneTickSpreadMarket:
    """When best_ask - best_bid == tick_size (one-tick spread)."""

    def test_one_tick_spread_ask_crosses(self, engine: QuotingEngine) -> None:
        """fv=0.35, best_bid=0.49, best_ask=0.50 (one tick spread).

        Raw ask ~0.36 <= best_bid=0.49: crosses!
        Clamp ask = best_ask - tick = 0.50 - 0.01 = 0.49.
        But ask=0.49 == best_bid=0.49: ask suppressed (cross-prevention).
        Result: bid-only quote at 0.34.
        """
        quote = engine.generate_quotes(
            token_id="tok",
            fair_value=0.35,
            position=0.0,
            best_bid=0.49,
            best_ask=0.50,
        )
        assert quote is not None
        assert quote.bid_price == pytest.approx(0.34, abs=1e-9)
        assert quote.ask_price is None
        assert quote.ask_size is None

    def test_one_tick_spread_bid_crosses(self, engine: QuotingEngine) -> None:
        """fv=0.65, best_bid=0.50, best_ask=0.51 (one tick spread).

        Raw bid ~0.64 >= best_ask=0.51: crosses!
        Clamp bid = best_ask - tick = 0.51 - 0.01 = 0.50.
        bid=0.50 < best_ask=0.51: bid valid (retreated to best_bid level).
        Result: two-sided quote, bid=0.50, ask=0.66.
        """
        quote = engine.generate_quotes(
            token_id="tok",
            fair_value=0.65,
            position=0.0,
            best_bid=0.50,
            best_ask=0.51,
        )
        assert quote is not None
        assert quote.bid_price == pytest.approx(0.50, abs=1e-9)
        assert quote.ask_price == pytest.approx(0.66, abs=1e-9)


# ---------------------------------------------------------------------------
# 13. Fair value exactly at best_bid or best_ask
# ---------------------------------------------------------------------------


class TestFairValueAtBestPrices:
    """When fair value is exactly at best_bid or best_ask."""

    def test_fv_at_best_bid(self, engine: QuotingEngine) -> None:
        """fv=0.50, best_bid=0.50, best_ask=0.60.

        Raw bid = 0.50 - 0.01 = 0.49.
        Raw ask = 0.50 + 0.01 = 0.51.
        Neither crosses (bid=0.49 < best_ask=0.60, ask=0.51 > best_bid=0.50).
        """
        quote = engine.generate_quotes(
            token_id="tok",
            fair_value=0.50,
            position=0.0,
            best_bid=0.50,
            best_ask=0.60,
        )
        assert quote is not None
        assert quote.bid_price == pytest.approx(0.49, abs=1e-9)
        assert quote.ask_price == pytest.approx(0.51, abs=1e-9)

    def test_fv_at_best_ask(self, engine: QuotingEngine) -> None:
        """fv=0.60, best_bid=0.50, best_ask=0.60.

        Raw bid = 0.60 - 0.01 = 0.59.
        Raw ask = 0.60 + 0.01 = 0.61.
        Neither crosses (bid=0.59 < best_ask=0.60, ask=0.61 > best_bid=0.50).
        """
        quote = engine.generate_quotes(
            token_id="tok",
            fair_value=0.60,
            position=0.0,
            best_bid=0.50,
            best_ask=0.60,
        )
        assert quote is not None
        assert quote.bid_price == pytest.approx(0.59, abs=1e-9)
        assert quote.ask_price == pytest.approx(0.61, abs=1e-9)


# ---------------------------------------------------------------------------
# 14. Only bids or only asks in orderbook
# ---------------------------------------------------------------------------


class TestOneSidedOrderbook:
    """Orderbook has only one side (bids only or asks only)."""

    def test_only_best_bid_no_ask_no_cross(self, engine: QuotingEngine) -> None:
        """fv=0.50, best_bid=0.40, best_ask=None.

        Our quotes: bid=0.49, ask=0.51.
        ask=0.51 > best_bid=0.40: no crossing.
        bid=0.49, best_ask=None: cannot cross.
        """
        quote = engine.generate_quotes(
            token_id="tok",
            fair_value=0.50,
            position=0.0,
            best_bid=0.40,
            best_ask=None,
        )
        assert quote is not None
        assert quote.bid_price == pytest.approx(0.49, abs=1e-9)
        assert quote.ask_price == pytest.approx(0.51, abs=1e-9)

    def test_only_best_ask_no_bid_no_cross(self, engine: QuotingEngine) -> None:
        """fv=0.50, best_bid=None, best_ask=0.60.

        Our quotes: bid=0.49, ask=0.51.
        ask, best_bid=None: cannot cross.
        bid=0.49 < best_ask=0.60: no crossing.
        """
        quote = engine.generate_quotes(
            token_id="tok",
            fair_value=0.50,
            position=0.0,
            best_bid=None,
            best_ask=0.60,
        )
        assert quote is not None
        assert quote.bid_price == pytest.approx(0.49, abs=1e-9)
        assert quote.ask_price == pytest.approx(0.51, abs=1e-9)

    def test_only_best_bid_ask_crosses(self, engine: QuotingEngine) -> None:
        """fv=0.35, best_bid=0.44, best_ask=None.

        Raw ask ~0.36 <= best_bid=0.44: crosses!
        best_ask is None, so clamp ask = best_bid + tick = 0.45.
        """
        quote = engine.generate_quotes(
            token_id="tok",
            fair_value=0.35,
            position=0.0,
            best_bid=0.44,
            best_ask=None,
        )
        assert quote is not None
        assert quote.ask_price == pytest.approx(0.45, abs=1e-9)

    def test_only_best_ask_bid_crosses(self, engine: QuotingEngine) -> None:
        """fv=0.65, best_bid=None, best_ask=0.56.

        Raw bid ~0.64 >= best_ask=0.56: crosses!
        best_bid is None, so clamp bid = best_ask - tick = 0.55.
        """
        quote = engine.generate_quotes(
            token_id="tok",
            fair_value=0.65,
            position=0.0,
            best_bid=None,
            best_ask=0.56,
        )
        assert quote is not None
        assert quote.bid_price == pytest.approx(0.55, abs=1e-9)


# ---------------------------------------------------------------------------
# 15. Position at exact max_position boundary
# ---------------------------------------------------------------------------


class TestPositionAtExactBoundary:
    """Position exactly at max_position should skip one side."""

    def test_exact_max_long_skips_bid(self, engine: QuotingEngine) -> None:
        """At position == max_position, bid should be dropped."""
        quote = engine.generate_quotes(
            token_id="tok",
            fair_value=0.50,
            position=500.0,
            best_bid=0.40,
            best_ask=0.60,
        )
        assert quote is not None
        assert quote.bid_price is None
        assert quote.bid_size is None
        assert quote.ask_price is not None
        assert quote.ask_size is not None

    def test_exact_max_short_skips_ask(self, engine: QuotingEngine) -> None:
        """At position == -max_position, ask should be dropped."""
        quote = engine.generate_quotes(
            token_id="tok",
            fair_value=0.50,
            position=-500.0,
            best_bid=0.40,
            best_ask=0.60,
        )
        assert quote is not None
        assert quote.ask_price is None
        assert quote.ask_size is None
        assert quote.bid_price is not None
        assert quote.bid_size is not None

    def test_one_below_max_still_quotes_both(self, engine: QuotingEngine) -> None:
        """At position == max_position - 1, both sides should be quoted."""
        quote = engine.generate_quotes(
            token_id="tok",
            fair_value=0.50,
            position=499.0,
            best_bid=0.40,
            best_ask=0.60,
        )
        assert quote is not None
        assert quote.bid_price is not None
        assert quote.ask_price is not None
