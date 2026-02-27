"""Quoting engine — generates bid/ask quotes with inventory skew."""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from typing import Optional

from .config import MMConfig
from .models import Quote

logger = logging.getLogger(__name__)

# C-3: Default price bounds for Polymarket binary contracts.
# These are used as defaults when tick_size is 0.01. For other tick sizes,
# min_price=tick_size and max_price=1.0-tick_size are computed dynamically.
DEFAULT_MIN_PRICE = 0.01
DEFAULT_MAX_PRICE = 0.99


def _get_price_bounds(tick_size: float) -> tuple[float, float]:
    """Compute min/max price bounds from tick_size.

    For tick_size=0.01: (0.01, 0.99)
    For tick_size=0.1:  (0.1, 0.9)
    For tick_size=0.001: (0.001, 0.999)
    """
    min_price = tick_size
    max_price = round(1.0 - tick_size, 10)
    return min_price, max_price


def _tick_decimals(tick_size: float) -> int:
    """Return the number of decimal places for a tick_size.

    C-2: Used to round prices to exact decimal representation after
    multiply-back, avoiding IEEE 754 artifacts like 0.30000000000000004.
    """
    if tick_size >= 1.0:
        return 0
    # Count decimal places: 0.01 -> 2, 0.001 -> 3, 0.1 -> 1, 0.0001 -> 4
    return max(0, -int(math.floor(math.log10(tick_size) + 1e-9)))


def _round_to_tick(price: float, tick_size: float, direction: str = "nearest") -> float:
    """Round a price to the tick grid.

    Args:
        price: The raw price.
        tick_size: Minimum price increment.
        direction: ``"down"`` floors to the tick (for bids),
            ``"up"`` ceils to the tick (for asks),
            ``"nearest"`` rounds to the closest tick.
    """
    decimals = _tick_decimals(tick_size)
    if direction == "down":
        return round(math.floor(price / tick_size + 1e-9) * tick_size, decimals)
    elif direction == "up":
        return round(math.ceil(price / tick_size - 1e-9) * tick_size, decimals)
    return round(round(price / tick_size) * tick_size, decimals)


def _clamp(price: float, lo: float = DEFAULT_MIN_PRICE, hi: float = DEFAULT_MAX_PRICE) -> float:
    """Clamp a price to [lo, hi]."""
    return max(lo, min(hi, price))


class QuotingEngine:
    """Generates two-sided quotes based on fair value and inventory.

    The spread is symmetric around fair value, then shifted by an
    inventory skew that encourages mean-reversion of the position.
    """

    def __init__(self, config: MMConfig) -> None:
        self._config = config

    @property
    def config(self) -> MMConfig:
        return self._config

    def generate_accumulation_quote(
        self,
        token_id: str,
        fair_value: float,
        tick_size: float,
        best_bid: float | None = None,
        best_ask: float | None = None,
    ) -> Optional[Quote]:
        """Generate aggressive bid-only quote for inventory accumulation.

        Used when the engine starts with zero inventory and needs to build
        a position before it can sell tokens (Polymarket constraint). Uses
        a tighter spread and larger size to fill faster.

        Args:
            token_id: Token identifier.
            fair_value: Current fair probability.
            tick_size: Minimum price increment.
            best_bid: Market's current best bid price. Used to place the
                accumulation bid at the top of the book.
            best_ask: Market's current best ask price. When cross_spread is
                enabled and this is provided, the bid is placed at best_ask
                to cross the spread for immediate fill.

        Returns:
            A bid-only Quote, or None if the bid price is invalid.
        """
        cfg = self._config
        price_cap = cfg.accumulation_max_price_cents / 100.0

        # Determine bid price
        if cfg.accumulation_cross_spread and best_ask is not None:
            # C-6: Cap cross-spread bid at fair_value + slippage to avoid paying
            # wildly above fair value when best_ask is far from fair_value.
            # I-1: Use max(fair_value * 0.10, 0.02) so low-probability tokens
            # (e.g. fair_value=0.05) still have at least 2 cents of slippage tolerance.
            max_accumulation_slippage = max(fair_value * 0.10, 0.02)
            raw_bid = min(best_ask, fair_value + max_accumulation_slippage)
        elif best_bid is not None and best_ask is not None:
            # Place at top of book: match best_bid or improve by one tick
            # (whichever is closer to fair_value), but never cross the ask.
            raw_bid = best_bid
            # Cap at best_ask - tick_size to avoid crossing the book with post_only orders.
            raw_bid = min(raw_bid, best_ask - tick_size)
            # Safety: never bid above fair_value + accumulation_spread/2
            spread = cfg.accumulation_spread_bps / 10_000.0
            raw_bid = min(raw_bid, fair_value + spread / 2.0)
        else:
            spread = cfg.accumulation_spread_bps / 10_000.0
            raw_bid = fair_value - spread / 2.0
            # Cap at best_ask - tick_size to avoid crossing the book with post_only orders.
            if best_ask is not None:
                raw_bid = min(raw_bid, best_ask - tick_size)

        # Round bid price DOWN to tick grid.
        # Add a small epsilon before flooring to avoid floating-point errors
        # where exact tick values (e.g. 0.29/0.01 = 28.9999...) lose a tick.
        # C-2: Round to correct decimal places to avoid IEEE 754 artifacts.
        decimals = _tick_decimals(tick_size)
        bid_price = round(math.floor(raw_bid / tick_size + 1e-9) * tick_size, decimals)

        # I-BUG-1: Dynamic price bounds from tick_size. For tick_size=0.1,
        # max_price=0.9 — bids above this are invalid on the exchange.
        _acc_min_price, _acc_max_price = _get_price_bounds(tick_size)

        # Validate bid price
        if bid_price < _acc_min_price:
            return None
        if bid_price > _acc_max_price:
            return None
        if bid_price > price_cap:
            return None

        bid_size = cfg.order_size * cfg.accumulation_size_multiplier

        quote = Quote(
            token_id=token_id,
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=None,
            ask_size=None,
            timestamp=datetime.utcfromtimestamp(time.time()),
        )

        logger.debug(
            "Accumulation quote: token=%s bid=%.4f size=%.1f fv=%.4f",
            token_id,
            bid_price,
            bid_size,
            fair_value,
        )

        return quote

    def generate_quotes(
        self,
        token_id: str,
        fair_value: float,
        position: float,
        tick_size: float | None = None,
        market_mid: float | None = None,
        best_bid: float | None = None,
        best_ask: float | None = None,
        divergence_ema_bps: float | None = None,
        allow_cross: bool = False,
    ) -> Optional[Quote]:
        """Generate a two-sided quote for a token.

        Args:
            token_id: Token identifier.
            fair_value: Current fair probability.
            position: Current net position (positive = long).
            tick_size: Override tick size (defaults to config value).
            market_mid: Current market mid price for divergence calculation.
                If None, no divergence adjustment is applied.
            best_bid: Market's current best bid price from the orderbook.
                Used for cross-prevention clamping. If None, no clamping.
            best_ask: Market's current best ask price from the orderbook.
                Used for cross-prevention clamping. If None, no clamping.
            divergence_ema_bps: EMA-smoothed divergence in basis points.
                When provided, the hard stop (max_bps) uses this EMA value
                instead of the instantaneous divergence — making quoting
                resistant to single-tick spikes.  Spread widening always
                uses the instantaneous value (conservative).

        Returns:
            A Quote, or None if the resulting spread is inverted or
            divergence exceeds the maximum threshold.
        """
        cfg = self._config
        ts = tick_size if tick_size is not None else cfg.tick_size

        # Per-token overrides for divergence thresholds
        overrides = cfg.divergence_overrides.get(token_id, {})
        effective_max_bps = overrides.get("max_bps", cfg.divergence_max_bps)
        effective_widen_bps = overrides.get("widen_bps", cfg.divergence_widen_bps)

        # Divergence-aware spread/size multipliers
        spread_mult = 1.0
        size_mult = 1.0

        if market_mid is not None:
            divergence = abs(fair_value - market_mid)
            divergence_bps = divergence * 10_000.0

            # Hard stop: use EMA if available (spike-resistant), else instantaneous
            check_bps = divergence_ema_bps if divergence_ema_bps is not None else divergence_bps
            if check_bps >= effective_max_bps:
                logger.warning(
                    "Divergence too high for %s: check=%.0f bps >= max %d bps "
                    "(instant=%.0f ema=%s fv=%.4f mid=%.4f)",
                    token_id, check_bps, effective_max_bps,
                    divergence_bps,
                    f"{divergence_ema_bps:.0f}" if divergence_ema_bps is not None else "N/A",
                    fair_value, market_mid,
                )
                return None

            # Spread widening: always uses instantaneous (conservative)
            if effective_widen_bps <= 0:
                logger.warning(
                    "divergence_widen_bps is %d (<= 0) for %s, skipping divergence adjustment",
                    effective_widen_bps, token_id,
                )
                # Leave spread_mult and size_mult at 1.0 (no adjustment)
            else:
                ratio = divergence_bps / effective_widen_bps
                spread_mult = 1.0 + ratio
                size_mult = max(0.1, 1.0 - ratio * (1.0 - cfg.divergence_size_reduction))

        spread = (cfg.spread_bps / 10_000.0) * spread_mult

        # Inventory skew: positive position -> skew pushes prices down
        # (encouraging sells / discouraging buys)
        # Uses the widened spread intentionally — more aggressive reversion under uncertainty
        skew = cfg.inventory_skew_factor * (position / cfg.max_position) * spread

        # C-3: Dynamic price bounds based on tick_size
        min_price, max_price = _get_price_bounds(ts)

        # C-7: Cap skew so the position-reducing side always stays valid.
        # When long (position > 0), the ask (reducing side) = fair_value + spread/2 - skew.
        # If skew is too large, ask gets pushed above max_price. Cap skew so ask >= min_price.
        # When short (position < 0, defensive), bid = fair_value - spread/2 - skew.
        # Negative skew pushes bid up; cap so bid <= max_price.
        half_spread = spread / 2.0
        if position > 0:
            # Reducing side is ask: raw_ask = fair_value + half_spread - skew
            # Need raw_ask >= min_price, so skew <= fair_value + half_spread - min_price
            max_skew = fair_value + half_spread - min_price
            if skew > max_skew:
                skew = max_skew
        elif position < 0:
            # Reducing side is bid: raw_bid = fair_value - half_spread - skew
            # Need raw_bid <= max_price, so skew >= fair_value - half_spread - max_price
            min_skew = fair_value - half_spread - max_price
            if skew < min_skew:
                skew = min_skew

        raw_bid = fair_value - spread / 2.0 - skew
        raw_ask = fair_value + spread / 2.0 - skew

        bid_price = _clamp(_round_to_tick(raw_bid, ts, direction="down"), lo=min_price, hi=max_price)
        ask_price = _clamp(_round_to_tick(raw_ask, ts, direction="up"), lo=min_price, hi=max_price)

        # Determine which sides to quote
        bid_size: float | None = cfg.order_size * size_mult
        ask_size: float | None = cfg.order_size * size_mult

        # Skip bid if at max long position
        if position >= cfg.max_position:
            bid_price = None  # type: ignore[assignment]
            bid_size = None
            logger.info("Skipping bid for %s: position %.1f >= max %.1f", token_id, position, cfg.max_position)

        # Skip ask if at max short position
        if position <= -cfg.max_position:
            ask_price = None  # type: ignore[assignment]
            ask_size = None
            logger.info("Skipping ask for %s: position %.1f <= -max %.1f", token_id, position, cfg.max_position)

        # Cross-prevention: clamp quotes that would cross the market.
        # When allow_cross=True and we're long, let the ask cross (we want to
        # take the bid to sell inventory). Similarly when short, let the bid cross.
        ask_may_cross = allow_cross and position > 0
        bid_may_cross = allow_cross and position < 0

        if not ask_may_cross and ask_price is not None and best_bid is not None and ask_price <= best_bid:
            if best_ask is not None:
                ask_price = _clamp(_round_to_tick(best_ask - ts, ts, direction="up"), lo=min_price, hi=max_price)
            else:
                ask_price = _clamp(_round_to_tick(best_bid + ts, ts, direction="up"), lo=min_price, hi=max_price)
            logger.info(
                "Ask crossed market for %s: clamped to %.4f (best_bid=%.4f best_ask=%s)",
                token_id, ask_price, best_bid,
                f"{best_ask:.4f}" if best_ask is not None else "None",
            )
            # If still crossing or equal to best_bid, suppress the ask
            if ask_price is not None and best_bid is not None and ask_price <= best_bid:
                ask_price = None
                ask_size = None

        if not bid_may_cross and bid_price is not None and best_ask is not None and bid_price >= best_ask:
            # Retreat: place bid one tick below best_ask (mirror of ask-side logic)
            bid_price = _clamp(_round_to_tick(best_ask - ts, ts, direction="down"), lo=min_price, hi=max_price)
            logger.info(
                "Bid crossed market for %s: clamped to %.4f (best_bid=%s best_ask=%.4f)",
                token_id, bid_price,
                f"{best_bid:.4f}" if best_bid is not None else "None",
                best_ask,
            )
            # If still crossing or equal to best_ask, suppress the bid
            if bid_price is not None and best_ask is not None and bid_price >= best_ask:
                bid_price = None
                bid_size = None

        # If both sides present, check for inverted spread
        if bid_price is not None and ask_price is not None and bid_price >= ask_price:
            logger.warning(
                "Inverted spread for %s: bid=%.4f >= ask=%.4f (fv=%.4f pos=%.1f) — keeping reducing side only",
                token_id, bid_price, ask_price, fair_value, position,
            )
            # Keep the position-reducing side, drop the position-increasing side
            if position > 0:
                # Long → keep ask (reducing), drop bid (increasing)
                bid_price = None
                bid_size = None
            elif position < 0:
                # Short → keep bid (reducing), drop ask (increasing)
                ask_price = None
                ask_size = None
            else:
                # Flat → drop both (no preferred side)
                return None

        # If neither side is quotable, return None
        if bid_price is None and ask_price is None:
            return None

        quote = Quote(
            token_id=token_id,
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size,
            market_mid=market_mid,
            timestamp=datetime.utcnow(),
        )

        logger.debug(
            "Quote generated: token=%s bid=%.4f/%s ask=%.4f/%s fv=%.4f pos=%.1f skew=%.6f spread_mult=%.2f size_mult=%.2f",
            token_id,
            bid_price or 0,
            bid_size or 0,
            ask_price or 0,
            ask_size or 0,
            fair_value,
            position,
            skew,
            spread_mult,
            size_mult,
        )

        return quote

    def should_requote(
        self,
        current: Quote,
        new: Quote,
        threshold_bps: int | None = None,
    ) -> bool:
        """Determine whether a new quote differs enough to warrant requoting.

        Args:
            current: The currently live quote.
            new: The newly generated quote.
            threshold_bps: Minimum price change in basis points. Defaults to config value.

        Returns:
            True if the new quote differs meaningfully from the current.
        """
        if threshold_bps is None:
            threshold_bps = self._config.requote_threshold_bps

        threshold = threshold_bps / 10_000.0

        # If one side appeared or disappeared, requote
        if (current.bid_price is None) != (new.bid_price is None):
            return True
        if (current.ask_price is None) != (new.ask_price is None):
            return True

        # Check bid price movement
        if current.bid_price is not None and new.bid_price is not None:
            if abs(current.bid_price - new.bid_price) >= threshold:
                return True

        # Check ask price movement
        if current.ask_price is not None and new.ask_price is not None:
            if abs(current.ask_price - new.ask_price) >= threshold:
                return True

        # M-1: Check size changes with threshold comparison (half a unit)
        # instead of exact float equality to avoid unnecessary requotes from
        # floating-point rounding differences.
        if abs((current.bid_size or 0) - (new.bid_size or 0)) >= 0.5:
            return True
        # I-1 fix: Always requote when ask size DECREASED, regardless of
        # threshold. A decreased ask_size means our position shrunk (fill
        # arrived). If we don't requote, we have a live order for more
        # shares than we hold, risking over-commitment.
        current_ask = current.ask_size or 0
        new_ask = new.ask_size or 0
        if current_ask > new_ask + 1e-9:
            return True
        if abs(current_ask - new_ask) >= 0.5:
            return True

        return False
