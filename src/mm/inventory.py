"""Inventory manager — tracks positions, PnL, and exposure."""

from __future__ import annotations

import logging
import threading
from typing import Dict

from .config import MMConfig
from .models import Fill, Position, Side

logger = logging.getLogger(__name__)


class InventoryManager:
    """Thread-safe position and PnL tracker.

    Maintains per-token positions updated by fill events and computes
    unrealized PnL against mark prices.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._positions: Dict[str, Position] = {}
        self._mark_prices: Dict[str, float] = {}
        self._pairs: Dict[str, str] = {}  # token_id -> complement_token_id

    def register_pair(self, token_a: str, token_b: str) -> None:
        """Register two tokens as a binary pair (e.g. Yes/No on same condition).

        Paired tokens share net exposure for limits and reporting.

        I-8: This method must only be called BEFORE ``engine.run()`` starts.
        Calling it during runtime would acquire ``_lock`` from the registering
        thread while ``is_within_limits()`` or ``get_total_exposure()`` may
        already hold ``_lock`` on the engine thread (``threading.Lock`` is not
        re-entrant). Registration during initialization is safe because no
        other thread is accessing the inventory yet.

        Args:
            token_a: First token ID.
            token_b: Second token ID.
        """
        with self._lock:
            self._pairs[token_a] = token_b
            self._pairs[token_b] = token_a

    def _ensure_position(self, token_id: str) -> Position:
        """Return the position for a token, creating one if needed."""
        if token_id not in self._positions:
            self._positions[token_id] = Position(token_id=token_id)
        return self._positions[token_id]

    def update_fill(self, fill: Fill) -> None:
        """Process a fill and update the corresponding position.

        For buys, position increases. For sells, position decreases.
        Average entry price is updated on position-increasing fills.
        Realized PnL is booked on position-reducing fills.

        Args:
            fill: The fill event to process.
        """
        with self._lock:
            pos = self._ensure_position(fill.token_id)

            signed_size = fill.size if fill.side == Side.BUY else -fill.size
            old_size = pos.size
            new_size = old_size + signed_size

            # Determine if this fill increases or reduces the position
            if old_size == 0.0:
                # Opening from flat
                pos.avg_entry_price = fill.price
            elif (old_size > 0 and signed_size > 0) or (old_size < 0 and signed_size < 0):
                # Increasing position — update average entry
                total_cost = abs(old_size) * pos.avg_entry_price + fill.size * fill.price
                pos.avg_entry_price = total_cost / (abs(old_size) + fill.size)
            else:
                # Reducing or flipping position — book realized PnL
                reduce_size = min(fill.size, abs(old_size))
                if old_size > 0:
                    # Was long, selling
                    realized = reduce_size * (fill.price - pos.avg_entry_price)
                else:
                    # Was short, buying
                    realized = reduce_size * (pos.avg_entry_price - fill.price)
                pos.realized_pnl += realized

                # If we flipped, set new avg entry to fill price for the overshoot
                if abs(signed_size) > abs(old_size):
                    pos.avg_entry_price = fill.price

            pos.size = new_size

            # C-2: Always deduct fee regardless of fill direction.
            # Previously fees were only captured in the reducing branch.
            pos.realized_pnl -= fill.fee

            # I-11: Warn if position goes negative. Polymarket does not allow
            # short positions, so a negative position indicates a tracking error
            # which would cause incorrect PnL and risk checks.
            if new_size < 0:
                logger.warning(
                    "NEGATIVE POSITION detected: token=%s size=%.1f — "
                    "Polymarket does not allow short positions. "
                    "This may indicate a fill tracking error.",
                    fill.token_id,
                    new_size,
                )

            # Recalculate unrealized PnL if we have a mark price
            mark = self._mark_prices.get(fill.token_id)
            if mark is not None:
                self._update_unrealized(pos, mark)

            logger.info(
                "Fill processed: token=%s side=%s price=%.4f size=%.1f -> pos=%.1f avg=%.4f rpnl=%.2f",
                fill.token_id,
                fill.side.value,
                fill.price,
                fill.size,
                pos.size,
                pos.avg_entry_price,
                pos.realized_pnl,
            )

    def reset_position(self, token_id: str) -> None:
        """Reset a token's position to zero, preserving realized PnL.

        Clears position size, average entry price, and unrealized PnL for
        the token.  Used before re-seeding positions from trade history on
        reconnect to avoid double-counting (C-1).

        I-5: Preserves realized_pnl from the old position so that the
        max_loss risk check is not defeated by reconnect re-seeding.

        Args:
            token_id: The token whose position should be reset.
        """
        with self._lock:
            old_realized_pnl = 0.0
            old_pos = self._positions.get(token_id)
            if old_pos is not None:
                old_realized_pnl = old_pos.realized_pnl
            new_pos = Position(token_id=token_id)
            new_pos.realized_pnl = old_realized_pnl
            self._positions[token_id] = new_pos
            logger.info(
                "Reset position for token %s (preserved realized_pnl=%.2f)",
                token_id, old_realized_pnl,
            )

    def get_position(self, token_id: str) -> Position:
        """Return the current position for a token.

        Args:
            token_id: The token to query.

        Returns:
            The Position (zero-initialized if no fills have occurred).
        """
        with self._lock:
            return self._ensure_position(token_id).model_copy()

    def get_total_exposure(self) -> float:
        """Return the total exposure across all tokens.

        For paired tokens, exposure is ``abs(pos_a - pos_b)`` counted once
        (not both individually).  Unpaired tokens use ``abs(pos)`` as before.

        Returns:
            Total absolute exposure.
        """
        with self._lock:
            seen: set[str] = set()
            total = 0.0
            for tid, pos in self._positions.items():
                if tid in seen:
                    continue
                complement = self._pairs.get(tid)
                if complement is not None and complement in self._positions:
                    seen.add(tid)
                    seen.add(complement)
                    total += abs(pos.size - self._positions[complement].size)
                else:
                    seen.add(tid)
                    total += abs(pos.size)
            return total

    def get_pnl(self) -> float:
        """Return the total PnL (realized + unrealized) across all tokens.

        Returns:
            Combined PnL.
        """
        with self._lock:
            return sum(p.realized_pnl + p.unrealized_pnl for p in self._positions.values())

    def update_mark_price(self, token_id: str, price: float) -> None:
        """Set the mark price for unrealized PnL calculation.

        Args:
            token_id: The token to update.
            price: The current mark/mid price.
        """
        with self._lock:
            self._mark_prices[token_id] = price
            pos = self._ensure_position(token_id)
            self._update_unrealized(pos, price)

    def _update_unrealized(self, pos: Position, mark_price: float) -> None:
        """Recalculate unrealized PnL for a position against a mark price."""
        if pos.size == 0.0:
            pos.unrealized_pnl = 0.0
        elif pos.size > 0:
            pos.unrealized_pnl = pos.size * (mark_price - pos.avg_entry_price)
        else:
            pos.unrealized_pnl = abs(pos.size) * (pos.avg_entry_price - mark_price)

    def is_within_limits(self, config: MMConfig) -> bool:
        """Check whether all positions are within configured limits.

        For paired tokens the per-token limit is checked against the net
        position ``abs(pos_a - pos_b)`` rather than each side independently.

        Args:
            config: The strategy configuration with position limits.

        Returns:
            True if all limits are satisfied.
        """
        with self._lock:
            checked: set[str] = set()
            for pos in self._positions.values():
                if pos.token_id in checked:
                    continue
                complement_id = self._pairs.get(pos.token_id)
                if complement_id is not None and complement_id in self._positions:
                    checked.add(pos.token_id)
                    checked.add(complement_id)
                    net = abs(pos.size - self._positions[complement_id].size)
                else:
                    checked.add(pos.token_id)
                    net = abs(pos.size)

                if net > config.max_position:
                    logger.warning(
                        "Position limit breached: token=%s net=%.1f max=%.1f",
                        pos.token_id,
                        net,
                        config.max_position,
                    )
                    return False

            # Total exposure (pair-aware) — reuse the method to stay DRY.
            # We already hold the lock, so call the inner logic directly.
            seen: set[str] = set()
            total = 0.0
            for tid, p in self._positions.items():
                if tid in seen:
                    continue
                complement = self._pairs.get(tid)
                if complement is not None and complement in self._positions:
                    seen.add(tid)
                    seen.add(complement)
                    total += abs(p.size - self._positions[complement].size)
                else:
                    seen.add(tid)
                    total += abs(p.size)

            if total > config.max_total_position:
                logger.warning(
                    "Total position limit breached: total=%.1f max=%.1f",
                    total,
                    config.max_total_position,
                )
                return False

        return True
