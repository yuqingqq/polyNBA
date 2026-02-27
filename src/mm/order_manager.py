"""Order manager — places, cancels, and tracks orders via the Polymarket CLOB."""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .models import OrderState, OrderStatus, Quote, Side

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# C-5: Polymarket minimum order size
MIN_ORDER_SIZE = 5.0


class OrderManager:
    """Manages the full lifecycle of orders on the Polymarket CLOB.

    In dry-run mode all actions are logged but no actual API calls are made.
    """

    def __init__(self, client: Optional[Any] = None, dry_run: bool = True) -> None:
        """Initialize the order manager.

        Args:
            client: A ``ClobClient`` instance, or None for dry-run mode.
            dry_run: If True, simulate all order operations without hitting the CLOB.
        """
        self._client = client
        self._dry_run = dry_run
        self._orders: Dict[str, OrderState] = {}  # order_id -> OrderState
        # I-2: Thread-safety for _orders dict
        self._lock = threading.Lock()

        if dry_run:
            logger.info("OrderManager initialized in DRY-RUN mode")
        else:
            if client is None:
                raise ValueError("ClobClient required when dry_run=False")
            logger.info("OrderManager initialized in LIVE mode")

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    def place_order(
        self,
        token_id: str,
        side: Side,
        price: float,
        size: float,
        tick_size: str | None = None,
        neg_risk: bool | None = None,
        min_order_size: float | None = None,
        post_only: bool = True,
    ) -> OrderState:
        """Place a single order.

        Args:
            token_id: The token to trade.
            side: BUY or SELL.
            price: Limit price.
            size: Order size.
            tick_size: Tick size string for create_order options (e.g. "0.01").
            neg_risk: Whether this token uses the neg_risk exchange.
            min_order_size: Per-token minimum order size override.
            post_only: If True, order is post-only (no taker fees). Default True.

        Returns:
            The resulting OrderState.
        """
        # C-4: Use per-token min_order_size if provided, else fall back to global
        effective_min_size = min_order_size if min_order_size is not None else MIN_ORDER_SIZE
        # C-5: Validate minimum order size before sending to exchange
        if size < effective_min_size:
            logger.warning(
                "Order size %.1f below minimum %.1f — rejecting: token=%s side=%s price=%.4f",
                size, effective_min_size, token_id, side.value, price,
            )
            state = OrderState(
                order_id=f"rejected-{uuid.uuid4().hex[:8]}",
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                status=OrderStatus.FAILED,
                created_at=datetime.utcnow(),
            )
            with self._lock:
                self._orders[state.order_id] = state
            return state

        if self._dry_run:
            order_id = f"dry-{uuid.uuid4().hex[:12]}"
            state = OrderState(
                order_id=order_id,
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                status=OrderStatus.LIVE,
                created_at=datetime.utcnow(),
            )
            with self._lock:
                self._orders[order_id] = state
            logger.info(
                "[DRY-RUN] Placed order: id=%s token=%s side=%s price=%.4f size=%.1f",
                order_id,
                token_id,
                side.value,
                price,
                size,
            )
            return state

        # Live mode — call the CLOB
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions
            from py_clob_client.order_builder.constants import BUY as CLOB_BUY
            from py_clob_client.order_builder.constants import SELL as CLOB_SELL

            clob_side = CLOB_BUY if side == Side.BUY else CLOB_SELL
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=clob_side,
            )
            # C-2: Pass tick_size and neg_risk via options to avoid extra API lookups.
            # C-BUG-1: Only include neg_risk when True — py_clob_client treats False
            # as falsy/"not set" and falls through to an API lookup. Omitting it when
            # False produces the correct default behavior.
            options = None
            opts_kwargs: dict[str, Any] = {}
            if tick_size is not None:
                opts_kwargs["tick_size"] = tick_size
            if neg_risk:
                opts_kwargs["neg_risk"] = True
            if opts_kwargs:
                options = PartialCreateOrderOptions(**opts_kwargs)
            signed_order = self._client.create_order(order_args, options=options)
            # C-5: Pass post_only flag to prevent taker execution for MM quotes
            resp = self._client.post_order(signed_order, OrderType.GTC, post_only=post_only)

            success = resp.get("success", False) if isinstance(resp, dict) else False
            order_id = resp.get("orderID", resp.get("orderId", "")) if isinstance(resp, dict) else ""

            status = OrderStatus.LIVE if success else OrderStatus.FAILED
            state = OrderState(
                order_id=order_id or f"unknown-{uuid.uuid4().hex[:8]}",
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                status=status,
                created_at=datetime.utcnow(),
            )
            with self._lock:
                self._orders[state.order_id] = state

            logger.info(
                "Placed order: id=%s token=%s side=%s price=%.4f size=%.1f status=%s",
                state.order_id,
                token_id,
                side.value,
                price,
                size,
                status.value,
            )
            return state

        except Exception:
            logger.exception("Failed to place order: token=%s side=%s price=%.4f size=%.1f", token_id, side.value, price, size)
            state = OrderState(
                order_id=f"error-{uuid.uuid4().hex[:8]}",
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                status=OrderStatus.FAILED,
                created_at=datetime.utcnow(),
            )
            with self._lock:
                self._orders[state.order_id] = state
            return state

    def _cleanup_terminal_orders(self, include_stale: bool = True, active_token_ids: set[str] | None = None) -> tuple[list[str], set[str]]:
        """Remove old orders in terminal states to prevent memory leak.

        C-3: This method does NOT make HTTP calls. Stale LIVE/PENDING orders
        are marked as CANCELLED locally and removed from tracking. The actual
        exchange cancellation is handled by cancel_stale_orders_on_exchange()
        which must be called OUTSIDE the engine lock.

        C-6: The include_stale parameter controls whether stale LIVE/PENDING
        orders are also cleaned up. When called from place_quotes or
        cancel_all_for_token, pass include_stale=False so that stale order
        handling ONLY happens from the top-level cleanup() call in engine._tick.
        This prevents stale order IDs from being silently lost (the return value
        was previously discarded in those call sites).

        Args:
            include_stale: If True (default), also clean up stale LIVE/PENDING
                orders and return their IDs. If False, only clean terminal orders.

        Returns:
            Tuple of (stale_order_ids, stale_token_ids). Both empty if
            include_stale=False.
        """
        with self._lock:
            terminal = (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.FAILED)
            cutoff = datetime.utcnow() - timedelta(minutes=5)
            to_remove = [
                oid for oid, o in self._orders.items()
                if o.status in terminal and o.created_at < cutoff
            ]
            for oid in to_remove:
                del self._orders[oid]
            if to_remove:
                logger.debug("Cleaned up %d terminal orders", len(to_remove))

            if not include_stale:
                return [], set()

            # C-4: Also clean up stale LIVE/PENDING orders (older than 10 minutes).
            # The engine requotes every ~5s, so orders older than 10min are definitely
            # stale — they were likely cancelled server-side but we never got confirmation.
            # C-3: Mark as CANCELLED locally and remove. Do NOT make HTTP calls here
            # since this method may be called from within the engine lock chain.
            # C-1: Return the stale IDs BEFORE deleting so the caller can cancel
            # them on the exchange. Previously collect_stale_order_ids() was called
            # after cleanup which always returned empty since orders were already gone.
            stale_cutoff = datetime.utcnow() - timedelta(minutes=10)
            _protected = active_token_ids or set()
            stale_orders = [
                (oid, o.token_id) for oid, o in self._orders.items()
                if o.status in (OrderStatus.PENDING, OrderStatus.LIVE)
                and o.created_at < stale_cutoff
                and o.token_id not in _protected
            ]
            stale = [oid for oid, _ in stale_orders]
            stale_token_ids = {tid for _, tid in stale_orders}
            for oid in stale:
                self._orders[oid].status = OrderStatus.CANCELLED
                del self._orders[oid]
            if stale:
                logger.info(
                    "Stale LIVE/PENDING cleanup: marked %d orders as locally cancelled", len(stale)
                )

            return stale, stale_token_ids

    def cancel_stale_orders_on_exchange(self, stale_ids: List[str]) -> None:
        """Cancel stale orders on the exchange. Must be called OUTSIDE engine lock.

        C-3: Separated from _cleanup_terminal_orders to avoid HTTP calls
        under the engine lock chain.
        """
        if self._dry_run or not stale_ids:
            return
        for oid in stale_ids:
            try:
                self._client.cancel(order_id=oid)
                logger.info("Cancelled stale order %s on exchange", oid)
            except Exception:
                logger.warning(
                    "Failed to cancel stale order %s on exchange — already removed locally", oid
                )

    def place_quotes(
        self,
        quotes: List[Quote],
        token_tick_sizes: Optional[Dict[str, str]] = None,
        token_neg_risk: Optional[Dict[str, bool]] = None,
        token_min_sizes: Optional[Dict[str, float]] = None,
        post_only: bool = True,
    ) -> List[OrderState]:
        """Place bid and ask orders for a list of quotes.

        Each quote can produce up to two orders (bid + ask).
        In live mode, uses batch placement when possible.

        Args:
            quotes: Quotes to place.
            token_tick_sizes: Per-token tick size strings for create_order options.
            token_neg_risk: Per-token neg_risk flags for create_order options.
            token_min_sizes: Per-token minimum order sizes.
            post_only: If True, orders are post-only (no taker fees). Default True.

        Returns:
            List of resulting OrderState objects.
        """
        results: List[OrderState] = []
        tick_sizes = token_tick_sizes or {}
        neg_risks = token_neg_risk or {}
        min_sizes = token_min_sizes or {}

        if self._dry_run:
            for q in quotes:
                q_min = min_sizes.get(q.token_id)
                if q.bid_price is not None and q.bid_size is not None:
                    results.append(self.place_order(
                        q.token_id, Side.BUY, q.bid_price, q.bid_size,
                        min_order_size=q_min,
                    ))
                if q.ask_price is not None and q.ask_size is not None:
                    results.append(self.place_order(
                        q.token_id, Side.SELL, q.ask_price, q.ask_size,
                        min_order_size=q_min,
                    ))
            # C-6: Only clean terminal orders here; stale LIVE/PENDING handling
            # is done exclusively from the top-level cleanup() in engine._tick.
            self._cleanup_terminal_orders(include_stale=False)
            return results

        # Live mode — build batch
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType, PostOrdersArgs, PartialCreateOrderOptions
            from py_clob_client.order_builder.constants import BUY as CLOB_BUY
            from py_clob_client.order_builder.constants import SELL as CLOB_SELL

            batch: List[Any] = []
            order_meta: List[Dict[str, Any]] = []  # track metadata for each batch entry

            for q in quotes:
                # C-4: Use per-token min_order_size if available
                q_min_size = min_sizes.get(q.token_id, MIN_ORDER_SIZE)
                # C-2: Build options with tick_size and neg_risk to avoid extra API lookups.
                # C-BUG-1: Only include neg_risk when True (see place_order comment).
                q_tick_size = tick_sizes.get(q.token_id)
                q_neg_risk = neg_risks.get(q.token_id)
                options = None
                q_opts: dict[str, Any] = {}
                if q_tick_size is not None:
                    q_opts["tick_size"] = q_tick_size
                if q_neg_risk:
                    q_opts["neg_risk"] = True
                if q_opts:
                    options = PartialCreateOrderOptions(**q_opts)

                # C-5: Filter out sub-minimum orders before building batch
                if q.bid_price is not None and q.bid_size is not None and q.bid_size >= q_min_size:
                    args = OrderArgs(token_id=q.token_id, price=q.bid_price, size=q.bid_size, side=CLOB_BUY)
                    signed = self._client.create_order(args, options=options)
                    # C-5: Use post_only to prevent taker execution for MM quotes
                    batch.append(PostOrdersArgs(order=signed, orderType=OrderType.GTC, postOnly=post_only))
                    order_meta.append({"token_id": q.token_id, "side": Side.BUY, "price": q.bid_price, "size": q.bid_size})

                if q.ask_price is not None and q.ask_size is not None and q.ask_size >= q_min_size:
                    args = OrderArgs(token_id=q.token_id, price=q.ask_price, size=q.ask_size, side=CLOB_SELL)
                    signed = self._client.create_order(args, options=options)
                    batch.append(PostOrdersArgs(order=signed, orderType=OrderType.GTC, postOnly=post_only))
                    order_meta.append({"token_id": q.token_id, "side": Side.SELL, "price": q.ask_price, "size": q.ask_size})

            if not batch:
                return results

            # Submit in chunks of 15 (API limit)
            for i in range(0, len(batch), 15):
                chunk = batch[i : i + 15]
                chunk_meta = order_meta[i : i + 15]

                resp = self._client.post_orders(chunk)

                # Parse response — list of {orderID, success, errorMsg, status, ...}
                resp_list = resp if isinstance(resp, list) else []

                # C-1 (round 5): If response length doesn't match, mark all as FAILED.
                # Do NOT retry individual placement — the server may have already
                # placed some orders, so retrying risks duplicate exposure (same
                # fix pattern as C-7 for the exception path).
                if len(resp_list) != len(chunk_meta):
                    logger.warning(
                        "Batch response length mismatch: expected %d, got %d — marking all as FAILED (no retry to avoid duplicates)",
                        len(chunk_meta), len(resp_list),
                    )
                    for meta in chunk_meta:
                        state = OrderState(
                            order_id=f"batch-mismatch-{uuid.uuid4().hex[:8]}",
                            token_id=meta["token_id"],
                            side=meta["side"],
                            price=meta["price"],
                            size=meta["size"],
                            status=OrderStatus.FAILED,
                            created_at=datetime.utcnow(),
                        )
                        with self._lock:
                            self._orders[state.order_id] = state
                        results.append(state)
                    continue  # skip the normal response parsing

                for j, meta in enumerate(chunk_meta):
                    order_resp = resp_list[j] if j < len(resp_list) else {}
                    success = order_resp.get("success", False) if isinstance(order_resp, dict) else False
                    order_id = (order_resp.get("orderID", "") or order_resp.get("orderId", "")) if isinstance(order_resp, dict) else ""
                    error_msg = order_resp.get("errorMsg", "") if isinstance(order_resp, dict) else ""

                    # Treat empty orderID with error message as failure
                    # (Polymarket batch API returns success=true even on errors)
                    if not order_id and error_msg:
                        success = False
                    status = OrderStatus.LIVE if success and order_id else OrderStatus.FAILED

                    final_order_id = order_id or f"batch-{uuid.uuid4().hex[:8]}"

                    state = OrderState(
                        order_id=final_order_id,
                        token_id=meta["token_id"],
                        side=meta["side"],
                        price=meta["price"],
                        size=meta["size"],
                        status=status,
                        created_at=datetime.utcnow(),
                    )
                    with self._lock:
                        self._orders[final_order_id] = state
                    results.append(state)

                    if not success:
                        logger.warning(
                            "Batch order failed: token=%s side=%s price=%.4f error=%s",
                            meta["token_id"], meta["side"].value, meta["price"], error_msg,
                        )

            placed = sum(1 for r in results if r.status == OrderStatus.LIVE)
            logger.info("Batch placed %d/%d orders from %d quotes", placed, len(results), len(quotes))

        except Exception:
            # C-7: Don't retry on batch exception — partial server processing
            # may have already placed some orders. Retrying all orders risks
            # duplicates. Mark all as FAILED and let the next tick cycle requote.
            logger.exception("Failed to batch place orders — marking all as FAILED (no retry to avoid duplicates)")
            for q in quotes:
                # I-BUG-2: Use per-token min_order_size, not global MIN_ORDER_SIZE
                q_min_size_exc = min_sizes.get(q.token_id, MIN_ORDER_SIZE)
                if q.bid_price is not None and q.bid_size is not None and q.bid_size >= q_min_size_exc:
                    state = OrderState(
                        order_id=f"batch-fail-{uuid.uuid4().hex[:8]}",
                        token_id=q.token_id,
                        side=Side.BUY,
                        price=q.bid_price,
                        size=q.bid_size,
                        status=OrderStatus.FAILED,
                        created_at=datetime.utcnow(),
                    )
                    with self._lock:
                        self._orders[state.order_id] = state
                    results.append(state)
                if q.ask_price is not None and q.ask_size is not None and q.ask_size >= q_min_size_exc:
                    state = OrderState(
                        order_id=f"batch-fail-{uuid.uuid4().hex[:8]}",
                        token_id=q.token_id,
                        side=Side.SELL,
                        price=q.ask_price,
                        size=q.ask_size,
                        status=OrderStatus.FAILED,
                        created_at=datetime.utcnow(),
                    )
                    with self._lock:
                        self._orders[state.order_id] = state
                    results.append(state)

        # C-6: Only clean terminal orders here; stale LIVE/PENDING handling
        # is done exclusively from the top-level cleanup() in engine._tick.
        self._cleanup_terminal_orders(include_stale=False)
        return results

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order.

        I-5: Only transitions PENDING/LIVE orders to CANCELLED. If the order
        is already in a terminal state (FILLED/CANCELLED/FAILED), the cancel
        is a no-op and returns True (the order is already done).

        Args:
            order_id: The order to cancel.

        Returns:
            True if the cancellation succeeded (or was simulated), or if the
            order was already in a terminal state.
        """
        # I-5: Check if order is already in a terminal state — skip cancel
        terminal = (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.FAILED)
        with self._lock:
            order = self._orders.get(order_id)
            if order is not None and order.status in terminal:
                logger.debug(
                    "Order %s already in terminal state %s — skipping cancel",
                    order_id, order.status.value,
                )
                return True

        if self._dry_run:
            with self._lock:
                if order_id in self._orders:
                    # Re-check terminal status under lock — on_fill may have
                    # transitioned to FILLED between the earlier check and now.
                    if self._orders[order_id].status not in terminal:
                        self._orders[order_id].status = OrderStatus.CANCELLED
            logger.info("[DRY-RUN] Cancelled order: %s", order_id)
            return True

        try:
            self._client.cancel(order_id=order_id)
            with self._lock:
                if order_id in self._orders:
                    # I-5: Re-check status under lock — on_fill may have set it
                    # to FILLED between our earlier check and now.
                    if self._orders[order_id].status not in terminal:
                        self._orders[order_id].status = OrderStatus.CANCELLED
            logger.info("Cancelled order: %s", order_id)
            return True
        except Exception:
            logger.exception("Failed to cancel order: %s", order_id)
            return False

    def cancel_all_for_token(self, token_id: str) -> int:
        """Cancel all open orders for a specific token.

        I-1: Uses batch cancel API (cancel_orders) for a single HTTP call
        instead of sequential per-order cancels.

        Args:
            token_id: The token whose orders to cancel.

        Returns:
            Number of orders cancelled.
        """
        with self._lock:
            open_orders = [
                o for o in self._orders.values()
                if o.token_id == token_id and o.status in (OrderStatus.PENDING, OrderStatus.LIVE)
            ]

        if not open_orders:
            return 0

        order_ids = [o.order_id for o in open_orders]

        if self._dry_run:
            # In dry-run, just mark all as cancelled
            with self._lock:
                for oid in order_ids:
                    if oid in self._orders:
                        self._orders[oid].status = OrderStatus.CANCELLED
            logger.info("[DRY-RUN] Batch cancelled %d orders for token %s", len(order_ids), token_id)
            self._cleanup_terminal_orders(include_stale=False)
            return len(order_ids)

        # Live mode — use batch cancel
        try:
            self._client.cancel_orders(order_ids)
            terminal = (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.FAILED)
            with self._lock:
                for oid in order_ids:
                    if oid in self._orders and self._orders[oid].status not in terminal:
                        self._orders[oid].status = OrderStatus.CANCELLED
            logger.info("Batch cancelled %d orders for token %s", len(order_ids), token_id)
            self._cleanup_terminal_orders(include_stale=False)
            return len(order_ids)
        except Exception:
            logger.exception("Batch cancel failed for token %s — falling back to sequential", token_id)
            # Fallback to sequential cancel
            count = 0
            failed = 0
            for order in open_orders:
                if self.cancel_order(order.order_id):
                    count += 1
                else:
                    failed += 1
            if failed > 0:
                logger.warning(
                    "Failed to cancel %d/%d orders for token %s",
                    failed, len(open_orders), token_id,
                )
            logger.info("Sequential fallback cancelled %d orders for token %s", count, token_id)
            self._cleanup_terminal_orders(include_stale=False)
            return count

    def cancel_all(self) -> int:
        """Cancel all open orders across all tokens.

        Returns:
            Number of orders cancelled.
        """
        if not self._dry_run:
            try:
                self._client.cancel_all()
                with self._lock:
                    count = 0
                    for order in self._orders.values():
                        if order.status in (OrderStatus.PENDING, OrderStatus.LIVE):
                            order.status = OrderStatus.CANCELLED
                            count += 1
                logger.info("Cancelled all orders: %d total", count)
                # I-2 (round 5): Pass include_stale=False — the blanket cancel_all
                # already handles live orders; cleanup should only remove terminal ones.
                self._cleanup_terminal_orders(include_stale=False)
                return count
            except Exception:
                logger.exception(
                    "Failed to cancel all orders via API — marking local orders as cancelled"
                )
                with self._lock:
                    count = 0
                    for order in self._orders.values():
                        if order.status in (OrderStatus.PENDING, OrderStatus.LIVE):
                            order.status = OrderStatus.CANCELLED
                            count += 1
                logger.warning(
                    "Marked %d orders as locally cancelled (exchange state unknown)", count
                )
                # I-2 (round 5): Pass include_stale=False — same reasoning as success path.
                self._cleanup_terminal_orders(include_stale=False)
                return count

        # Dry-run
        with self._lock:
            open_order_ids = [
                o.order_id for o in self._orders.values()
                if o.status in (OrderStatus.PENDING, OrderStatus.LIVE)
            ]
        count = 0
        for oid in open_order_ids:
            if self.cancel_order(oid):
                count += 1

        logger.info("Cancelled all orders: %d total", count)
        # I-2 (round 5): Pass include_stale=False — same reasoning as live paths.
        self._cleanup_terminal_orders(include_stale=False)
        return count

    def cleanup(self, active_token_ids: set[str] | None = None) -> tuple[list[str], set[str]]:
        """Public interface to remove old terminal orders.

        Args:
            active_token_ids: Token IDs with intentionally live quotes.
                Orders for these tokens are not stale — they're waiting
                to fill — and will be skipped by the stale cleanup.

        Returns:
            Tuple of (stale_order_ids, stale_token_ids):
            - stale_order_ids: order IDs to cancel on exchange via
              cancel_stale_orders_on_exchange().
            - stale_token_ids: token IDs affected by the stale cleanup,
              so the engine can clear _current_quotes and re-place orders.
        """
        return self._cleanup_terminal_orders(active_token_ids=active_token_ids)

    def get_tracked_order_ids(self) -> set[str]:
        """Return the set of all order IDs currently tracked (any status).

        Used to match incoming user trade events against our orders so we
        only process fills that belong to us.
        """
        with self._lock:
            return set(self._orders.keys())

    def is_tracked(self, order_id: str) -> bool:
        """Check if an order ID is currently tracked.

        I-6: Used by engine.on_fill() to check order membership inside the
        engine lock, avoiding the race where get_tracked_order_ids() is
        called outside the lock and orders are cleaned up before on_fill.

        Note: This method acquires its own _lock internally. When called
        from engine.on_fill (which holds engine._lock), this is safe because
        OrderManager._lock is a different lock — no deadlock risk.

        Args:
            order_id: The order ID to check.

        Returns:
            True if the order is tracked.
        """
        with self._lock:
            return order_id in self._orders

    def get_open_orders(self, token_id: Optional[str] = None) -> List[OrderState]:
        """Return all currently open orders.

        Args:
            token_id: If provided, filter to this token only.

        Returns:
            List of open OrderState objects.
        """
        with self._lock:
            open_statuses = (OrderStatus.PENDING, OrderStatus.LIVE)
            orders = [
                o for o in self._orders.values()
                if o.status in open_statuses
            ]
            if token_id is not None:
                orders = [o for o in orders if o.token_id == token_id]
            return orders

    def on_fill(self, order_id: str, filled_size: float, price: float) -> bool:
        """Handle a fill notification for an order.

        Updates the order's filled_size and status.

        Args:
            order_id: The order that was filled.
            filled_size: The size that was filled in this event.
            price: The fill price.

        Returns:
            True if the order was tracked and processed, False if unknown.
        """
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                logger.debug("Fill for untracked order %s — ignoring", order_id)
                return False

            # I-7: If order is already in a terminal state (CANCELLED/FAILED),
            # update filled_size for accounting but don't overwrite the status.
            # A partial fill arriving after cancel should not resurrect the order.
            if order.status in (OrderStatus.CANCELLED, OrderStatus.FAILED):
                logger.warning(
                    "Fill for terminal order %s (status=%s) — updating filled_size only",
                    order_id, order.status.value,
                )
                order.filled_size += filled_size
                return True

            order.filled_size += filled_size
            # Use epsilon tolerance for float comparison. Accumulated filled_size
            # via repeated += can have floating-point imprecision (e.g., three
            # fills of 3.33 sum to 9.99 not 10.0), leaving orders stuck as LIVE.
            if order.filled_size >= order.size - 1e-9:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.LIVE  # partial fill, still live

            # I-2: Capture log values inside lock to avoid reading mutable
            # state after lock release from another thread.
            log_filled = order.filled_size
            log_total = order.size
            log_status = order.status.value

        logger.info(
            "Order fill: id=%s filled=%.1f/%.1f price=%.4f status=%s",
            order_id,
            log_filled,
            log_total,
            price,
            log_status,
        )
        return True

    def on_cancel(self, order_id: str) -> Optional[str]:
        """Handle an external cancellation notification for an order.

        Marks the order as CANCELLED if it is currently in a non-terminal
        state. Returns the token_id so the caller (engine) can clear stale
        quote state and enqueue a requote.

        If the order is unknown or already in a terminal state, returns None
        (idempotent — processing the same cancellation twice is harmless).

        Args:
            order_id: The order that was cancelled.

        Returns:
            The token_id of the cancelled order, or None if no action taken.
        """
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                logger.debug("Cancel for untracked order %s — ignoring", order_id)
                return None

            terminal = (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.FAILED)
            if order.status in terminal:
                logger.debug(
                    "Cancel for already-terminal order %s (status=%s) — ignoring",
                    order_id, order.status.value,
                )
                return None

            order.status = OrderStatus.CANCELLED
            token_id = order.token_id

        logger.info("Order cancelled (external): id=%s token=%s", order_id, token_id)
        return token_id
