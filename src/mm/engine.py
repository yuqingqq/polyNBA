"""Main market making engine — orchestrates the quoting loop."""

from __future__ import annotations

import logging
import time
import threading
from typing import Any, Dict, Optional

from .config import MMConfig
from .divergence import DivergenceTracker, compute_ema
from .fair_value import FairValueEngine
from .inventory import InventoryManager
from .latency import LatencyTracker
from .models import Fill, OrderStatus, Quote, Side
from .order_manager import OrderManager
from .quoting import QuotingEngine
from .risk import RiskManager

# Optional import — OrderbookManager lives in src.data
try:
    from src.data.orderbook import OrderbookManager
except ImportError:
    OrderbookManager = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class MarketMakingEngine:
    """Top-level orchestrator for the market making strategy.

    Runs a simple ``while not stopped`` loop that:

    1. Performs a risk check and halts if any limit is breached.
    2. For each registered token, computes a fair value, checks
       the current position, generates quotes, and requotes if
       the price has moved significantly.
    3. Logs the current state.

    The engine catches all exceptions inside the loop so that transient
    errors do not crash the process.

    Lock ordering (must always be acquired in this order to prevent deadlock):

    1. ``engine._lock`` (outermost)
    2. ``inventory._lock`` (via ``self._inventory`` methods)
    3. ``order_manager._lock`` (via ``self._order_manager`` methods)

    Never acquire a higher-numbered lock and then attempt to acquire a
    lower-numbered one.  All current code paths respect this ordering.
    """

    def __init__(
        self,
        config: MMConfig,
        client: Optional[Any] = None,
        orderbook_manager: Optional[Any] = None,
    ) -> None:
        """Initialize the engine.

        Args:
            config: Strategy configuration.
            client: A ``ClobClient`` instance, or None for dry-run mode.
            orderbook_manager: An ``OrderbookManager`` instance for market mid
                lookups, or None to skip divergence adjustments.
        """
        self._config = config
        self._fair_value = FairValueEngine()
        self._inventory = InventoryManager()
        self._quoting = QuotingEngine(config)
        self._order_manager = OrderManager(client=client, dry_run=config.dry_run)
        self._risk = RiskManager(config)
        self._orderbook_manager = orderbook_manager

        # token_id -> tick_size override
        self._tokens: Dict[str, float] = {}
        # C-2: token_id -> neg_risk flag
        self._neg_risk: Dict[str, bool] = {}
        # C-4: token_id -> minimum order size
        self._min_sizes: Dict[str, float] = {}
        # token_id -> complement token_id (binary pair)
        self._pairs: Dict[str, str] = {}
        # token_id -> last sent Quote
        self._current_quotes: Dict[str, Quote] = {}

        self._stopped = threading.Event()
        self._lock = threading.Lock()
        self._risk_halted = False
        # I-4: Hysteresis — after a risk halt, wait this long before resuming
        self._risk_halt_until: float = 0.0

        # Latency instrumentation
        self._latency = LatencyTracker()
        self._last_requote_time: Dict[str, float] = {}

        # Divergence EMA state
        self._divergence_emas: Dict[str, float] = {}
        self._divergence_tracker = DivergenceTracker(max_bps=config.divergence_max_bps)

        # I-3: Accumulation start time per token — used for timeout
        self._accumulation_start: Dict[str, float] = {}
        # I-2: Tokens that exhausted accumulation timeout — permanently skip
        self._accumulation_exhausted: set[str] = set()

        # C-3: Pending requotes from on_fill — processed by tick loop outside lock.
        # on_fill adds token_ids here under lock; _tick drains and requotes them.
        self._pending_requotes: set[str] = set()
        # C-5: Per-token requoting flag to prevent concurrent requotes.
        # Set True while _requote_token_unlocked is in progress for a token.
        self._requoting: set[str] = set()

        # C-1 (runner): External gate for temporarily blocking quoting.
        # When cleared, _tick() skips all token processing. The runner
        # clears this during reconnect-cancel to prevent the tick loop
        # from placing new orders that the cancel thread would then remove.
        self._quoting_gate = threading.Event()
        self._quoting_gate.set()  # initially open — quoting allowed

        logger.info(
            "MarketMakingEngine initialized: dry_run=%s spread=%dbps interval=%.1fs orderbook=%s",
            config.dry_run,
            config.spread_bps,
            config.update_interval_seconds,
            "yes" if orderbook_manager is not None else "no",
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def fair_value_engine(self) -> FairValueEngine:
        return self._fair_value

    @property
    def inventory_manager(self) -> InventoryManager:
        return self._inventory

    @property
    def order_manager(self) -> OrderManager:
        return self._order_manager

    @property
    def risk_manager(self) -> RiskManager:
        return self._risk

    @property
    def quoting_engine(self) -> QuotingEngine:
        return self._quoting

    @property
    def latency_tracker(self) -> LatencyTracker:
        return self._latency

    def clear_divergence_emas(self) -> None:
        """Clear divergence EMA state.

        I-10: Should be called on orderbook reconnect so stale EMA values
        from pre-reconnect data don't affect quoting after the reconnect.
        """
        with self._lock:
            self._divergence_emas.clear()
        logger.info("Divergence EMAs cleared (orderbook reconnect)")

    def clear_quotes(self) -> None:
        """Clear stale quotes so the next tick forces a full requote.

        I-3: Clears both ``_current_quotes`` and ``_last_requote_time``
        atomically under ``_lock``. Should be called on orderbook reconnect
        so that ``should_requote()`` doesn't skip retrying when the fair
        value hasn't changed but the orderbook is now empty.

        R23-I2: Also clears divergence EMAs. These two operations must
        always happen together on reconnect — stale high-EMA values from
        before the reconnect cause unnecessarily widened spreads for
        5-10 ticks during recovery. Consolidating here prevents callers
        from forgetting to call ``clear_divergence_emas()`` separately.
        """
        with self._lock:
            self._current_quotes.clear()
            self._last_requote_time.clear()
            self._divergence_emas.clear()
        logger.info("Quotes and divergence EMAs cleared")

    def update_tick_size(self, asset_id: str, new_tick_size: float) -> None:
        """Update tick_size for a token and its complement (if paired).

        I-NEW-1: Public method so that callers don't need to access
        ``_lock``, ``_tokens``, or ``_pairs`` directly.

        Args:
            asset_id: The token whose tick_size changed.
            new_tick_size: The new tick_size value from the exchange.
        """
        with self._lock:
            if asset_id not in self._tokens:
                return
            old_tick_size = self._tokens[asset_id]
            self._tokens[asset_id] = new_tick_size
            logger.info(
                "Updated tick_size for %s: %.6f -> %.6f",
                asset_id, old_tick_size, new_tick_size,
            )
            # Update complement token if this is part of a binary pair.
            # Polymarket changes tick_size per condition_id (market), so
            # both tokens in a pair should always share the same tick_size.
            complement = self._pairs.get(asset_id)
            if complement is not None and complement in self._tokens:
                old_complement_tick = self._tokens[complement]
                self._tokens[complement] = new_tick_size
                logger.info(
                    "Updated tick_size for complement %s: %.6f -> %.6f",
                    complement, old_complement_tick, new_tick_size,
                )

    def seed_positions(
        self,
        computed_positions: Dict[str, tuple[float, float]],
    ) -> None:
        """Reset positions and inject synthetic fills from trade history.

        I-NEW-2: Public method so that callers don't need to access
        ``_lock`` or ``_inventory`` directly.

        Acquires ``_lock`` to make the reset + inject atomic with respect
        to ``on_fill()``.

        Args:
            computed_positions: Mapping of token_id to (net_size, avg_price)
                tuples computed from trade history. Only tokens present in
                this dict are reset (RN-1).
        """
        with self._lock:
            # RN-1 fix: Only reset positions for tokens present in computed_positions.
            # Previously all token_ids were reset, which wiped live-accumulated fills
            # for tokens not in computed_positions (e.g., tokens with very recent fills
            # not yet in the API, or tokens with net position < 0.01 filtered out).
            for token_id in computed_positions:
                self._inventory.reset_position(token_id)

            # Inject synthetic fills for non-zero positions
            for token_id, (net_size, avg_price) in computed_positions.items():
                # R13 fix: Skip zero-size positions to avoid misleading logs
                # and unnecessary zero-size fill processing.
                if abs(net_size) < 1e-9:
                    continue
                side = Side.BUY if net_size > 0 else Side.SELL
                fill = Fill(
                    order_id=f"seed-{token_id[:8]}",
                    token_id=token_id,
                    side=side,
                    price=avg_price,
                    size=abs(net_size),
                )
                self._inventory.update_fill(fill)
                logger.info(
                    "Seeded position: %s %.2f %s @ %.4f",
                    side.value, abs(net_size), token_id[-10:], avg_price,
                )

    # ------------------------------------------------------------------
    # Token registration
    # ------------------------------------------------------------------

    def add_token(
        self,
        token_id: str,
        tick_size: float = 0.01,
        neg_risk: bool = False,
        min_order_size: float = 5.0,
    ) -> None:
        """Register a token for quoting.

        C-1 fix: Acquires ``_lock`` to prevent dict mutation races with
        ``_tick()`` (which iterates ``_tokens`` under lock). Consistent
        with ``update_tick_size()`` which also acquires the lock.

        Args:
            token_id: The Polymarket token/outcome identifier.
            tick_size: The minimum price increment for this token.
            neg_risk: Whether this token uses the neg_risk exchange.
            min_order_size: Minimum order size for this token's market.
        """
        with self._lock:
            self._tokens[token_id] = tick_size
            self._neg_risk[token_id] = neg_risk
            self._min_sizes[token_id] = min_order_size
        logger.info(
            "Registered token %s with tick_size=%.4f neg_risk=%s min_order_size=%.1f",
            token_id, tick_size, neg_risk, min_order_size,
        )

    def add_token_pair(
        self,
        token_a: str,
        token_b: str,
        tick_size: float = 0.01,
        neg_risk: bool = False,
        min_order_size: float = 5.0,
    ) -> None:
        """Register two tokens as a binary pair sharing one condition_id.

        Both tokens are added for quoting and linked so that inventory
        skew, exposure, and limits use the net position ``pos_a - pos_b``.

        C-1 fix: Acquires ``_lock`` to prevent dict mutation races with
        ``_tick()`` (which iterates ``_tokens`` under lock). Consistent
        with ``update_tick_size()`` which also acquires the lock.

        Args:
            token_a: First token ID.
            token_b: Second token ID.
            tick_size: The minimum price increment for both tokens.
            neg_risk: Whether these tokens use the neg_risk exchange.
            min_order_size: Minimum order size for this market.
        """
        with self._lock:
            self._tokens[token_a] = tick_size
            self._tokens[token_b] = tick_size
            self._neg_risk[token_a] = neg_risk
            self._neg_risk[token_b] = neg_risk
            self._min_sizes[token_a] = min_order_size
            self._min_sizes[token_b] = min_order_size
            self._pairs[token_a] = token_b
            self._pairs[token_b] = token_a
            self._inventory.register_pair(token_a, token_b)
        logger.info(
            "Registered token pair %s <-> %s with tick_size=%.4f neg_risk=%s min_order_size=%.1f",
            token_a, token_b, tick_size, neg_risk, min_order_size,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the main market making loop.

        Blocks until ``stop()`` is called from another thread or a
        KeyboardInterrupt is received.
        """
        logger.info("MarketMakingEngine starting main loop")
        self._stopped.clear()

        try:
            while not self._stopped.is_set():
                try:
                    self._tick()
                except Exception:
                    logger.exception("Error in main loop tick — continuing")

                self._stopped.wait(timeout=self._config.update_interval_seconds)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received — stopping")
        finally:
            self._shutdown()

    def stop(self) -> None:
        """Signal the engine to stop after the current tick."""
        logger.info("Stop requested")
        self._stopped.set()

    # ------------------------------------------------------------------
    # Fill handling
    # ------------------------------------------------------------------

    def on_fill(self, fill: Fill) -> None:
        """Handle an incoming fill event.

        Updates inventory and the order manager's state, then enqueues
        the affected token for requoting by the tick loop.

        Thread-safe: acquires ``_lock`` so that inventory/order mutations
        do not race with the main tick loop.

        C-3: Does NOT call _requote_token_unlocked under lock. Instead,
        adds the token to _pending_requotes which the tick loop drains
        and processes outside the lock. This prevents blocking the tick
        loop during HTTP calls (cancel + place).

        C-5: If the token is currently being requoted (_requoting flag),
        skip adding to pending requotes — the in-progress requote will
        use fresh position data.

        I-6: The tracked order check and fill processing happen atomically
        under the engine lock, avoiding the race where orders are cleaned
        up between an external tracked-ID check and fill processing.

        R16-C1: OrderManager.on_fill is called BEFORE inventory update.
        user_trade_to_fills emits fill candidates for ALL orders in a trade
        event (taker + all makers), including other users' orders. Without
        this guard, phantom fills from other participants would corrupt our
        inventory, inflating position size and causing wrong risk checks.

        Args:
            fill: The fill to process.
        """
        t0 = time.monotonic()
        with self._lock:
            try:
                # R16-C1: Check order membership FIRST — only update inventory
                # for our own tracked orders. Preserves I-6 atomicity (both
                # check and update inside engine lock, no TOCTOU race).
                tracked = self._order_manager.on_fill(fill.order_id, fill.size, fill.price)
                if not tracked:
                    return
                self._inventory.update_fill(fill)
                # I-14: Update mark price to current fair value (not fill price).
                # Fill price is bid/ask, not fair value mid, so using it would
                # make PnL incorrect until the next tick updates it.
                fair_value = self._fair_value.get_fair_value(fill.token_id)
                mark_price = fair_value if fair_value is not None else fill.price
                self._inventory.update_mark_price(fill.token_id, mark_price)
                logger.info("Fill handled: order=%s token=%s", fill.order_id, fill.token_id)

                # C-3: Enqueue requote for the tick loop instead of requoting here.
                # C-5: Skip if token is already being requoted to avoid duplicate orders.
                if fill.token_id in self._tokens and not self._risk_halted:
                    if fill.token_id not in self._requoting:
                        self._pending_requotes.add(fill.token_id)
                    else:
                        logger.debug(
                            "Token %s is being requoted — skipping pending requote from fill",
                            fill.token_id,
                        )
                    # MM-2 fix: Also enqueue the complement token for requote.
                    # For binary pairs, a fill on one token changes the net_position
                    # used for inventory skew on the complement. Without this, the
                    # complement's quotes remain stale for up to one full tick interval,
                    # enabling adverse selection on the stale skew.
                    complement_id = self._pairs.get(fill.token_id)
                    if complement_id is not None and complement_id in self._tokens:
                        if complement_id not in self._requoting:
                            self._pending_requotes.add(complement_id)
            except Exception:
                logger.exception("Error handling fill: %s", fill)
        # Record latency outside the engine lock (LatencyTracker has its own)
        self._latency.record("on_fill", time.monotonic() - t0)

    def on_order_cancel(self, order_id: str) -> None:
        """Handle an external order cancellation (e.g. game-start auto-cancel).

        Marks the order as cancelled via OrderManager, clears the stale quote
        entry from ``_current_quotes``, and enqueues both the affected token
        and its complement for requoting on the next tick.

        Thread-safe: acquires ``_lock`` (outermost per lock ordering).
        Processing the same cancellation twice is harmless — ``on_cancel``
        returns None for already-terminal orders.

        Args:
            order_id: The cancelled order ID from a UserOrderEvent.
        """
        with self._lock:
            try:
                token_id = self._order_manager.on_cancel(order_id)
                if token_id is None:
                    return

                # Clear stale quote so the next tick re-places fresh orders
                self._current_quotes.pop(token_id, None)

                # Enqueue requote (same guards as on_fill)
                if token_id in self._tokens and not self._risk_halted:
                    if token_id not in self._requoting:
                        self._pending_requotes.add(token_id)
                    else:
                        logger.debug(
                            "Token %s is being requoted — skipping pending requote from cancel",
                            token_id,
                        )
                    # MM-2: Also enqueue complement token for requote
                    complement_id = self._pairs.get(token_id)
                    if complement_id is not None and complement_id in self._tokens:
                        if complement_id not in self._requoting:
                            self._pending_requotes.add(complement_id)
            except Exception:
                logger.exception("Error handling order cancel: %s", order_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """Execute one cycle of the market making loop."""
        t0 = time.monotonic()

        # 1. Risk check — hold lock for state mutation only.
        #    I-4: The kill_switch HTTP call (cancel_all) is done OUTSIDE the lock
        #    to avoid blocking fill processing. The critical state mutations
        #    (_risk_halted flag + _current_quotes.clear()) are atomic under lock.
        _need_kill_switch = False
        _cooldown_active = False
        _stale_tokens_to_cancel: list[str] = []
        stale_set: set[str] = set()
        tokens_snapshot: Dict[str, float] = {}
        with self._lock:
            risk_result = self._risk.check_all(
                self._inventory, self._fair_value,
                active_token_ids=set(self._tokens.keys()),
            )
            if risk_result.should_halt:
                logger.warning("Risk halt — cancelling all orders. Reasons: %s", risk_result.reasons)
                self._risk_halted = True
                self._risk_halt_until = time.monotonic() + 30.0
                self._current_quotes.clear()
                _need_kill_switch = True

            # I-4: Hysteresis — stay halted during cooldown even if conditions cleared
            elif time.monotonic() < self._risk_halt_until:
                logger.info("Risk halt cooldown active (%.1fs remaining) — staying halted",
                            self._risk_halt_until - time.monotonic())
                # I-3 fix: Run cleanup and drain pending requotes during cooldown
                # to prevent memory leak from accumulating terminal orders and
                # pending requotes that would all fire at once when cooldown ends.
                self._pending_requotes.clear()
                self._latency.record("tick", time.monotonic() - t0)
                _cooldown_active = True
            else:
                self._risk_halted = False

            if not _need_kill_switch:
                # I-1 (round 5): Collect stale token IDs under the lock and clean
                # up _current_quotes, but defer the actual HTTP cancel calls until
                # after the lock is released (same pattern as kill_switch fix).
                stale_set = set(risk_result.stale_tokens) if risk_result.stale_tokens else set()
                if stale_set:
                    logger.warning("Stale tokens — cancelling orders for: %s", stale_set)
                    for stale_id in stale_set:
                        if stale_id in self._current_quotes:
                            _stale_tokens_to_cancel.append(stale_id)
                            del self._current_quotes[stale_id]

                # C-1: Snapshot tokens under lock so on_fill cannot mutate _tokens
                # between lock release and _process_token calls. Also snapshot
                # _risk_halted state (already set above).
                tokens_snapshot = dict(self._tokens)

        # I-1 (round 5): Execute stale token cancellation OUTSIDE the engine lock
        # to avoid blocking fill processing with HTTP calls.
        # R13 fix: Cooldown cleanup OUTSIDE engine lock to avoid blocking
        # fill processing with HTTP calls (same pattern as normal path).
        if _cooldown_active:
            with self._lock:
                _active_tids = set(self._current_quotes.keys())
            stale_ids, stale_tids = self._order_manager.cleanup(active_token_ids=_active_tids)
            if stale_ids:
                self._order_manager.cancel_stale_orders_on_exchange(stale_ids)
            if stale_tids:
                with self._lock:
                    for tid in stale_tids:
                        self._current_quotes.pop(tid, None)
            return

        if not _need_kill_switch and _stale_tokens_to_cancel:
            for stale_id in _stale_tokens_to_cancel:
                try:
                    self._order_manager.cancel_all_for_token(stale_id)
                except Exception:
                    logger.exception("Failed to cancel orders for stale token %s", stale_id)

        # I-4: Execute kill_switch (HTTP cancel_all) OUTSIDE the engine lock
        if _need_kill_switch:
            self._risk.kill_switch(self._order_manager)
            self._latency.record("tick", time.monotonic() - t0)
            return

        # C-1: Cleanup old terminal orders and collect stale LIVE/PENDING IDs
        # in one call. cleanup() returns stale order IDs before removing them
        # from tracking, so we can cancel them on the exchange.
        # Pass active token IDs so orders the engine intentionally placed
        # (and is tracking in _current_quotes) are not treated as stale.
        with self._lock:
            _active_tids = set(self._current_quotes.keys())
        stale_ids, stale_tids = self._order_manager.cleanup(active_token_ids=_active_tids)
        if stale_ids:
            # Cancel stale orders on exchange OUTSIDE the engine lock (C-3).
            self._order_manager.cancel_stale_orders_on_exchange(stale_ids)
        # Clear _current_quotes for stale tokens so the engine re-places orders
        # on the next tick instead of thinking it still has live quotes.
        if stale_tids:
            with self._lock:
                for tid in stale_tids:
                    self._current_quotes.pop(tid, None)

        # C-1 (runner): If the quoting gate is closed (reconnect-cancel in
        # progress), skip all quoting this tick — both pending requotes AND
        # regular token processing — to prevent placing orders that the
        # cancel thread would then remove.
        if not self._quoting_gate.is_set():
            logger.info("Quoting gate closed (reconnect-cancel in progress) — skipping tick")
            self._latency.record("tick", time.monotonic() - t0)
            return

        # C-3: Process pending requotes from on_fill before the regular tick.
        # Drain the set under lock, then requote each token individually
        # (each _requote_token call acquires lock internally).
        pending: set[str] = set()
        with self._lock:
            pending = set(self._pending_requotes)
            self._pending_requotes.clear()
        # C-6: Track tokens requoted during pending drain to avoid double requote
        requoted_this_tick: set[str] = set()
        for pending_token in pending:
            if pending_token in stale_set:
                continue
            if pending_token in tokens_snapshot:
                try:
                    self._requote_token(pending_token)
                    requoted_this_tick.add(pending_token)
                except Exception:
                    logger.exception("Error processing pending requote for %s", pending_token)

        # 2. For each token: fair value -> position -> quote -> requote
        for token_id, tick_size in tokens_snapshot.items():
            try:
                # Skip stale tokens — they were already cancelled above
                if token_id in stale_set:
                    logger.debug("Skipping stale token %s", token_id)
                    continue
                # C-6: Skip tokens already requoted during pending drain
                if token_id in requoted_this_tick:
                    continue
                self._process_token(token_id, tick_size)
            except Exception:
                logger.exception("Error processing token %s", token_id)

        # 3. Log state summary
        # I-2: Pass tokens_snapshot (computed under lock) to avoid reading
        # self._tokens without the lock, which could cause RuntimeError if
        # a tick-size change callback modifies _tokens during iteration.
        self._log_state(tokens_snapshot)

        # Record tick latency outside the engine lock
        self._latency.record("tick", time.monotonic() - t0)

    def _process_token(self, token_id: str, tick_size: float) -> None:
        """Process a single token in the quoting cycle. Delegates to _requote_token."""
        self._requote_token(token_id)

    def _get_orderbook_info(self, token_id: str) -> tuple[float | None, float | None, float | None]:
        """Return (market_mid, best_bid, best_ask) from the orderbook manager."""
        if self._orderbook_manager is None:
            return None, None, None
        book = self._orderbook_manager.get(token_id)
        if book is None:
            return None, None, None
        return book.get_bbo()

    def _requote_token(self, token_id: str) -> None:
        """Thread-safe requoting for a single token.

        Called from ``_tick()`` via ``_process_token``.  Acquires ``_lock``.
        """
        with self._lock:
            self._requote_token_unlocked(token_id)

    def _is_accumulating(self, token_id: str) -> bool:
        """Check whether the engine is in accumulation mode for a token.

        Caller must hold ``_lock``.  This method reads and mutates
        ``_accumulation_start`` and ``_accumulation_exhausted``.

        Returns True when accumulation is enabled, we are not in dry-run
        mode, the current per-token position is below the target initial
        position, and the accumulation timeout has not expired.

        Uses raw position.size per token (not net exposure) because
        accumulation builds per-token inventory so you can quote both
        bid AND ask. Net exposure is irrelevant here -- you need physical
        shares of each token to sell.
        """
        if not self._config.accumulation_enabled:
            return False
        if self._config.dry_run:
            return False
        # I-2: Permanently skip tokens that exhausted accumulation timeout
        if token_id in self._accumulation_exhausted:
            return False
        position = self._inventory.get_position(token_id)
        if position.size >= self._config.target_initial_position:
            # Target reached — clear accumulation start time
            self._accumulation_start.pop(token_id, None)
            return False

        # I-3: Track when accumulation started and enforce timeout.
        # A timeout of 0 means no limit — accumulation continues until filled.
        now = time.monotonic()
        if token_id not in self._accumulation_start:
            self._accumulation_start[token_id] = now
        timeout = self._config.accumulation_timeout_seconds
        if timeout > 0:
            elapsed = now - self._accumulation_start[token_id]
            if elapsed > timeout:
                logger.warning(
                    "Accumulation timeout for %s: %.0fs elapsed (limit %ds), pos=%.1f/%.1f — "
                    "permanently switching to normal quoting",
                    token_id, elapsed, timeout,
                    position.size, self._config.target_initial_position,
                )
                # I-2: Mark as exhausted so we don't retry accumulation cycles
                # for this token. Clears only on engine restart.
                self._accumulation_exhausted.add(token_id)
                del self._accumulation_start[token_id]
                return False

        return True

    def _requote_token_unlocked(self, token_id: str) -> None:
        """Requote logic — caller must hold ``_lock``.

        C-5: Sets _requoting flag for this token to prevent concurrent
        requotes from on_fill. If on_fill sees the flag, it skips
        adding to _pending_requotes — this requote will use fresh data.
        """
        if self._risk_halted:
            return

        tick_size = self._tokens.get(token_id)
        if tick_size is None:
            return

        # C-5: Mark token as being requoted to prevent concurrent requotes.
        # Must be cleared in the finally block below.
        self._requoting.add(token_id)
        try:
            self._requote_token_unlocked_inner(token_id, tick_size)
        finally:
            self._requoting.discard(token_id)

    def _requote_token_unlocked_inner(self, token_id: str, tick_size: float) -> None:
        """Inner requote logic, called by _requote_token_unlocked with _requoting guard."""
        fair_value = self._fair_value.get_fair_value(token_id)
        if fair_value is None:
            logger.debug("No fair value for %s, skipping", token_id)
            return

        # Keep mark prices up to date on every tick for accurate unrealized PnL
        self._inventory.update_mark_price(token_id, fair_value)

        position = self._inventory.get_position(token_id)

        # For paired tokens, use net position for inventory skew
        complement_id = self._pairs.get(token_id)
        if complement_id is not None:
            complement_pos = self._inventory.get_position(complement_id)
            net_position = position.size - complement_pos.size
        else:
            net_position = position.size

        _accum_pos = self._inventory.get_position(token_id)
        logger.info(
            "TRACE %s: fv=%.4f pos=%.1f net=%.1f accum=%s accum_pos=%.6f target=%.6f exhausted=%s",
            token_id[-10:], fair_value, position.size, net_position,
            self._is_accumulating(token_id),
            _accum_pos.size, self._config.target_initial_position,
            token_id in self._accumulation_exhausted,
        )

        # Accumulation branch: use aggressive bid-only quoting when position
        # is below the target initial position (building inventory from zero).
        if self._is_accumulating(token_id):
            _acc_mid, _acc_best_bid, _acc_best_ask = self._get_orderbook_info(token_id)
            new_quote = self._quoting.generate_accumulation_quote(
                token_id=token_id,
                fair_value=fair_value,
                tick_size=tick_size,
                best_bid=_acc_best_bid,
                best_ask=_acc_best_ask,
            )
            if new_quote is None:
                if token_id in self._current_quotes:
                    self._order_manager.cancel_all_for_token(token_id)
                    # MM-1 fix: Only clear _current_quotes after verifying cancel succeeded.
                    remaining_open = self._order_manager.get_open_orders(token_id)
                    if not remaining_open:
                        del self._current_quotes[token_id]
                    else:
                        logger.warning("Failed to cancel %d orders for %s during accumulation no-quote", len(remaining_open), token_id[-10:])
                return

            # I-11: Cap accumulation bid size to remaining amount needed.
            # Uses raw position.size (not net exposure) consistent with
            # _is_accumulating -- we need physical shares per token.
            # I-5: When remaining < MIN_ORDER_SIZE (5.0), stop accumulating
            # instead of over-buying with max(remaining, 5.0).
            if new_quote.bid_size is not None:
                remaining = self._config.target_initial_position - position.size
                token_min_acc = self._min_sizes.get(token_id, 5.0)
                if remaining < token_min_acc:
                    # I-5: Can't place a valid order for less than per-token min_order_size.
                    # Mark as exhausted so next tick transitions to normal MM quoting
                    # instead of being stuck in accumulation limbo forever.
                    logger.info(
                        "Accumulation near-complete for %s: pos=%.4f remaining=%.4f < min=%.1f — switching to MM",
                        token_id[-10:], position.size, remaining, token_min_acc,
                    )
                    self._accumulation_exhausted.add(token_id)
                    self._accumulation_start.pop(token_id, None)
                    return  # next tick will use MM quoting
                elif new_quote.bid_size > remaining:
                    new_quote = new_quote.model_copy(update={"bid_size": remaining})

            # I-5: After capping, if no sides are quotable, cancel and return
            if new_quote.bid_price is None and new_quote.ask_price is None:
                if token_id in self._current_quotes:
                    self._order_manager.cancel_all_for_token(token_id)
                    # MM-1 fix: Verify cancel succeeded before clearing quotes
                    remaining_open = self._order_manager.get_open_orders(token_id)
                    if not remaining_open:
                        del self._current_quotes[token_id]
                    else:
                        logger.warning("Failed to cancel %d orders for %s during accumulation size-cap", len(remaining_open), token_id[-10:])
                return

            current_quote = self._current_quotes.get(token_id)
            if current_quote is not None:
                # In accumulation mode, requote whenever the bid price changes
                # (even by 1 tick) to stay at top of book. The standard
                # requote_threshold_bps (50 bps) is too coarse — if someone
                # outbids us by 1 cent, we'd sit on 2nd level until FV moves.
                bid_changed = (
                    current_quote.bid_price != new_quote.bid_price
                    if current_quote.bid_price is not None and new_quote.bid_price is not None
                    else current_quote.bid_price is not new_quote.bid_price
                )
                if not bid_changed:
                    logger.debug("Accumulation quote for %s unchanged, skipping requote", token_id)
                    return

            if current_quote is not None:
                self._order_manager.cancel_all_for_token(token_id)

                # C-7: Check for remaining open orders BEFORE clearing _current_quotes.
                # If cancel failed, restore the old quote so the next tick doesn't
                # place new orders alongside still-live old ones.
                remaining_open = self._order_manager.get_open_orders(token_id)
                if remaining_open:
                    # Don't clear _current_quotes — old quote still represents live orders
                    logger.warning("Failed to cancel %d orders for %s — skipping requote", len(remaining_open), token_id[-10:])
                    return

                # Only clear after successful cancel
                self._current_quotes.pop(token_id, None)

            # I-NEW-3: Re-check position after cancel — a fill may have arrived
            # during the cancel HTTP call, making the pre-computed bid_size stale.
            # Re-cap to avoid over-accumulating by one order size.
            if new_quote.bid_size is not None:
                refreshed_pos = self._inventory.get_position(token_id)
                remaining = self._config.target_initial_position - refreshed_pos.size
                token_min_recheck = self._min_sizes.get(token_id, 5.0)
                if remaining < token_min_recheck:
                    new_quote = new_quote.model_copy(update={"bid_price": None, "bid_size": None})
                elif new_quote.bid_size > remaining:
                    new_quote = new_quote.model_copy(update={"bid_size": remaining})
                # After re-cap, if no sides are quotable, bail out
                if new_quote.bid_price is None and new_quote.ask_price is None:
                    return

            # C-BUG-2: When accumulation_cross_spread is enabled, bids are meant to
            # cross the spread for immediate fill. post_only=True would reject these
            # orders on the exchange. Use post_only=False for cross-spread accumulation.
            acc_post_only = not self._config.accumulation_cross_spread

            # C-BUG-1: Only pass neg_risk when True. py_clob_client treats False as
            # falsy/"not set" and falls back to an API lookup. When neg_risk is False,
            # omitting it produces the correct default behavior without the extra call.
            acc_neg_risk = self._neg_risk.get(token_id, False)
            acc_neg_risk_map = {token_id: True} if acc_neg_risk else {}

            placed = self._order_manager.place_quotes(
                [new_quote],
                token_tick_sizes={token_id: str(tick_size)},
                token_neg_risk=acc_neg_risk_map,
                token_min_sizes={token_id: self._min_sizes.get(token_id, 5.0)},
                post_only=acc_post_only,
            )
            live_orders = [o for o in placed if o.status == OrderStatus.LIVE]
            if live_orders:
                # C-3: Reconcile stored quote to reflect actually placed sides
                acc_live_sides = {o.side for o in live_orders}
                stored_acc_quote = new_quote
                if Side.BUY not in acc_live_sides and new_quote.bid_price is not None:
                    stored_acc_quote = stored_acc_quote.model_copy(update={"bid_price": None, "bid_size": None})
                if Side.SELL not in acc_live_sides and new_quote.ask_price is not None:
                    stored_acc_quote = stored_acc_quote.model_copy(update={"ask_price": None, "ask_size": None})
                self._current_quotes[token_id] = stored_acc_quote
                self._last_requote_time[token_id] = time.monotonic()
                logger.info(
                    "ACCUMULATION %s: bid=%.4f/%.1f pos=%.1f target=%.1f post_only=%s",
                    token_id,
                    stored_acc_quote.bid_price or 0,
                    stored_acc_quote.bid_size or 0,
                    self._inventory.get_position(token_id).size,
                    self._config.target_initial_position,
                    acc_post_only,
                )
            return

        market_mid, best_bid, best_ask = self._get_orderbook_info(token_id)

        # Compute divergence EMA when orderbook data is available
        divergence_ema_bps: float | None = None
        if market_mid is not None:
            divergence_bps = abs(fair_value - market_mid) * 10_000.0

            prev_ema = self._divergence_emas.get(token_id)
            if prev_ema is None:
                # Seed with first observation
                ema = divergence_bps
            else:
                ema = compute_ema(prev_ema, divergence_bps, self._config.divergence_ema_alpha)
            self._divergence_emas[token_id] = ema
            divergence_ema_bps = ema

            # Record in tracker (tracker has its own lock, safe under engine lock)
            self._divergence_tracker.record(token_id, divergence_bps, ema)

        # Allow crossing the spread when holding large inventory to reduce position
        position_ratio = abs(net_position) / self._config.max_position if self._config.max_position > 0 else 0
        allow_cross = position_ratio >= 0.3

        new_quote = self._quoting.generate_quotes(
            token_id=token_id,
            fair_value=fair_value,
            position=net_position,
            tick_size=tick_size,
            market_mid=market_mid,
            best_bid=best_bid,
            best_ask=best_ask,
            divergence_ema_bps=divergence_ema_bps,
            allow_cross=allow_cross,
        )

        if new_quote is None:
            # Inverted or no-quote situation — cancel existing
            if token_id in self._current_quotes:
                self._order_manager.cancel_all_for_token(token_id)
                # MM-1 fix: Verify cancel succeeded before clearing quotes
                remaining_open = self._order_manager.get_open_orders(token_id)
                if not remaining_open:
                    del self._current_quotes[token_id]
                else:
                    logger.warning("Failed to cancel %d orders for %s during no-quote transition", len(remaining_open), token_id[-10:])
            return

        # Polymarket constraint (live only): can only SELL tokens we hold.
        # Cap ask size to actual position (not net — raw token balance).
        if not self._config.dry_run and new_quote.ask_price is not None and position.size <= 0:
            new_quote = Quote(
                token_id=new_quote.token_id,
                bid_price=new_quote.bid_price,
                bid_size=new_quote.bid_size,
                ask_price=None,
                ask_size=None,
                market_mid=new_quote.market_mid,
                timestamp=new_quote.timestamp,
            )
            if new_quote.bid_price is None:
                # No sides to quote at all
                if token_id in self._current_quotes:
                    self._order_manager.cancel_all_for_token(token_id)
                    # MM-1 fix: Verify cancel succeeded before clearing quotes
                    remaining_open = self._order_manager.get_open_orders(token_id)
                    if not remaining_open:
                        del self._current_quotes[token_id]
                    else:
                        logger.warning("Failed to cancel %d orders for %s during ask-suppression", len(remaining_open), token_id[-10:])
                return

        # Cap ask size to position when we hold some but less than order_size
        if not self._config.dry_run and position.size > 0 and new_quote.ask_size is not None:
            if new_quote.ask_size > position.size:
                new_quote = new_quote.model_copy(update={"ask_size": position.size})
            token_min = self._min_sizes.get(token_id, 5.0)
            if new_quote.ask_size < token_min:
                # Only suppress if the constraint is position cap (can't sell shares
                # you don't have). If size is below min due to divergence reduction
                # but position is large enough, clamp up to min instead.
                if position.size < token_min:
                    new_quote = new_quote.model_copy(update={"ask_price": None, "ask_size": None})
                else:
                    new_quote = new_quote.model_copy(update={"ask_size": token_min})

        # C-1: Clamp bid size to per-token minimum. Divergence size_mult can
        # reduce bid_size below min_order_size. Instead of dropping the bid
        # entirely (which stops all quoting during sustained divergence),
        # clamp up to min_order_size. The hard stop at divergence_max_bps
        # still protects against extreme divergence.
        if not self._config.dry_run and new_quote.bid_size is not None:
            token_min_bid = self._min_sizes.get(token_id, 5.0)
            if new_quote.bid_size < token_min_bid:
                new_quote = new_quote.model_copy(update={"bid_size": token_min_bid})

        # Suppress bid when raw individual position exceeds max_position.
        # For paired tokens, net position may be small (e.g. 163 Celtics vs 194
        # Nuggets = net -31), but each individual token can grow unbounded since
        # the quoting skew only sees net. This caps gross capital usage by
        # stopping buys when a single token is already at max.
        if position.size >= self._config.max_position and new_quote.bid_size is not None:
            logger.info(
                "Suppressing bid for %s: raw position %.1f >= max %.1f (net=%.1f)",
                token_id[-10:], position.size, self._config.max_position, net_position,
            )
            new_quote = new_quote.model_copy(update={"bid_price": None, "bid_size": None})

        # After min-size filtering, check if we still have any quotable sides
        if new_quote.bid_price is None and new_quote.ask_price is None:
            if token_id in self._current_quotes:
                self._order_manager.cancel_all_for_token(token_id)
                # MM-1 fix: Verify cancel succeeded before clearing quotes
                remaining_open = self._order_manager.get_open_orders(token_id)
                if not remaining_open:
                    del self._current_quotes[token_id]
                else:
                    logger.warning("Failed to cancel %d orders for %s during min-size filter", len(remaining_open), token_id[-10:])
            return

        # Check if we should requote
        current_quote = self._current_quotes.get(token_id)
        if current_quote is not None and not self._quoting.should_requote(current_quote, new_quote):
            logger.debug("Quote for %s unchanged, skipping requote", token_id)
            return

        # R21-C1: Pre-flight check for stale orders on first-time placement.
        # After a failed reconnect cancel, _current_quotes is empty but old
        # orders may still be live on the exchange.  Without this guard,
        # place_quotes would create duplicates alongside the still-live orders.
        if current_quote is None:
            stale_orders = self._order_manager.get_open_orders(token_id)
            if stale_orders:
                logger.warning(
                    "R21-C1: Found %d open orders for %s on first-time placement — cancelling first",
                    len(stale_orders), token_id[-10:],
                )
                self._order_manager.cancel_all_for_token(token_id)
                remaining = self._order_manager.get_open_orders(token_id)
                if remaining:
                    logger.warning(
                        "Failed to cancel %d stale orders for %s — skipping placement",
                        len(remaining), token_id[-10:],
                    )
                    return

        # Cancel old orders and place new ones
        if current_quote is not None:
            self._order_manager.cancel_all_for_token(token_id)

            # C-7: Check for remaining open orders BEFORE clearing _current_quotes.
            # If cancel failed, restore the old quote so the next tick doesn't
            # place new orders alongside still-live old ones.
            remaining_open = self._order_manager.get_open_orders(token_id)
            if remaining_open:
                # Don't clear _current_quotes — old quote still represents live orders
                logger.warning("Failed to cancel %d orders for %s — skipping requote", len(remaining_open), token_id[-10:])
                return

            # Only clear after successful cancel
            self._current_quotes.pop(token_id, None)

            # I-5: Re-verify tick_size after cancel — it may have changed via
            # tick_size_change callback while the cancel HTTP call was in flight.
            refreshed_tick = self._tokens.get(token_id)
            if refreshed_tick is None:
                logger.warning("Token %s unregistered during requote — aborting", token_id)
                return
            if refreshed_tick != tick_size:
                logger.info("tick_size changed for %s during requote: %.6f -> %.6f", token_id, tick_size, refreshed_tick)
                tick_size = refreshed_tick
                # C-2 fix: Re-generate quote with new tick_size. The old
                # new_quote was rounded to the old tick grid, which may produce
                # off-grid prices for the new tick_size (e.g., 0.505 for
                # tick_size=0.01 after change to 0.1). Re-compute entirely.
                # R16-C2: Re-fetch position after cancel for accurate skew.
                # A fill during the cancel HTTP call changes the real position,
                # but the old net_position was computed before the cancel.
                pos_refreshed_ts = self._inventory.get_position(token_id)
                if complement_id is not None:
                    comp_pos_refreshed = self._inventory.get_position(complement_id)
                    net_position = pos_refreshed_ts.size - comp_pos_refreshed.size
                else:
                    net_position = pos_refreshed_ts.size
                # Recompute allow_cross with refreshed position
                position_ratio = abs(net_position) / self._config.max_position if self._config.max_position > 0 else 0
                allow_cross = position_ratio >= 0.3
                new_quote = self._quoting.generate_quotes(
                    token_id=token_id,
                    fair_value=fair_value,
                    position=net_position,
                    tick_size=tick_size,
                    market_mid=market_mid,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    divergence_ema_bps=divergence_ema_bps,
                    allow_cross=allow_cross,
                )
                if new_quote is None:
                    return
                # Re-apply live-mode position constraints
                if not self._config.dry_run and new_quote.ask_price is not None:
                    pos_check = self._inventory.get_position(token_id)
                    if pos_check.size <= 0:
                        new_quote = new_quote.model_copy(update={"ask_price": None, "ask_size": None})
                        if new_quote.bid_price is None:
                            return
                    elif new_quote.ask_size is not None and new_quote.ask_size > pos_check.size:
                        new_quote = new_quote.model_copy(update={"ask_size": pos_check.size})
                        token_min_ts = self._min_sizes.get(token_id, 5.0)
                        if new_quote.ask_size < token_min_ts:
                            new_quote = new_quote.model_copy(update={"ask_price": None, "ask_size": None})
                if not self._config.dry_run and new_quote.bid_size is not None:
                    token_min_bid_ts = self._min_sizes.get(token_id, 5.0)
                    if new_quote.bid_size < token_min_bid_ts:
                        new_quote = new_quote.model_copy(update={"bid_price": None, "bid_size": None})
                if new_quote.bid_price is None and new_quote.ask_price is None:
                    return

            # C-8 block moved below — now runs unconditionally (first-time + requote).

        # Suppress bid when raw individual position exceeds max_position
        # (same check as primary path, applied after tick_size re-computation).
        refreshed_raw = self._inventory.get_position(token_id)
        if refreshed_raw.size >= self._config.max_position and new_quote.bid_size is not None:
            logger.info(
                "Suppressing bid for %s: raw position %.1f >= max %.1f (net=%.1f)",
                token_id[-10:], refreshed_raw.size, self._config.max_position, net_position,
            )
            new_quote = new_quote.model_copy(update={"bid_price": None, "bid_size": None})

        # R16-C1: Position re-check runs for BOTH first-time placement and
        # requotes. Previously nested inside `if current_quote is not None`,
        # skipping first-time placements where a fill could arrive between
        # the initial position fetch and place_quotes, producing a stale
        # ask_size that sells more than we hold.
        if not self._config.dry_run and new_quote.ask_price is not None:
            refreshed_pos = self._inventory.get_position(token_id)
            if refreshed_pos.size <= 0:
                new_quote = new_quote.model_copy(update={"ask_price": None, "ask_size": None})
                if new_quote.bid_price is None:
                    return
            elif new_quote.ask_size is not None and new_quote.ask_size > refreshed_pos.size:
                new_quote = new_quote.model_copy(update={"ask_size": refreshed_pos.size})
                token_min_refreshed = self._min_sizes.get(token_id, 5.0)
                if new_quote.ask_size < token_min_refreshed:
                    new_quote = new_quote.model_copy(update={"ask_price": None, "ask_size": None})

        # C-BUG-1: Only pass neg_risk when True to avoid py_clob_client
        # treating False as falsy and making an unnecessary API lookup.
        norm_neg_risk = self._neg_risk.get(token_id, False)
        norm_neg_risk_map = {token_id: True} if norm_neg_risk else {}

        # When holding large inventory, allow the ask to cross the spread
        # (post_only=False) so we can take the bid and actually reduce position.
        # The bid side won't cross because skew pushes it far below the ask.
        position_ratio = abs(net_position) / self._config.max_position if self._config.max_position > 0 else 0
        mm_post_only = position_ratio < 0.3

        placed = self._order_manager.place_quotes(
            [new_quote],
            token_tick_sizes={token_id: str(tick_size)},
            token_neg_risk=norm_neg_risk_map,
            token_min_sizes={token_id: self._min_sizes.get(token_id, 5.0)},
            post_only=mm_post_only,
        )
        live_orders = [o for o in placed if o.status == OrderStatus.LIVE]
        if live_orders:
            # C-3: Reconcile _current_quotes to reflect what was actually placed.
            # place_quotes may have silently filtered one side (e.g. bid below
            # min_order_size). Store only the sides that actually have LIVE orders
            # so should_requote doesn't skip retrying the missing side.
            live_sides = {o.side for o in live_orders}
            stored_quote = new_quote
            if Side.BUY not in live_sides and new_quote.bid_price is not None:
                stored_quote = stored_quote.model_copy(update={"bid_price": None, "bid_size": None})
            if Side.SELL not in live_sides and new_quote.ask_price is not None:
                stored_quote = stored_quote.model_copy(update={"ask_price": None, "ask_size": None})
            self._current_quotes[token_id] = stored_quote
            self._last_requote_time[token_id] = time.monotonic()
            logger.info(
                "Requoted %s: bid=%.4f/%s ask=%.4f/%s",
                token_id,
                stored_quote.bid_price or 0,
                stored_quote.bid_size or 0,
                stored_quote.ask_price or 0,
                stored_quote.ask_size or 0,
            )

    def _log_state(self, tokens_snapshot: Dict[str, float]) -> None:
        """Log a summary of the current engine state.

        I-2: Accepts tokens_snapshot (computed under engine._lock in _tick)
        instead of reading self._tokens directly, which would be unsafe
        outside the lock — a tick-size change callback could mutate _tokens
        during iteration, causing RuntimeError.

        Args:
            tokens_snapshot: Snapshot of self._tokens taken under lock.
        """
        total_exposure = self._inventory.get_total_exposure()
        pnl = self._inventory.get_pnl()
        open_orders = len(self._order_manager.get_open_orders())
        fair_values = self._fair_value.get_all_fair_values()

        logger.info(
            "State: tokens=%d open_orders=%d exposure=%.1f pnl=%.2f fair_values=%s",
            len(tokens_snapshot),
            open_orders,
            total_exposure,
            pnl,
            {k: f"{v:.4f}" for k, v in fair_values.items()},
        )

        # Latency summary
        logger.info(self._latency.format_summary())

        # Divergence summary
        logger.info(self._divergence_tracker.format_summary())

        # Quote age
        now = time.monotonic()
        quote_ages = {}
        for token_id in tokens_snapshot:
            last = self._last_requote_time.get(token_id)
            if last is not None:
                quote_ages[token_id] = f"{now - last:.1f}s"
            else:
                quote_ages[token_id] = "never"
        logger.info("Quote age: %s", quote_ages)

    def _shutdown(self) -> None:
        """Clean up on shutdown.

        Acquires ``_lock`` to prevent concurrent ``on_fill`` from placing
        new orders after the cancel-all sweep.

        Known limitation: cancel_all() makes an HTTP call while holding
        ``_lock``, blocking fill processing for the duration.  Fills
        arriving during shutdown are queued and effectively lost when the
        engine exits.  If the process restarts, position re-seeding from
        trade history will reconcile.
        """
        logger.info("Shutting down — cancelling all orders")
        with self._lock:
            self._risk_halted = True
            try:
                self._order_manager.cancel_all()
            except KeyboardInterrupt:
                # R15-I2 fix: SIGINT during cancel_all leaves orders on exchange.
                # Retry once so orders are cleaned up before shutdown completes.
                logger.warning("Interrupted during shutdown cancel — retrying")
                try:
                    self._order_manager.cancel_all()
                except Exception:
                    logger.exception("Retry cancel also failed — orders may remain on exchange")
            except Exception:
                logger.exception("Error during shutdown cancel")
            self._current_quotes.clear()
        logger.info("MarketMakingEngine shut down")
