"""Risk manager — pre-trade and continuous risk checks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Set

from .config import MMConfig
from .models import RiskCheckResult

if TYPE_CHECKING:
    from .fair_value import FairValueEngine
    from .inventory import InventoryManager
    from .order_manager import OrderManager

logger = logging.getLogger(__name__)


class RiskManager:
    """Evaluates risk conditions and can trigger a kill switch.

    The risk manager inspects inventory limits, PnL thresholds, and
    data freshness. When any condition is breached it signals the engine
    to halt quoting.
    """

    def __init__(self, config: MMConfig) -> None:
        self._config = config

    def check_all(
        self,
        inventory: InventoryManager,
        fair_value_engine: FairValueEngine,
        active_token_ids: Optional[Set[str]] = None,
    ) -> RiskCheckResult:
        """Run all risk checks and return the combined result.

        Args:
            inventory: The inventory manager with current positions.
            fair_value_engine: The fair value engine to check for stale data.
            active_token_ids: Set of token IDs that are actively being
                traded.  When provided, stale data checks are limited to
                these tokens so that an unrelated stale token does not
                halt all trading.

        Returns:
            A RiskCheckResult summarizing the risk state.
        """
        result = RiskCheckResult()
        cfg = self._config

        # 1. Per-token position limits
        if not inventory.is_within_limits(cfg):
            result.max_position_breached = True
            result.reasons.append("Per-token position limit breached")

        # 2. Total exposure limit
        total_exposure = inventory.get_total_exposure()
        if total_exposure > cfg.max_total_position:
            result.max_total_breached = True
            result.reasons.append(
                f"Total exposure {total_exposure:.1f} exceeds limit {cfg.max_total_position:.1f}"
            )

        # 3. Max loss check
        pnl = inventory.get_pnl()
        if pnl < -cfg.max_loss:
            result.max_loss_breached = True
            result.reasons.append(
                f"PnL {pnl:.2f} below max loss limit -{cfg.max_loss:.2f}"
            )

        # 4. Stale data check — collect stale token IDs for per-token handling.
        #    Staleness no longer triggers a full halt; the engine skips stale
        #    tokens individually so that one finished game doesn't kill all quoting.
        tokens_to_check = active_token_ids if active_token_ids is not None else None
        fair_values = fair_value_engine.get_all_fair_values()

        if tokens_to_check is not None:
            # Only check the tokens we are registered to trade
            relevant_fvs = {t: v for t, v in fair_values.items() if t in tokens_to_check}
            # If we have registered tokens but no fair values for any of them, that's stale
            if not relevant_fvs and tokens_to_check:
                result.stale_data = True
                result.stale_tokens = list(tokens_to_check)
                result.reasons.append("No fair value data for any active token")
            else:
                for token_id in relevant_fvs:
                    if fair_value_engine.is_stale(token_id, cfg.stale_data_timeout_seconds):
                        result.stale_data = True
                        result.stale_tokens.append(token_id)
                        result.reasons.append(f"Stale data for token {token_id}")
        else:
            # Legacy behavior: check all fair values
            if not fair_values:
                result.stale_data = True
                result.reasons.append("No fair value data available")
            else:
                for token_id in fair_values:
                    if fair_value_engine.is_stale(token_id, cfg.stale_data_timeout_seconds):
                        result.stale_data = True
                        result.stale_tokens.append(token_id)
                        result.reasons.append(f"Stale data for token {token_id}")

        # Determine overall halt — staleness is handled per-token, not as a halt
        result.should_halt = (
            result.max_position_breached
            or result.max_total_breached
            or result.max_loss_breached
        )

        if result.should_halt:
            logger.warning("Risk check HALT: %s", "; ".join(result.reasons))
        else:
            logger.debug("Risk check OK: exposure=%.1f pnl=%.2f", total_exposure, pnl)

        return result

    def kill_switch(self, order_manager: OrderManager) -> None:
        """Emergency kill switch — cancel all orders immediately.

        Args:
            order_manager: The order manager to use for cancellation.
        """
        logger.critical("KILL SWITCH ACTIVATED — cancelling all orders")
        try:
            cancelled = order_manager.cancel_all()
            logger.critical("Kill switch: cancelled %d orders", cancelled)
        except Exception:
            logger.exception("Kill switch: failed to cancel orders")
