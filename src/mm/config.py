"""Strategy configuration for the market making engine."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)


class MMConfig(BaseModel):
    """Market making strategy parameters."""

    # Spread and sizing
    spread_bps: int = 200  # 2% total spread
    order_size: float = 50.0  # size per order

    # Position limits
    max_position: float = 500.0  # max position per contract
    max_total_position: float = 2000.0  # max total across all contracts

    # Inventory management
    inventory_skew_factor: float = 0.5  # how aggressively to skew quotes

    # Requoting
    requote_threshold_bps: int = 50  # requote if fair value moves by this much

    # Staleness
    stale_data_timeout_seconds: int = 60  # seconds before data is stale

    # Risk
    max_loss: float = 200.0  # max cumulative loss before halt

    # Timing
    update_interval_seconds: float = 5.0  # seconds between quote updates

    # Operating mode
    dry_run: bool = True  # True = log only, no real orders

    # Price precision
    tick_size: float = 0.01  # standard tick size

    # Quote depth
    num_levels: int = 1  # number of price levels per side

    # Divergence (fair value vs market mid)
    divergence_widen_bps: int = 500  # |fv - mid| at which spread doubles (5%)
    divergence_max_bps: int = 1500  # |fv - mid| beyond which quoting stops (15%)
    divergence_size_reduction: float = 0.5  # size multiplier at divergence_widen_bps
    divergence_ema_alpha: float = 0.3  # EMA weight on new observation (0.3 = 30%)
    divergence_overrides: dict[str, dict[str, int]] = {}  # per-token threshold overrides

    # Accumulation phase
    accumulation_enabled: bool = True
    accumulation_spread_bps: int = 50
    accumulation_cross_spread: bool = False
    target_initial_position: float = 30.0
    accumulation_size_multiplier: float = 1.5
    accumulation_max_price_cents: int = 95
    accumulation_timeout_seconds: int = 600  # 10 min — give up accumulating after this

    @field_validator("max_position")
    @classmethod
    def _max_position_must_be_positive(cls, v: float) -> float:
        if v < 1:
            raise ValueError("max_position must be >= 1 to avoid division by zero in skew calculation")
        return v

    @field_validator("divergence_size_reduction")
    @classmethod
    def _divergence_size_reduction_range(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("divergence_size_reduction must be in [0.0, 1.0]")
        return v

    @field_validator("divergence_ema_alpha")
    @classmethod
    def _divergence_ema_alpha_range(cls, v: float) -> float:
        if v <= 0.0 or v > 1.0:
            raise ValueError("divergence_ema_alpha must be in (0.0, 1.0]")
        return v


def load_config(path: Union[str, Path]) -> MMConfig:
    """Load MMConfig from a JSON file, using defaults for missing fields.

    Args:
        path: Path to a JSON configuration file.

    Returns:
        An MMConfig instance with values from the file merged with defaults.
    """
    config_path = Path(path)
    if not config_path.exists():
        logger.warning("Config file %s not found, using defaults", config_path)
        return MMConfig()

    with open(config_path, "r") as f:
        data = json.load(f)

    config = MMConfig(**data)
    logger.info("Loaded config from %s: %s", config_path, config.model_dump())
    return config
