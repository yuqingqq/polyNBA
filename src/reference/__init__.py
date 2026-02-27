"""Reference price source module for NBA market making.

Connects to external odds APIs, scans Polymarket for NBA markets,
maps them together, and adapts odds into Polymarket-compatible probabilities.
"""

from .models import (
    ExternalOdds,
    MappedMarket,
    MarketType,
    PolymarketContract,
    ReferencePrice,
)
from .staleness import StalenessChecker
from .vig_removal import (
    american_to_probability,
    compute_overround,
    decimal_to_probability,
    proportional_vig_removal,
    shin_vig_removal,
)

__all__ = [
    "ExternalOdds",
    "MappedMarket",
    "MarketType",
    "PolymarketContract",
    "ReferencePrice",
    "StalenessChecker",
    "american_to_probability",
    "compute_overround",
    "decimal_to_probability",
    "proportional_vig_removal",
    "shin_vig_removal",
]
