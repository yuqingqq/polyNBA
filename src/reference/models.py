"""Unified data models for the reference price source module."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MarketType(str, Enum):
    """Types of NBA markets we track."""
    CHAMPIONSHIP = "championship"
    GAME_ML = "game_ml"
    SPREAD = "spread"
    TOTAL = "total"
    CONFERENCE = "conference"
    MVP = "mvp"
    PLAYER_PROP = "player_prop"


class ExternalOdds(BaseModel):
    """A single outcome's odds from an external bookmaker."""
    team: str = Field(..., description="Team or outcome name (e.g., 'Los Angeles Lakers')")
    american_odds: Optional[int] = Field(None, description="American odds (e.g., +350, -150)")
    decimal_odds: Optional[float] = Field(None, description="Decimal odds (e.g., 4.50)")
    implied_probability: float = Field(..., description="Raw implied probability from odds (includes vig)")
    bookmaker: str = Field(..., description="Source bookmaker name (e.g., 'DraftKings')")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    point: Optional[float] = Field(None, description="Point spread or total line (e.g., -4.5, 215.5)")
    description: Optional[str] = Field(None, description="Additional description (e.g., 'Over 215.5')")
    market_key: Optional[str] = Field(None, description="Market type key ('h2h', 'spreads', 'totals', 'outrights')")


class PolymarketContract(BaseModel):
    """A single Polymarket binary contract."""
    token_id: str = Field(..., description="CLOB token ID for trading")
    condition_id: str = Field(..., description="CTF condition ID")
    slug: str = Field("", description="Market URL slug")
    question: str = Field(..., description="Market question text")
    outcome: str = Field(..., description="Outcome this contract represents (e.g., 'Yes', 'Lakers')")
    current_price: Optional[float] = Field(None, description="Current mid price (0-1)")
    volume: Optional[float] = Field(None, description="Total volume traded in USD")
    event_slug: Optional[str] = Field(None, description="Parent event slug")
    end_date: Optional[str] = Field(None, description="Market end/resolution date")
    # C-3: Per-market tick size from Gamma API (minimum_tick_size field)
    minimum_tick_size: float = Field(0.01, description="Minimum price increment for this market")
    # C-2: Per-market neg_risk flag from Gamma API
    neg_risk: bool = Field(False, description="Whether this market uses neg_risk exchange")
    # C-4: Per-market minimum order size
    min_order_size: float = Field(5.0, description="Minimum order size for this market")


class MappedMarket(BaseModel):
    """Pairs an external market with its corresponding Polymarket contract(s)."""
    external_event_id: Optional[str] = Field(None, description="External API event ID")
    external_odds: list[ExternalOdds] = Field(default_factory=list, description="All outcomes from external source")
    polymarket_contracts: list[PolymarketContract] = Field(default_factory=list, description="Matched Polymarket contracts")
    market_type: MarketType = Field(..., description="Type of market")
    event_name: Optional[str] = Field(None, description="Human-readable event name")
    commence_time: Optional[datetime] = Field(None, description="Event start time (for games)")
    mapping_confidence: float = Field(1.0, description="Confidence in the mapping (0-1)")


class ReferencePrice(BaseModel):
    """A fair reference price for a specific Polymarket contract."""
    token_id: str = Field(..., description="Polymarket CLOB token ID")
    fair_probability: float = Field(..., description="Vig-removed fair probability (0.01-0.99)")
    raw_probability: float = Field(..., description="Raw implied probability before vig removal")
    source: str = Field(..., description="Source of the odds (e.g., 'the-odds-api:pinnacle')")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    market_type: MarketType = Field(..., description="Type of market")
    bookmaker: Optional[str] = Field(None, description="Primary bookmaker used")
    vig_removal_method: str = Field("proportional", description="Vig removal method used")
