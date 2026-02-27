"""Raw API response models for The Odds API.

These models match the JSON structure returned by https://the-odds-api.com/
so we can validate and parse responses cleanly.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class OddsApiOutcome(BaseModel):
    """A single outcome within a market from The Odds API."""
    name: str = Field(..., description="Outcome name (e.g., 'Los Angeles Lakers')")
    price: int | float = Field(..., description="Odds value (American or decimal depending on request)")
    point: Optional[float] = Field(None, description="Point line for spreads/totals (e.g., -4.5, 215.5)")
    description: Optional[str] = Field(None, description="Additional description")


class OddsApiMarket(BaseModel):
    """A single market type (h2h, spreads, totals, outrights) from a bookmaker."""
    key: str = Field(..., description="Market key (e.g., 'h2h', 'spreads', 'totals', 'outrights')")
    last_update: Optional[datetime] = Field(None, description="When this bookmaker last updated odds")
    outcomes: list[OddsApiOutcome] = Field(default_factory=list)


class OddsApiBookmaker(BaseModel):
    """A single bookmaker's odds for an event."""
    key: str = Field(..., description="Bookmaker key (e.g., 'draftkings', 'fanduel')")
    title: str = Field("", description="Bookmaker display name")
    last_update: Optional[datetime] = Field(None, description="Last update timestamp")
    markets: list[OddsApiMarket] = Field(default_factory=list)


class OddsApiEvent(BaseModel):
    """A single event (game or futures market) from The Odds API."""
    id: str = Field(..., description="Unique event identifier")
    sport_key: str = Field(..., description="Sport key (e.g., 'basketball_nba')")
    sport_title: str = Field("", description="Sport display title")
    commence_time: Optional[datetime] = Field(None, description="Event start time (ISO 8601)")
    home_team: Optional[str] = Field(None, description="Home team name (null for futures)")
    away_team: Optional[str] = Field(None, description="Away team name (null for futures)")
    bookmakers: list[OddsApiBookmaker] = Field(default_factory=list)


class OddsApiSport(BaseModel):
    """A sport available on The Odds API."""
    key: str = Field(..., description="Sport key (e.g., 'basketball_nba')")
    group: str = Field("", description="Sport group (e.g., 'Basketball')")
    title: str = Field("", description="Sport display title (e.g., 'NBA')")
    description: str = Field("", description="Sport description")
    active: bool = Field(True, description="Whether the sport is currently active")
    has_outrights: bool = Field(False, description="Whether the sport has outright/futures markets")


class OddsApiResponse(BaseModel):
    """Wrapper for API response metadata (from headers)."""
    events: list[OddsApiEvent] = Field(default_factory=list)
    requests_remaining: Optional[int] = Field(None, description="API calls remaining this month")
    requests_used: Optional[int] = Field(None, description="API calls used this month")
