"""Pydantic models for Kalshi API responses.

These models match the JSON structure returned by Kalshi's public REST API
for market and event data.
"""

from typing import Optional

from pydantic import BaseModel, Field


class KalshiMarket(BaseModel):
    """A single market (contract) on Kalshi."""
    ticker: str = Field(..., description="Unique market ticker (e.g., 'KXNBA-26FEB27-LAL')")
    event_ticker: str = Field("", description="Parent event ticker")
    title: str = Field("", description="Market title (e.g., 'Lakers vs Celtics: Lakers win?')")
    subtitle: str = Field("", description="Market subtitle")
    status: str = Field("", description="Market status: 'open', 'closed', 'settled'")
    yes_bid: Optional[int] = Field(None, description="Best YES bid in cents (0-100)")
    yes_ask: Optional[int] = Field(None, description="Best YES ask in cents (0-100)")
    no_bid: Optional[int] = Field(None, description="Best NO bid in cents (0-100)")
    no_ask: Optional[int] = Field(None, description="Best NO ask in cents (0-100)")
    last_price: Optional[int] = Field(None, description="Last trade price in cents")
    volume: Optional[int] = Field(None, description="Total contracts traded")
    open_interest: Optional[int] = Field(None, description="Open interest")
    result: Optional[str] = Field(None, description="Settlement result if settled")


class KalshiEvent(BaseModel):
    """A Kalshi event grouping related markets."""
    event_ticker: str = Field(..., description="Unique event ticker")
    title: str = Field("", description="Event title")
    category: str = Field("", description="Event category")
    markets: list[KalshiMarket] = Field(default_factory=list, description="Markets in this event")


class KalshiMarketsResponse(BaseModel):
    """Response from GET /trade-api/v2/markets."""
    markets: list[KalshiMarket] = Field(default_factory=list)
    cursor: Optional[str] = Field(None, description="Pagination cursor for next page")


class KalshiEventsResponse(BaseModel):
    """Response from GET /trade-api/v2/events."""
    events: list[KalshiEvent] = Field(default_factory=list)
    cursor: Optional[str] = Field(None, description="Pagination cursor for next page")
