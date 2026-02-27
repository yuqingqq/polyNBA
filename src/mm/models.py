"""Data models for the market making engine."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


class Side(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, enum.Enum):
    PENDING = "PENDING"
    LIVE = "LIVE"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class Quote(BaseModel):
    """A two-sided quote for a single token."""

    token_id: str
    bid_price: float | None = None
    bid_size: float | None = None
    ask_price: float | None = None
    ask_size: float | None = None
    market_mid: float | None = None  # market mid used for divergence calc
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Fill(BaseModel):
    """A single fill event."""

    order_id: str
    token_id: str
    side: Side
    price: float
    size: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    fee: float = 0.0


class Position(BaseModel):
    """Current position for a single token."""

    token_id: str
    size: float = 0.0  # positive = long, negative = short
    avg_entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class OrderState(BaseModel):
    """Lifecycle state of a single order."""

    order_id: str
    token_id: str
    side: Side
    price: float
    size: float
    filled_size: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RiskCheckResult(BaseModel):
    """Result of a full risk check."""

    should_halt: bool = False
    max_position_breached: bool = False
    max_total_breached: bool = False
    max_loss_breached: bool = False
    stale_data: bool = False
    stale_tokens: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)
