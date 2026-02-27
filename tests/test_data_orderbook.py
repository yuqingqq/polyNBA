"""Tests for local orderbook maintenance."""

import pytest

from src.data.models import OrderbookLevel, OrderbookSnapshot, PriceChange, Side
from src.data.orderbook import Orderbook, OrderbookManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_snapshot(
    asset_id: str = "asset_1",
    bids: list[tuple[float, float]] | None = None,
    asks: list[tuple[float, float]] | None = None,
) -> OrderbookSnapshot:
    """Helper to build an OrderbookSnapshot."""
    if bids is None:
        bids = [(0.48, 30.0), (0.47, 50.0), (0.46, 100.0)]
    if asks is None:
        asks = [(0.52, 25.0), (0.53, 40.0), (0.54, 80.0)]

    return OrderbookSnapshot(
        asset_id=asset_id,
        market="0xmarket1",
        bids=[OrderbookLevel(price=p, size=s) for p, s in bids],
        asks=[OrderbookLevel(price=p, size=s) for p, s in asks],
        timestamp="1700000000000",
        hash="0xhash1",
    )


def _make_change(
    asset_id: str = "asset_1",
    price: float = 0.50,
    size: float = 100.0,
    side: Side = Side.BUY,
) -> PriceChange:
    return PriceChange(
        asset_id=asset_id,
        price=price,
        size=size,
        side=side,
        hash="0xhash_change",
    )


# ---------------------------------------------------------------------------
# Orderbook tests
# ---------------------------------------------------------------------------


class TestOrderbookSnapshot:
    """Tests for applying full snapshots."""

    def test_apply_snapshot_sets_levels(self) -> None:
        book = Orderbook("asset_1")
        snap = _make_snapshot()
        book.apply_snapshot(snap)

        assert book.best_bid == 0.48
        assert book.best_ask == 0.52
        assert book.market == "0xmarket1"

    def test_apply_snapshot_replaces_previous(self) -> None:
        book = Orderbook("asset_1")
        book.apply_snapshot(_make_snapshot(bids=[(0.40, 10.0)], asks=[(0.60, 10.0)]))
        assert book.best_bid == 0.40

        # Apply new snapshot — old levels should be gone
        book.apply_snapshot(_make_snapshot(bids=[(0.45, 20.0)], asks=[(0.55, 20.0)]))
        assert book.best_bid == 0.45
        depth = book.get_depth(10)
        assert len(depth["bids"]) == 1
        assert len(depth["asks"]) == 1

    def test_snapshot_skips_zero_size(self) -> None:
        book = Orderbook("asset_1")
        book.apply_snapshot(
            _make_snapshot(bids=[(0.48, 0.0), (0.47, 10.0)], asks=[(0.52, 25.0)])
        )
        assert book.best_bid == 0.47
        depth = book.get_depth(10)
        assert len(depth["bids"]) == 1


class TestOrderbookIncrementalUpdates:
    """Tests for applying price_change deltas."""

    def test_add_new_level(self) -> None:
        book = Orderbook("asset_1")
        book.apply_snapshot(_make_snapshot())

        # Add a new bid level at 0.49
        book.apply_update(_make_change(price=0.49, size=200.0, side=Side.BUY))
        assert book.best_bid == 0.49

    def test_update_existing_level(self) -> None:
        book = Orderbook("asset_1")
        book.apply_snapshot(_make_snapshot())

        # Update the 0.48 bid
        book.apply_update(_make_change(price=0.48, size=999.0, side=Side.BUY))
        depth = book.get_depth(1)
        assert depth["bids"][0].size == 999.0

    def test_remove_level_size_zero(self) -> None:
        book = Orderbook("asset_1")
        book.apply_snapshot(_make_snapshot())

        # Remove the 0.48 bid by setting size=0
        book.apply_update(_make_change(price=0.48, size=0.0, side=Side.BUY))
        assert book.best_bid == 0.47

    def test_remove_nonexistent_level_is_safe(self) -> None:
        book = Orderbook("asset_1")
        book.apply_snapshot(_make_snapshot())
        # Removing a level that doesn't exist should be a no-op
        book.apply_update(_make_change(price=0.99, size=0.0, side=Side.BUY))
        assert book.best_bid == 0.48

    def test_ask_side_update(self) -> None:
        book = Orderbook("asset_1")
        book.apply_snapshot(_make_snapshot())
        # Add a tighter ask
        book.apply_update(_make_change(price=0.51, size=10.0, side=Side.SELL))
        assert book.best_ask == 0.51


class TestOrderbookProperties:
    """Tests for derived properties."""

    def test_mid_price(self) -> None:
        book = Orderbook("asset_1")
        book.apply_snapshot(_make_snapshot())
        assert book.mid_price == pytest.approx(0.50)

    def test_spread(self) -> None:
        book = Orderbook("asset_1")
        book.apply_snapshot(_make_snapshot())
        assert book.spread == pytest.approx(0.04)

    def test_get_depth_limits_levels(self) -> None:
        book = Orderbook("asset_1")
        book.apply_snapshot(_make_snapshot())
        depth = book.get_depth(2)
        assert len(depth["bids"]) == 2
        assert len(depth["asks"]) == 2
        # Best bid first
        assert depth["bids"][0].price == 0.48
        assert depth["bids"][1].price == 0.47
        # Best ask first
        assert depth["asks"][0].price == 0.52
        assert depth["asks"][1].price == 0.53


class TestEmptyBook:
    """Tests for an orderbook with no levels."""

    def test_empty_best_bid(self) -> None:
        book = Orderbook("empty")
        assert book.best_bid is None

    def test_empty_best_ask(self) -> None:
        book = Orderbook("empty")
        assert book.best_ask is None

    def test_empty_mid_price(self) -> None:
        book = Orderbook("empty")
        assert book.mid_price is None

    def test_empty_spread(self) -> None:
        book = Orderbook("empty")
        assert book.spread is None

    def test_empty_depth(self) -> None:
        book = Orderbook("empty")
        depth = book.get_depth(5)
        assert depth["bids"] == []
        assert depth["asks"] == []

    def test_one_sided_book(self) -> None:
        """Only bids, no asks -> mid/spread are None."""
        book = Orderbook("one_side")
        book.apply_snapshot(_make_snapshot(bids=[(0.48, 10.0)], asks=[]))
        assert book.best_bid == 0.48
        assert book.best_ask is None
        assert book.mid_price is None
        assert book.spread is None


# ---------------------------------------------------------------------------
# OrderbookManager tests
# ---------------------------------------------------------------------------


class TestOrderbookManager:
    def test_get_or_create(self) -> None:
        mgr = OrderbookManager()
        b1 = mgr.get_or_create("a1")
        b2 = mgr.get_or_create("a1")
        assert b1 is b2

    def test_handle_snapshot(self) -> None:
        mgr = OrderbookManager()
        mgr.handle_snapshot(_make_snapshot(asset_id="x"))
        book = mgr.get("x")
        assert book is not None
        assert book.best_bid == 0.48

    def test_handle_update(self) -> None:
        mgr = OrderbookManager()
        mgr.handle_snapshot(_make_snapshot(asset_id="x"))
        mgr.handle_update(_make_change(asset_id="x", price=0.49, size=50.0, side=Side.BUY))
        book = mgr.get("x")
        assert book is not None
        assert book.best_bid == 0.49

    def test_len_and_asset_ids(self) -> None:
        mgr = OrderbookManager()
        mgr.get_or_create("a")
        mgr.get_or_create("b")
        assert len(mgr) == 2
        assert set(mgr.asset_ids) == {"a", "b"}

    def test_get_nonexistent_returns_none(self) -> None:
        mgr = OrderbookManager()
        assert mgr.get("nonexistent") is None
