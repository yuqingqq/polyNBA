"""Unit and integration tests for LatencyTracker and engine latency instrumentation.

Covers: record + snapshot, percentile math, rolling window trim, empty state,
thread safety, and engine integration (on_fill / tick / quote age).
"""

from __future__ import annotations

import threading

import pytest

from src.mm.latency import LatencyTracker
from src.mm.config import MMConfig
from src.mm.engine import MarketMakingEngine
from src.mm.models import Fill, Side


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker() -> LatencyTracker:
    return LatencyTracker(window=1000)


@pytest.fixture
def engine_config() -> MMConfig:
    return MMConfig(
        dry_run=True,
        spread_bps=200,
        order_size=50,
        max_position=500,
        max_total_position=2000,
        update_interval_seconds=2,
        stale_data_timeout_seconds=60,
        max_loss=200,
    )


# ---------------------------------------------------------------------------
# LatencyTracker unit tests
# ---------------------------------------------------------------------------


class TestLatencyTrackerBasic:
    """Core record + snapshot functionality."""

    def test_record_and_snapshot(self, tracker: LatencyTracker) -> None:
        """Recording a sample and retrieving a snapshot."""
        tracker.record("test", 0.005)
        snap = tracker.snapshot("test")
        assert snap is not None
        assert snap.name == "test"
        assert snap.count == 1
        assert snap.last_ms == pytest.approx(5.0)
        assert snap.max_ms == pytest.approx(5.0)

    def test_empty_snapshot_returns_none(self, tracker: LatencyTracker) -> None:
        """Snapshot for an unrecorded name returns None."""
        assert tracker.snapshot("nonexistent") is None

    def test_empty_snapshots_returns_empty(self, tracker: LatencyTracker) -> None:
        """snapshots() with no data returns an empty list."""
        assert tracker.snapshots() == []

    def test_empty_format_summary(self, tracker: LatencyTracker) -> None:
        """format_summary with no data returns a placeholder."""
        assert "no data" in tracker.format_summary()

    def test_multiple_samples_percentiles(self, tracker: LatencyTracker) -> None:
        """Percentile computation across multiple samples."""
        # Record 100 samples: 0.001, 0.002, ..., 0.100 seconds
        for i in range(1, 101):
            tracker.record("op", i / 1000.0)
        snap = tracker.snapshot("op")
        assert snap is not None
        assert snap.count == 100
        # p50 should be around 50ms
        assert 45.0 <= snap.p50_ms <= 55.0
        # p95 should be around 95ms
        assert 90.0 <= snap.p95_ms <= 100.0
        # max should be 100ms
        assert snap.max_ms == pytest.approx(100.0)
        # last should be 100ms
        assert snap.last_ms == pytest.approx(100.0)

    def test_single_sample_all_percentiles_equal(self, tracker: LatencyTracker) -> None:
        """With one sample, all percentiles should equal that sample."""
        tracker.record("single", 0.010)
        snap = tracker.snapshot("single")
        assert snap is not None
        assert snap.p50_ms == pytest.approx(10.0)
        assert snap.p95_ms == pytest.approx(10.0)
        assert snap.p99_ms == pytest.approx(10.0)
        assert snap.max_ms == pytest.approx(10.0)


class TestLatencyTrackerWindow:
    """Rolling window trimming."""

    def test_window_trims_oldest(self) -> None:
        """Excess samples beyond window are trimmed (oldest removed)."""
        tracker = LatencyTracker(window=5)
        for i in range(10):
            tracker.record("x", float(i))
        snap = tracker.snapshot("x")
        assert snap is not None
        assert snap.count == 5
        # Only samples 5..9 should remain, so min is 5.0
        # last should be 9.0 → 9000ms
        assert snap.last_ms == pytest.approx(9000.0)

    def test_window_exact_boundary(self) -> None:
        """Exactly window samples should not trim."""
        tracker = LatencyTracker(window=3)
        for i in range(3):
            tracker.record("y", 0.001 * (i + 1))
        snap = tracker.snapshot("y")
        assert snap is not None
        assert snap.count == 3


class TestLatencyTrackerMultipleNames:
    """Multiple named measurements are independent."""

    def test_independent_names(self, tracker: LatencyTracker) -> None:
        """Different names maintain separate buffers."""
        tracker.record("a", 0.001)
        tracker.record("b", 0.002)
        tracker.record("a", 0.003)

        snap_a = tracker.snapshot("a")
        snap_b = tracker.snapshot("b")
        assert snap_a is not None and snap_b is not None
        assert snap_a.count == 2
        assert snap_b.count == 1

    def test_snapshots_returns_all_sorted(self, tracker: LatencyTracker) -> None:
        """snapshots() returns all names, sorted alphabetically."""
        tracker.record("z_op", 0.001)
        tracker.record("a_op", 0.002)
        all_snaps = tracker.snapshots()
        assert len(all_snaps) == 2
        assert all_snaps[0].name == "a_op"
        assert all_snaps[1].name == "z_op"


class TestLatencyTrackerFormatSummary:
    """format_summary output."""

    def test_format_with_data(self, tracker: LatencyTracker) -> None:
        """Summary includes name, percentiles, and count."""
        tracker.record("tick", 0.002)
        summary = tracker.format_summary()
        assert "Latency:" in summary
        assert "tick:" in summary
        assert "p50=" in summary
        assert "n=1" in summary


class TestLatencyTrackerThreadSafety:
    """Concurrent access must not crash."""

    def test_concurrent_record_and_snapshot(self) -> None:
        """Multiple threads recording and snapshotting concurrently."""
        tracker = LatencyTracker(window=100)
        errors: list[Exception] = []

        def writer(name: str, count: int) -> None:
            try:
                for i in range(count):
                    tracker.record(name, 0.001 * i)
            except Exception as e:
                errors.append(e)

        def reader(count: int) -> None:
            try:
                for _ in range(count):
                    tracker.snapshots()
                    tracker.format_summary()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=("a", 200)),
            threading.Thread(target=writer, args=("b", 200)),
            threading.Thread(target=reader, args=(100,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0, f"Thread safety errors: {errors}"


# ---------------------------------------------------------------------------
# Engine integration tests
# ---------------------------------------------------------------------------


class TestEngineLatencyOnFill:
    """on_fill() records latency."""

    def test_on_fill_records_latency(self, engine_config: MMConfig) -> None:
        """After on_fill, latency tracker should have an 'on_fill' entry."""
        eng = MarketMakingEngine(config=engine_config, client=None)
        eng.add_token("tok_a")
        eng.fair_value_engine.update("tok_a", 0.50)
        eng._tick()

        buy_order = next(o for o in eng.order_manager.get_open_orders("tok_a") if o.side == Side.BUY)
        fill = Fill(
            order_id=buy_order.order_id,
            token_id="tok_a",
            side=Side.BUY,
            price=0.49,
            size=10.0,
        )
        eng.on_fill(fill)

        snap = eng.latency_tracker.snapshot("on_fill")
        assert snap is not None
        assert snap.count == 1
        assert snap.last_ms > 0

    def test_on_fill_latency_positive(self, engine_config: MMConfig) -> None:
        """on_fill latency should be a positive number of milliseconds."""
        eng = MarketMakingEngine(config=engine_config, client=None)
        eng.add_token("tok_a")
        eng.fair_value_engine.update("tok_a", 0.50)
        eng._tick()

        sell_order = next(o for o in eng.order_manager.get_open_orders("tok_a") if o.side == Side.SELL)
        fill = Fill(
            order_id=sell_order.order_id,
            token_id="tok_a",
            side=Side.SELL,
            price=0.51,
            size=10.0,
        )
        eng.on_fill(fill)

        snap = eng.latency_tracker.snapshot("on_fill")
        assert snap is not None
        assert snap.p50_ms > 0


class TestEngineLatencyTick:
    """_tick() records latency."""

    def test_tick_records_latency(self, engine_config: MMConfig) -> None:
        """After _tick, latency tracker should have a 'tick' entry."""
        eng = MarketMakingEngine(config=engine_config, client=None)
        eng.add_token("tok_a")
        eng.fair_value_engine.update("tok_a", 0.50)
        eng._tick()

        snap = eng.latency_tracker.snapshot("tick")
        assert snap is not None
        assert snap.count == 1
        assert snap.last_ms > 0


class TestEngineQuoteAge:
    """Quote age tracking after requote."""

    def test_quote_age_set_after_requote(self, engine_config: MMConfig) -> None:
        """After a successful requote, quote age should be tracked."""
        eng = MarketMakingEngine(config=engine_config, client=None)
        eng.add_token("tok_a")
        eng.fair_value_engine.update("tok_a", 0.50)
        eng._tick()

        # _last_requote_time should have an entry for tok_a
        assert "tok_a" in eng._last_requote_time
        assert eng._last_requote_time["tok_a"] > 0

    def test_quote_age_not_set_without_tick(self, engine_config: MMConfig) -> None:
        """Before any tick, quote age dict should be empty."""
        eng = MarketMakingEngine(config=engine_config, client=None)
        eng.add_token("tok_a")
        assert "tok_a" not in eng._last_requote_time
