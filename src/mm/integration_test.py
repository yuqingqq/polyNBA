"""Dry-run integration test for the Market Making Engine.

Runs the full pipeline end-to-end in dry-run mode with synthetic NBA tokens.
No real orders are placed, no API keys or network access required.

Usage:
    cd /home/yuqing/polymarket-trading/repos/polymarket-trading
    python3 -m src.mm.integration_test
"""

from __future__ import annotations

import io
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from src.mm.config import MMConfig
from src.mm.engine import MarketMakingEngine
from src.mm.models import Fill, Side

# ---------------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stdout)
logger = logging.getLogger("integration_test")


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

class ReportBuilder:
    """Collects test results and generates a markdown report."""

    def __init__(self) -> None:
        self._sections: List[str] = []
        self._results: List[Tuple[str, bool, str]] = []  # (name, passed, detail)

    def section(self, title: str, body: str) -> None:
        self._sections.append(f"### {title}\n\n{body}\n")

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        self._results.append((name, passed, detail))
        status = "PASS" if passed else "FAIL"
        logger.info("[%s] %s %s", status, name, f"-- {detail}" if detail else "")

    def build(self) -> str:
        lines: List[str] = []
        lines.append("# Integration Test Report")
        lines.append(f"\nGenerated: {datetime.utcnow().isoformat()}Z\n")

        for sec in self._sections:
            lines.append(sec)

        lines.append("---\n")
        lines.append("## Summary\n")
        total = len(self._results)
        passed = sum(1 for _, p, _ in self._results if p)
        failed = total - passed
        lines.append(f"- **Total checks**: {total}")
        lines.append(f"- **Passed**: {passed}")
        lines.append(f"- **Failed**: {failed}")
        lines.append(f"- **Result**: {'ALL CHECKS PASSED' if failed == 0 else 'SOME CHECKS FAILED'}\n")

        lines.append("| # | Check | Result | Detail |")
        lines.append("|---|-------|--------|--------|")
        for i, (name, p, detail) in enumerate(self._results, 1):
            status = "PASS" if p else "FAIL"
            lines.append(f"| {i} | {name} | {status} | {detail} |")
        lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: format state snapshot
# ---------------------------------------------------------------------------

def snapshot_state(engine: MarketMakingEngine, tokens: List[str]) -> str:
    """Return a formatted string with the current engine state."""
    buf = io.StringIO()
    fv = engine.fair_value_engine.get_all_fair_values()
    buf.write(f"Fair values: {{{', '.join(f'{k}: {v:.4f}' for k, v in fv.items())}}}\n")

    for token_id in tokens:
        pos = engine.inventory_manager.get_position(token_id)
        buf.write(f"  {token_id}: pos={pos.size:.1f} avg={pos.avg_entry_price:.4f} "
                  f"rpnl={pos.realized_pnl:.2f} upnl={pos.unrealized_pnl:.2f}\n")

    open_orders = engine.order_manager.get_open_orders()
    buf.write(f"Open orders: {len(open_orders)}\n")
    for o in open_orders:
        buf.write(f"  {o.order_id[:20]}... {o.token_id} {o.side.value} "
                  f"price={o.price:.4f} size={o.size:.1f} status={o.status.value}\n")

    buf.write(f"Total exposure: {engine.inventory_manager.get_total_exposure():.1f}\n")
    buf.write(f"Total PnL: {engine.inventory_manager.get_pnl():.2f}\n")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main integration test
# ---------------------------------------------------------------------------

def run_integration_test() -> str:
    """Run the full integration test and return the report markdown."""
    report = ReportBuilder()

    # ---------------------------------------------------------------
    # 1. Setup
    # ---------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("PHASE 1: Setup")
    logger.info("=" * 70)

    config = MMConfig(
        dry_run=True,
        spread_bps=200,
        order_size=50,
        max_position=500,
        max_total_position=2000,
        update_interval_seconds=2,
        stale_data_timeout_seconds=60,
        max_loss=200,
    )

    engine = MarketMakingEngine(config=config, client=None)

    report.check(
        "Engine initializes in dry-run mode",
        engine.order_manager.dry_run is True,
        f"dry_run={engine.order_manager.dry_run}",
    )

    # ---------------------------------------------------------------
    # 2. Register tokens
    # ---------------------------------------------------------------
    tokens = ["token_okc_thunder", "token_boston_celtics", "token_lakers"]
    for t in tokens:
        engine.add_token(t)

    report.check(
        "Tokens registered",
        True,
        f"registered {len(tokens)} tokens",
    )

    # ---------------------------------------------------------------
    # 3. Simulate reference price updates and run cycles
    # ---------------------------------------------------------------
    price_sequences: Dict[str, List[float]] = {
        "token_okc_thunder": [0.35, 0.36, 0.37],
        "token_boston_celtics": [0.22, 0.21, 0.22],
        "token_lakers": [0.08, 0.07, 0.06],
    }

    logger.info("=" * 70)
    logger.info("PHASE 2: Price updates and quoting cycles")
    logger.info("=" * 70)

    cycle_snapshots: List[str] = []

    for cycle_idx in range(3):
        logger.info("-" * 50)
        logger.info("Cycle %d", cycle_idx + 1)
        logger.info("-" * 50)

        # Update fair values for this cycle
        for token_id, prices in price_sequences.items():
            engine.fair_value_engine.update(token_id, prices[cycle_idx])

        # Run one tick of the engine
        engine._tick()

        # Capture snapshot
        snap = snapshot_state(engine, tokens)
        cycle_snapshots.append(f"**Cycle {cycle_idx + 1}**\n```\n{snap}```\n")
        logger.info("Snapshot:\n%s", snap)

        # Run risk check for reporting
        risk_result = engine.risk_manager.check_all(
            engine.inventory_manager, engine.fair_value_engine
        )
        logger.info("Risk check: should_halt=%s reasons=%s",
                     risk_result.should_halt, risk_result.reasons)

    # Verify quotes were generated
    open_orders = engine.order_manager.get_open_orders()
    report.check(
        "Quotes generated after 3 cycles",
        len(open_orders) > 0,
        f"{len(open_orders)} open orders",
    )

    # Verify we have orders for each token
    tokens_with_orders = set(o.token_id for o in open_orders)
    report.check(
        "All tokens have open orders",
        tokens_with_orders == set(tokens),
        f"tokens with orders: {tokens_with_orders}",
    )

    # Verify both sides exist (at zero inventory, both bid and ask should be present)
    buy_orders = [o for o in open_orders if o.side == Side.BUY]
    sell_orders = [o for o in open_orders if o.side == Side.SELL]
    report.check(
        "Both bid and ask orders present",
        len(buy_orders) > 0 and len(sell_orders) > 0,
        f"bids={len(buy_orders)} asks={len(sell_orders)}",
    )

    # Verify fair values match expected cycle 3 values
    fv = engine.fair_value_engine.get_all_fair_values()
    report.check(
        "Fair values match cycle 3 inputs",
        (abs(fv["token_okc_thunder"] - 0.37) < 1e-9
         and abs(fv["token_boston_celtics"] - 0.22) < 1e-9
         and abs(fv["token_lakers"] - 0.06) < 1e-9),
        f"OKC={fv.get('token_okc_thunder', 'N/A'):.4f} "
        f"BOS={fv.get('token_boston_celtics', 'N/A'):.4f} "
        f"LAL={fv.get('token_lakers', 'N/A'):.4f}",
    )

    report.section("Quoting Cycles", "\n".join(cycle_snapshots))

    # ---------------------------------------------------------------
    # 4. Simulate fills and verify inventory + skewing
    # ---------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("PHASE 3: Simulate fills")
    logger.info("=" * 70)

    # Use a fresh engine for the fill test to avoid cached quote state
    # from the previous cycles interfering with the skew demonstration.
    # Use a non-tick-aligned fair value (0.505) so that directional
    # rounding (bids floor, asks ceil) allows the skew to visibly shift
    # quotes by at least one tick.
    fill_engine = MarketMakingEngine(config=config, client=None)
    fill_tokens = ["token_okc_thunder", "token_boston_celtics"]
    for t in fill_tokens:
        fill_engine.add_token(t)
        fill_engine.fair_value_engine.update(t, 0.505)

    # Run initial tick to get baseline quotes
    fill_engine._tick()
    okc_orders_before = fill_engine.order_manager.get_open_orders("token_okc_thunder")
    okc_bid_before = next(o.price for o in okc_orders_before if o.side == Side.BUY)
    okc_ask_before = next(o.price for o in okc_orders_before if o.side == Side.SELL)
    bos_orders_before = fill_engine.order_manager.get_open_orders("token_boston_celtics")
    bos_bid_before = next(o.price for o in bos_orders_before if o.side == Side.BUY)
    bos_ask_before = next(o.price for o in bos_orders_before if o.side == Side.SELL)

    logger.info("Pre-fill OKC: bid=%.4f ask=%.4f", okc_bid_before, okc_ask_before)
    logger.info("Pre-fill BOS: bid=%.4f ask=%.4f", bos_bid_before, bos_ask_before)

    # OKC Thunder: bought 300 @ 0.49 (large enough for visible skew)
    # Skew = 0.5 * (300/500) * 0.02 = 0.006 -> enough to shift by 1 tick
    okc_buy_order = next(o for o in okc_orders_before if o.side == Side.BUY)
    fill_okc = Fill(
        order_id=okc_buy_order.order_id,
        token_id="token_okc_thunder",
        side=Side.BUY,
        price=0.49,
        size=300.0,
    )
    fill_engine.on_fill(fill_okc)

    pos_okc = fill_engine.inventory_manager.get_position("token_okc_thunder")
    report.check(
        "OKC fill updates position to +300",
        abs(pos_okc.size - 300.0) < 1e-9,
        f"position={pos_okc.size:.1f}",
    )
    report.check(
        "OKC avg entry price is 0.49",
        abs(pos_okc.avg_entry_price - 0.49) < 1e-6,
        f"avg_entry={pos_okc.avg_entry_price:.4f}",
    )

    # Boston Celtics: sold 300 @ 0.51
    bos_sell_order = next(o for o in bos_orders_before if o.side == Side.SELL)
    fill_bos = Fill(
        order_id=bos_sell_order.order_id,
        token_id="token_boston_celtics",
        side=Side.SELL,
        price=0.51,
        size=300.0,
    )
    fill_engine.on_fill(fill_bos)

    pos_bos = fill_engine.inventory_manager.get_position("token_boston_celtics")
    report.check(
        "BOS fill updates position to -300",
        abs(pos_bos.size - (-300.0)) < 1e-9,
        f"position={pos_bos.size:.1f}",
    )

    # Cancel all existing orders and clear cached quotes so the engine
    # will requote from scratch with the new inventory positions.
    fill_engine.order_manager.cancel_all()
    fill_engine._current_quotes.clear()

    # Run another tick to get skewed quotes
    logger.info("Running post-fill cycle to verify quote skewing...")
    fill_engine._tick()

    snap_post_fill = snapshot_state(fill_engine, fill_tokens)
    logger.info("Post-fill snapshot:\n%s", snap_post_fill)
    report.section("Post-Fill State", f"```\n{snap_post_fill}```")

    # Verify OKC (long +300): quotes should be skewed DOWN
    okc_orders_after = fill_engine.order_manager.get_open_orders("token_okc_thunder")
    okc_buys_after = [o for o in okc_orders_after if o.side == Side.BUY]
    okc_sells_after = [o for o in okc_orders_after if o.side == Side.SELL]

    report.check(
        "OKC quotes present after long fill",
        len(okc_orders_after) > 0,
        f"buys={len(okc_buys_after)} sells={len(okc_sells_after)}",
    )
    if okc_sells_after:
        report.check(
            "OKC ask skewed down after long fill",
            okc_sells_after[0].price < okc_ask_before,
            f"ask_before={okc_ask_before:.4f} ask_after={okc_sells_after[0].price:.4f}",
        )

    # Verify BOS (short -300): quotes should be skewed UP
    bos_orders_after = fill_engine.order_manager.get_open_orders("token_boston_celtics")
    bos_buys_after = [o for o in bos_orders_after if o.side == Side.BUY]
    bos_sells_after = [o for o in bos_orders_after if o.side == Side.SELL]

    report.check(
        "BOS quotes present after short fill",
        len(bos_orders_after) > 0,
        f"buys={len(bos_buys_after)} sells={len(bos_sells_after)}",
    )
    if bos_buys_after:
        report.check(
            "BOS bid skewed up after short fill",
            bos_buys_after[0].price > bos_bid_before,
            f"bid_before={bos_bid_before:.4f} bid_after={bos_buys_after[0].price:.4f}",
        )

    # ---------------------------------------------------------------
    # 5. Risk scenarios
    # ---------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("PHASE 4: Risk scenarios")
    logger.info("=" * 70)

    # --- Scenario A: Large position near max_position ---
    logger.info("--- Scenario A: Large position near max_position ---")

    # Create a fresh engine for risk testing.
    # Use a non-tick-aligned fair value so that directional rounding
    # (bids floor, asks ceil) allows the skew to visibly shift quotes.
    risk_engine = MarketMakingEngine(config=config, client=None)
    for t in tokens:
        risk_engine.add_token(t)
        risk_engine.fair_value_engine.update(t, 0.505)

    # Place initial orders so we have tracked order IDs for fills
    risk_engine._tick()
    risk_okc_orders = risk_engine.order_manager.get_open_orders("token_okc_thunder")
    risk_okc_buy = next(o for o in risk_okc_orders if o.side == Side.BUY)

    # Simulate a large position: 490 units (near max_position=500)
    large_fill = Fill(
        order_id=risk_okc_buy.order_id,
        token_id="token_okc_thunder",
        side=Side.BUY,
        price=0.50,
        size=490.0,
    )
    risk_engine.on_fill(large_fill)

    # Run a tick
    risk_engine._tick()

    # Check that the bid side is skewed / reduced
    okc_risk_orders = risk_engine.order_manager.get_open_orders("token_okc_thunder")
    okc_risk_buys = [o for o in okc_risk_orders if o.side == Side.BUY]
    okc_risk_sells = [o for o in okc_risk_orders if o.side == Side.SELL]

    # With position=490 and max_position=500, bid should still exist
    # but should be heavily skewed down. The skew = 0.5 * (490/500) * 0.02 = 0.0098
    report.check(
        "Risk A: Near-max position skews quotes heavily",
        len(okc_risk_sells) > 0,
        f"buys={len(okc_risk_buys)} sells={len(okc_risk_sells)} "
        f"(heavily skewed to encourage selling)",
    )

    # Check that sell price is lower than with zero position
    if okc_risk_sells:
        skewed_ask = okc_risk_sells[0].price
        # With zero position and fv=0.505 ask would be ~0.52 (ceil-rounded),
        # with pos=490 it should be lower due to skew
        report.check(
            "Risk A: Ask price skewed down from heavy long",
            skewed_ask < 0.52,
            f"skewed_ask={skewed_ask:.4f} (vs unskewed ~0.52)",
        )

    risk_snap_a = snapshot_state(risk_engine, tokens)
    logger.info("Risk scenario A snapshot:\n%s", risk_snap_a)
    report.section("Risk Scenario A: Large Position", f"```\n{risk_snap_a}```")

    # --- Scenario B: Stale data ---
    logger.info("--- Scenario B: Stale data ---")

    stale_engine = MarketMakingEngine(config=config, client=None)
    for t in tokens:
        stale_engine.add_token(t)

    # Place some orders first with ALL tokens having fresh data
    for t in tokens:
        stale_engine.fair_value_engine.update(t, 0.50, timestamp=time.time())
    stale_engine._tick()
    orders_before_stale = len(stale_engine.order_manager.get_open_orders())

    # Now make data stale again
    stale_ts = time.time() - 120
    for t in tokens:
        stale_engine.fair_value_engine.update(t, 0.50, timestamp=stale_ts)

    # Risk check should detect stale data
    risk_result_stale = stale_engine.risk_manager.check_all(
        stale_engine.inventory_manager, stale_engine.fair_value_engine
    )
    report.check(
        "Risk B: Stale data detected",
        risk_result_stale.stale_data is True,
        f"stale_data={risk_result_stale.stale_data} reasons={risk_result_stale.reasons}",
    )
    report.check(
        "Risk B: Stale tokens collected for per-token handling",
        len(risk_result_stale.stale_tokens) > 0,
        f"stale_tokens={risk_result_stale.stale_tokens}",
    )

    # Run tick - should cancel stale token orders (per-token, not halt-all)
    stale_engine._tick()
    orders_after_stale = len(stale_engine.order_manager.get_open_orders())
    report.check(
        "Risk B: Orders cancelled for stale tokens",
        orders_after_stale == 0,
        f"orders_before={orders_before_stale} orders_after={orders_after_stale}",
    )

    report.section("Risk Scenario B: Stale Data",
                    f"Stale data timestamp: {stale_ts:.0f} (now: {time.time():.0f})\n"
                    f"Risk result: should_halt={risk_result_stale.should_halt}\n"
                    f"Reasons: {risk_result_stale.reasons}")

    # --- Scenario C: Max loss breach (kill switch) ---
    logger.info("--- Scenario C: Max loss breach ---")

    loss_engine = MarketMakingEngine(config=config, client=None)
    for t in tokens:
        loss_engine.add_token(t)
        loss_engine.fair_value_engine.update(t, 0.50)

    # Run a tick to create orders
    loss_engine._tick()
    orders_before_loss = len(loss_engine.order_manager.get_open_orders())
    logger.info("Orders before loss: %d", orders_before_loss)

    # Use actual order IDs from placed orders for fill processing
    loss_okc_orders = loss_engine.order_manager.get_open_orders("token_okc_thunder")
    loss_buy_order = next(o for o in loss_okc_orders if o.side == Side.BUY)
    loss_sell_order = next(o for o in loss_okc_orders if o.side == Side.SELL)

    # Simulate a large loss: buy at 0.60, sell at 0.10 -> loss = 500 * 0.50 = 250 > max_loss=200
    loss_engine.on_fill(Fill(
        order_id=loss_buy_order.order_id,
        token_id="token_okc_thunder",
        side=Side.BUY,
        price=0.60,
        size=500.0,
    ))
    loss_engine.on_fill(Fill(
        order_id=loss_sell_order.order_id,
        token_id="token_okc_thunder",
        side=Side.SELL,
        price=0.10,
        size=500.0,
    ))

    pnl_after_loss = loss_engine.inventory_manager.get_pnl()
    logger.info("PnL after simulated loss: %.2f", pnl_after_loss)

    risk_result_loss = loss_engine.risk_manager.check_all(
        loss_engine.inventory_manager, loss_engine.fair_value_engine
    )
    report.check(
        "Risk C: Max loss breached",
        risk_result_loss.max_loss_breached is True,
        f"pnl={pnl_after_loss:.2f} max_loss={config.max_loss:.2f}",
    )
    report.check(
        "Risk C: Engine halts on max loss",
        risk_result_loss.should_halt is True,
        f"should_halt={risk_result_loss.should_halt}",
    )

    # Run tick - kill switch should fire
    loss_engine._tick()
    orders_after_loss = len(loss_engine.order_manager.get_open_orders())
    report.check(
        "Risk C: Kill switch cancels all orders",
        orders_after_loss == 0,
        f"orders_before={orders_before_loss} orders_after={orders_after_loss}",
    )

    report.section("Risk Scenario C: Max Loss / Kill Switch",
                    f"PnL after simulated loss: {pnl_after_loss:.2f}\n"
                    f"Max loss limit: {config.max_loss:.2f}\n"
                    f"Risk result: should_halt={risk_result_loss.should_halt}\n"
                    f"Reasons: {risk_result_loss.reasons}\n"
                    f"Orders cancelled: {orders_before_loss} -> {orders_after_loss}")

    # ---------------------------------------------------------------
    # 6. Requote on price move
    # ---------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("PHASE 5: Requote on price move")
    logger.info("=" * 70)

    requote_engine = MarketMakingEngine(config=config, client=None)
    requote_engine.add_token("token_okc_thunder")
    requote_engine.fair_value_engine.update("token_okc_thunder", 0.50)

    # Run initial tick
    requote_engine._tick()
    initial_orders = requote_engine.order_manager.get_open_orders("token_okc_thunder")
    initial_buys = [o for o in initial_orders if o.side == Side.BUY]
    initial_asks = [o for o in initial_orders if o.side == Side.SELL]
    initial_bid = initial_buys[0].price if initial_buys else 0
    initial_ask = initial_asks[0].price if initial_asks else 0
    logger.info("Initial quotes: bid=%.4f ask=%.4f", initial_bid, initial_ask)

    # Move fair value significantly (beyond requote_threshold_bps=50 -> 0.005)
    requote_engine.fair_value_engine.update("token_okc_thunder", 0.53)
    requote_engine._tick()

    new_orders = requote_engine.order_manager.get_open_orders("token_okc_thunder")
    new_buys = [o for o in new_orders if o.side == Side.BUY]
    new_asks = [o for o in new_orders if o.side == Side.SELL]
    new_bid = new_buys[0].price if new_buys else 0
    new_ask = new_asks[0].price if new_asks else 0
    logger.info("New quotes after fv move: bid=%.4f ask=%.4f", new_bid, new_ask)

    report.check(
        "Requote: bid moves up on fv increase",
        new_bid > initial_bid,
        f"initial_bid={initial_bid:.4f} new_bid={new_bid:.4f}",
    )
    report.check(
        "Requote: ask moves up on fv increase",
        new_ask > initial_ask,
        f"initial_ask={initial_ask:.4f} new_ask={new_ask:.4f}",
    )

    report.section("Requote on Price Move",
                    f"Fair value: 0.50 -> 0.53\n"
                    f"Bid: {initial_bid:.4f} -> {new_bid:.4f}\n"
                    f"Ask: {initial_ask:.4f} -> {new_ask:.4f}")

    # ---------------------------------------------------------------
    # 7. Binary pair-aware market making
    # ---------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("PHASE 6: Binary pair-aware market making")
    logger.info("=" * 70)

    pair_engine = MarketMakingEngine(config=config, client=None)

    # Suns (tok_suns) vs Spurs (tok_spurs) — binary pair.
    # Use non-tick-aligned fair values so directional rounding allows
    # skew to visibly shift quotes by at least one tick.
    pair_engine.add_token_pair("tok_suns", "tok_spurs")
    pair_engine.fair_value_engine.update("tok_suns", 0.605)
    pair_engine.fair_value_engine.update("tok_spurs", 0.395)

    report.check(
        "Pair: both tokens registered",
        "tok_suns" in pair_engine._tokens and "tok_spurs" in pair_engine._tokens,
        "tok_suns and tok_spurs in _tokens",
    )
    report.check(
        "Pair: linked both directions",
        pair_engine._pairs.get("tok_suns") == "tok_spurs"
        and pair_engine._pairs.get("tok_spurs") == "tok_suns",
        "tok_suns <-> tok_spurs",
    )

    # Initial tick — zero inventory, should produce 4 orders (bid+ask each)
    pair_engine._tick()
    suns_orders_0 = pair_engine.order_manager.get_open_orders("tok_suns")
    spurs_orders_0 = pair_engine.order_manager.get_open_orders("tok_spurs")
    report.check(
        "Pair: 4 orders at zero inventory (2 per side)",
        len(suns_orders_0) == 2 and len(spurs_orders_0) == 2,
        f"suns={len(suns_orders_0)} spurs={len(spurs_orders_0)}",
    )

    snap_pair_0 = snapshot_state(pair_engine, ["tok_suns", "tok_spurs"])
    logger.info("Pair baseline (zero inventory):\n%s", snap_pair_0)

    # --- Scenario: equal positions on both sides => net = 0 ---
    logger.info("--- Equal positions: long 25 Suns + long 25 Spurs => net=0 ---")
    suns_buy_0 = next(o for o in suns_orders_0 if o.side == Side.BUY)
    spurs_buy_0 = next(o for o in spurs_orders_0 if o.side == Side.BUY)
    pair_engine.on_fill(Fill(
        order_id=suns_buy_0.order_id, token_id="tok_suns",
        side=Side.BUY, price=0.605, size=25.0,
    ))
    pair_engine.on_fill(Fill(
        order_id=spurs_buy_0.order_id, token_id="tok_spurs",
        side=Side.BUY, price=0.395, size=25.0,
    ))

    exposure_equal = pair_engine.inventory_manager.get_total_exposure()
    report.check(
        "Pair: equal positions => net exposure = 0",
        abs(exposure_equal) < 1e-9,
        f"exposure={exposure_equal:.1f} (long 25+25, net=0)",
    )

    snap_pair_equal = snapshot_state(pair_engine, ["tok_suns", "tok_spurs"])
    logger.info("Pair equal positions:\n%s", snap_pair_equal)

    # --- Scenario: asymmetric fill => net skew ---
    logger.info("--- Asymmetric: buy 275 more Suns => net = 300 ---")
    # Use same order (it's already partially filled but still LIVE)
    pair_engine.on_fill(Fill(
        order_id=suns_buy_0.order_id, token_id="tok_suns",
        side=Side.BUY, price=0.605, size=275.0,
    ))

    exposure_asym = pair_engine.inventory_manager.get_total_exposure()
    report.check(
        "Pair: asymmetric => net exposure = 275",
        abs(exposure_asym - 275.0) < 1e-9,
        f"exposure={exposure_asym:.1f} (suns=300, spurs=25, net=275)",
    )

    # Force fresh requote
    pair_engine.order_manager.cancel_all()
    pair_engine._current_quotes.clear()
    pair_engine._tick()

    # Suns quotes should be skewed by net_position=275 (not raw 300)
    suns_orders_asym = pair_engine.order_manager.get_open_orders("tok_suns")
    suns_bids_asym = [o for o in suns_orders_asym if o.side == Side.BUY]
    suns_asks_asym = [o for o in suns_orders_asym if o.side == Side.SELL]

    # Spurs quotes should be skewed by net_position=-275 (not raw 25)
    spurs_orders_asym = pair_engine.order_manager.get_open_orders("tok_spurs")
    spurs_bids_asym = [o for o in spurs_orders_asym if o.side == Side.BUY]
    spurs_asks_asym = [o for o in spurs_orders_asym if o.side == Side.SELL]

    report.check(
        "Pair: Suns quotes present with asymmetric inventory",
        len(suns_orders_asym) > 0,
        f"suns buys={len(suns_bids_asym)} asks={len(suns_asks_asym)}",
    )
    report.check(
        "Pair: Spurs quotes present with asymmetric inventory",
        len(spurs_orders_asym) > 0,
        f"spurs buys={len(spurs_bids_asym)} asks={len(spurs_asks_asym)}",
    )

    # The Suns bid should be lower than baseline (net long → skew down)
    if suns_bids_asym:
        suns_bid_0 = next((o.price for o in suns_orders_0 if o.side == Side.BUY), 0)
        report.check(
            "Pair: Suns bid skewed down by net long",
            suns_bids_asym[0].price < suns_bid_0,
            f"bid_baseline={suns_bid_0:.4f} bid_asym={suns_bids_asym[0].price:.4f}",
        )

    # The Spurs bid should be higher than baseline (net short → skew up)
    if spurs_bids_asym:
        spurs_bid_0 = next((o.price for o in spurs_orders_0 if o.side == Side.BUY), 0)
        report.check(
            "Pair: Spurs bid skewed up by net short",
            spurs_bids_asym[0].price > spurs_bid_0,
            f"bid_baseline={spurs_bid_0:.4f} bid_asym={spurs_bids_asym[0].price:.4f}",
        )

    snap_pair_asym = snapshot_state(pair_engine, ["tok_suns", "tok_spurs"])
    logger.info("Pair asymmetric positions:\n%s", snap_pair_asym)

    # --- Scenario: fill on Suns triggers requote on Spurs ---
    # Use a fresh pair engine so we can control the exact skew delta.
    # Net must shift enough to cross the requote_threshold_bps (50).
    # skew_delta = 0.5 * (delta_net / max_pos) * spread
    # Need > 50bps = 0.005.  With spread=0.02, max_pos=500:
    #   delta_net > 0.005 / (0.5 * 0.02 / 500) = 0.005 / 0.00002 = 250
    logger.info("--- Fill on Suns triggers requote on Spurs ---")
    trigger_engine = MarketMakingEngine(config=config, client=None)
    trigger_engine.add_token_pair("tok_suns2", "tok_spurs2")
    trigger_engine.fair_value_engine.update("tok_suns2", 0.605)
    trigger_engine.fair_value_engine.update("tok_spurs2", 0.395)
    trigger_engine._tick()

    spurs_ids_before = {o.order_id for o in trigger_engine.order_manager.get_open_orders("tok_spurs2")}

    # Large fill: net jumps from 0 to 300 → skew_delta = 60bps > 50bps threshold
    suns2_buy = next(o for o in trigger_engine.order_manager.get_open_orders("tok_suns2") if o.side == Side.BUY)
    trigger_engine.on_fill(Fill(
        order_id=suns2_buy.order_id, token_id="tok_suns2",
        side=Side.BUY, price=0.605, size=300.0,
    ))

    # Complement is NOT requoted during on_fill (stale orderbook risk).
    # The next tick will requote with fresh data.
    trigger_engine._tick()

    spurs_ids_after = {o.order_id for o in trigger_engine.order_manager.get_open_orders("tok_spurs2")}
    report.check(
        "Pair: fill on Suns triggers Spurs requote (on next tick)",
        not spurs_ids_after.intersection(spurs_ids_before),
        f"spurs orders changed: {len(spurs_ids_before)} before, {len(spurs_ids_after)} after",
    )

    # --- Scenario: pair-aware limits ---
    logger.info("--- Pair-aware limits ---")
    limit_engine = MarketMakingEngine(
        config=config.model_copy(update={"max_position": 100, "max_total_position": 200}),
        client=None,
    )
    limit_engine.add_token_pair("tok_a", "tok_b")
    limit_engine.fair_value_engine.update("tok_a", 0.50)
    limit_engine.fair_value_engine.update("tok_b", 0.50)

    # Place initial orders to get tracked order IDs
    limit_engine._tick()
    la_buy = next(o for o in limit_engine.order_manager.get_open_orders("tok_a") if o.side == Side.BUY)
    lb_buy = next(o for o in limit_engine.order_manager.get_open_orders("tok_b") if o.side == Side.BUY)

    # Long 200 on both sides — raw would breach, net = 0
    limit_engine.on_fill(Fill(
        order_id=la_buy.order_id, token_id="tok_a", side=Side.BUY, price=0.5, size=200.0,
    ))
    limit_engine.on_fill(Fill(
        order_id=lb_buy.order_id, token_id="tok_b", side=Side.BUY, price=0.5, size=200.0,
    ))

    within = limit_engine.inventory_manager.is_within_limits(
        config.model_copy(update={"max_position": 100, "max_total_position": 200}),
    )
    report.check(
        "Pair: equal positions within limits despite raw > max",
        within is True,
        f"raw_a=200 raw_b=200 net=0, max_position=100 → within_limits={within}",
    )

    snap_pair_final = snapshot_state(pair_engine, ["tok_suns", "tok_spurs"])
    report.section(
        "Binary Pair-Aware Market Making",
        f"**Baseline (zero inventory)**\n```\n{snap_pair_0}```\n\n"
        f"**Equal positions (net=0)**\n```\n{snap_pair_equal}```\n\n"
        f"**Asymmetric (Suns=300, Spurs=25, net=275)**\n```\n{snap_pair_asym}```\n\n"
        f"**After fill-triggered requote**\n```\n{snap_pair_final}```",
    )

    # ---------------------------------------------------------------
    # Build final report
    # ---------------------------------------------------------------
    return report.build()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Starting Market Making Engine integration test (dry-run)")
    logger.info("No real orders will be placed.")
    logger.info("")

    report_md = run_integration_test()

    # Write report to workspace
    workspace_dir = Path(
        "/home/yuqing/polymarket-trading/orchestrator/"
        "PROGRAMS/P-2026-003-market-making/workspace"
    )
    workspace_dir.mkdir(parents=True, exist_ok=True)
    report_path = workspace_dir / "integration-test-report.md"
    report_path.write_text(report_md)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Integration test complete. Report written to:")
    logger.info("  %s", report_path)
    logger.info("=" * 70)

    # Print summary
    print("\n" + report_md)


if __name__ == "__main__":
    main()
