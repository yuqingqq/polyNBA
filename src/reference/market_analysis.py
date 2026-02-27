"""Market analysis tool for evaluating NBA markets for market making.

Scans Polymarket for NBA contracts, fetches external odds, computes
divergence/spread/edge metrics, and ranks markets by MM suitability.

Usage:
    python3 -m src.reference.market_analysis [--json output.json]
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

from .market_mapper import MarketMapper
from .models import PolymarketContract, ReferencePrice
from .odds_client import OddsClient, OddsClientError, parse_event_to_external_odds
from .polymarket_scanner import PolymarketScanner, PolymarketScannerError
from .price_adapter import PriceAdapter

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Tier thresholds
# -------------------------------------------------------------------

TIER1_MIN_VOLUME = 1_000_000  # $1M
TIER1_MAX_DIVERGENCE_BPS = 500
TIER1_MIN_FAIR_VALUE = 0.03

TIER2_MIN_VOLUME = 100_000  # $100K
TIER2_MAX_DIVERGENCE_BPS = 1000
TIER2_MIN_FAIR_VALUE = 0.01


@dataclass
class MarketMetrics:
    """Computed metrics for a single Polymarket contract."""

    token_id: str
    question: str
    outcome: str
    poly_price: Optional[float]
    fair_value: Optional[float]
    divergence_bps: Optional[float]
    volume_usd: Optional[float]
    spread_estimate: Optional[float]
    edge_bps: Optional[float]
    tier: int  # 1=Best, 2=Good, 3=Avoid


def compute_metrics(
    contract: PolymarketContract,
    ref_price: Optional[ReferencePrice],
) -> MarketMetrics:
    """Compute analysis metrics for a single contract.

    Args:
        contract: Polymarket contract with current price/volume.
        ref_price: Matched reference price (vig-removed fair value),
            or None if no external odds available.

    Returns:
        MarketMetrics with all computed fields.
    """
    poly_price = contract.current_price
    fair_value = ref_price.fair_probability if ref_price else None

    # Divergence: |poly_price - fair_value| in basis points
    divergence_bps: Optional[float] = None
    if poly_price is not None and fair_value is not None:
        divergence_bps = abs(poly_price - fair_value) * 10_000

    # Spread estimate: 2x distance from 0.50 (proxy for bid-ask width)
    spread_estimate: Optional[float] = None
    if poly_price is not None:
        spread_estimate = 2.0 * abs(poly_price - 0.50)

    # Edge: half-spread capture potential
    edge_bps: Optional[float] = None
    if spread_estimate is not None:
        edge_bps = spread_estimate * 10_000 / 2.0

    volume_usd = contract.volume
    tier = _classify_tier(volume_usd, fair_value, divergence_bps)

    return MarketMetrics(
        token_id=contract.token_id,
        question=contract.question,
        outcome=contract.outcome,
        poly_price=poly_price,
        fair_value=fair_value,
        divergence_bps=divergence_bps,
        volume_usd=volume_usd,
        spread_estimate=spread_estimate,
        edge_bps=edge_bps,
        tier=tier,
    )


def _classify_tier(
    volume_usd: Optional[float],
    fair_value: Optional[float],
    divergence_bps: Optional[float],
) -> int:
    """Classify a market into a tier for MM suitability.

    Returns:
        1 (Best), 2 (Good), or 3 (Avoid).
    """
    vol = volume_usd or 0
    fv = fair_value or 0
    div = divergence_bps if divergence_bps is not None else float("inf")

    if vol >= TIER1_MIN_VOLUME and fv >= TIER1_MIN_FAIR_VALUE and div <= TIER1_MAX_DIVERGENCE_BPS:
        return 1
    if vol >= TIER2_MIN_VOLUME and fv >= TIER2_MIN_FAIR_VALUE and div <= TIER2_MAX_DIVERGENCE_BPS:
        return 2
    return 3


def run_analysis(json_path: Optional[str] = None) -> list[MarketMetrics]:
    """Run the full market analysis pipeline.

    1. Scan Polymarket for NBA contracts.
    2. Fetch external odds (championship + game) and map to reference prices.
    3. Compute per-contract metrics.
    4. Print ranked table and optionally write JSON.

    Args:
        json_path: Optional path to write JSON output.

    Returns:
        Sorted list of MarketMetrics (tier ascending, divergence ascending).
    """
    # --- Step 1: Scan Polymarket ---
    print("Scanning Polymarket for NBA contracts...")
    scanner = PolymarketScanner()
    try:
        contracts = scanner.get_all_nba_contracts()
    except PolymarketScannerError as e:
        print(f"ERROR: Scanner failed: {e}", file=sys.stderr)
        return []

    if not contracts:
        print("No NBA contracts found.")
        return []
    print(f"  Found {len(contracts)} NBA contracts")

    # --- Step 2: Fetch external odds ---
    odds_client = OddsClient()
    mapper = MarketMapper()
    adapter = PriceAdapter()
    ref_by_token: dict[str, ReferencePrice] = {}

    # 2a: Championship / futures odds
    print("Fetching external championship odds...")
    try:
        champ_events = odds_client.get_nba_championship_odds(regions="us,eu")
    except (OddsClientError, ValueError) as e:
        print(f"  Championship odds unavailable: {e}", file=sys.stderr)
        champ_events = []

    champ_odds = []
    for event in champ_events:
        champ_odds.extend(parse_event_to_external_odds(event))

    if champ_odds:
        print(f"  Got {len(champ_odds)} championship odds entries")
        mapped = mapper.map_championship(champ_odds, contracts)
        for mm in mapped:
            for rp in adapter.adapt(mm):
                ref_by_token[rp.token_id] = rp
        print(f"  Mapped {len(ref_by_token)} contracts to championship reference prices")

    # 2b: Game odds (h2h moneylines)
    print("Fetching external game odds...")
    try:
        game_events = odds_client.get_nba_game_odds(regions="us,eu")
    except (OddsClientError, ValueError) as e:
        print(f"  Game odds unavailable: {e}", file=sys.stderr)
        game_events = []

    if game_events:
        print(f"  Got {len(game_events)} upcoming game events")
        game_mapped = mapper.map_all_games(game_events, contracts)
        before = len(ref_by_token)
        for mm in game_mapped:
            for rp in adapter.adapt(mm):
                ref_by_token[rp.token_id] = rp
        print(f"  Mapped {len(ref_by_token) - before} additional contracts to game reference prices")

    if ref_by_token:
        print(f"  Total: {len(ref_by_token)} contracts with reference prices")
    else:
        print("  No external odds available — metrics will be partial")

    # --- Step 3: Compute metrics ---
    metrics: list[MarketMetrics] = []
    for contract in contracts:
        ref = ref_by_token.get(contract.token_id)
        m = compute_metrics(contract, ref)
        metrics.append(m)

    # Sort: tier ascending, then divergence ascending (None last)
    metrics.sort(key=lambda m: (m.tier, m.divergence_bps if m.divergence_bps is not None else float("inf")))

    # --- Step 4: Output ---
    _print_table(metrics)

    if json_path:
        _write_json(metrics, json_path)
        print(f"\nJSON output written to {json_path} ({len(metrics)} entries)")
        print("  Tip: Edit this file to select markets, then: "
              f"python3 -m src.runner --markets {json_path}")

    return metrics


def _print_table(metrics: list[MarketMetrics]) -> None:
    """Print a formatted table of market metrics."""
    tier_labels = {1: "T1-Best", 2: "T2-Good", 3: "T3-Avoid"}

    header = (
        f"{'Tier':<10} {'Outcome':<25} {'Poly':>6} {'Fair':>6} "
        f"{'Div(bp)':>8} {'Vol($)':>12} {'Spread':>7} {'Edge(bp)':>9}"
    )
    print(f"\n{'=' * len(header)}")
    print("NBA Market Analysis — MM Suitability Ranking")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))

    for m in metrics:
        tier = tier_labels.get(m.tier, "??")
        outcome = m.outcome[:24]
        poly = f"{m.poly_price:.3f}" if m.poly_price is not None else "  N/A"
        fair = f"{m.fair_value:.3f}" if m.fair_value is not None else "  N/A"
        div = f"{m.divergence_bps:.0f}" if m.divergence_bps is not None else "    N/A"
        vol = f"{m.volume_usd:,.0f}" if m.volume_usd is not None else "        N/A"
        spread = f"{m.spread_estimate:.3f}" if m.spread_estimate is not None else "  N/A"
        edge = f"{m.edge_bps:.0f}" if m.edge_bps is not None else "    N/A"

        print(f"{tier:<10} {outcome:<25} {poly:>6} {fair:>6} {div:>8} {vol:>12} {spread:>7} {edge:>9}")

    # Summary
    by_tier = {1: 0, 2: 0, 3: 0}
    for m in metrics:
        by_tier[m.tier] = by_tier.get(m.tier, 0) + 1
    print(f"\nTotal: {len(metrics)} contracts — "
          f"T1: {by_tier[1]}, T2: {by_tier[2]}, T3: {by_tier[3]}")


def _write_json(metrics: list[MarketMetrics], path: str) -> None:
    """Write metrics to a JSON file."""
    data = []
    for m in metrics:
        data.append({
            "token_id": m.token_id,
            "question": m.question,
            "outcome": m.outcome,
            "poly_price": m.poly_price,
            "fair_value": m.fair_value,
            "divergence_bps": m.divergence_bps,
            "volume_usd": m.volume_usd,
            "spread_estimate": m.spread_estimate,
            "edge_bps": m.edge_bps,
            "tier": m.tier,
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    """CLI entry point."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Analyze NBA markets on Polymarket for MM suitability"
    )
    parser.add_argument(
        "--json",
        metavar="FILE",
        help="Write results to a JSON file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    run_analysis(json_path=args.json)


if __name__ == "__main__":
    main()
