"""Live validation script for reference price source.

Fetches live NBA markets from Polymarket, fetches or synthesizes
external reference odds, maps and adapts them, then compares
reference prices vs Polymarket mid-prices to measure divergence.

Usage:
    cd /home/yuqing/polymarket-trading/repos/polymarket-trading
    python3 -m src.reference.validate

If ODDS_API_KEY is set, uses live odds from The Odds API.
Otherwise, generates synthetic reference odds for testing.
"""

import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from .market_mapper import MarketMapper, get_canonical_name, normalize_team_name
from .models import (
    ExternalOdds,
    MappedMarket,
    MarketType,
    PolymarketContract,
    ReferencePrice,
)
from .polymarket_scanner import PolymarketScanner
from .price_adapter import PriceAdapter
from .staleness import StalenessChecker

logger = logging.getLogger(__name__)

# Where to write the validation report
WORKSPACE_DIR = Path(
    "/home/yuqing/polymarket-trading/orchestrator/"
    "PROGRAMS/P-2026-001-nba-market-research/workspace"
)
REPORT_PATH = WORKSPACE_DIR / "validation-report.md"


def fetch_polymarket_contracts(max_pages: int = 5) -> list[PolymarketContract]:
    """Fetch active NBA contracts from Polymarket's Gamma API.

    Limits pagination to avoid rate limiting, and filters results
    to only include contracts that are actually about NBA basketball.

    Args:
        max_pages: Maximum number of pages to fetch (100 contracts per page).

    Returns:
        List of PolymarketContract models for NBA markets.
    """
    scanner = PolymarketScanner(timeout=30)

    # Fetch with pagination limit to avoid 429 rate limiting
    all_contracts: list[PolymarketContract] = []
    page_size = 100

    for page in range(max_pages):
        offset = page * page_size
        try:
            markets = scanner.get_nba_markets(limit=page_size, offset=offset)
        except Exception as e:
            logger.warning("Failed to fetch page %d: %s", page, e)
            break

        if not markets:
            break

        for market in markets:
            contracts = scanner._parse_market(market)
            all_contracts.extend(contracts)

        if len(markets) < page_size:
            break

    # Filter to only NBA-relevant contracts (the 'nba' tag can be broad)
    nba_contracts = _filter_nba_contracts(all_contracts)
    logger.info(
        "Fetched %d contracts from Polymarket, %d are NBA-relevant",
        len(all_contracts),
        len(nba_contracts),
    )
    return nba_contracts


# NBA-related keywords for filtering contract questions
_NBA_KEYWORDS = [
    "nba", "basketball", "championship", "playoffs", "finals",
    "celtics", "lakers", "warriors", "bucks", "76ers", "sixers",
    "nuggets", "heat", "suns", "mavericks", "mavs", "thunder",
    "knicks", "nets", "clippers", "bulls", "cavaliers", "cavs",
    "rockets", "pacers", "hawks", "grizzlies", "pelicans",
    "timberwolves", "wolves", "raptors", "spurs", "kings",
    "trail blazers", "blazers", "pistons", "hornets", "magic",
    "jazz", "wizards",
]


def _filter_nba_contracts(contracts: list[PolymarketContract]) -> list[PolymarketContract]:
    """Filter contracts to only those about NBA basketball.

    The Gamma API 'nba' tag can be broad, so we check questions
    for NBA-related keywords.

    Args:
        contracts: Raw list of contracts from the API.

    Returns:
        Filtered list of NBA-relevant contracts.
    """
    result = []
    for c in contracts:
        q_lower = c.question.lower()
        if any(kw in q_lower for kw in _NBA_KEYWORDS):
            result.append(c)
    return result


def fetch_external_odds() -> tuple[list[ExternalOdds], str]:
    """Fetch external odds from The Odds API if available.

    Returns:
        Tuple of (list of ExternalOdds, source description string).

    Raises:
        ValueError: If the API key is set but the request fails.
    """
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        return [], "none"

    from .odds_client import OddsClient, OddsClientError, parse_event_to_external_odds

    try:
        client = OddsClient(api_key=api_key)

        # Fetch championship odds
        champ_events = client.get_nba_championship_odds()
        all_odds: list[ExternalOdds] = []
        for event in champ_events:
            odds = parse_event_to_external_odds(event)
            all_odds.extend(odds)

        # Fetch game odds
        game_events = client.get_nba_game_odds()
        for event in game_events:
            odds = parse_event_to_external_odds(event)
            all_odds.extend(odds)

        source = f"the-odds-api (live, {len(champ_events)} champ events, {len(game_events)} game events)"
        logger.info("Fetched %d external odds entries from The Odds API", len(all_odds))
        return all_odds, source

    except OddsClientError as e:
        logger.warning("Odds API error: %s — falling back to synthetic", e)
        return [], "none (API error)"


def generate_synthetic_odds(
    contracts: list[PolymarketContract],
    noise_std: float = 0.05,
    seed: int = 42,
) -> list[ExternalOdds]:
    """Generate synthetic external odds based on Polymarket prices plus noise.

    Creates fake bookmaker odds by taking Polymarket mid-prices, adding
    random noise, and converting back to American odds format. This allows
    validation testing without an API key.

    Args:
        contracts: Polymarket contracts to base synthetic odds on.
        noise_std: Standard deviation of the Gaussian noise added to
                   Polymarket prices (default 0.05 = 5 percentage points).
        seed: Random seed for reproducibility.

    Returns:
        List of synthetic ExternalOdds entries.
    """
    rng = random.Random(seed)
    synthetic: list[ExternalOdds] = []

    for contract in contracts:
        if contract.current_price is None:
            continue

        # Add noise to simulate external bookmaker divergence
        base_prob = contract.current_price
        noise = rng.gauss(0, noise_std)
        noisy_prob = max(0.02, min(0.98, base_prob + noise))

        # Add vig (inflate probability by ~5%)
        vig_prob = noisy_prob * 1.05

        # Convert to American odds for realism
        if noisy_prob >= 0.5:
            american = int(-100 * noisy_prob / (1 - noisy_prob))
        else:
            american = int(100 * (1 - noisy_prob) / noisy_prob)

        # Try to extract a team name from the contract
        team = _extract_team_or_outcome(contract)

        odds = ExternalOdds(
            team=team,
            american_odds=american,
            implied_probability=vig_prob,
            bookmaker="synthetic",
            timestamp=datetime.now(timezone.utc),
        )
        synthetic.append(odds)

    logger.info("Generated %d synthetic external odds entries", len(synthetic))
    return synthetic


def _extract_team_or_outcome(contract: PolymarketContract) -> str:
    """Extract the best team or outcome label from a contract.

    Args:
        contract: A Polymarket contract.

    Returns:
        The canonical team name if found, otherwise the contract outcome.
    """
    # Try to normalize from outcome first
    canonical = normalize_team_name(contract.outcome)
    if canonical:
        return canonical

    # Try from question
    canonical = normalize_team_name(contract.question)
    if canonical:
        return canonical

    # Fallback to raw outcome
    return contract.outcome


def run_validation() -> str:
    """Run the full validation pipeline and return the report as a string.

    Steps:
        1. Fetch Polymarket contracts
        2. Fetch or synthesize external odds
        3. Map markets
        4. Adapt prices
        5. Compare and compute divergence
        6. Generate report

    Returns:
        Markdown-formatted validation report string.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Step 1: Fetch Polymarket contracts
    print("Step 1: Fetching Polymarket NBA contracts...")
    try:
        contracts = fetch_polymarket_contracts()
    except Exception as e:
        contracts = []
        print(f"  Warning: Failed to fetch Polymarket contracts: {e}")

    print(f"  Found {len(contracts)} contracts")

    # Step 2: Fetch or synthesize external odds
    print("Step 2: Fetching external odds...")
    external_odds, odds_source = fetch_external_odds()

    using_synthetic = False
    if not external_odds:
        print("  No ODDS_API_KEY set — generating synthetic reference odds")
        external_odds = generate_synthetic_odds(contracts)
        odds_source = f"synthetic (noise_std=0.05, based on {len(contracts)} Polymarket contracts)"
        using_synthetic = True
    else:
        print(f"  Fetched {len(external_odds)} odds from {odds_source}")

    # Step 3: Map markets
    print("Step 3: Mapping markets...")
    mapper = MarketMapper()

    # For championship mapping, filter to championship-looking contracts
    champ_contracts = _filter_championship_contracts(contracts)
    champ_external = _filter_championship_odds(external_odds)

    mapped_markets: list[MappedMarket] = []

    if champ_contracts and champ_external:
        champ_mapped = mapper.map_championship(champ_external, champ_contracts)
        mapped_markets.extend(champ_mapped)
        print(f"  Championship: {len(champ_mapped)} markets mapped")

    # For synthetic odds of non-championship contracts, do direct matching
    if using_synthetic:
        non_champ = [c for c in contracts if c not in champ_contracts]
        non_champ_odds = [o for o in external_odds if o not in champ_external]
        direct_mapped = _direct_map_synthetic(non_champ, non_champ_odds)
        mapped_markets.extend(direct_mapped)
        print(f"  Direct (synthetic): {len(direct_mapped)} markets mapped")

    total_mapped = len(mapped_markets)
    print(f"  Total mapped: {total_mapped}")

    # Unmatched analysis
    unmatched_ext = mapper.get_unmatched_external(external_odds, mapped_markets)
    unmatched_poly = mapper.get_unmatched_polymarket(contracts, mapped_markets)
    print(f"  Unmatched external: {len(unmatched_ext)}")
    print(f"  Unmatched Polymarket: {len(unmatched_poly)}")

    # Step 4: Adapt prices
    print("Step 4: Adapting prices (vig removal)...")
    adapter = PriceAdapter(vig_method="proportional", source=odds_source.split(" ")[0])

    all_ref_prices: list[ReferencePrice] = []
    comparison_rows: list[dict] = []

    for mm in mapped_markets:
        try:
            ref_prices = adapter.adapt(mm)
            all_ref_prices.extend(ref_prices)

            # Build comparison rows
            for rp in ref_prices:
                # Find the matching Polymarket contract
                matching_contract = next(
                    (c for c in mm.polymarket_contracts if c.token_id == rp.token_id),
                    None,
                )
                poly_mid = matching_contract.current_price if matching_contract else None

                divergence = None
                if poly_mid is not None:
                    divergence = rp.fair_probability - poly_mid

                row = {
                    "event_name": mm.event_name or "Unknown",
                    "market_type": mm.market_type.value,
                    "outcome": matching_contract.outcome if matching_contract else "?",
                    "ref_prob": rp.fair_probability,
                    "raw_prob": rp.raw_probability,
                    "poly_mid": poly_mid,
                    "divergence": divergence,
                    "abs_divergence": abs(divergence) if divergence is not None else None,
                    "token_id": rp.token_id[:16] + "..." if len(rp.token_id) > 16 else rp.token_id,
                    "bookmaker": rp.bookmaker or "?",
                }
                comparison_rows.append(row)
        except Exception as e:
            logger.warning("Failed to adapt market %s: %s", mm.event_name, e)

    print(f"  Adapted {len(all_ref_prices)} reference prices")

    # Step 5: Staleness check
    print("Step 5: Checking staleness...")
    checker = StalenessChecker()
    staleness_report = checker.get_staleness_report(all_ref_prices, max_age_seconds=300)
    print(f"  Fresh: {staleness_report['fresh_count']}, Stale: {staleness_report['stale_count']}")

    # Step 6: Compute summary statistics
    print("Step 6: Computing summary statistics...")
    valid_divergences = [
        r["abs_divergence"] for r in comparison_rows if r["abs_divergence"] is not None
    ]

    stats = {
        "total_polymarket_contracts": len(contracts),
        "total_external_odds": len(external_odds),
        "total_mapped_markets": total_mapped,
        "total_ref_prices": len(all_ref_prices),
        "total_comparisons": len(comparison_rows),
        "comparisons_with_price": len(valid_divergences),
        "unmatched_external": len(unmatched_ext),
        "unmatched_polymarket": len(unmatched_poly),
        "mean_abs_divergence": (
            sum(valid_divergences) / len(valid_divergences) if valid_divergences else None
        ),
        "max_abs_divergence": max(valid_divergences) if valid_divergences else None,
        "min_abs_divergence": min(valid_divergences) if valid_divergences else None,
        "median_abs_divergence": _median(valid_divergences) if valid_divergences else None,
    }

    print(f"  Mean abs divergence: {stats['mean_abs_divergence']}")
    print(f"  Max abs divergence: {stats['max_abs_divergence']}")

    # Step 7: Generate report
    print("Step 7: Generating report...")
    report = _generate_report(
        timestamp=timestamp,
        odds_source=odds_source,
        using_synthetic=using_synthetic,
        stats=stats,
        comparison_rows=comparison_rows,
        unmatched_ext=unmatched_ext,
        unmatched_poly=unmatched_poly,
        staleness_report=staleness_report,
    )

    return report


def _filter_championship_contracts(
    contracts: list[PolymarketContract],
) -> list[PolymarketContract]:
    """Filter contracts that look like championship/futures markets.

    Heuristic: question contains 'championship', 'title', or 'win the NBA'.

    Args:
        contracts: All Polymarket contracts.

    Returns:
        Contracts that appear to be championship markets.
    """
    keywords = ["championship", "win the nba", "nba title", "nba champion"]
    result = []
    for c in contracts:
        q_lower = c.question.lower()
        if any(kw in q_lower for kw in keywords):
            result.append(c)
    return result


def _filter_championship_odds(
    external_odds: list[ExternalOdds],
) -> list[ExternalOdds]:
    """Filter external odds that look like championship/futures odds.

    Heuristic: for synthetic odds, we match by team name being a
    canonical NBA team. For live odds, we accept all of them as
    championship odds come from a separate endpoint.

    Args:
        external_odds: All external odds.

    Returns:
        Odds entries likely for championship markets.
    """
    result = []
    for o in external_odds:
        canonical = normalize_team_name(o.team)
        if canonical is not None:
            result.append(o)
    return result


def _direct_map_synthetic(
    contracts: list[PolymarketContract],
    synthetic_odds: list[ExternalOdds],
) -> list[MappedMarket]:
    """Directly map synthetic odds to contracts by team name matching.

    Since synthetic odds are generated from Polymarket contracts,
    we can match them back directly by team/outcome name.

    Args:
        contracts: Polymarket contracts (non-championship).
        synthetic_odds: Synthetic odds entries.

    Returns:
        List of MappedMarket with 1:1 pairings.
    """
    mapped: list[MappedMarket] = []

    # Build lookup from team name to synthetic odds
    odds_by_team: dict[str, ExternalOdds] = {}
    for o in synthetic_odds:
        canonical = get_canonical_name(o.team)
        odds_by_team[canonical] = o

    for contract in contracts:
        team = normalize_team_name(contract.outcome) or normalize_team_name(contract.question)
        if team and team in odds_by_team:
            mm = MappedMarket(
                external_odds=[odds_by_team[team]],
                polymarket_contracts=[contract],
                market_type=MarketType.GAME_ML,
                event_name=f"Direct match: {team}",
                mapping_confidence=0.7,
            )
            mapped.append(mm)

    return mapped


def _median(values: list[float]) -> float:
    """Compute median of a list of floats.

    Args:
        values: List of numeric values. Must not be empty.

    Returns:
        Median value.
    """
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0


def _generate_report(
    timestamp: str,
    odds_source: str,
    using_synthetic: bool,
    stats: dict,
    comparison_rows: list[dict],
    unmatched_ext: list[ExternalOdds],
    unmatched_poly: list[PolymarketContract],
    staleness_report: dict,
) -> str:
    """Generate the markdown validation report.

    Args:
        timestamp: When the validation was run.
        odds_source: Description of the odds source used.
        using_synthetic: Whether synthetic odds were used.
        stats: Summary statistics dict.
        comparison_rows: List of comparison row dicts.
        unmatched_ext: Unmatched external odds.
        unmatched_poly: Unmatched Polymarket contracts.
        staleness_report: Staleness analysis dict.

    Returns:
        Markdown-formatted report string.
    """
    lines: list[str] = []

    lines.append("# Validation Report: Reference Prices vs Polymarket Mid-Prices")
    lines.append("")
    lines.append(f"**Generated:** {timestamp}")
    lines.append(f"**Odds Source:** {odds_source}")
    if using_synthetic:
        lines.append("")
        lines.append("> **Note:** Using synthetic reference odds (ODDS_API_KEY not set).")
        lines.append("> Synthetic odds are based on Polymarket prices with Gaussian noise (std=0.05).")
        lines.append("> This tests the pipeline end-to-end but divergence values reflect the added noise, not real market differences.")
    lines.append("")

    # Summary statistics
    lines.append("## Summary Statistics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Polymarket contracts scanned | {stats['total_polymarket_contracts']} |")
    lines.append(f"| External odds entries | {stats['total_external_odds']} |")
    lines.append(f"| Markets mapped | {stats['total_mapped_markets']} |")
    lines.append(f"| Reference prices generated | {stats['total_ref_prices']} |")
    lines.append(f"| Comparisons with both prices | {stats['comparisons_with_price']} |")
    lines.append(f"| Unmatched external odds | {stats['unmatched_external']} |")
    lines.append(f"| Unmatched Polymarket contracts | {stats['unmatched_polymarket']} |")

    if stats["mean_abs_divergence"] is not None:
        lines.append(f"| Mean absolute divergence | {stats['mean_abs_divergence']:.4f} ({stats['mean_abs_divergence']*100:.2f}%) |")
        lines.append(f"| Median absolute divergence | {stats['median_abs_divergence']:.4f} ({stats['median_abs_divergence']*100:.2f}%) |")
        lines.append(f"| Max absolute divergence | {stats['max_abs_divergence']:.4f} ({stats['max_abs_divergence']*100:.2f}%) |")
        lines.append(f"| Min absolute divergence | {stats['min_abs_divergence']:.4f} ({stats['min_abs_divergence']*100:.2f}%) |")
    else:
        lines.append("| Mean absolute divergence | N/A (no valid comparisons) |")

    lines.append("")

    # Staleness
    lines.append("## Staleness Analysis")
    lines.append("")
    lines.append(f"- Total reference prices: {staleness_report['total']}")
    lines.append(f"- Fresh (< 300s): {staleness_report['fresh_count']}")
    lines.append(f"- Stale (>= 300s): {staleness_report['stale_count']}")
    if staleness_report["oldest_age_seconds"] is not None:
        lines.append(f"- Oldest age: {staleness_report['oldest_age_seconds']:.1f}s")
        lines.append(f"- Newest age: {staleness_report['newest_age_seconds']:.1f}s")
    lines.append("")

    # Detailed comparison table
    lines.append("## Detailed Comparison")
    lines.append("")
    if comparison_rows:
        # Sort by absolute divergence descending
        sorted_rows = sorted(
            comparison_rows,
            key=lambda r: r["abs_divergence"] if r["abs_divergence"] is not None else -1,
            reverse=True,
        )

        lines.append("| Event | Type | Outcome | Ref Prob | Poly Mid | Divergence | Bookmaker |")
        lines.append("|-------|------|---------|----------|----------|------------|-----------|")

        for row in sorted_rows:
            ref_prob = f"{row['ref_prob']:.4f}" if row['ref_prob'] is not None else "N/A"
            poly_mid = f"{row['poly_mid']:.4f}" if row['poly_mid'] is not None else "N/A"
            if row['divergence'] is not None:
                div_sign = "+" if row['divergence'] >= 0 else ""
                divergence = f"{div_sign}{row['divergence']:.4f} ({div_sign}{row['divergence']*100:.2f}%)"
            else:
                divergence = "N/A"

            lines.append(
                f"| {row['event_name'][:40]} | {row['market_type']} | {row['outcome'][:20]} "
                f"| {ref_prob} | {poly_mid} | {divergence} | {row['bookmaker']} |"
            )
    else:
        lines.append("*No comparisons available.*")

    lines.append("")

    # Unmatched external odds
    lines.append("## Unmatched External Odds")
    lines.append("")
    if unmatched_ext:
        lines.append(f"These {len(unmatched_ext)} external odds entries had no matching Polymarket contract:")
        lines.append("")
        # Deduplicate by team name
        unmatched_teams = sorted({get_canonical_name(o.team) for o in unmatched_ext})
        for team in unmatched_teams[:30]:  # Cap at 30
            lines.append(f"- {team}")
        if len(unmatched_teams) > 30:
            lines.append(f"- ... and {len(unmatched_teams) - 30} more")
    else:
        lines.append("*All external odds were matched.*")

    lines.append("")

    # Unmatched Polymarket contracts
    lines.append("## Unmatched Polymarket Contracts")
    lines.append("")
    if unmatched_poly:
        lines.append(f"These {len(unmatched_poly)} Polymarket contracts had no matching external odds:")
        lines.append("")
        for c in unmatched_poly[:30]:  # Cap at 30
            price_str = f" (price: {c.current_price:.4f})" if c.current_price is not None else ""
            lines.append(f"- {c.question[:60]} / {c.outcome}{price_str}")
        if len(unmatched_poly) > 30:
            lines.append(f"- ... and {len(unmatched_poly) - 30} more")
    else:
        lines.append("*All Polymarket contracts were matched.*")

    lines.append("")
    lines.append("---")
    lines.append(f"*Report generated by `src.reference.validate` at {timestamp}*")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Entry point for the validation script."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 70)
    print("Reference Price Validation")
    print("=" * 70)
    print()

    try:
        report = run_validation()
    except Exception as e:
        print(f"\nFATAL: Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print to stdout
    print()
    print("=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)
    print()
    print(report)

    # Write to file
    try:
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(f"\nReport written to: {REPORT_PATH}")
    except Exception as e:
        print(f"\nWarning: Could not write report to {REPORT_PATH}: {e}")
        # Try current directory as fallback
        fallback = Path("validation-report.md")
        fallback.write_text(report, encoding="utf-8")
        print(f"Report written to fallback: {fallback.resolve()}")


if __name__ == "__main__":
    main()
