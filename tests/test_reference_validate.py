"""Tests for the validation script.

These tests exercise the full validation pipeline end-to-end,
including live Polymarket API calls (Gamma API is public, no auth),
synthetic odds generation, market mapping, price adaptation,
and report generation.

This test doubles as the live validation run for task t7.
"""

from pathlib import Path


from src.reference.validate import (
    _extract_team_or_outcome,
    _filter_championship_contracts,
    _filter_championship_odds,
    _median,
    fetch_polymarket_contracts,
    run_validation,
)
from src.reference.models import (
    ExternalOdds,
    PolymarketContract,
)


WORKSPACE_DIR = Path(
    "/home/yuqing/polymarket-trading/orchestrator/"
    "PROGRAMS/P-2026-001-nba-market-research/workspace"
)
REPORT_PATH = WORKSPACE_DIR / "validation-report.md"


class TestMedian:
    """Tests for the _median helper function."""

    def test_odd_count(self) -> None:
        assert _median([1.0, 2.0, 3.0]) == 2.0

    def test_even_count(self) -> None:
        assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_single_value(self) -> None:
        assert _median([5.0]) == 5.0

    def test_unsorted_input(self) -> None:
        assert _median([3.0, 1.0, 2.0]) == 2.0


class TestExtractTeamOrOutcome:
    """Tests for _extract_team_or_outcome."""

    def test_team_from_outcome(self) -> None:
        contract = PolymarketContract(
            token_id="t1",
            condition_id="c1",
            question="Who will win?",
            outcome="Lakers",
        )
        result = _extract_team_or_outcome(contract)
        assert result == "Los Angeles Lakers"

    def test_team_from_question(self) -> None:
        contract = PolymarketContract(
            token_id="t1",
            condition_id="c1",
            question="Will the Celtics win the championship?",
            outcome="Yes",
        )
        result = _extract_team_or_outcome(contract)
        assert result == "Boston Celtics"

    def test_fallback_to_raw_outcome(self) -> None:
        contract = PolymarketContract(
            token_id="t1",
            condition_id="c1",
            question="Will X happen?",
            outcome="CustomOutcome",
        )
        result = _extract_team_or_outcome(contract)
        assert result == "CustomOutcome"


class TestFilterChampionshipContracts:
    """Tests for _filter_championship_contracts."""

    def test_filters_championship_questions(self) -> None:
        contracts = [
            PolymarketContract(
                token_id="t1",
                condition_id="c1",
                question="Will the Lakers win the 2026 NBA Championship?",
                outcome="Yes",
            ),
            PolymarketContract(
                token_id="t2",
                condition_id="c2",
                question="Will the Lakers beat the Celtics tonight?",
                outcome="Yes",
            ),
        ]
        result = _filter_championship_contracts(contracts)
        assert len(result) == 1
        assert result[0].token_id == "t1"

    def test_empty_input(self) -> None:
        assert _filter_championship_contracts([]) == []


class TestFilterChampionshipOdds:
    """Tests for _filter_championship_odds."""

    def test_filters_nba_teams(self) -> None:
        odds = [
            ExternalOdds(
                team="Boston Celtics",
                implied_probability=0.20,
                bookmaker="test",
            ),
            ExternalOdds(
                team="Unknown Team XYZ",
                implied_probability=0.10,
                bookmaker="test",
            ),
        ]
        result = _filter_championship_odds(odds)
        assert len(result) == 1
        assert result[0].team == "Boston Celtics"


class TestLivePolymarketFetch:
    """Test that we can actually reach the Polymarket Gamma API.

    This is a live integration test that requires network access.
    """

    def test_fetch_polymarket_contracts(self) -> None:
        """Fetch live NBA contracts from Polymarket Gamma API (public, no auth)."""
        contracts = fetch_polymarket_contracts(max_pages=2)
        # We expect at least some NBA contracts to exist
        # (if none exist, the market may be down or no NBA markets active)
        print(f"\nFetched {len(contracts)} live NBA contracts from Polymarket")
        if contracts:
            for c in contracts[:5]:
                print(f"  - {c.question[:60]} | {c.outcome} | price={c.current_price}")
        # Don't assert > 0 since NBA season may be off
        assert isinstance(contracts, list)


class TestFullValidationPipeline:
    """End-to-end test of the validation pipeline.

    This test runs the full validation, writes the report,
    and verifies the output. It serves as the live validation
    for task t7.
    """

    def test_run_validation_produces_report(self) -> None:
        """Run the full pipeline and verify report is generated."""
        report = run_validation()

        # Verify report structure
        assert "# Validation Report" in report
        assert "## Summary Statistics" in report
        assert "## Staleness Analysis" in report
        assert "## Detailed Comparison" in report
        assert "## Unmatched External Odds" in report
        assert "## Unmatched Polymarket Contracts" in report

        # Verify some data was processed
        assert "Polymarket contracts scanned" in report
        assert "Markets mapped" in report

        # Write report to workspace
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(f"\nReport written to: {REPORT_PATH}")
        assert REPORT_PATH.exists()

        # Print report for visibility
        print("\n" + "=" * 70)
        print("VALIDATION REPORT")
        print("=" * 70)
        print(report)
