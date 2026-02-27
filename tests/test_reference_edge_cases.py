"""Tests for edge case handling in the reference price module.

Tests staleness detection, unmatched market detection,
and synthetic odds generation.
"""

from datetime import datetime, timedelta

import pytest

from src.reference.market_mapper import MarketMapper
from src.reference.models import (
    ExternalOdds,
    MappedMarket,
    MarketType,
    PolymarketContract,
    ReferencePrice,
)
from src.reference.staleness import StalenessChecker


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@pytest.fixture()
def fixed_now() -> datetime:
    """A fixed 'now' time for deterministic testing."""
    return datetime(2026, 2, 14, 12, 0, 0)


@pytest.fixture()
def checker(fixed_now: datetime) -> StalenessChecker:
    """StalenessChecker with a fixed clock."""
    return StalenessChecker(clock=lambda: fixed_now)


@pytest.fixture()
def mapper() -> MarketMapper:
    """MarketMapper with no overrides file."""
    return MarketMapper(overrides_path="/nonexistent/path.json")


@pytest.fixture()
def fresh_price(fixed_now: datetime) -> ReferencePrice:
    """A reference price that is 60 seconds old (fresh)."""
    return ReferencePrice(
        token_id="token_fresh",
        fair_probability=0.45,
        raw_probability=0.48,
        source="test",
        timestamp=fixed_now - timedelta(seconds=60),
        market_type=MarketType.CHAMPIONSHIP,
    )


@pytest.fixture()
def stale_price(fixed_now: datetime) -> ReferencePrice:
    """A reference price that is 600 seconds old (stale at 300s threshold)."""
    return ReferencePrice(
        token_id="token_stale",
        fair_probability=0.30,
        raw_probability=0.33,
        source="test",
        timestamp=fixed_now - timedelta(seconds=600),
        market_type=MarketType.CHAMPIONSHIP,
    )


@pytest.fixture()
def borderline_price(fixed_now: datetime) -> ReferencePrice:
    """A reference price that is exactly 300 seconds old (on the boundary)."""
    return ReferencePrice(
        token_id="token_borderline",
        fair_probability=0.20,
        raw_probability=0.22,
        source="test",
        timestamp=fixed_now - timedelta(seconds=300),
        market_type=MarketType.CHAMPIONSHIP,
    )


@pytest.fixture()
def sample_external_odds() -> list[ExternalOdds]:
    """External odds for multiple teams."""
    return [
        ExternalOdds(
            team="Boston Celtics",
            american_odds=350,
            implied_probability=0.222,
            bookmaker="pinnacle",
        ),
        ExternalOdds(
            team="Oklahoma City Thunder",
            american_odds=400,
            implied_probability=0.200,
            bookmaker="pinnacle",
        ),
        ExternalOdds(
            team="Denver Nuggets",
            american_odds=800,
            implied_probability=0.111,
            bookmaker="pinnacle",
        ),
        ExternalOdds(
            team="Sacramento Kings",
            american_odds=5000,
            implied_probability=0.020,
            bookmaker="pinnacle",
        ),
        ExternalOdds(
            team="Charlotte Hornets",
            american_odds=10000,
            implied_probability=0.010,
            bookmaker="pinnacle",
        ),
    ]


@pytest.fixture()
def sample_poly_contracts() -> list[PolymarketContract]:
    """Polymarket contracts for some teams (not all)."""
    return [
        PolymarketContract(
            token_id="token_celtics",
            condition_id="cond_1",
            question="Will the Celtics win the 2026 NBA Championship?",
            outcome="Celtics",
            current_price=0.20,
        ),
        PolymarketContract(
            token_id="token_thunder",
            condition_id="cond_2",
            question="Will the Thunder win the 2026 NBA Championship?",
            outcome="Thunder",
            current_price=0.18,
        ),
        PolymarketContract(
            token_id="token_nuggets",
            condition_id="cond_3",
            question="Will the Nuggets win the 2026 NBA Championship?",
            outcome="Nuggets",
            current_price=0.10,
        ),
        PolymarketContract(
            token_id="token_lakers",
            condition_id="cond_4",
            question="Will the Lakers win the 2026 NBA Championship?",
            outcome="Lakers",
            current_price=0.08,
        ),
        PolymarketContract(
            token_id="token_warriors",
            condition_id="cond_5",
            question="Will the Warriors win the 2026 NBA Championship?",
            outcome="Warriors",
            current_price=0.05,
        ),
    ]


# -------------------------------------------------------------------
# Tests for StalenessChecker.is_stale
# -------------------------------------------------------------------


class TestIsStale:
    """Tests for StalenessChecker.is_stale."""

    def test_fresh_price_is_not_stale(
        self, checker: StalenessChecker, fresh_price: ReferencePrice
    ) -> None:
        """A 60-second-old price should not be stale at 300s threshold."""
        assert checker.is_stale(fresh_price, max_age_seconds=300) is False

    def test_stale_price_is_stale(
        self, checker: StalenessChecker, stale_price: ReferencePrice
    ) -> None:
        """A 600-second-old price should be stale at 300s threshold."""
        assert checker.is_stale(stale_price, max_age_seconds=300) is True

    def test_borderline_price_is_not_stale(
        self, checker: StalenessChecker, borderline_price: ReferencePrice
    ) -> None:
        """A price exactly at the threshold (300s) should NOT be stale."""
        assert checker.is_stale(borderline_price, max_age_seconds=300) is False

    def test_just_over_threshold_is_stale(
        self, checker: StalenessChecker, fixed_now: datetime
    ) -> None:
        """A price 301 seconds old should be stale at 300s threshold."""
        price = ReferencePrice(
            token_id="token_301",
            fair_probability=0.50,
            raw_probability=0.55,
            source="test",
            timestamp=fixed_now - timedelta(seconds=301),
            market_type=MarketType.GAME_ML,
        )
        assert checker.is_stale(price, max_age_seconds=300) is True

    def test_custom_threshold(
        self, checker: StalenessChecker, fresh_price: ReferencePrice
    ) -> None:
        """With a 30s threshold, a 60s-old price should be stale."""
        assert checker.is_stale(fresh_price, max_age_seconds=30) is True

    def test_very_large_threshold(
        self, checker: StalenessChecker, stale_price: ReferencePrice
    ) -> None:
        """With a very large threshold, even old prices are fresh."""
        assert checker.is_stale(stale_price, max_age_seconds=10000) is False

    def test_zero_age_is_fresh(
        self, checker: StalenessChecker, fixed_now: datetime
    ) -> None:
        """A price with timestamp equal to now should be fresh."""
        price = ReferencePrice(
            token_id="token_now",
            fair_probability=0.50,
            raw_probability=0.55,
            source="test",
            timestamp=fixed_now,
            market_type=MarketType.GAME_ML,
        )
        assert checker.is_stale(price, max_age_seconds=300) is False


# -------------------------------------------------------------------
# Tests for StalenessChecker.filter_stale
# -------------------------------------------------------------------


class TestFilterStale:
    """Tests for StalenessChecker.filter_stale."""

    def test_filters_stale_keeps_fresh(
        self,
        checker: StalenessChecker,
        fresh_price: ReferencePrice,
        stale_price: ReferencePrice,
    ) -> None:
        """Should keep fresh prices and remove stale ones."""
        result = checker.filter_stale([fresh_price, stale_price], max_age_seconds=300)
        assert len(result) == 1
        assert result[0].token_id == "token_fresh"

    def test_empty_list(self, checker: StalenessChecker) -> None:
        """Empty input should return empty output."""
        result = checker.filter_stale([], max_age_seconds=300)
        assert result == []

    def test_all_fresh(
        self,
        checker: StalenessChecker,
        fresh_price: ReferencePrice,
        borderline_price: ReferencePrice,
    ) -> None:
        """When all prices are fresh, all should be returned."""
        result = checker.filter_stale(
            [fresh_price, borderline_price], max_age_seconds=300
        )
        assert len(result) == 2

    def test_all_stale(
        self,
        checker: StalenessChecker,
        stale_price: ReferencePrice,
    ) -> None:
        """When all prices are stale, result should be empty."""
        result = checker.filter_stale([stale_price], max_age_seconds=300)
        assert result == []

    def test_preserves_order(
        self,
        checker: StalenessChecker,
        fixed_now: datetime,
    ) -> None:
        """Returned prices should preserve the input order."""
        prices = [
            ReferencePrice(
                token_id=f"token_{i}",
                fair_probability=0.1 * (i + 1),
                raw_probability=0.1 * (i + 1),
                source="test",
                timestamp=fixed_now - timedelta(seconds=i * 10),
                market_type=MarketType.CHAMPIONSHIP,
            )
            for i in range(5)
        ]
        result = checker.filter_stale(prices, max_age_seconds=300)
        assert len(result) == 5
        assert [r.token_id for r in result] == [f"token_{i}" for i in range(5)]


# -------------------------------------------------------------------
# Tests for StalenessChecker.get_staleness_report
# -------------------------------------------------------------------


class TestStalenessReport:
    """Tests for StalenessChecker.get_staleness_report."""

    def test_empty_list(self, checker: StalenessChecker) -> None:
        """Empty input should return zeroed report."""
        report = checker.get_staleness_report([])
        assert report["total"] == 0
        assert report["fresh_count"] == 0
        assert report["stale_count"] == 0
        assert report["oldest_age_seconds"] is None

    def test_mixed_staleness(
        self,
        checker: StalenessChecker,
        fresh_price: ReferencePrice,
        stale_price: ReferencePrice,
    ) -> None:
        """Report should correctly count fresh and stale prices."""
        report = checker.get_staleness_report(
            [fresh_price, stale_price], max_age_seconds=300
        )
        assert report["total"] == 2
        assert report["fresh_count"] == 1
        assert report["stale_count"] == 1
        assert report["oldest_age_seconds"] == 600.0
        assert report["newest_age_seconds"] == 60.0


# -------------------------------------------------------------------
# Tests for MarketMapper.get_unmatched_external
# -------------------------------------------------------------------


class TestGetUnmatchedExternal:
    """Tests for detecting external odds with no Polymarket match."""

    def test_identifies_unmatched_external(
        self,
        mapper: MarketMapper,
        sample_external_odds: list[ExternalOdds],
        sample_poly_contracts: list[PolymarketContract],
    ) -> None:
        """Kings and Hornets have no Polymarket contracts, so should be unmatched."""
        mapped = mapper.map_championship(sample_external_odds, sample_poly_contracts)
        unmatched = mapper.get_unmatched_external(sample_external_odds, mapped)

        unmatched_teams = {o.team for o in unmatched}
        assert "Sacramento Kings" in unmatched_teams
        assert "Charlotte Hornets" in unmatched_teams

    def test_all_matched_returns_empty(
        self,
        mapper: MarketMapper,
    ) -> None:
        """When all external odds are matched, unmatched should be empty."""
        ext = [
            ExternalOdds(
                team="Boston Celtics",
                american_odds=350,
                implied_probability=0.222,
                bookmaker="pinnacle",
            ),
        ]
        poly = [
            PolymarketContract(
                token_id="token_celtics",
                condition_id="cond_1",
                question="Will the Celtics win the 2026 NBA Championship?",
                outcome="Celtics",
                current_price=0.20,
            ),
        ]
        mapped = mapper.map_championship(ext, poly)
        unmatched = mapper.get_unmatched_external(ext, mapped)
        assert len(unmatched) == 0

    def test_empty_mapped_returns_all(
        self,
        mapper: MarketMapper,
        sample_external_odds: list[ExternalOdds],
    ) -> None:
        """With no mapped markets, all external odds should be unmatched."""
        unmatched = mapper.get_unmatched_external(sample_external_odds, [])
        assert len(unmatched) == len(sample_external_odds)


# -------------------------------------------------------------------
# Tests for MarketMapper.get_unmatched_polymarket
# -------------------------------------------------------------------


class TestGetUnmatchedPolymarket:
    """Tests for detecting Polymarket contracts with no external match."""

    def test_identifies_unmatched_polymarket(
        self,
        mapper: MarketMapper,
        sample_external_odds: list[ExternalOdds],
        sample_poly_contracts: list[PolymarketContract],
    ) -> None:
        """Lakers and Warriors have no external odds, so should be unmatched."""
        mapped = mapper.map_championship(sample_external_odds, sample_poly_contracts)
        unmatched = mapper.get_unmatched_polymarket(sample_poly_contracts, mapped)

        unmatched_tokens = {c.token_id for c in unmatched}
        assert "token_lakers" in unmatched_tokens
        assert "token_warriors" in unmatched_tokens

    def test_all_matched_returns_empty(
        self,
        mapper: MarketMapper,
    ) -> None:
        """When all contracts are matched, unmatched should be empty."""
        ext = [
            ExternalOdds(
                team="Boston Celtics",
                american_odds=350,
                implied_probability=0.222,
                bookmaker="pinnacle",
            ),
        ]
        poly = [
            PolymarketContract(
                token_id="token_celtics",
                condition_id="cond_1",
                question="Will the Celtics win the 2026 NBA Championship?",
                outcome="Celtics",
                current_price=0.20,
            ),
        ]
        mapped = mapper.map_championship(ext, poly)
        unmatched = mapper.get_unmatched_polymarket(poly, mapped)
        assert len(unmatched) == 0

    def test_empty_mapped_returns_all(
        self,
        mapper: MarketMapper,
        sample_poly_contracts: list[PolymarketContract],
    ) -> None:
        """With no mapped markets, all contracts should be unmatched."""
        unmatched = mapper.get_unmatched_polymarket(sample_poly_contracts, [])
        assert len(unmatched) == len(sample_poly_contracts)

    def test_partial_match(
        self,
        mapper: MarketMapper,
    ) -> None:
        """Only contracts not in any mapped market should be returned."""
        poly = [
            PolymarketContract(
                token_id="token_A",
                condition_id="cond_A",
                question="Will the Celtics win?",
                outcome="Celtics",
            ),
            PolymarketContract(
                token_id="token_B",
                condition_id="cond_B",
                question="Will the Heat win?",
                outcome="Heat",
            ),
        ]
        # Only token_A is mapped
        mapped = [
            MappedMarket(
                external_odds=[],
                polymarket_contracts=[poly[0]],
                market_type=MarketType.CHAMPIONSHIP,
            ),
        ]
        unmatched = mapper.get_unmatched_polymarket(poly, mapped)
        assert len(unmatched) == 1
        assert unmatched[0].token_id == "token_B"


# -------------------------------------------------------------------
# Tests for synthetic odds generation
# -------------------------------------------------------------------


class TestSyntheticOdds:
    """Tests for the synthetic odds generator used in validation."""

    def test_generates_odds_for_priced_contracts(self) -> None:
        """Synthetic odds should be generated for contracts with prices."""
        from src.reference.validate import generate_synthetic_odds

        contracts = [
            PolymarketContract(
                token_id="t1",
                condition_id="c1",
                question="Will the Lakers win?",
                outcome="Lakers",
                current_price=0.30,
            ),
            PolymarketContract(
                token_id="t2",
                condition_id="c2",
                question="Will the Celtics win?",
                outcome="Celtics",
                current_price=0.25,
            ),
        ]
        odds = generate_synthetic_odds(contracts)
        assert len(odds) == 2

    def test_skips_contracts_without_price(self) -> None:
        """Contracts with no current_price should be skipped."""
        from src.reference.validate import generate_synthetic_odds

        contracts = [
            PolymarketContract(
                token_id="t1",
                condition_id="c1",
                question="Will the Lakers win?",
                outcome="Lakers",
                current_price=None,
            ),
        ]
        odds = generate_synthetic_odds(contracts)
        assert len(odds) == 0

    def test_probabilities_in_valid_range(self) -> None:
        """Synthetic odds probabilities should be in (0, 1)."""
        from src.reference.validate import generate_synthetic_odds

        contracts = [
            PolymarketContract(
                token_id=f"t{i}",
                condition_id=f"c{i}",
                question=f"Will team {i} win?",
                outcome=f"Team{i}",
                current_price=p,
            )
            for i, p in enumerate([0.01, 0.10, 0.50, 0.90, 0.99])
        ]
        odds = generate_synthetic_odds(contracts)
        for o in odds:
            assert 0.0 < o.implied_probability < 1.5  # vig-inflated but bounded

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed should produce identical results."""
        from src.reference.validate import generate_synthetic_odds

        contracts = [
            PolymarketContract(
                token_id="t1",
                condition_id="c1",
                question="Will the Lakers win?",
                outcome="Lakers",
                current_price=0.30,
            ),
        ]
        odds1 = generate_synthetic_odds(contracts, seed=42)
        odds2 = generate_synthetic_odds(contracts, seed=42)
        assert odds1[0].implied_probability == odds2[0].implied_probability

    def test_different_seeds_produce_different_results(self) -> None:
        """Different seeds should produce different results."""
        from src.reference.validate import generate_synthetic_odds

        contracts = [
            PolymarketContract(
                token_id="t1",
                condition_id="c1",
                question="Will the Lakers win?",
                outcome="Lakers",
                current_price=0.30,
            ),
        ]
        odds1 = generate_synthetic_odds(contracts, seed=42)
        odds2 = generate_synthetic_odds(contracts, seed=99)
        assert odds1[0].implied_probability != odds2[0].implied_probability

    def test_bookmaker_is_synthetic(self) -> None:
        """Synthetic odds should be marked with bookmaker='synthetic'."""
        from src.reference.validate import generate_synthetic_odds

        contracts = [
            PolymarketContract(
                token_id="t1",
                condition_id="c1",
                question="Will the Lakers win?",
                outcome="Lakers",
                current_price=0.30,
            ),
        ]
        odds = generate_synthetic_odds(contracts)
        assert odds[0].bookmaker == "synthetic"
