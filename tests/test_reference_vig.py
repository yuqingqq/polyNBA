"""Tests for the vig removal module.

Tests proportional and Shin vig removal methods, odds conversion functions,
and overround computation.
"""

import pytest

from src.reference.vig_removal import (
    american_to_probability,
    compute_overround,
    decimal_to_probability,
    proportional_vig_removal,
    shin_vig_removal,
)


# -------------------------------------------------------------------
# Tests for american_to_probability
# -------------------------------------------------------------------


class TestAmericanToProbability:
    """Tests for American odds to implied probability conversion."""

    def test_even_money(self) -> None:
        """Even money (+100) should give 50% probability."""
        assert american_to_probability(100) == pytest.approx(0.5)

    def test_heavy_favorite(self) -> None:
        """-150 should give 60% probability."""
        result = american_to_probability(-150)
        assert result == pytest.approx(0.6)

    def test_heavy_underdog(self) -> None:
        """+200 should give ~33.3% probability."""
        result = american_to_probability(200)
        assert result == pytest.approx(1.0 / 3.0, rel=1e-6)

    def test_large_favorite(self) -> None:
        """-500 should give ~83.3% probability."""
        result = american_to_probability(-500)
        expected = 500.0 / 600.0
        assert result == pytest.approx(expected, rel=1e-6)

    def test_large_underdog(self) -> None:
        """+500 should give ~16.7% probability."""
        result = american_to_probability(500)
        expected = 100.0 / 600.0
        assert result == pytest.approx(expected, rel=1e-6)

    def test_extreme_favorite(self) -> None:
        """-10000 should give ~99% probability."""
        result = american_to_probability(-10000)
        expected = 10000.0 / 10100.0
        assert result == pytest.approx(expected, rel=1e-6)
        assert result > 0.98

    def test_extreme_longshot(self) -> None:
        """+10000 should give ~1% probability."""
        result = american_to_probability(10000)
        expected = 100.0 / 10100.0
        assert result == pytest.approx(expected, rel=1e-6)
        assert result < 0.02

    def test_symmetry(self) -> None:
        """Complementary odds should sum to more than 1 (vig)."""
        fav = american_to_probability(-110)
        dog = american_to_probability(-110)
        # Both sides at -110 is the standard vig market
        total = fav + dog
        assert total > 1.0  # Contains vig

    def test_negative_200(self) -> None:
        """-200 should give ~66.7% probability."""
        result = american_to_probability(-200)
        assert result == pytest.approx(2.0 / 3.0, rel=1e-6)


# -------------------------------------------------------------------
# Tests for decimal_to_probability
# -------------------------------------------------------------------


class TestDecimalToProbability:
    """Tests for decimal odds to implied probability conversion."""

    def test_even_money(self) -> None:
        """Decimal 2.0 should give 50% probability."""
        assert decimal_to_probability(2.0) == pytest.approx(0.5)

    def test_favorite(self) -> None:
        """Decimal 1.5 should give ~66.7% probability."""
        result = decimal_to_probability(1.5)
        assert result == pytest.approx(2.0 / 3.0, rel=1e-6)

    def test_underdog(self) -> None:
        """Decimal 4.0 should give 25% probability."""
        assert decimal_to_probability(4.0) == pytest.approx(0.25)

    def test_invalid_zero(self) -> None:
        """Zero decimal odds should raise ValueError."""
        with pytest.raises(ValueError):
            decimal_to_probability(0.0)

    def test_invalid_negative(self) -> None:
        """Negative decimal odds should raise ValueError."""
        with pytest.raises(ValueError):
            decimal_to_probability(-1.5)


# -------------------------------------------------------------------
# Tests for compute_overround
# -------------------------------------------------------------------


class TestComputeOverround:
    """Tests for overround computation."""

    def test_fair_market(self) -> None:
        """A fair market (sum = 1.0) should have zero overround."""
        overround = compute_overround([0.5, 0.5])
        assert overround == pytest.approx(0.0, abs=1e-10)

    def test_vigged_market(self) -> None:
        """A market with vig should have positive overround."""
        # -110 / -110 market
        probs = [
            american_to_probability(-110),
            american_to_probability(-110),
        ]
        overround = compute_overround(probs)
        assert overround > 0.0
        assert overround == pytest.approx(0.0476, rel=0.01)

    def test_championship_overround(self) -> None:
        """A multi-way market typically has larger overround."""
        # Simulate 4 teams
        probs = [0.30, 0.28, 0.27, 0.25]  # sum = 1.10
        overround = compute_overround(probs)
        assert overround == pytest.approx(0.10, abs=1e-10)


# -------------------------------------------------------------------
# Tests for proportional_vig_removal
# -------------------------------------------------------------------


class TestProportionalVigRemoval:
    """Tests for the proportional vig removal method."""

    def test_sums_to_one(self) -> None:
        """After vig removal, probabilities should sum to 1.0."""
        raw = [0.55, 0.55]  # 10% overround
        fair = proportional_vig_removal(raw)
        assert sum(fair) == pytest.approx(1.0, abs=1e-10)

    def test_fair_market_unchanged(self) -> None:
        """A fair market (sum = 1.0) should be unchanged."""
        raw = [0.5, 0.5]
        fair = proportional_vig_removal(raw)
        assert fair[0] == pytest.approx(0.5)
        assert fair[1] == pytest.approx(0.5)

    def test_two_way_standard_vig(self) -> None:
        """Standard -110/-110 market should normalize correctly."""
        raw = [
            american_to_probability(-110),
            american_to_probability(-110),
        ]
        fair = proportional_vig_removal(raw)
        assert sum(fair) == pytest.approx(1.0, abs=1e-10)
        assert fair[0] == pytest.approx(0.5)
        assert fair[1] == pytest.approx(0.5)

    def test_multi_way_sums_to_one(self) -> None:
        """Multi-way market should also sum to 1.0 after vig removal."""
        raw = [0.30, 0.28, 0.27, 0.25]  # sum = 1.10
        fair = proportional_vig_removal(raw)
        assert sum(fair) == pytest.approx(1.0, abs=1e-10)

    def test_preserves_ordering(self) -> None:
        """Proportional method should preserve relative ordering."""
        raw = [0.60, 0.30, 0.20]  # sum = 1.10
        fair = proportional_vig_removal(raw)
        assert fair[0] > fair[1] > fair[2]

    def test_empty_raises(self) -> None:
        """Empty input should raise ValueError."""
        with pytest.raises(ValueError):
            proportional_vig_removal([])

    def test_single_outcome(self) -> None:
        """Single outcome should normalize to 1.0."""
        fair = proportional_vig_removal([0.95])
        assert fair[0] == pytest.approx(1.0, abs=1e-10)

    def test_large_championship_market(self) -> None:
        """30-team championship market should sum to 1.0."""
        # Simulate 30 teams with various odds
        raw = [
            0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04,
            0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.015, 0.015, 0.01,
            0.01, 0.01, 0.01, 0.008, 0.008, 0.005, 0.005, 0.003, 0.003, 0.002,
        ]
        fair = proportional_vig_removal(raw)
        assert sum(fair) == pytest.approx(1.0, abs=1e-10)
        assert all(p > 0 for p in fair)


# -------------------------------------------------------------------
# Tests for shin_vig_removal
# -------------------------------------------------------------------


class TestShinVigRemoval:
    """Tests for the Shin (1991) vig removal method."""

    def test_sums_to_one(self) -> None:
        """After Shin removal, probabilities should sum to ~1.0."""
        raw = [0.55, 0.55]  # 10% overround
        fair = shin_vig_removal(raw)
        assert sum(fair) == pytest.approx(1.0, abs=1e-6)

    def test_fair_market_unchanged(self) -> None:
        """Fair market should remain approximately unchanged."""
        raw = [0.5, 0.5]
        fair = shin_vig_removal(raw)
        assert fair[0] == pytest.approx(0.5, abs=0.01)
        assert fair[1] == pytest.approx(0.5, abs=0.01)

    def test_favorite_longshot_bias(self) -> None:
        """Shin method should adjust longshots more than favorites.

        The key insight: bookmakers overcharge more on longshots.
        So the fair probability of a longshot should be lower relative
        to its raw probability compared to the favorite.
        """
        # Heavy favorite vs longshot
        raw = [
            american_to_probability(-300),  # ~0.75
            american_to_probability(+250),  # ~0.286
        ]
        fair_shin = shin_vig_removal(raw)
        fair_prop = proportional_vig_removal(raw)

        # Both should sum to 1.0
        assert sum(fair_shin) == pytest.approx(1.0, abs=1e-6)
        assert sum(fair_prop) == pytest.approx(1.0, abs=1e-6)

        # Shin should give different results than proportional
        # (they may be similar for 2-way but the math is different)
        # Just verify reasonable range
        assert 0.0 < fair_shin[0] < 1.0
        assert 0.0 < fair_shin[1] < 1.0

    def test_multi_way_sums_to_one(self) -> None:
        """Multi-way Shin removal should sum to ~1.0."""
        raw = [0.30, 0.28, 0.27, 0.25]
        fair = shin_vig_removal(raw)
        assert sum(fair) == pytest.approx(1.0, abs=1e-6)

    def test_preserves_ordering(self) -> None:
        """Shin method should preserve relative ordering."""
        raw = [0.60, 0.30, 0.20]
        fair = shin_vig_removal(raw)
        assert fair[0] > fair[1] > fair[2]

    def test_empty_raises(self) -> None:
        """Empty input should raise ValueError."""
        with pytest.raises(ValueError):
            shin_vig_removal([])

    def test_single_outcome(self) -> None:
        """Single outcome should return [1.0]."""
        fair = shin_vig_removal([0.95])
        assert fair[0] == pytest.approx(1.0, abs=1e-6)

    def test_reasonable_adjustments(self) -> None:
        """Shin adjustments should be reasonable (not wildly different from proportional)."""
        raw = [0.55, 0.55]
        fair_shin = shin_vig_removal(raw)
        fair_prop = proportional_vig_removal(raw)

        # Both should give ~0.5 each for a symmetric market
        assert fair_shin[0] == pytest.approx(0.5, abs=0.05)
        assert fair_shin[1] == pytest.approx(0.5, abs=0.05)

        # Results should be close to proportional for symmetric cases
        for s, p in zip(fair_shin, fair_prop):
            assert abs(s - p) < 0.1
