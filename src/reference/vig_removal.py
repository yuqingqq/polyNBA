"""Vig removal algorithms for converting bookmaker odds to fair probabilities.

Bookmaker odds include a margin (vig/juice/overround). These functions strip
the margin to estimate true implied probabilities.
"""

import math


def american_to_probability(american_odds: int) -> float:
    """Convert American odds to raw implied probability.

    Args:
        american_odds: American odds value (e.g., +350, -150).
            Positive odds: probability = 100 / (odds + 100)
            Negative odds: probability = |odds| / (|odds| + 100)

    Returns:
        Raw implied probability (0-1), includes vig.

    Examples:
        >>> american_to_probability(-150)  # Strong favorite
        0.6
        >>> american_to_probability(+200)  # Underdog
        0.3333...
        >>> american_to_probability(+100)  # Even money
        0.5
    """
    # I-2: Guard against american_odds == 0 (undefined in American odds).
    # Malformed API data could produce this; treat as even money (50%).
    if american_odds == 0:
        return 0.5
    if american_odds >= 0:
        return 100.0 / (american_odds + 100.0)
    else:
        return abs(american_odds) / (abs(american_odds) + 100.0)


def decimal_to_probability(decimal_odds: float) -> float:
    """Convert decimal odds to raw implied probability.

    Args:
        decimal_odds: Decimal odds value (e.g., 4.50, 1.67).
            probability = 1 / decimal_odds

    Returns:
        Raw implied probability (0-1), includes vig.

    Raises:
        ValueError: If decimal_odds is <= 0.

    Examples:
        >>> decimal_to_probability(2.0)  # Even money
        0.5
        >>> decimal_to_probability(1.5)  # Favorite
        0.6667...
    """
    if decimal_odds <= 0:
        raise ValueError(f"Decimal odds must be positive, got {decimal_odds}")
    return 1.0 / decimal_odds


def compute_overround(probabilities: list[float]) -> float:
    """Compute the total overround (vig) from a list of implied probabilities.

    A fair market sums to 1.0. The overround is the amount above 1.0,
    representing the bookmaker's margin.

    Args:
        probabilities: List of raw implied probabilities for all outcomes.

    Returns:
        Total overround. E.g., 0.05 means 5% vig on a market summing to 1.05.
    """
    return sum(probabilities) - 1.0


def proportional_vig_removal(probabilities: list[float]) -> list[float]:
    """Remove vig using proportional (multiplicative) method.

    The simplest approach: divide each probability by the sum of all
    probabilities so they normalize to 1.0. Assumes the vig is distributed
    proportionally across all outcomes.

    Args:
        probabilities: List of raw implied probabilities (with vig).

    Returns:
        List of fair probabilities summing to ~1.0.

    Raises:
        ValueError: If probabilities list is empty or sums to zero.

    Example:
        >>> proportional_vig_removal([0.55, 0.55])  # 10% overround
        [0.5, 0.5]
    """
    if not probabilities:
        raise ValueError("Probabilities list cannot be empty")

    total = sum(probabilities)
    if total <= 0:
        raise ValueError(f"Sum of probabilities must be positive, got {total}")

    return [p / total for p in probabilities]


def shin_vig_removal(probabilities: list[float]) -> list[float]:
    """Remove vig using the Shin (1991) method.

    The Shin method accounts for the favorite-longshot bias: bookmakers
    tend to overcharge more on longshots than on favorites. This method
    uses a model where informed bettors create a wedge that inflates
    longshot odds more than favorite odds.

    The method solves for the insider trading proportion z, then adjusts
    each probability accordingly.

    Reference:
        Shin, H.S. (1991). "Optimal Betting Odds Against Insider Traders."
        The Economic Journal, 101(408), 1179-1185.

    Args:
        probabilities: List of raw implied probabilities (with vig).

    Returns:
        List of fair probabilities summing to ~1.0.

    Raises:
        ValueError: If probabilities list is empty.
    """
    if not probabilities:
        raise ValueError("Probabilities list cannot be empty")

    n = len(probabilities)

    if n == 1:
        return [1.0]

    # Solve for z (insider proportion) using the Shin formula.
    # For a two-outcome market, there's a closed-form solution.
    # For multi-outcome, we use an iterative approach.
    z = _solve_shin_z(probabilities, n)

    # Adjust each probability using the Shin formula with raw probabilities:
    # fair_p = (sqrt(z^2 + 4*(1-z)*p_i^2) - z) / (2*(1-z))
    # where p_i are the raw implied probabilities (including vig).
    # R-2 fix: Use raw probabilities (not p/S). The solver now also uses
    # raw probabilities, so z is calibrated to bring the raw sum down to 1.0.
    fair_probs = []

    for p in probabilities:
        numerator = (math.sqrt(z ** 2 + 4.0 * (1.0 - z) * p ** 2) - z)
        denominator = 2.0 * (1.0 - z)
        if denominator == 0:
            # Fallback to proportional if z == 1 (degenerate case)
            total = sum(probabilities)
            fair_probs.append(p / total)
        else:
            fair_probs.append(numerator / denominator)

    # Normalize to ensure they sum exactly to 1.0 (numerical precision)
    fair_total = sum(fair_probs)
    if fair_total > 0:
        fair_probs = [p / fair_total for p in fair_probs]

    return fair_probs


def _solve_shin_z(probabilities: list[float], n: int) -> float:
    """Solve for the Shin insider proportion z via binary search.

    Finds z such that the sum of Shin-adjusted fair probabilities equals 1.0.
    The Shin fair probability for outcome i is:
        fair_i = (sqrt(z^2 + 4*(1-z)*p_i^2) - z) / (2*(1-z))
    where p_i are the raw implied probabilities (including vig).

    R-2 fix: Uses raw probabilities instead of normalized (p/S). With
    normalized probabilities, at z=0 the sum was already 1.0, so the solver
    always converged to z≈0 and the Shin method degenerated to proportional.
    With raw probabilities, at z=0 the sum equals S > 1.0, requiring z > 0
    to bring it down to 1.0, which correctly models the insider proportion.

    We search for z in [0, 0.99] such that sum(fair_i) = 1.0.

    Args:
        probabilities: Raw implied probabilities.
        n: Number of outcomes.

    Returns:
        Estimated z (insider proportion), typically 0.01-0.10.
    """
    S = sum(probabilities)

    if S <= 1.0 or n <= 1:
        return 0.0

    z_lo, z_hi = 0.0, 0.99
    for _ in range(50):
        z = (z_lo + z_hi) / 2
        fair_sum = sum(
            (math.sqrt(z ** 2 + 4 * (1 - z) * p ** 2) - z) / (2 * (1 - z))
            for p in probabilities
        )
        if fair_sum > 1.0:
            z_lo = z
        else:
            z_hi = z

    return (z_lo + z_hi) / 2
