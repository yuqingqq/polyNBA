"""Convert external odds to Polymarket-compatible probabilities.

Applies vig removal to external odds and outputs fair probabilities
clamped to the Polymarket tick range [0.01, 0.99].
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from .models import (
    ExternalOdds,
    MappedMarket,
    MarketType,
    PolymarketContract,
    ReferencePrice,
)
from .vig_removal import (
    proportional_vig_removal,
    shin_vig_removal,
)

logger = logging.getLogger(__name__)

# Polymarket probability bounds
MIN_PROBABILITY = 0.01
MAX_PROBABILITY = 0.99


def clamp_probability(p: float) -> float:
    """Clamp a probability to the Polymarket tick range [0.01, 0.99].

    Args:
        p: Raw probability value.

    Returns:
        Clamped probability.
    """
    return max(MIN_PROBABILITY, min(MAX_PROBABILITY, p))


class PriceAdapter:
    """Convert external odds to Polymarket reference prices.

    Applies vig removal to extract fair probabilities, then maps them
    to the corresponding Polymarket contracts.

    Usage:
        adapter = PriceAdapter(vig_method="proportional")
        ref_prices = adapter.adapt(mapped_market)
        for rp in ref_prices:
            print(rp.token_id, rp.fair_probability)
    """

    def __init__(
        self,
        vig_method: str = "proportional",
        source: str = "the-odds-api",
        preferred_bookmakers: Optional[list[str]] = None,
        vig_free_bookmakers: Optional[set[str]] = None,
    ) -> None:
        """Initialize the PriceAdapter.

        Args:
            vig_method: Vig removal method to use.
                'proportional' — simple normalization (default).
                'shin' — Shin (1991) method for favorite-longshot bias.
            source: Source identifier for reference prices.
            preferred_bookmakers: Ordered list of preferred bookmaker keys
                for multi-way markets. Falls back to most-outcomes bookmaker.
            vig_free_bookmakers: Set of bookmaker keys whose prices are
                already vig-free (e.g., exchange mid-prices from Kalshi,
                Betfair). For these, probabilities are normalized to sum=1.0
                but Shin/proportional vig removal is skipped.
        """
        if vig_method not in ("proportional", "shin"):
            raise ValueError(f"Unknown vig method: {vig_method}. Use 'proportional' or 'shin'.")
        self.vig_method = vig_method
        self.source = source
        # R-5 fix: Use configurable preferred bookmakers instead of hardcoded list.
        self.preferred_bookmakers = preferred_bookmakers or [
            "pinnacle", "draftkings", "fanduel", "betmgm", "bovada",
        ]
        self.vig_free_bookmakers = vig_free_bookmakers or {"kalshi", "betfair"}

    def adapt(self, mapped_market: MappedMarket) -> list[ReferencePrice]:
        """Convert a mapped market's external odds to reference prices.

        For each Polymarket contract in the mapped market, finds the
        corresponding external odds, applies vig removal, and returns
        a ReferencePrice.

        Args:
            mapped_market: A MappedMarket with both external odds and
                Polymarket contracts.

        Returns:
            List of ReferencePrice, one per matched Polymarket contract.
        """
        if not mapped_market.external_odds:
            logger.warning("No external odds for market: %s", mapped_market.event_name)
            return []

        if not mapped_market.polymarket_contracts:
            logger.warning("No Polymarket contracts for market: %s", mapped_market.event_name)
            return []

        if mapped_market.market_type == MarketType.CHAMPIONSHIP:
            return self._adapt_multi_way(mapped_market)
        else:
            return self._adapt_two_way(mapped_market)

    def _adapt_two_way(self, mapped_market: MappedMarket) -> list[ReferencePrice]:
        """Adapt a 2-way market (game moneyline, spread, total).

        Groups odds by bookmaker, performs vig removal on each pair,
        then maps to Polymarket contracts.

        Args:
            mapped_market: Mapped 2-way market.

        Returns:
            List of ReferencePrice.
        """
        # Group external odds by bookmaker to do vig removal per bookmaker
        odds_by_bookmaker: dict[str, list[ExternalOdds]] = {}
        for odds in mapped_market.external_odds:
            odds_by_bookmaker.setdefault(odds.bookmaker, []).append(odds)

        # Use the first bookmaker that has a complete set of outcomes
        selected_odds: Optional[list[ExternalOdds]] = None
        selected_bookmaker: Optional[str] = None

        # I-3 fix: Prefer bookmakers with exactly 2 outcomes for 2-way markets.
        # Using >2 entries with proportional vig removal would halve probabilities.
        # R15-I5 fix: Iterate preferred_bookmakers first (e.g., Pinnacle before
        # regional books) for more accurate vig removal, matching _adapt_multi_way.
        for pref_bm in self.preferred_bookmakers:
            if pref_bm in odds_by_bookmaker and len(odds_by_bookmaker[pref_bm]) == 2:
                selected_odds = odds_by_bookmaker[pref_bm]
                selected_bookmaker = pref_bm
                break
        # If no preferred bookmaker has exactly 2, try any bookmaker
        if selected_odds is None:
            for bm, bm_odds in odds_by_bookmaker.items():
                if len(bm_odds) == 2:
                    selected_odds = bm_odds
                    selected_bookmaker = bm
                    break
        # Fallback: accept >=2 but take first 2 with warning
        if selected_odds is None:
            for bm, bm_odds in odds_by_bookmaker.items():
                if len(bm_odds) >= 2:
                    logger.warning(
                        "Bookmaker %s has %d outcomes for 2-way market '%s'; "
                        "expected 2. Using first 2 only.",
                        bm, len(bm_odds), mapped_market.event_name,
                    )
                    selected_odds = bm_odds[:2]
                    selected_bookmaker = bm
                    break

        if selected_odds is None:
            # I-3 fix: No single bookmaker has a complete set of outcomes.
            # Mixing odds from different bookmakers produces incorrect vig
            # removal.  Log a warning and return empty instead.
            bookmakers_found = list(odds_by_bookmaker.keys())
            logger.warning(
                "No single bookmaker has >=2 outcomes for market '%s'. "
                "Bookmakers found: %s. Skipping to avoid mixed-bookmaker vig removal.",
                mapped_market.event_name,
                bookmakers_found,
            )
            return []

        # Extract raw probabilities
        raw_probs = [o.implied_probability for o in selected_odds]

        # Apply vig removal — skip for exchange bookmakers (already vig-free)
        is_exchange = selected_bookmaker in self.vig_free_bookmakers
        if is_exchange:
            # Normalize to sum=1.0 without applying vig model
            total = sum(raw_probs)
            fair_probs = [p / total for p in raw_probs] if total > 0 else list(raw_probs)
            vig_method_used = "exchange_passthrough"
        else:
            fair_probs = self._remove_vig(raw_probs)
            vig_method_used = self.vig_method

        # Map fair probabilities to Polymarket contracts
        ref_prices: list[ReferencePrice] = []
        now = datetime.now(timezone.utc)

        for contract in mapped_market.polymarket_contracts:
            # Find the matching external odds entry for this contract
            match_idx = self._find_matching_outcome(
                contract, selected_odds, mapped_market.market_type
            )

            if match_idx is not None and match_idx < len(fair_probs):
                fair_p = clamp_probability(fair_probs[match_idx])
                raw_p = raw_probs[match_idx]

                ref_price = ReferencePrice(
                    token_id=contract.token_id,
                    fair_probability=fair_p,
                    raw_probability=raw_p,
                    source=f"{self.source}:{selected_bookmaker}",
                    timestamp=now,
                    market_type=mapped_market.market_type,
                    bookmaker=selected_bookmaker,
                    vig_removal_method=vig_method_used,
                )
                ref_prices.append(ref_price)
            else:
                logger.debug(
                    "No matching external odds for contract %s (%s)",
                    contract.token_id,
                    contract.outcome,
                )

        return ref_prices

    def _adapt_multi_way(self, mapped_market: MappedMarket) -> list[ReferencePrice]:
        """Adapt a multi-way market (championship, conference).

        For multi-way markets, we need all outcomes' probabilities to
        do vig removal properly. Each Polymarket contract represents
        one outcome in the multi-way market.

        Args:
            mapped_market: Mapped multi-way market.

        Returns:
            List of ReferencePrice.
        """
        # Group external odds by team (canonical name)
        from .market_mapper import get_canonical_name

        # For championship markets, each ExternalOdds entry is one team
        # We need to collect all teams' odds from the same bookmaker
        odds_by_bookmaker: dict[str, list[ExternalOdds]] = {}
        for odds in mapped_market.external_odds:
            odds_by_bookmaker.setdefault(odds.bookmaker, []).append(odds)

        # R-5 fix: Select preferred bookmaker from configurable list
        selected_bookmaker = None
        for pref in self.preferred_bookmakers:
            if pref in odds_by_bookmaker and len(odds_by_bookmaker[pref]) >= 2:
                selected_bookmaker = pref
                break
        if selected_bookmaker is None:
            selected_bookmaker = max(
                odds_by_bookmaker.keys(),
                key=lambda bm: len(odds_by_bookmaker[bm]),
            )
        selected_odds = odds_by_bookmaker[selected_bookmaker]

        # Extract raw probabilities
        raw_probs = [o.implied_probability for o in selected_odds]

        # Apply vig removal — skip for exchange bookmakers (already vig-free)
        is_exchange = selected_bookmaker in self.vig_free_bookmakers

        # C-1 fix: map_championship() creates one MappedMarket per team,
        # so we often have only 1 outcome here.  Vig removal on a single
        # probability normalises to 1.0 (clamped to 0.99), which is wrong.
        # When we can't do proper normalisation, use the raw probability.
        if is_exchange:
            total = sum(raw_probs)
            fair_probs = [p / total for p in raw_probs] if total > 0 else list(raw_probs)
            vig_method_used = "exchange_passthrough"
        elif len(raw_probs) >= 2:
            fair_probs = self._remove_vig(raw_probs)
            vig_method_used = self.vig_method
        else:
            logger.debug(
                "Skipping vig removal for single-outcome multi-way market: %s",
                mapped_market.event_name,
            )
            fair_probs = list(raw_probs)
            vig_method_used = self.vig_method

        # Build mapping: canonical team name -> (raw_prob, fair_prob)
        team_probs: dict[str, tuple[float, float]] = {}
        for i, odds in enumerate(selected_odds):
            canonical = get_canonical_name(odds.team)
            team_probs[canonical] = (raw_probs[i], fair_probs[i])

        # Map to Polymarket contracts
        ref_prices: list[ReferencePrice] = []
        now = datetime.now(timezone.utc)

        for contract in mapped_market.polymarket_contracts:
            # Normalize the contract's outcome to a canonical team name
            from .market_mapper import normalize_team_name
            canonical = normalize_team_name(contract.outcome)
            if canonical is None:
                canonical = normalize_team_name(contract.question)

            if canonical and canonical in team_probs:
                raw_p, fair_p = team_probs[canonical]

                ref_price = ReferencePrice(
                    token_id=contract.token_id,
                    fair_probability=clamp_probability(fair_p),
                    raw_probability=raw_p,
                    source=f"{self.source}:{selected_bookmaker}",
                    timestamp=now,
                    market_type=mapped_market.market_type,
                    bookmaker=selected_bookmaker,
                    vig_removal_method=vig_method_used,
                )
                ref_prices.append(ref_price)
            else:
                logger.debug(
                    "No matching external odds for championship contract: %s",
                    contract.outcome,
                )

        return ref_prices

    def _remove_vig(self, probabilities: list[float]) -> list[float]:
        """Apply the configured vig removal method.

        Args:
            probabilities: Raw implied probabilities.

        Returns:
            Fair probabilities after vig removal.
        """
        if self.vig_method == "shin":
            return shin_vig_removal(probabilities)
        else:
            return proportional_vig_removal(probabilities)

    def _find_matching_outcome(
        self,
        contract: PolymarketContract,
        odds_list: list[ExternalOdds],
        market_type: MarketType,
    ) -> Optional[int]:
        """Find which external odds entry matches a Polymarket contract.

        Args:
            contract: The Polymarket contract to match.
            odds_list: List of external odds entries.
            market_type: Type of market for context.

        Returns:
            Index into odds_list if matched, None otherwise.
        """
        from .market_mapper import normalize_team_name, _SPREAD_POINT_RE

        outcome_lower = contract.outcome.lower()

        # For totals markets, match Over/Under by name
        if market_type == MarketType.TOTAL:
            for i, odds in enumerate(odds_list):
                if outcome_lower == "over" and odds.team.lower() == "over":
                    return i
                if outcome_lower == "under" and odds.team.lower() == "under":
                    return i
            return None

        # For h2h / spread: match by team name
        contract_team = normalize_team_name(contract.outcome)
        if contract_team is None:
            contract_team = normalize_team_name(contract.question)

        for i, odds in enumerate(odds_list):
            odds_team = normalize_team_name(odds.team)

            # Direct team name match (skip for Yes/No — need special handling below)
            if outcome_lower not in ("yes", "no") and contract_team and odds_team and contract_team == odds_team:
                return i

            # I-4 fix: For Yes/No contracts in game markets, parse the
            # contract question to determine which team "Yes" refers to,
            # then match by team name instead of relying on position.
            if outcome_lower in ("yes", "no"):
                # R2-6 fix: For spread markets, the question mentions both teams
                # (e.g., "Spread: Rockets (-4.5) vs. Hornets"). Parsing the
                # full question picks the longest-alias team, which may be wrong.
                # Instead, extract the team from the text BEFORE the spread point
                # — that's the team the spread applies to.
                if market_type == MarketType.SPREAD:
                    spread_match = _SPREAD_POINT_RE.search(contract.question)
                    if spread_match:
                        pre_point_text = contract.question[:spread_match.start()]
                        question_team = normalize_team_name(pre_point_text)
                    else:
                        question_team = normalize_team_name(contract.question)
                else:
                    question_team = normalize_team_name(contract.question)

                if question_team:
                    # C-1 fix: Find the "Yes" index first, then derive "No"
                    # as the complement. This avoids mapping "No" to the wrong
                    # entry when >2 odds exist (e.g., duplicate data or alternate lines).
                    yes_idx = None
                    for j, od in enumerate(odds_list):
                        if normalize_team_name(od.team) == question_team:
                            yes_idx = j
                            break

                    if yes_idx is not None:
                        if outcome_lower == "yes":
                            return yes_idx
                        else:
                            # "No" = complement in a 2-way market
                            if len(odds_list) == 2:
                                return 1 - yes_idx
                            # Fallback for >2 entries: first non-matching team
                            for j, od in enumerate(odds_list):
                                if j != yes_idx:
                                    return j

                    # Break out of the outer loop since we already searched all odds
                    break

        return None
