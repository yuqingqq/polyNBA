"""Map external odds markets to Polymarket contracts.

Handles team name normalization, fuzzy matching, and manual overrides
to pair external bookmaker markets with Polymarket binary contracts.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

from .models import (
    ExternalOdds,
    MappedMarket,
    MarketType,
    PolymarketContract,
)
from .odds_models import OddsApiEvent
from .odds_client import parse_event_to_external_odds

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# NBA Team Aliases: canonical name -> set of known aliases
# All 30 NBA teams with full names, city, short names, abbreviations
# -------------------------------------------------------------------
NBA_TEAM_ALIASES: dict[str, set[str]] = {
    "Atlanta Hawks": {
        "Atlanta Hawks", "Hawks", "ATL", "Atlanta",
    },
    "Boston Celtics": {
        "Boston Celtics", "Celtics", "BOS", "Boston",
    },
    "Brooklyn Nets": {
        "Brooklyn Nets", "Nets", "BKN", "BRK", "Brooklyn",
    },
    "Charlotte Hornets": {
        "Charlotte Hornets", "Hornets", "CHA", "Charlotte",
    },
    "Chicago Bulls": {
        "Chicago Bulls", "Bulls", "CHI", "Chicago",
    },
    "Cleveland Cavaliers": {
        "Cleveland Cavaliers", "Cavaliers", "Cavs", "CLE", "Cleveland",
    },
    "Dallas Mavericks": {
        "Dallas Mavericks", "Mavericks", "Mavs", "DAL", "Dallas",
    },
    "Denver Nuggets": {
        "Denver Nuggets", "Nuggets", "DEN", "Denver",
    },
    "Detroit Pistons": {
        "Detroit Pistons", "Pistons", "DET", "Detroit",
    },
    "Golden State Warriors": {
        "Golden State Warriors", "Warriors", "GSW", "GS", "Golden State",
    },
    "Houston Rockets": {
        "Houston Rockets", "Rockets", "HOU", "Houston",
    },
    "Indiana Pacers": {
        "Indiana Pacers", "Pacers", "IND", "Indiana",
    },
    "Los Angeles Clippers": {
        "Los Angeles Clippers", "LA Clippers", "Clippers", "LAC",
        "Los Angeles C",
    },
    "Los Angeles Lakers": {
        "Los Angeles Lakers", "LA Lakers", "Lakers", "LAL",
        "Los Angeles L",
    },
    "Memphis Grizzlies": {
        "Memphis Grizzlies", "Grizzlies", "MEM", "Memphis",
    },
    "Miami Heat": {
        "Miami Heat", "Heat", "MIA", "Miami",
    },
    "Milwaukee Bucks": {
        "Milwaukee Bucks", "Bucks", "MIL", "Milwaukee",
    },
    "Minnesota Timberwolves": {
        "Minnesota Timberwolves", "Timberwolves", "Wolves", "MIN", "Minnesota",
    },
    "New Orleans Pelicans": {
        "New Orleans Pelicans", "Pelicans", "NOP", "NO", "New Orleans",
    },
    "New York Knicks": {
        "New York Knicks", "Knicks", "NYK", "NY Knicks", "New York",
    },
    "Oklahoma City Thunder": {
        "Oklahoma City Thunder", "Thunder", "OKC", "Oklahoma City",
    },
    "Orlando Magic": {
        "Orlando Magic", "Magic", "ORL", "Orlando",
    },
    "Philadelphia 76ers": {
        "Philadelphia 76ers", "76ers", "Sixers", "PHI", "Philadelphia", "Philly",
    },
    "Phoenix Suns": {
        "Phoenix Suns", "Suns", "PHX", "Phoenix",
    },
    "Portland Trail Blazers": {
        "Portland Trail Blazers", "Trail Blazers", "Blazers", "POR", "Portland",
    },
    "Sacramento Kings": {
        "Sacramento Kings", "Kings", "SAC", "Sacramento",
    },
    "San Antonio Spurs": {
        "San Antonio Spurs", "Spurs", "SAS", "SA", "San Antonio",
    },
    "Toronto Raptors": {
        "Toronto Raptors", "Raptors", "TOR", "Toronto",
    },
    "Utah Jazz": {
        "Utah Jazz", "Jazz", "UTA", "Utah",
    },
    "Washington Wizards": {
        "Washington Wizards", "Wizards", "WAS", "Washington",
    },
}

# Build reverse lookup: any alias (lowered) -> canonical name
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical, aliases in NBA_TEAM_ALIASES.items():
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias.lower()] = canonical


_SPREAD_POINT_RE = re.compile(r"\(([+-]?\d+\.?\d*)\)")
_TOTAL_POINT_RE = re.compile(r"O/U\s+(\d+\.?\d*)", re.IGNORECASE)


def classify_contract(contract: PolymarketContract) -> tuple[Optional[str], Optional[float]]:
    """Classify a contract's market type and extract point/line from question text.

    Returns (market_key, point):
      - ("h2h", None) for moneyline
      - ("spreads", 4.5) for spread with extracted absolute point
      - ("totals", 216.5) for O/U with extracted total line
      - (None, None) for 1H, props, or unrecognized
    """
    q = contract.question
    # I-2 fix: Use case-insensitive matching for market type keywords.
    # Polymarket could use variations like "spread", "o/u", "1h".
    q_upper = q.upper()

    # 1H markets have no external data — skip
    if "1H" in q_upper:
        return (None, None)

    # Spread contracts: "Spread: Rockets (-4.5)"
    if "SPREAD" in q_upper:
        m = _SPREAD_POINT_RE.search(q)
        point = abs(float(m.group(1))) if m else None
        return ("spreads", point)

    # Over/Under contracts: "Rockets vs. Hornets: O/U 216.5"
    if "O/U" in q_upper:
        m = _TOTAL_POINT_RE.search(q)
        point = float(m.group(1)) if m else None
        return ("totals", point)

    # Moneyline: " vs. " in question, outcomes are team names (not Yes/No)
    # R-6 fix: Use q_upper for case-insensitive matching, consistent with other checks.
    if " VS. " in q_upper and contract.outcome.lower() not in ("yes", "no"):
        return ("h2h", None)

    return (None, None)


def _word_match(alias: str, text: str) -> bool:
    """Check if alias appears as a whole word in text.

    Uses regex word boundaries to prevent substring false positives
    (e.g., 'nets' matching 'hornets').
    """
    return bool(re.search(r'\b' + re.escape(alias) + r'\b', text))


def _mentions_other_team(text: str, *exclude_teams: Optional[str]) -> bool:
    """Check if text mentions any NBA team other than the excluded ones.

    Used to reject single-team matches when the contract is clearly about
    a different game (e.g., 'Cavaliers vs. Hornets' matching the Rockets-Hornets
    game because it shares 'Hornets').
    """
    exclude = {t for t in exclude_teams if t}
    for team, aliases in NBA_TEAM_ALIASES.items():
        if team in exclude:
            continue
        for alias in aliases:
            if len(alias) < 4:
                continue
            if _word_match(alias.lower(), text):
                return True
    return False


# Common English words that collide with short team aliases.
# "no" = New Orleans Pelicans alias, but also Yes/No contract outcomes.
# "sa" = San Antonio Spurs alias, but also common abbreviation.
_COMMON_WORDS: set[str] = {"no", "or", "in", "on", "at", "to", "do", "go", "so", "sa"}


def normalize_team_name(name: str) -> Optional[str]:
    """Normalize a team name to its canonical form.

    Tries exact match first, then checks if the input contains any
    known alias as a substring.

    Args:
        name: Raw team name from any source (e.g., "Lakers", "LAL",
              "Los Angeles Lakers", "Will the Lakers win?").

    Returns:
        Canonical team name (e.g., "Los Angeles Lakers") or None if no match.
    """
    cleaned = name.strip()

    # Exact match (case-insensitive).
    # R-1 fix: Skip matches on common English words that collide with
    # short team aliases. E.g., "No" (Yes/No outcome) matches "NO"
    # (New Orleans Pelicans), corrupting championship contract mapping.
    lowered = cleaned.lower()
    if lowered in _ALIAS_TO_CANONICAL and lowered not in _COMMON_WORDS:
        return _ALIAS_TO_CANONICAL[lowered]

    # Substring match — check if any alias is contained in the input
    # Sort by length descending to prefer longer (more specific) matches
    # Skip aliases shorter than 4 chars to avoid false positives
    # (e.g., "NO" in "unknown", "ATL" in "Atlantic", "MIL" in "Miller")
    for alias in sorted(_ALIAS_TO_CANONICAL.keys(), key=len, reverse=True):
        if len(alias) < 4:
            continue
        if alias in cleaned.lower():
            return _ALIAS_TO_CANONICAL[alias]

    return None


def get_canonical_name(name: str) -> str:
    """Get canonical team name, returning original if no match found.

    Args:
        name: Raw team name.

    Returns:
        Canonical name or original name if no match.
    """
    canonical = normalize_team_name(name)
    return canonical if canonical else name


class MarketMapper:
    """Maps external odds markets to Polymarket contracts.

    Usage:
        mapper = MarketMapper()
        mapped = mapper.map_championship(external_odds, poly_contracts)
        for m in mapped:
            print(m.event_name, len(m.polymarket_contracts))
    """

    def __init__(
        self,
        overrides_path: Optional[str] = None,
        preferred_bookmakers: Optional[list[str]] = None,
    ) -> None:
        """Initialize the MarketMapper.

        Args:
            overrides_path: Path to market_overrides.json.
                Defaults to config/market_overrides.json relative to project root.
            preferred_bookmakers: Ordered list of preferred bookmaker keys.
                The first bookmaker found will be used as primary source.
        """
        if overrides_path is None:
            # Default path: <project_root>/config/market_overrides.json
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            overrides_path = os.path.join(project_root, "config", "market_overrides.json")

        self.overrides: dict = self._load_overrides(overrides_path)
        self.preferred_bookmakers = preferred_bookmakers or [
            "pinnacle", "draftkings", "fanduel", "betmgm", "bovada",
        ]

    def _load_overrides(self, path: str) -> dict:
        """Load manual market mapping overrides from JSON file.

        Args:
            path: Path to overrides JSON file.

        Returns:
            Dict of overrides. Empty dict if file not found or invalid.
        """
        try:
            with open(path, "r") as f:
                data = json.load(f)
                logger.info("Loaded %d market overrides from %s", len(data), path)
                return data
        except FileNotFoundError:
            logger.debug("No overrides file at %s", path)
            return {}
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in overrides file %s: %s", path, e)
            return {}

    def map_championship(
        self,
        external_odds: list[ExternalOdds],
        poly_contracts: list[PolymarketContract],
    ) -> list[MappedMarket]:
        """Map championship/futures odds to Polymarket contracts.

        Matches by normalizing team names on both sides.

        Args:
            external_odds: Flat list of ExternalOdds from championship market.
            poly_contracts: List of Polymarket contracts for championship markets.

        Returns:
            List of MappedMarket, one per successfully matched team.
        """
        # Group external odds by canonical team name
        external_by_team: dict[str, list[ExternalOdds]] = {}
        for odds in external_odds:
            canonical = get_canonical_name(odds.team)
            external_by_team.setdefault(canonical, []).append(odds)

        # Group Polymarket contracts by canonical team name
        poly_by_team: dict[str, list[PolymarketContract]] = {}
        for contract in poly_contracts:
            # Try to match from question or outcome
            canonical = normalize_team_name(contract.outcome)
            if canonical is None:
                canonical = normalize_team_name(contract.question)
            if canonical is None:
                logger.debug(
                    "Could not normalize Polymarket contract: %s / %s",
                    contract.question,
                    contract.outcome,
                )
                continue
            poly_by_team.setdefault(canonical, []).append(contract)

        # Check overrides
        override_map = self.overrides.get("championship", {})

        # Select preferred bookmaker from ALL external odds so _adapt_multi_way
        # receives all outcomes for proper vig removal (30-team normalization).
        best_odds_all = self._select_preferred_odds(external_odds)

        # Match teams
        mapped_markets: list[MappedMarket] = []
        matched_teams = set(external_by_team.keys()) & set(poly_by_team.keys())

        for team in matched_teams:
            poly_cs = poly_by_team[team]

            mapped = MappedMarket(
                external_odds=best_odds_all,
                polymarket_contracts=poly_cs,
                market_type=MarketType.CHAMPIONSHIP,
                event_name=f"NBA Championship - {team}",
                mapping_confidence=1.0,
            )
            mapped_markets.append(mapped)

        # Apply manual overrides
        for override_key, override_val in override_map.items():
            ext_team = override_val.get("external_team")
            poly_token = override_val.get("polymarket_token_id")
            if ext_team and poly_token:
                canonical_ext = get_canonical_name(ext_team)
                if canonical_ext in external_by_team:
                    matching_poly = [
                        c for c in poly_contracts if c.token_id == poly_token
                    ]
                    if matching_poly:
                        mapped = MappedMarket(
                            external_odds=best_odds_all,
                            polymarket_contracts=matching_poly,
                            market_type=MarketType.CHAMPIONSHIP,
                            event_name=f"NBA Championship - {canonical_ext} (override)",
                            mapping_confidence=0.9,
                        )
                        mapped_markets.append(mapped)

        # Log unmapped teams
        ext_only = set(external_by_team.keys()) - set(poly_by_team.keys())
        poly_only = set(poly_by_team.keys()) - set(external_by_team.keys())
        if ext_only:
            logger.info("Teams with external odds but no Polymarket match: %s", ext_only)
        if poly_only:
            logger.info("Polymarket teams with no external odds: %s", poly_only)

        logger.info(
            "Championship mapping: %d matched, %d ext-only, %d poly-only",
            len(matched_teams),
            len(ext_only),
            len(poly_only),
        )

        return mapped_markets

    # Mapping from odds API market key to our MarketType enum
    _MARKET_KEY_TO_TYPE: dict[str, MarketType] = {
        "h2h": MarketType.GAME_ML,
        "spreads": MarketType.SPREAD,
        "totals": MarketType.TOTAL,
    }

    def map_game(
        self,
        external_event: OddsApiEvent,
        poly_contracts: list[PolymarketContract],
        bookmaker_filter: Optional[list[str]] = None,
    ) -> list[MappedMarket]:
        """Map a single game event's odds to Polymarket contracts.

        Creates one MappedMarket per market type (h2h, spreads, totals)
        so that vig removal is applied to each pair independently.

        Args:
            external_event: An OddsApiEvent for a specific game.
            poly_contracts: All available Polymarket NBA contracts.
            bookmaker_filter: Optional list of bookmaker keys to include.

        Returns:
            List of MappedMarket, one per market type found. Empty if no match.
        """
        home = normalize_team_name(external_event.home_team or "")
        away = normalize_team_name(external_event.away_team or "")

        if not home and not away:
            logger.debug("Cannot normalize teams for event %s", external_event.id)
            return []

        # Find Polymarket contracts that mention both teams
        matched_contracts: list[PolymarketContract] = []
        for contract in poly_contracts:
            q_lower = contract.question.lower()
            outcome_lower = contract.outcome.lower()
            text = q_lower + " " + outcome_lower

            # Check if contract is about this game (word-boundary matching)
            home_match = False
            away_match = False

            if home:
                home_aliases = NBA_TEAM_ALIASES.get(home, {home})
                home_match = any(
                    _word_match(alias.lower(), text)
                    for alias in home_aliases
                    if len(alias) >= 4
                )

            if away:
                away_aliases = NBA_TEAM_ALIASES.get(away, {away})
                away_match = any(
                    _word_match(alias.lower(), text)
                    for alias in away_aliases
                    if len(alias) >= 4
                )

            if home_match and away_match:
                matched_contracts.append(contract)
            elif home_match or away_match:
                # Single team match — only accept if date matches AND
                # the question doesn't mention another NBA team (which
                # would indicate a different game sharing one team).
                if self._date_matches(external_event, contract):
                    if not _mentions_other_team(text, home, away):
                        matched_contracts.append(contract)

        if not matched_contracts:
            # Check overrides
            override_key = f"{home} vs {away}"
            game_overrides = self.overrides.get("games", {})
            if override_key in game_overrides:
                override_token = game_overrides[override_key].get("polymarket_token_id")
                override_contracts = [
                    c for c in poly_contracts if c.token_id == override_token
                ]
                if override_contracts:
                    matched_contracts = override_contracts

        if not matched_contracts:
            logger.debug(
                "No Polymarket match for game: %s vs %s (%s)",
                home,
                away,
                external_event.commence_time,
            )
            return []

        # Parse external odds and select preferred bookmaker
        ext_odds = parse_event_to_external_odds(external_event, bookmaker_filter)
        best_odds = self._select_preferred_odds(ext_odds)

        # Group preferred odds by market_key
        odds_by_market: dict[str, list[ExternalOdds]] = {}
        for odds in best_odds:
            key = odds.market_key or "h2h"
            odds_by_market.setdefault(key, []).append(odds)

        # Create one MappedMarket per market type
        results: list[MappedMarket] = []
        event_name = f"{away} @ {home}"

        for market_key, market_odds in odds_by_market.items():
            market_type = self._MARKET_KEY_TO_TYPE.get(market_key)
            if market_type is None:
                logger.debug("Skipping unknown market key: %s", market_key)
                continue

            # Extract the line from external odds (e.g., 4.5 for spreads, 216.5 for totals)
            ext_point = abs(market_odds[0].point) if market_odds[0].point is not None else None

            # R17-I2: Skip spreads/totals entirely when the external odds
            # have no point value — we can't correctly match contract lines.
            if market_key in ("spreads", "totals") and ext_point is None:
                logger.warning(
                    "No point value in external %s odds for %s — skipping",
                    market_key, external_event.home_team,
                )
                continue

            # Filter contracts to only those matching this market type + line
            filtered: list[PolymarketContract] = []
            for c in matched_contracts:
                c_key, c_point = classify_contract(c)
                if c_key != market_key:
                    continue
                # For spreads/totals, also require point match
                if ext_point is not None and c_point is not None:
                    if abs(c_point - ext_point) > 0.01:
                        continue
                filtered.append(c)

            if not filtered:
                continue  # no matching Poly contracts for this market type

            mapped = MappedMarket(
                external_event_id=external_event.id,
                external_odds=market_odds,
                polymarket_contracts=filtered,
                market_type=market_type,
                event_name=event_name,
                commence_time=external_event.commence_time,
                mapping_confidence=1.0 if len(filtered) >= 2 else 0.8,
            )
            results.append(mapped)

        return results

    def map_all_games(
        self,
        external_events: list[OddsApiEvent],
        poly_contracts: list[PolymarketContract],
        bookmaker_filter: Optional[list[str]] = None,
        skip_in_progress: bool = True,
    ) -> list[MappedMarket]:
        """Map all game events to Polymarket contracts.

        Args:
            external_events: List of OddsApiEvent for upcoming games.
            poly_contracts: All available Polymarket NBA contracts.
            bookmaker_filter: Optional list of bookmaker keys to include.
            skip_in_progress: If True (default), skip games that have already
                started. Set to False to include in-progress games and use
                live odds for quoting.

        Returns:
            List of successfully mapped MappedMarket objects.
        """
        mapped: list[MappedMarket] = []
        skipped_live = 0
        for event in external_events:
            # Skip in-progress games if requested
            if skip_in_progress and event.commence_time is not None:
                ct = event.commence_time if event.commence_time.tzinfo else event.commence_time.replace(tzinfo=timezone.utc)
                if ct < datetime.now(timezone.utc):
                    skipped_live += 1
                    logger.info(
                        "Skipping in-progress game: %s @ %s (commenced %s)",
                        event.away_team,
                        event.home_team,
                        event.commence_time,
                    )
                    continue
            results = self.map_game(event, poly_contracts, bookmaker_filter)
            mapped.extend(results)

        logger.info(
            "Game mapping: %d of %d events matched (%d skipped in-progress)",
            len(mapped),
            len(external_events),
            skipped_live,
        )
        return mapped

    def _select_preferred_odds(
        self, odds_list: list[ExternalOdds]
    ) -> list[ExternalOdds]:
        """Select odds from the most preferred bookmaker.

        If no preferred bookmaker is found, returns all odds.

        Args:
            odds_list: All available odds for a market.

        Returns:
            Filtered list of odds from the preferred bookmaker.
        """
        for bookmaker in self.preferred_bookmakers:
            filtered = [o for o in odds_list if o.bookmaker == bookmaker]
            if filtered:
                return filtered

        # Fallback: return all odds
        return odds_list

    def get_unmatched_external(
        self,
        external_odds: list[ExternalOdds],
        mapped_markets: list[MappedMarket],
    ) -> list[ExternalOdds]:
        """Return external odds that have no corresponding Polymarket match.

        Compares external odds against the set of already-mapped markets
        to find odds entries that were not successfully matched to any
        Polymarket contract.

        Args:
            external_odds: Full list of external odds entries.
            mapped_markets: List of MappedMarket objects from a mapping run.

        Returns:
            List of ExternalOdds that do not appear in any mapped market.
        """
        # Collect teams that have Polymarket contracts in mapped markets.
        # (Championship MappedMarkets include all-team external odds for vig
        # removal, so checking external_odds would mark all teams as matched.)
        mapped_teams: set[str] = set()
        for mm in mapped_markets:
            for contract in mm.polymarket_contracts:
                canonical = normalize_team_name(contract.outcome)
                if canonical is None:
                    canonical = normalize_team_name(contract.question)
                if canonical:
                    mapped_teams.add(canonical)

        # Find external odds whose canonical team is not in the mapped set
        unmatched: list[ExternalOdds] = []
        for odds in external_odds:
            canonical = get_canonical_name(odds.team)
            if canonical not in mapped_teams:
                unmatched.append(odds)

        if unmatched:
            logger.info(
                "Unmatched external odds: %d entries (%s)",
                len(unmatched),
                ", ".join(sorted({get_canonical_name(o.team) for o in unmatched})),
            )

        return unmatched

    def get_unmatched_polymarket(
        self,
        poly_contracts: list[PolymarketContract],
        mapped_markets: list[MappedMarket],
    ) -> list[PolymarketContract]:
        """Return Polymarket contracts that have no external odds match.

        Compares Polymarket contracts against the set of already-mapped
        markets to find contracts that were not successfully matched to
        any external odds source.

        Args:
            poly_contracts: Full list of Polymarket contracts.
            mapped_markets: List of MappedMarket objects from a mapping run.

        Returns:
            List of PolymarketContract that do not appear in any mapped market.
        """
        # Collect all Polymarket token IDs that are present in mapped markets
        mapped_token_ids: set[str] = set()
        for mm in mapped_markets:
            for contract in mm.polymarket_contracts:
                mapped_token_ids.add(contract.token_id)

        # Find contracts whose token_id is not in the mapped set
        unmatched: list[PolymarketContract] = [
            c for c in poly_contracts if c.token_id not in mapped_token_ids
        ]

        if unmatched:
            logger.info(
                "Unmatched Polymarket contracts: %d of %d total",
                len(unmatched),
                len(poly_contracts),
            )

        return unmatched

    def _date_matches(
        self,
        event: OddsApiEvent,
        contract: PolymarketContract,
        tolerance_days: int = 1,
    ) -> bool:
        """Check if an event's date roughly matches a contract's end date.

        Args:
            event: External event with commence_time.
            contract: Polymarket contract with optional end_date.
            tolerance_days: Number of days of tolerance for matching.

        Returns:
            True if dates are within tolerance, or if dates can't be compared.
        """
        if not event.commence_time or not contract.end_date:
            return True

        try:
            # I-1 fix: Normalize both datetimes to timezone-aware before comparison.
            # event.commence_time can be naive (e.g., from test fixtures) while
            # contract_date parsed from "Z"-suffixed ISO string is always aware.
            # Subtracting naive from aware raises TypeError, silently disabling
            # the date guard.
            ct = event.commence_time
            if ct.tzinfo is None:
                ct = ct.replace(tzinfo=timezone.utc)
            contract_date = datetime.fromisoformat(
                contract.end_date.replace("Z", "+00:00")
            )
            if contract_date.tzinfo is None:
                contract_date = contract_date.replace(tzinfo=timezone.utc)
            # R15-I1 fix: Use total_seconds() for symmetric tolerance.
            # timedelta.days truncates differently for positive vs negative deltas
            # (e.g., -36h → .days=-2 but +36h → .days=1), causing asymmetric matching.
            delta_days = abs((ct - contract_date).total_seconds()) / 86400
            return delta_days <= tolerance_days
        except (ValueError, TypeError):
            # Unparseable dates treated same as missing — allow the match
            return True
