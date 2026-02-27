"""Tests for the market mapper module.

Tests team name normalization, championship market mapping,
and game market mapping by date and team.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.reference.market_mapper import (
    NBA_TEAM_ALIASES,
    MarketMapper,
    classify_contract,
    get_canonical_name,
    normalize_team_name,
)
from src.reference.models import (
    ExternalOdds,
    MappedMarket,
    MarketType,
    PolymarketContract,
)
from src.reference.odds_models import (
    OddsApiBookmaker,
    OddsApiEvent,
    OddsApiMarket,
    OddsApiOutcome,
)


# -------------------------------------------------------------------
# Tests for NBA_TEAM_ALIASES completeness
# -------------------------------------------------------------------


class TestTeamAliases:
    """Verify that all 30 NBA teams are represented."""

    def test_all_30_teams(self) -> None:
        """NBA_TEAM_ALIASES should contain all 30 NBA teams."""
        assert len(NBA_TEAM_ALIASES) == 30

    def test_each_team_has_aliases(self) -> None:
        """Each team should have at least 3 aliases (full, short, abbreviation)."""
        for team, aliases in NBA_TEAM_ALIASES.items():
            assert len(aliases) >= 3, f"{team} has fewer than 3 aliases: {aliases}"

    def test_known_teams_present(self) -> None:
        """Spot-check that specific well-known teams are present."""
        expected_teams = [
            "Los Angeles Lakers",
            "Boston Celtics",
            "Golden State Warriors",
            "Chicago Bulls",
            "New York Knicks",
            "Miami Heat",
        ]
        for team in expected_teams:
            assert team in NBA_TEAM_ALIASES, f"Missing team: {team}"


# -------------------------------------------------------------------
# Tests for normalize_team_name
# -------------------------------------------------------------------


class TestNormalizeTeamName:
    """Tests for team name normalization."""

    def test_full_name(self) -> None:
        """Full team name should resolve to canonical."""
        assert normalize_team_name("Los Angeles Lakers") == "Los Angeles Lakers"

    def test_short_name(self) -> None:
        """Short team name should resolve to canonical."""
        assert normalize_team_name("Lakers") == "Los Angeles Lakers"

    def test_abbreviation(self) -> None:
        """Abbreviation should resolve to canonical."""
        assert normalize_team_name("LAL") == "Los Angeles Lakers"

    def test_case_insensitive(self) -> None:
        """Matching should be case-insensitive."""
        assert normalize_team_name("lakers") == "Los Angeles Lakers"
        assert normalize_team_name("CELTICS") == "Boston Celtics"

    def test_whitespace_handling(self) -> None:
        """Leading/trailing whitespace should be stripped."""
        assert normalize_team_name("  Lakers  ") == "Los Angeles Lakers"

    def test_unknown_team(self) -> None:
        """Unknown team name should return None."""
        assert normalize_team_name("Gotham City Rogues") is None

    def test_city_name(self) -> None:
        """City names should resolve where unambiguous."""
        assert normalize_team_name("Denver") == "Denver Nuggets"
        assert normalize_team_name("Miami") == "Miami Heat"

    def test_nickname(self) -> None:
        """Common nicknames should resolve."""
        assert normalize_team_name("Cavs") == "Cleveland Cavaliers"
        assert normalize_team_name("Sixers") == "Philadelphia 76ers"
        assert normalize_team_name("Wolves") == "Minnesota Timberwolves"
        assert normalize_team_name("Mavs") == "Dallas Mavericks"

    def test_substring_match(self) -> None:
        """Should match when a team name is contained in a longer string."""
        result = normalize_team_name("Will the Lakers win the championship?")
        assert result == "Los Angeles Lakers"

    def test_all_abbreviations(self) -> None:
        """All standard 3-letter abbreviations should resolve."""
        abbrev_map = {
            "ATL": "Atlanta Hawks",
            "BOS": "Boston Celtics",
            "BKN": "Brooklyn Nets",
            "CHA": "Charlotte Hornets",
            "CHI": "Chicago Bulls",
            "CLE": "Cleveland Cavaliers",
            "DAL": "Dallas Mavericks",
            "DEN": "Denver Nuggets",
            "DET": "Detroit Pistons",
            "GSW": "Golden State Warriors",
            "HOU": "Houston Rockets",
            "IND": "Indiana Pacers",
            "LAC": "Los Angeles Clippers",
            "LAL": "Los Angeles Lakers",
            "MEM": "Memphis Grizzlies",
            "MIA": "Miami Heat",
            "MIL": "Milwaukee Bucks",
            "MIN": "Minnesota Timberwolves",
            "NOP": "New Orleans Pelicans",
            "NYK": "New York Knicks",
            "OKC": "Oklahoma City Thunder",
            "ORL": "Orlando Magic",
            "PHI": "Philadelphia 76ers",
            "PHX": "Phoenix Suns",
            "POR": "Portland Trail Blazers",
            "SAC": "Sacramento Kings",
            "SAS": "San Antonio Spurs",
            "TOR": "Toronto Raptors",
            "UTA": "Utah Jazz",
            "WAS": "Washington Wizards",
        }
        for abbr, expected in abbrev_map.items():
            result = normalize_team_name(abbr)
            assert result == expected, f"Abbreviation '{abbr}' resolved to '{result}', expected '{expected}'"


class TestGetCanonicalName:
    """Tests for get_canonical_name helper."""

    def test_known_team(self) -> None:
        """Known team should return canonical name."""
        assert get_canonical_name("Lakers") == "Los Angeles Lakers"

    def test_unknown_returns_original(self) -> None:
        """Unknown name should return original string."""
        assert get_canonical_name("Unknown Team") == "Unknown Team"


# -------------------------------------------------------------------
# Tests for MarketMapper — Championship mapping
# -------------------------------------------------------------------


class TestChampionshipMapping:
    """Tests for championship market mapping."""

    @pytest.fixture()
    def mapper(self) -> MarketMapper:
        """Create a MarketMapper with no overrides."""
        return MarketMapper(overrides_path="/nonexistent/path.json")

    @pytest.fixture()
    def sample_external_odds(self) -> list[ExternalOdds]:
        """Create sample championship external odds."""
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
        ]

    @pytest.fixture()
    def sample_poly_contracts(self) -> list[PolymarketContract]:
        """Create sample Polymarket championship contracts."""
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
        ]

    def test_maps_matching_teams(
        self,
        mapper: MarketMapper,
        sample_external_odds: list[ExternalOdds],
        sample_poly_contracts: list[PolymarketContract],
    ) -> None:
        """Should map teams with matching names."""
        mapped = mapper.map_championship(sample_external_odds, sample_poly_contracts)
        assert len(mapped) == 3

    def test_mapped_market_type(
        self,
        mapper: MarketMapper,
        sample_external_odds: list[ExternalOdds],
        sample_poly_contracts: list[PolymarketContract],
    ) -> None:
        """Mapped markets should have CHAMPIONSHIP type."""
        mapped = mapper.map_championship(sample_external_odds, sample_poly_contracts)
        for m in mapped:
            assert m.market_type == MarketType.CHAMPIONSHIP

    def test_unmatched_teams_skipped(
        self,
        mapper: MarketMapper,
    ) -> None:
        """Teams without a Polymarket match should be skipped."""
        ext = [
            ExternalOdds(
                team="Boston Celtics",
                american_odds=350,
                implied_probability=0.222,
                bookmaker="pinnacle",
            ),
            ExternalOdds(
                team="Sacramento Kings",
                american_odds=5000,
                implied_probability=0.02,
                bookmaker="pinnacle",
            ),
        ]
        poly = [
            PolymarketContract(
                token_id="token_celtics",
                condition_id="cond_1",
                question="Will the Celtics win?",
                outcome="Celtics",
            ),
        ]
        mapped = mapper.map_championship(ext, poly)
        assert len(mapped) == 1
        assert mapped[0].event_name == "NBA Championship - Boston Celtics"


# -------------------------------------------------------------------
# Tests for MarketMapper — Game mapping
# -------------------------------------------------------------------


class TestGameMapping:
    """Tests for game market mapping by team names and date."""

    @pytest.fixture()
    def mapper(self) -> MarketMapper:
        """Create a MarketMapper with no overrides."""
        return MarketMapper(overrides_path="/nonexistent/path.json")

    @pytest.fixture()
    def sample_event(self) -> OddsApiEvent:
        """Create a sample NBA game event."""
        return OddsApiEvent(
            id="event_123",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2026, 3, 15, 0, 0),
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            bookmakers=[
                OddsApiBookmaker(
                    key="pinnacle",
                    title="Pinnacle",
                    markets=[
                        OddsApiMarket(
                            key="h2h",
                            outcomes=[
                                OddsApiOutcome(name="Los Angeles Lakers", price=-150),
                                OddsApiOutcome(name="Boston Celtics", price=130),
                            ],
                        ),
                    ],
                ),
            ],
        )

    @pytest.fixture()
    def sample_poly_game_contracts(self) -> list[PolymarketContract]:
        """Create sample Polymarket game contracts."""
        return [
            PolymarketContract(
                token_id="token_lakers_celtics_lakers",
                condition_id="cond_game1",
                question="Lakers vs. Celtics",
                outcome="Lakers",
                current_price=0.58,
            ),
            PolymarketContract(
                token_id="token_lakers_celtics_celtics",
                condition_id="cond_game1",
                question="Lakers vs. Celtics",
                outcome="Celtics",
                current_price=0.42,
            ),
        ]

    def test_maps_game_by_team_names(
        self,
        mapper: MarketMapper,
        sample_event: OddsApiEvent,
        sample_poly_game_contracts: list[PolymarketContract],
    ) -> None:
        """Should map a game event to contracts mentioning both teams."""
        results = mapper.map_game(sample_event, sample_poly_game_contracts)
        assert len(results) >= 1
        assert len(results[0].polymarket_contracts) == 2

    def test_game_market_type(
        self,
        mapper: MarketMapper,
        sample_event: OddsApiEvent,
        sample_poly_game_contracts: list[PolymarketContract],
    ) -> None:
        """Mapped game with h2h-only event should have GAME_ML market type."""
        results = mapper.map_game(sample_event, sample_poly_game_contracts)
        assert len(results) == 1
        assert results[0].market_type == MarketType.GAME_ML

    def test_no_match_returns_empty(
        self,
        mapper: MarketMapper,
        sample_event: OddsApiEvent,
    ) -> None:
        """Should return empty list when no Polymarket contracts match."""
        unrelated = [
            PolymarketContract(
                token_id="token_other",
                condition_id="cond_other",
                question="Will the Hawks beat the Bulls?",
                outcome="Yes",
            ),
        ]
        result = mapper.map_game(sample_event, unrelated)
        assert result == []

    def test_event_name_format(
        self,
        mapper: MarketMapper,
        sample_event: OddsApiEvent,
        sample_poly_game_contracts: list[PolymarketContract],
    ) -> None:
        """Event name should be 'away @ home'."""
        results = mapper.map_game(sample_event, sample_poly_game_contracts)
        assert len(results) >= 1
        assert "Celtics" in results[0].event_name or "Boston" in results[0].event_name
        assert "Lakers" in results[0].event_name or "Los Angeles" in results[0].event_name

    def test_map_all_games(
        self,
        mapper: MarketMapper,
        sample_event: OddsApiEvent,
        sample_poly_game_contracts: list[PolymarketContract],
    ) -> None:
        """map_all_games should return a list of mapped markets."""
        results = mapper.map_all_games([sample_event], sample_poly_game_contracts)
        assert len(results) >= 1
        assert results[0].external_event_id == "event_123"


# -------------------------------------------------------------------
# Tests for split-by-market-type behavior
# -------------------------------------------------------------------


class TestGameMappingSplit:
    """Tests that map_game splits independent markets (h2h/spreads/totals)."""

    @pytest.fixture()
    def mapper(self) -> MarketMapper:
        return MarketMapper(overrides_path="/nonexistent/path.json")

    @pytest.fixture()
    def multi_market_event(self) -> OddsApiEvent:
        """Event with h2h, spreads, and totals from Pinnacle."""
        return OddsApiEvent(
            id="event_456",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2026, 3, 20, 0, 0),
            home_team="Houston Rockets",
            away_team="Charlotte Hornets",
            bookmakers=[
                OddsApiBookmaker(
                    key="pinnacle",
                    title="Pinnacle",
                    markets=[
                        OddsApiMarket(
                            key="h2h",
                            outcomes=[
                                OddsApiOutcome(name="Houston Rockets", price=-250),
                                OddsApiOutcome(name="Charlotte Hornets", price=210),
                            ],
                        ),
                        OddsApiMarket(
                            key="spreads",
                            outcomes=[
                                OddsApiOutcome(name="Houston Rockets", price=-110, point=-4.5),
                                OddsApiOutcome(name="Charlotte Hornets", price=-110, point=4.5),
                            ],
                        ),
                        OddsApiMarket(
                            key="totals",
                            outcomes=[
                                OddsApiOutcome(name="Over", price=-110, point=215.5),
                                OddsApiOutcome(name="Under", price=-105, point=215.5),
                            ],
                        ),
                    ],
                ),
            ],
        )

    @pytest.fixture()
    def rockets_hornets_contracts(self) -> list[PolymarketContract]:
        """Realistic contracts with proper question patterns for each market type."""
        return [
            # Moneyline
            PolymarketContract(
                token_id="token_ml_rockets",
                condition_id="cond_ml",
                question="Rockets vs. Hornets",
                outcome="Rockets",
                current_price=0.70,
            ),
            PolymarketContract(
                token_id="token_ml_hornets",
                condition_id="cond_ml",
                question="Rockets vs. Hornets",
                outcome="Hornets",
                current_price=0.30,
            ),
            # Spread matching external line (-4.5)
            PolymarketContract(
                token_id="token_spread_rockets",
                condition_id="cond_spread",
                question="Spread: Rockets (-4.5)",
                outcome="Rockets",
                current_price=0.50,
                end_date="2026-03-20T00:00:00",
            ),
            PolymarketContract(
                token_id="token_spread_hornets",
                condition_id="cond_spread",
                question="Spread: Hornets (+4.5)",
                outcome="Hornets",
                current_price=0.50,
                end_date="2026-03-20T00:00:00",
            ),
            # O/U matching external line (215.5)
            PolymarketContract(
                token_id="token_ou_over",
                condition_id="cond_ou",
                question="Rockets vs. Hornets: O/U 215.5",
                outcome="Over",
                current_price=0.52,
            ),
            PolymarketContract(
                token_id="token_ou_under",
                condition_id="cond_ou",
                question="Rockets vs. Hornets: O/U 215.5",
                outcome="Under",
                current_price=0.48,
            ),
        ]

    def test_returns_three_mapped_markets(
        self,
        mapper: MarketMapper,
        multi_market_event: OddsApiEvent,
        rockets_hornets_contracts: list[PolymarketContract],
    ) -> None:
        """map_game should return 3 MappedMarkets for h2h+spreads+totals."""
        results = mapper.map_game(multi_market_event, rockets_hornets_contracts)
        assert len(results) == 3

    def test_each_market_has_correct_type(
        self,
        mapper: MarketMapper,
        multi_market_event: OddsApiEvent,
        rockets_hornets_contracts: list[PolymarketContract],
    ) -> None:
        """Each MappedMarket should have the right MarketType."""
        results = mapper.map_game(multi_market_event, rockets_hornets_contracts)
        types = {m.market_type for m in results}
        assert types == {MarketType.GAME_ML, MarketType.SPREAD, MarketType.TOTAL}

    def test_each_market_has_two_odds(
        self,
        mapper: MarketMapper,
        multi_market_event: OddsApiEvent,
        rockets_hornets_contracts: list[PolymarketContract],
    ) -> None:
        """Each MappedMarket should contain exactly 2 odds entries (one pair)."""
        results = mapper.map_game(multi_market_event, rockets_hornets_contracts)
        for m in results:
            assert len(m.external_odds) == 2, (
                f"{m.market_type} has {len(m.external_odds)} odds, expected 2"
            )

    def test_odds_tagged_with_market_key(
        self,
        mapper: MarketMapper,
        multi_market_event: OddsApiEvent,
        rockets_hornets_contracts: list[PolymarketContract],
    ) -> None:
        """All odds within a MappedMarket should share the same market_key."""
        results = mapper.map_game(multi_market_event, rockets_hornets_contracts)
        for m in results:
            keys = {o.market_key for o in m.external_odds}
            assert len(keys) == 1, f"Mixed market_keys in {m.market_type}: {keys}"

    def test_moneyline_gets_only_moneyline_contracts(
        self,
        mapper: MarketMapper,
        multi_market_event: OddsApiEvent,
        rockets_hornets_contracts: list[PolymarketContract],
    ) -> None:
        """Moneyline MappedMarket should only contain moneyline contracts."""
        results = mapper.map_game(multi_market_event, rockets_hornets_contracts)
        ml_market = next(m for m in results if m.market_type == MarketType.GAME_ML)
        token_ids = {c.token_id for c in ml_market.polymarket_contracts}
        assert token_ids == {"token_ml_rockets", "token_ml_hornets"}

    def test_spread_gets_only_matching_spread_contracts(
        self,
        mapper: MarketMapper,
        multi_market_event: OddsApiEvent,
        rockets_hornets_contracts: list[PolymarketContract],
    ) -> None:
        """Spread MappedMarket should only contain spread contracts at the right line."""
        results = mapper.map_game(multi_market_event, rockets_hornets_contracts)
        spread_market = next(m for m in results if m.market_type == MarketType.SPREAD)
        token_ids = {c.token_id for c in spread_market.polymarket_contracts}
        assert token_ids == {"token_spread_rockets", "token_spread_hornets"}

    def test_total_gets_only_matching_ou_contracts(
        self,
        mapper: MarketMapper,
        multi_market_event: OddsApiEvent,
        rockets_hornets_contracts: list[PolymarketContract],
    ) -> None:
        """O/U MappedMarket should only contain O/U contracts at the right line."""
        results = mapper.map_game(multi_market_event, rockets_hornets_contracts)
        total_market = next(m for m in results if m.market_type == MarketType.TOTAL)
        token_ids = {c.token_id for c in total_market.polymarket_contracts}
        assert token_ids == {"token_ou_over", "token_ou_under"}

    def test_1h_contracts_excluded(
        self,
        mapper: MarketMapper,
        multi_market_event: OddsApiEvent,
    ) -> None:
        """1H contracts should not appear in any MappedMarket."""
        contracts = [
            PolymarketContract(
                token_id="token_ml_rockets",
                condition_id="cond_ml",
                question="Rockets vs. Hornets",
                outcome="Rockets",
            ),
            PolymarketContract(
                token_id="token_ml_hornets",
                condition_id="cond_ml",
                question="Rockets vs. Hornets",
                outcome="Hornets",
            ),
            PolymarketContract(
                token_id="token_1h_ml",
                condition_id="cond_1h_ml",
                question="Rockets vs. Hornets: 1H Moneyline",
                outcome="Rockets",
            ),
            PolymarketContract(
                token_id="token_1h_spread",
                condition_id="cond_1h_spread",
                question="1H Spread: Rockets (-1.5)",
                outcome="Rockets",
                end_date="2026-03-20T00:00:00",
            ),
            PolymarketContract(
                token_id="token_1h_ou",
                condition_id="cond_1h_ou",
                question="Rockets vs. Hornets: 1H O/U 103.5",
                outcome="Over",
            ),
        ]
        results = mapper.map_game(multi_market_event, contracts)
        all_token_ids = set()
        for m in results:
            for c in m.polymarket_contracts:
                all_token_ids.add(c.token_id)
        assert "token_1h_ml" not in all_token_ids
        assert "token_1h_spread" not in all_token_ids
        assert "token_1h_ou" not in all_token_ids

    def test_mismatched_spread_line_excluded(
        self,
        mapper: MarketMapper,
        multi_market_event: OddsApiEvent,
    ) -> None:
        """Spread contracts at a different line should not match."""
        contracts = [
            # External spread is -4.5, but contract is -2.5
            PolymarketContract(
                token_id="token_spread_wrong",
                condition_id="cond_spread_wrong",
                question="Spread: Rockets (-2.5)",
                outcome="Rockets",
                end_date="2026-03-20T00:00:00",
            ),
            PolymarketContract(
                token_id="token_spread_wrong2",
                condition_id="cond_spread_wrong",
                question="Spread: Hornets (+2.5)",
                outcome="Hornets",
                end_date="2026-03-20T00:00:00",
            ),
        ]
        results = mapper.map_game(multi_market_event, contracts)
        spread_markets = [m for m in results if m.market_type == MarketType.SPREAD]
        assert len(spread_markets) == 0

    def test_map_all_games_extends(
        self,
        mapper: MarketMapper,
        multi_market_event: OddsApiEvent,
        rockets_hornets_contracts: list[PolymarketContract],
    ) -> None:
        """map_all_games should include all per-type MappedMarkets."""
        results = mapper.map_all_games([multi_market_event], rockets_hornets_contracts)
        assert len(results) == 3


class TestFairValueAfterSplit:
    """Verify vig removal produces correct fair values after the split."""

    def test_h2h_fair_values_sum_to_one(self) -> None:
        """After vig removal on an h2h pair, fair probs should sum to ~1.0."""
        from src.reference.price_adapter import PriceAdapter
        from src.reference.vig_removal import american_to_probability

        rockets_prob = american_to_probability(-250)  # ~0.714
        hornets_prob = american_to_probability(210)    # ~0.323

        mapped = MappedMarket(
            external_event_id="evt_1",
            external_odds=[
                ExternalOdds(
                    team="Houston Rockets",
                    american_odds=-250,
                    implied_probability=rockets_prob,
                    bookmaker="pinnacle",
                    market_key="h2h",
                ),
                ExternalOdds(
                    team="Charlotte Hornets",
                    american_odds=210,
                    implied_probability=hornets_prob,
                    bookmaker="pinnacle",
                    market_key="h2h",
                ),
            ],
            polymarket_contracts=[
                PolymarketContract(
                    token_id="token_rockets",
                    condition_id="cond_1",
                    question="Will the Rockets beat the Hornets?",
                    outcome="Rockets",
                ),
                PolymarketContract(
                    token_id="token_hornets",
                    condition_id="cond_1",
                    question="Will the Rockets beat the Hornets?",
                    outcome="Hornets",
                ),
            ],
            market_type=MarketType.GAME_ML,
        )

        adapter = PriceAdapter(vig_method="proportional")
        ref_prices = adapter.adapt(mapped)
        assert len(ref_prices) == 2

        fair_sum = sum(rp.fair_probability for rp in ref_prices)
        assert 0.99 <= fair_sum <= 1.01, f"Fair values sum to {fair_sum}, expected ~1.0"

        # Rockets should be the heavy favourite
        rockets_fair = next(rp for rp in ref_prices if rp.token_id == "token_rockets")
        assert rockets_fair.fair_probability > 0.60


# -------------------------------------------------------------------
# Tests for parse_event_to_external_odds market_key tagging
# -------------------------------------------------------------------


class TestParseEventMarketKey:
    """Verify parse_event_to_external_odds tags market_key on each ExternalOdds."""

    def test_market_key_tagged(self) -> None:
        """Each ExternalOdds should carry the market_key from its source market."""
        from src.reference.odds_client import parse_event_to_external_odds

        event = OddsApiEvent(
            id="evt_mk",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2026, 3, 20, 0, 0),
            home_team="Houston Rockets",
            away_team="Charlotte Hornets",
            bookmakers=[
                OddsApiBookmaker(
                    key="pinnacle",
                    title="Pinnacle",
                    markets=[
                        OddsApiMarket(
                            key="h2h",
                            outcomes=[
                                OddsApiOutcome(name="Houston Rockets", price=-250),
                                OddsApiOutcome(name="Charlotte Hornets", price=210),
                            ],
                        ),
                        OddsApiMarket(
                            key="spreads",
                            outcomes=[
                                OddsApiOutcome(name="Houston Rockets", price=-110, point=-4.5),
                                OddsApiOutcome(name="Charlotte Hornets", price=-110, point=4.5),
                            ],
                        ),
                    ],
                ),
            ],
        )

        odds = parse_event_to_external_odds(event)
        h2h_odds = [o for o in odds if o.market_key == "h2h"]
        spread_odds = [o for o in odds if o.market_key == "spreads"]
        assert len(h2h_odds) == 2
        assert len(spread_odds) == 2


# -------------------------------------------------------------------
# Tests for classify_contract
# -------------------------------------------------------------------


class TestClassifyContract:
    """Tests for the classify_contract() helper."""

    def test_moneyline(self) -> None:
        c = PolymarketContract(
            token_id="t1", condition_id="c1",
            question="Rockets vs. Hornets", outcome="Rockets",
        )
        key, point = classify_contract(c)
        assert key == "h2h"
        assert point is None

    def test_spread_negative(self) -> None:
        c = PolymarketContract(
            token_id="t2", condition_id="c2",
            question="Spread: Rockets (-4.5)", outcome="Rockets",
        )
        key, point = classify_contract(c)
        assert key == "spreads"
        assert point == pytest.approx(4.5)

    def test_spread_positive(self) -> None:
        c = PolymarketContract(
            token_id="t3", condition_id="c3",
            question="Spread: Hornets (+4.5)", outcome="Hornets",
        )
        key, point = classify_contract(c)
        assert key == "spreads"
        assert point == pytest.approx(4.5)

    def test_total(self) -> None:
        c = PolymarketContract(
            token_id="t4", condition_id="c4",
            question="Rockets vs. Hornets: O/U 216.5", outcome="Over",
        )
        key, point = classify_contract(c)
        assert key == "totals"
        assert point == pytest.approx(216.5)

    def test_1h_moneyline(self) -> None:
        c = PolymarketContract(
            token_id="t5", condition_id="c5",
            question="Rockets vs. Hornets: 1H Moneyline", outcome="Rockets",
        )
        key, point = classify_contract(c)
        assert key is None
        assert point is None

    def test_1h_spread(self) -> None:
        c = PolymarketContract(
            token_id="t6", condition_id="c6",
            question="1H Spread: Rockets (-1.5)", outcome="Rockets",
        )
        key, point = classify_contract(c)
        assert key is None
        assert point is None

    def test_1h_ou(self) -> None:
        c = PolymarketContract(
            token_id="t7", condition_id="c7",
            question="Rockets vs. Hornets: 1H O/U 103.5", outcome="Over",
        )
        key, point = classify_contract(c)
        assert key is None
        assert point is None

    def test_yes_no_moneyline_excluded(self) -> None:
        """Yes/No outcomes in a 'vs.' question should not classify as h2h."""
        c = PolymarketContract(
            token_id="t8", condition_id="c8",
            question="Rockets vs. Hornets", outcome="Yes",
        )
        key, point = classify_contract(c)
        assert key is None

    def test_unrecognized(self) -> None:
        c = PolymarketContract(
            token_id="t9", condition_id="c9",
            question="Will Rockets make the playoffs?", outcome="Yes",
        )
        key, point = classify_contract(c)
        assert key is None
        assert point is None


# -------------------------------------------------------------------
# End-to-end: adapter produces correct fair values per market type
# -------------------------------------------------------------------


class TestAdapterEndToEnd:
    """Verify the adapter produces correct fair values for matched contracts only."""

    def test_totals_adapter(self) -> None:
        """O/U contracts should get Over/Under fair values from totals odds."""
        from src.reference.price_adapter import PriceAdapter
        from src.reference.vig_removal import american_to_probability

        over_prob = american_to_probability(-110)
        under_prob = american_to_probability(-105)

        mapped = MappedMarket(
            external_event_id="evt_ou",
            external_odds=[
                ExternalOdds(
                    team="Over",
                    american_odds=-110,
                    implied_probability=over_prob,
                    bookmaker="pinnacle",
                    market_key="totals",
                    point=215.5,
                ),
                ExternalOdds(
                    team="Under",
                    american_odds=-105,
                    implied_probability=under_prob,
                    bookmaker="pinnacle",
                    market_key="totals",
                    point=215.5,
                ),
            ],
            polymarket_contracts=[
                PolymarketContract(
                    token_id="token_over",
                    condition_id="cond_ou",
                    question="Rockets vs. Hornets: O/U 215.5",
                    outcome="Over",
                ),
                PolymarketContract(
                    token_id="token_under",
                    condition_id="cond_ou",
                    question="Rockets vs. Hornets: O/U 215.5",
                    outcome="Under",
                ),
            ],
            market_type=MarketType.TOTAL,
        )

        adapter = PriceAdapter(vig_method="proportional")
        ref_prices = adapter.adapt(mapped)
        assert len(ref_prices) == 2

        fair_sum = sum(rp.fair_probability for rp in ref_prices)
        assert 0.99 <= fair_sum <= 1.01, f"Fair values sum to {fair_sum}, expected ~1.0"

        over_rp = next(rp for rp in ref_prices if rp.token_id == "token_over")
        under_rp = next(rp for rp in ref_prices if rp.token_id == "token_under")
        # Over had slightly higher vig-implied prob, so should be slightly favored
        assert over_rp.fair_probability > 0.45
        assert under_rp.fair_probability > 0.45


# -------------------------------------------------------------------
# Tests for cross-game contamination prevention
# -------------------------------------------------------------------


class TestCrossGameContamination:
    """Verify that contracts from one game don't leak into another."""

    @pytest.fixture()
    def mapper(self) -> MarketMapper:
        return MarketMapper(overrides_path="/nonexistent/path.json")

    @pytest.fixture()
    def rockets_hornets_event(self) -> OddsApiEvent:
        return OddsApiEvent(
            id="evt_rockets_hornets",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2026, 3, 20, 0, 10),
            home_team="Charlotte Hornets",
            away_team="Houston Rockets",
            bookmakers=[
                OddsApiBookmaker(
                    key="pinnacle",
                    title="Pinnacle",
                    markets=[
                        OddsApiMarket(
                            key="h2h",
                            outcomes=[
                                OddsApiOutcome(name="Charlotte Hornets", price=155),
                                OddsApiOutcome(name="Houston Rockets", price=-177),
                            ],
                        ),
                    ],
                ),
            ],
        )

    @pytest.fixture()
    def nets_cavaliers_event(self) -> OddsApiEvent:
        return OddsApiEvent(
            id="evt_nets_cavs",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2026, 3, 20, 0, 10),
            home_team="Cleveland Cavaliers",
            away_team="Brooklyn Nets",
            bookmakers=[
                OddsApiBookmaker(
                    key="pinnacle",
                    title="Pinnacle",
                    markets=[
                        OddsApiMarket(
                            key="h2h",
                            outcomes=[
                                OddsApiOutcome(name="Cleveland Cavaliers", price=-1034),
                                OddsApiOutcome(name="Brooklyn Nets", price=692),
                            ],
                        ),
                    ],
                ),
            ],
        )

    @pytest.fixture()
    def all_contracts(self) -> list[PolymarketContract]:
        """Contracts for three different games — tests cross-contamination."""
        return [
            # Rockets vs Hornets moneyline
            PolymarketContract(
                token_id="tok_rh_rockets",
                condition_id="c_rh",
                question="Rockets vs. Hornets",
                outcome="Rockets",
            ),
            PolymarketContract(
                token_id="tok_rh_hornets",
                condition_id="c_rh",
                question="Rockets vs. Hornets",
                outcome="Hornets",
            ),
            # Nets vs Cavaliers moneyline
            PolymarketContract(
                token_id="tok_nc_nets",
                condition_id="c_nc",
                question="Nets vs. Cavaliers",
                outcome="Nets",
            ),
            PolymarketContract(
                token_id="tok_nc_cavs",
                condition_id="c_nc",
                question="Nets vs. Cavaliers",
                outcome="Cavaliers",
            ),
            # Cavaliers vs Hornets (DIFFERENT game — should not match either above)
            PolymarketContract(
                token_id="tok_ch_cavs",
                condition_id="c_ch",
                question="Cavaliers vs. Hornets",
                outcome="Cavaliers",
            ),
            PolymarketContract(
                token_id="tok_ch_hornets",
                condition_id="c_ch",
                question="Cavaliers vs. Hornets",
                outcome="Hornets",
            ),
        ]

    def test_rockets_hornets_no_contamination(
        self,
        mapper: MarketMapper,
        rockets_hornets_event: OddsApiEvent,
        all_contracts: list[PolymarketContract],
    ) -> None:
        """Rockets-Hornets game should not include 'Cavaliers vs. Hornets' contracts."""
        results = mapper.map_game(rockets_hornets_event, all_contracts)
        all_tokens = {c.token_id for m in results for c in m.polymarket_contracts}
        assert "tok_rh_rockets" in all_tokens
        assert "tok_rh_hornets" in all_tokens
        assert "tok_ch_cavs" not in all_tokens
        assert "tok_ch_hornets" not in all_tokens

    def test_nets_cavaliers_no_contamination(
        self,
        mapper: MarketMapper,
        nets_cavaliers_event: OddsApiEvent,
        all_contracts: list[PolymarketContract],
    ) -> None:
        """Nets-Cavaliers game should not include 'Cavaliers vs. Hornets' contracts."""
        results = mapper.map_game(nets_cavaliers_event, all_contracts)
        all_tokens = {c.token_id for m in results for c in m.polymarket_contracts}
        assert "tok_nc_nets" in all_tokens
        assert "tok_nc_cavs" in all_tokens
        assert "tok_ch_cavs" not in all_tokens
        assert "tok_ch_hornets" not in all_tokens

    def test_nets_not_substring_of_hornets(
        self,
        mapper: MarketMapper,
        nets_cavaliers_event: OddsApiEvent,
    ) -> None:
        """'Nets' should not match 'Hornets' via substring."""
        contracts = [
            PolymarketContract(
                token_id="tok_hornets_only",
                condition_id="c_h",
                question="Hornets vs. Pelicans",
                outcome="Hornets",
            ),
        ]
        results = mapper.map_game(nets_cavaliers_event, contracts)
        all_tokens = {c.token_id for m in results for c in m.polymarket_contracts}
        assert "tok_hornets_only" not in all_tokens


# -------------------------------------------------------------------
# Tests for in-progress game filtering in map_all_games
# -------------------------------------------------------------------


class TestInProgressGameFiltering:
    """Verify that map_all_games skips events whose commence_time is in the past."""

    @pytest.fixture()
    def mapper(self) -> MarketMapper:
        return MarketMapper(overrides_path="/nonexistent/path.json")

    @pytest.fixture()
    def poly_contracts(self) -> list[PolymarketContract]:
        """Contracts that match Lakers vs Celtics and Rockets vs Hornets."""
        return [
            PolymarketContract(
                token_id="tok_lakers",
                condition_id="c_lc",
                question="Lakers vs. Celtics",
                outcome="Lakers",
            ),
            PolymarketContract(
                token_id="tok_celtics",
                condition_id="c_lc",
                question="Lakers vs. Celtics",
                outcome="Celtics",
            ),
            PolymarketContract(
                token_id="tok_rockets",
                condition_id="c_rh",
                question="Rockets vs. Hornets",
                outcome="Rockets",
            ),
            PolymarketContract(
                token_id="tok_hornets",
                condition_id="c_rh",
                question="Rockets vs. Hornets",
                outcome="Hornets",
            ),
        ]

    def _make_event(self, event_id: str, home: str, away: str, commence_time: datetime) -> OddsApiEvent:
        """Helper to create a game event with h2h odds."""
        return OddsApiEvent(
            id=event_id,
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=commence_time,
            home_team=home,
            away_team=away,
            bookmakers=[
                OddsApiBookmaker(
                    key="pinnacle",
                    title="Pinnacle",
                    markets=[
                        OddsApiMarket(
                            key="h2h",
                            outcomes=[
                                OddsApiOutcome(name=home, price=-150),
                                OddsApiOutcome(name=away, price=130),
                            ],
                        ),
                    ],
                ),
            ],
        )

    def test_past_event_is_skipped(
        self,
        mapper: MarketMapper,
        poly_contracts: list[PolymarketContract],
    ) -> None:
        """An event with commence_time in the past should be skipped entirely."""
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        event = self._make_event("evt_past", "Los Angeles Lakers", "Boston Celtics", past_time)

        results = mapper.map_all_games([event], poly_contracts)
        assert len(results) == 0

    def test_future_event_is_processed(
        self,
        mapper: MarketMapper,
        poly_contracts: list[PolymarketContract],
    ) -> None:
        """An event with commence_time in the future should be processed normally."""
        future_time = datetime.now(timezone.utc) + timedelta(hours=3)
        event = self._make_event("evt_future", "Los Angeles Lakers", "Boston Celtics", future_time)

        results = mapper.map_all_games([event], poly_contracts)
        assert len(results) >= 1
        assert results[0].external_event_id == "evt_future"

    def test_mixed_events_only_future_mapped(
        self,
        mapper: MarketMapper,
        poly_contracts: list[PolymarketContract],
    ) -> None:
        """With mixed past/future events, only future ones should be mapped."""
        past_time = datetime.now(timezone.utc) - timedelta(hours=2)
        future_time = datetime.now(timezone.utc) + timedelta(hours=4)

        past_event = self._make_event("evt_past", "Los Angeles Lakers", "Boston Celtics", past_time)
        future_event = self._make_event("evt_future", "Houston Rockets", "Charlotte Hornets", future_time)

        results = mapper.map_all_games([past_event, future_event], poly_contracts)

        event_ids = {m.external_event_id for m in results}
        assert "evt_past" not in event_ids
        assert "evt_future" in event_ids

    def test_no_commence_time_is_processed(
        self,
        mapper: MarketMapper,
        poly_contracts: list[PolymarketContract],
    ) -> None:
        """An event with no commence_time (None) should still be processed."""
        event = self._make_event("evt_no_time", "Los Angeles Lakers", "Boston Celtics", None)

        results = mapper.map_all_games([event], poly_contracts)
        # Should not be skipped — None commence_time means we can't determine status
        assert len(results) >= 1

    def test_naive_past_datetime_is_skipped(
        self,
        mapper: MarketMapper,
        poly_contracts: list[PolymarketContract],
    ) -> None:
        """A naive (no tzinfo) datetime in the past should be treated as UTC and skipped."""
        # Naive datetime — no timezone info; code should assume UTC
        naive_past = datetime(2020, 1, 1, 0, 0, 0)
        event = self._make_event("evt_naive_past", "Los Angeles Lakers", "Boston Celtics", naive_past)

        results = mapper.map_all_games([event], poly_contracts)
        assert len(results) == 0
