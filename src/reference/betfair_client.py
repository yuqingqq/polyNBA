"""Client for Betfair Exchange via betfairlightweight.

Fetches NBA match odds from the Betfair Exchange and converts them to
OddsApiEvent format. Betfair exchange prices are vig-free (back/lay spread
is the only cost), so mid-prices represent fair value.

Requires:
  - betfairlightweight package
  - BETFAIR_USERNAME, BETFAIR_PASSWORD, BETFAIR_APP_KEY env vars
  - BETFAIR_CERT_DIR pointing to directory with client-ssn.crt and client-ssn.key
  - Optional: BETFAIR_PROXY_URL for routing through a non-US VPN

Graceful degradation: if credentials are missing, all methods return empty
results without raising exceptions.
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

from .market_mapper import normalize_team_name
from .odds_models import (
    OddsApiBookmaker,
    OddsApiEvent,
    OddsApiMarket,
    OddsApiOutcome,
)

logger = logging.getLogger(__name__)

# Betfair event type IDs
BASKETBALL_EVENT_TYPE_ID = "7522"

# Betfair market type for match winner
MATCH_ODDS_MARKET_TYPE = "MATCH_ODDS"

# Session keepalive interval — Betfair sessions expire after ~12h but we
# call keep_alive() proactively before each fetch if enough time has passed.
_KEEPALIVE_INTERVAL_S = 3600  # 1 hour

# Required cert file names inside BETFAIR_CERT_DIR
_CERT_FILES = ("client-ssn.crt", "client-ssn.key")


class BetfairClientError(Exception):
    """Raised when a Betfair API call fails."""
    pass


class BetfairClient:
    """Client for fetching NBA odds from the Betfair Exchange.

    Usage:
        client = BetfairClient()
        events = client.get_nba_game_events()
        for event in events:
            print(event.home_team, event.away_team)
    """

    def __init__(self) -> None:
        self._client = None
        self._available = False
        self._last_keepalive: float = 0.0
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the betfairlightweight API client.

        Reads credentials from environment variables. If any required
        credential is missing, marks the client as unavailable.
        """
        username = os.environ.get("BETFAIR_USERNAME", "")
        password = os.environ.get("BETFAIR_PASSWORD", "")
        app_key = os.environ.get("BETFAIR_APP_KEY", "")
        cert_dir = os.environ.get("BETFAIR_CERT_DIR", "")

        if not all([username, password, app_key]):
            logger.info(
                "Betfair credentials not configured — client disabled. "
                "Set BETFAIR_USERNAME, BETFAIR_PASSWORD, BETFAIR_APP_KEY."
            )
            return

        # Validate SSL certificates exist
        if cert_dir:
            missing = [f for f in _CERT_FILES if not os.path.isfile(os.path.join(cert_dir, f))]
            if missing:
                logger.warning(
                    "Betfair cert files missing in %s: %s — client disabled",
                    cert_dir, missing,
                )
                return
        else:
            logger.warning(
                "BETFAIR_CERT_DIR not set — Betfair requires SSL client certs. "
                "Client disabled."
            )
            return

        try:
            import betfairlightweight

            self._client = betfairlightweight.APIClient(
                username=username,
                password=password,
                app_key=app_key,
                certs=cert_dir,
            )

            # Configure proxy if provided
            proxy_url = os.environ.get("BETFAIR_PROXY_URL", "")
            if proxy_url:
                self._client.session.proxies = {
                    "https": proxy_url,
                    "http": proxy_url,
                }

            self._client.login()
            self._available = True
            self._last_keepalive = time.monotonic()
            logger.info("Betfair client initialized and logged in")

        except ImportError:
            logger.info("betfairlightweight not installed — Betfair client disabled")
        except Exception:
            logger.warning("Betfair login failed — client disabled", exc_info=True)

    @property
    def available(self) -> bool:
        """Whether the Betfair client is configured and logged in."""
        return self._available

    def _ensure_session(self) -> None:
        """Send keepalive or re-login if session may have expired.

        Called before each API request. Betfair sessions expire after ~12h;
        we send keep_alive() every hour as a safety margin.  If keep_alive
        fails, we attempt a full re-login once.
        """
        if self._client is None:
            return

        elapsed = time.monotonic() - self._last_keepalive
        if elapsed < _KEEPALIVE_INTERVAL_S:
            return

        try:
            self._client.keep_alive()
            self._last_keepalive = time.monotonic()
            logger.debug("Betfair session keepalive sent")
        except Exception:
            logger.warning("Betfair keepalive failed — attempting re-login", exc_info=True)
            try:
                self._client.login()
                self._last_keepalive = time.monotonic()
                logger.info("Betfair re-login successful")
            except Exception:
                logger.error("Betfair re-login failed — marking unavailable", exc_info=True)
                self._available = False

    def get_nba_game_events(self) -> list[OddsApiEvent]:
        """Fetch NBA match odds and convert to OddsApiEvent format.

        Returns:
            List of OddsApiEvent, one per NBA game on Betfair.
            Empty list if client is unavailable or on error.
        """
        if not self._available or self._client is None:
            return []

        self._ensure_session()
        if not self._available:
            return []

        try:
            return self._fetch_and_convert()
        except Exception:
            # On auth-like failures, try one re-login and retry
            logger.warning("Betfair fetch failed — attempting re-login", exc_info=True)
            try:
                self._client.login()
                self._last_keepalive = time.monotonic()
                return self._fetch_and_convert()
            except Exception:
                logger.error("Betfair fetch failed after re-login", exc_info=True)
                return []

    def _fetch_and_convert(self) -> list[OddsApiEvent]:
        """Internal: fetch NBA markets from Betfair and convert.

        Returns:
            List of OddsApiEvent.
        """
        import betfairlightweight.filters as filters

        # Step 1: Find NBA match odds market catalogues (OPEN markets only)
        market_filter = filters.market_filter(
            event_type_ids=[BASKETBALL_EVENT_TYPE_ID],
            market_type_codes=[MATCH_ODDS_MARKET_TYPE],
        )

        catalogues = self._client.betting.list_market_catalogue(
            filter=market_filter,
            market_projection=["RUNNER_DESCRIPTION", "EVENT", "MARKET_START_TIME"],
            max_results=100,  # NBA has at most ~15 games/day; 100 is plenty
        )

        if not catalogues:
            logger.debug("No NBA match odds markets found on Betfair")
            return []

        # Step 2: Fetch best prices for all markets
        market_ids = [c.market_id for c in catalogues]
        price_projection = filters.price_projection(
            price_data=["EX_BEST_OFFERS"],
        )

        market_books = self._client.betting.list_market_book(
            market_ids=market_ids,
            price_projection=price_projection,
        )

        # Index books by market_id for quick lookup
        books_by_id = {b.market_id: b for b in market_books}

        # Step 3: Convert each catalogue + book into an OddsApiEvent
        events: list[OddsApiEvent] = []
        for catalogue in catalogues:
            # Skip non-open markets from the book
            book = books_by_id.get(catalogue.market_id)
            if book is not None:
                status = getattr(book, "status", None)
                if status and status not in ("OPEN", "ACTIVE"):
                    logger.debug(
                        "Skipping Betfair market %s — status=%s",
                        catalogue.market_id, status,
                    )
                    continue

            event = self._convert_market(catalogue, book)
            if event is not None:
                events.append(event)

        logger.info("Converted %d Betfair NBA markets to OddsApiEvent format", len(events))
        return events

    def _convert_market(self, catalogue, book) -> Optional[OddsApiEvent]:
        """Convert a Betfair market catalogue + book into an OddsApiEvent.

        Filters out non-NBA-team runners (e.g., "The Draw") and requires
        exactly 2 recognized NBA teams.

        Args:
            catalogue: Market catalogue with runner descriptions and event info.
            book: Market book with current prices.

        Returns:
            OddsApiEvent or None if conversion fails.
        """
        if not catalogue.runners or len(catalogue.runners) < 2:
            return None

        # Extract runner names, filtering to only recognized NBA teams
        runner_names: dict[int, str] = {}
        team_runners: list[tuple[int, str]] = []  # (selection_id, canonical_name)

        for runner in catalogue.runners:
            name = getattr(runner, "runner_name", None) or str(runner.selection_id)
            runner_names[runner.selection_id] = name
            canonical = normalize_team_name(name)
            if canonical:
                team_runners.append((runner.selection_id, canonical))

        # Need exactly 2 recognized NBA teams (ignore "The Draw" etc.)
        if len(team_runners) != 2:
            if len(team_runners) > 2:
                logger.debug(
                    "Betfair market %s has %d NBA runners — expected 2, skipping",
                    getattr(catalogue, "market_id", "?"), len(team_runners),
                )
            else:
                logger.debug(
                    "Could not find 2 NBA teams in Betfair runners: %s",
                    list(runner_names.values()),
                )
            return None

        team1_canonical = team_runners[0][1]
        team2_canonical = team_runners[1][1]

        # Use event name to determine home/away
        # Betfair typically lists "Away @ Home" or "Team1 v Team2"
        event_name = ""
        if hasattr(catalogue, "event") and catalogue.event:
            event_name = getattr(catalogue.event, "name", "")

        if "@" in event_name:
            # "Away @ Home" format — first listed is away
            home_team = team2_canonical
            away_team = team1_canonical
        else:
            # Default: first listed = away, second = home
            home_team = team2_canonical
            away_team = team1_canonical

        # Extract actual commence_time from event data
        commence_time = self._extract_commence_time(catalogue)

        # Extract prices from the book (only for the 2 team runners)
        team_selection_ids = {tr[0] for tr in team_runners}
        outcomes = self._extract_outcomes(
            book, runner_names, home_team, away_team, team_selection_ids,
        )
        if not outcomes:
            return None

        now = datetime.now(timezone.utc)
        bookmaker = OddsApiBookmaker(
            key="betfair",
            title="Betfair Exchange",
            last_update=now,
            markets=[
                OddsApiMarket(
                    key="h2h",
                    last_update=now,
                    outcomes=outcomes,
                ),
            ],
        )

        event_id = f"betfair_{catalogue.market_id}"

        return OddsApiEvent(
            id=event_id,
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=commence_time or now,
            home_team=home_team,
            away_team=away_team,
            bookmakers=[bookmaker],
        )

    def _extract_outcomes(
        self,
        book,
        runner_names: dict,
        home_team: str,
        away_team: str,
        team_selection_ids: set[int],
    ) -> list[OddsApiOutcome]:
        """Extract OddsApiOutcome from Betfair market book prices.

        Computes mid-price from best back/lay prices for each runner.
        Only processes runners in team_selection_ids (skips "Draw" etc.).

        Args:
            book: Market book with runner prices.
            runner_names: Mapping of selection_id -> runner name.
            home_team: Canonical home team.
            away_team: Canonical away team.
            team_selection_ids: Selection IDs of the 2 NBA team runners.

        Returns:
            List of 2 OddsApiOutcome or empty list.
        """
        if book is None or not book.runners:
            return []

        outcomes_by_team: dict[str, float] = {}

        for runner in book.runners:
            # Skip non-team runners (e.g., "The Draw")
            if runner.selection_id not in team_selection_ids:
                continue

            name = runner_names.get(runner.selection_id, "")
            canonical = normalize_team_name(name)
            if not canonical:
                continue

            # Get best back and lay prices (with bounds check)
            back_price = None
            lay_price = None

            ex = getattr(runner, "ex", None)
            if ex:
                backs = getattr(ex, "available_to_back", None)
                if backs and len(backs) > 0:
                    back_price = backs[0].price
                lays = getattr(ex, "available_to_lay", None)
                if lays and len(lays) > 0:
                    lay_price = lays[0].price

            mid_prob = self._compute_mid_from_decimal(back_price, lay_price)
            if mid_prob is not None:
                outcomes_by_team[canonical] = mid_prob

        if home_team not in outcomes_by_team or away_team not in outcomes_by_team:
            return []

        home_prob = outcomes_by_team[home_team]
        away_prob = outcomes_by_team[away_team]

        # Normalize to sum to 1.0
        total = home_prob + away_prob
        if total > 0:
            home_prob /= total
            away_prob /= total

        home_american = _prob_to_american(home_prob)
        away_american = _prob_to_american(away_prob)

        return [
            OddsApiOutcome(name=home_team, price=home_american),
            OddsApiOutcome(name=away_team, price=away_american),
        ]

    @staticmethod
    def _extract_commence_time(catalogue) -> Optional[datetime]:
        """Extract event start time from Betfair catalogue.

        Checks market_start_time and event.open_date.

        Args:
            catalogue: Betfair market catalogue object.

        Returns:
            Timezone-aware datetime or None.
        """
        # market_start_time is the most reliable
        mst = getattr(catalogue, "market_start_time", None)
        if isinstance(mst, datetime):
            if mst.tzinfo is None:
                return mst.replace(tzinfo=timezone.utc)
            return mst

        # Fallback: event.open_date
        if hasattr(catalogue, "event") and catalogue.event:
            od = getattr(catalogue.event, "open_date", None)
            if isinstance(od, datetime):
                if od.tzinfo is None:
                    return od.replace(tzinfo=timezone.utc)
                return od

        return None

    @staticmethod
    def _compute_mid_from_decimal(
        back_price: Optional[float], lay_price: Optional[float]
    ) -> Optional[float]:
        """Compute mid-price probability from Betfair decimal prices.

        Betfair uses decimal odds (e.g., 2.0 = even money = 50%).
        Mid-price = average of 1/back and 1/lay.

        Args:
            back_price: Best available back (buy) decimal price.
            lay_price: Best available lay (sell) decimal price.

        Returns:
            Mid-price probability (0-1), or None.
        """
        if back_price is not None and lay_price is not None:
            if back_price > 0 and lay_price > 0:
                back_prob = 1.0 / back_price
                lay_prob = 1.0 / lay_price
                return (back_prob + lay_prob) / 2.0

        if back_price is not None and back_price > 0:
            return 1.0 / back_price

        if lay_price is not None and lay_price > 0:
            return 1.0 / lay_price

        return None


def _prob_to_american(prob: float) -> int:
    """Convert probability (0-1) to American odds."""
    prob = max(0.01, min(0.99, prob))
    if prob >= 0.5:
        return round(-100 * prob / (1 - prob))
    else:
        return round(100 * (1 - prob) / prob)
