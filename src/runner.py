"""
Production runner — single entry point that wires P-001 (reference prices),
P-002 (real-time data), and P-003 (market making) into a runnable system.

Usage:
    python3 -m src.runner                                     # dry-run (all markets)
    python3 -m src.runner --markets config/markets.json       # dry-run (selected markets)
    python3 -m src.runner --markets config/markets.json --live  # live (selected markets)
    python3 -m src.runner --config mm.json                    # custom config
    python3 -m src.runner --markets m.json --live --split 50  # split $50/condition into YES+NO
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
import uuid

from dotenv import load_dotenv

from src.data import (
    OrderbookManager,
    OrderEventType,
    PolymarketFeed,
    ReferenceFeed,
    ReferenceUpdate,
    TradeStatus,
    UserFeed,
    UserOrderEvent,
    UserTradeEvent,
)
from src.mm import MMConfig, MarketMakingEngine, Fill, Side, load_config
from src.reference import ReferencePrice
from src.reference.betfair_client import BetfairClient
from src.reference.composite_fetcher import CompositeReferenceFetcher
from src.reference.draftkings_client import DraftKingsClient
from src.reference.fanduel_client import FanDuelClient
from src.reference.kalshi_client import KalshiClient
from src.reference.market_mapper import MarketMapper
from src.reference.odds_client import OddsClient, OddsClientError, parse_event_to_external_odds
from src.reference.polymarket_scanner import PolymarketScanner
from src.reference.price_adapter import PriceAdapter

logger = logging.getLogger(__name__)

POLYMARKET_HOST = "https://clob.polymarket.com"
PREFERRED_BOOKMAKERS = ["fanduel", "betfair", "kalshi", "betfair_ex_eu", "pinnacle", "draftkings"]

# Heartbeat interval for the CLOB API (seconds).  Polymarket auto-cancels
# all orders if no heartbeat is received within 10s (with 5s buffer).
_HEARTBEAT_INTERVAL = 5.0


class ClobHeartbeat:
    """Background thread that sends CLOB API heartbeats to keep orders alive.

    Polymarket requires POST /v1/heartbeats every ~10s.  Without it, all
    open orders are automatically cancelled.  We send every 5s for margin.
    """

    def __init__(self, client, engine=None) -> None:
        self._client = client
        self._engine = engine  # optional reference to engine for quote clearing
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # First call uses empty string; server returns the actual heartbeat_id
        # which we store and reuse for all subsequent calls.
        self._heartbeat_id = ""
        self._consecutive_failures = 0
        # R23-I1: Track whether clear_quotes has been called for the current
        # failure episode. Prevents calling it on every heartbeat tick (every
        # 5s) during extended outages, which would cause repeated cancel-and-
        # replace thrashing in the engine.
        self._quotes_cleared_for_failure = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="clob-heartbeat",
        )
        self._thread.start()
        logger.info("CLOB heartbeat started (interval=%.0fs)", _HEARTBEAT_INTERVAL)

    def stop(self) -> None:
        self._stop.set()
        # R15-I1 fix: Capture thread ref before clearing to avoid
        # AttributeError if another thread sets _thread=None concurrently
        # (e.g., _force_stop_heartbeat timer racing with normal shutdown).
        t = self._thread
        self._thread = None
        if t is not None:
            t.join(timeout=10)
        logger.info("CLOB heartbeat stopped")

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                resp = self._client.post_heartbeat(self._heartbeat_id)
                # Server returns {"heartbeat_id": "..."} — use it for subsequent calls
                if isinstance(resp, dict) and "heartbeat_id" in resp:
                    self._heartbeat_id = resp["heartbeat_id"]
                self._consecutive_failures = 0
                self._quotes_cleared_for_failure = False
            except Exception:
                self._consecutive_failures += 1
                logger.warning(
                    "CLOB heartbeat failed (%d consecutive)",
                    self._consecutive_failures, exc_info=True,
                )
                # R23-I1: Only clear quotes ONCE per failure episode.
                # Repeated clear_quotes every 5s during extended outages
                # causes the engine to cancel-and-replace on every tick,
                # wasting API calls and risking partial fills from orders
                # placed then auto-cancelled by the exchange.
                if (self._consecutive_failures >= 3
                        and self._engine is not None
                        and not self._quotes_cleared_for_failure):
                    logger.critical(
                        "Heartbeat failed %d consecutive times — orders likely auto-cancelled by exchange. "
                        "Clearing quote state so engine will re-place on recovery.",
                        self._consecutive_failures,
                    )
                    self._engine.clear_quotes()
                    self._quotes_cleared_for_failure = True
            self._stop.wait(timeout=_HEARTBEAT_INTERVAL)


# ---------------------------------------------------------------------------
# 1. Config & credentials
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket market-making runner")
    parser.add_argument("--live", action="store_true", help="Enable live trading (default: dry-run)")
    parser.add_argument("--config", type=str, default=None, help="Path to MMConfig JSON file")
    parser.add_argument("--markets", type=str, default=None,
                        help="Path to markets JSON (from market_analysis --json)")
    parser.add_argument("--split", type=float, default=None,
                        help="Split USDC per condition into YES+NO tokens (requires --live)")
    return parser.parse_args()


def load_credentials(live: bool) -> dict:
    load_dotenv()
    creds = {
        "private_key": os.getenv("POLYMARKET_PRIVATE_KEY", ""),
        "api_key": os.getenv("POLYMARKET_API_KEY", ""),
        "api_secret": os.getenv("POLYMARKET_API_SECRET", ""),
        "api_passphrase": os.getenv("POLYMARKET_API_PASSPHRASE", ""),
        "chain_id": int(os.getenv("POLYMARKET_CHAIN_ID", "137")),
        "funder": os.getenv("POLYMARKET_FUNDER", ""),
        "signature_type": int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0")),
    }
    # OddsClient handles multi-key loading from ODDS_API_KEYS (comma-separated) or ODDS_API_KEY
    if not (os.getenv("ODDS_API_KEYS") or os.getenv("ODDS_API_KEY")):
        logger.error("No Odds API key found. Set ODDS_API_KEYS (comma-separated) or ODDS_API_KEY")
        sys.exit(1)
    if live:
        missing = [k for k in ("private_key", "api_key", "api_secret", "api_passphrase")
                   if not creds[k]]
        if missing:
            logger.error("Live mode requires credentials: %s", ", ".join(missing))
            sys.exit(1)
    # I-6: Validate signature_type when funder is set.
    # When using a proxy/funder wallet, signature_type must be 1 or 2 (not 0).
    if creds.get("funder") and creds["signature_type"] == 0:
        logger.error(
            "POLYMARKET_FUNDER is set but POLYMARKET_SIGNATURE_TYPE is 0. "
            "With a funder/proxy wallet, signature_type should be 1 or 2. "
            "Defaulting to signature_type=2 (POLY_GNOSIS_SAFE)."
        )
        creds["signature_type"] = 2
    return creds


def load_markets_file(path: str) -> set[str]:
    """Load selected token_ids from a markets JSON file."""
    with open(path) as f:
        data = json.load(f)
    token_ids = {entry["token_id"] for entry in data if "token_id" in entry}
    logger.info("Loaded %d target token_ids from %s", len(token_ids), path)
    return token_ids


# ---------------------------------------------------------------------------
# 2. ClobClient factory
# ---------------------------------------------------------------------------

def create_clob_client(creds: dict, live: bool):
    """Create a ClobClient for live mode, or None for dry-run."""
    if not live:
        logger.info("Dry-run mode — no ClobClient created")
        return None

    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds

    api_creds = ApiCreds(
        api_key=creds["api_key"],
        api_secret=creds["api_secret"],
        api_passphrase=creds["api_passphrase"],
    )
    kwargs = dict(
        host=POLYMARKET_HOST,
        key=creds["private_key"],
        chain_id=creds["chain_id"],
        creds=api_creds,
    )
    if creds.get("funder"):
        kwargs["funder"] = creds["funder"]
        kwargs["signature_type"] = creds.get("signature_type", 2)
        logger.info("Using proxy wallet: funder=%s signature_type=%d",
                    creds["funder"], kwargs["signature_type"])
    client = ClobClient(**kwargs)
    logger.info("ClobClient created (chain_id=%d)", creds["chain_id"])
    return client


# ---------------------------------------------------------------------------
# 3. Market discovery
# ---------------------------------------------------------------------------

def discover_markets(skip_championship: bool = False):
    """Scan Polymarket NBA contracts and map to external odds.

    Args:
        skip_championship: If True, skip the championship odds endpoint (saves 1 API call).
            Use when trading game markets only.
    """
    scanner = PolymarketScanner()
    poly_contracts = scanner.get_all_nba_contracts()
    logger.info("Found %d Polymarket NBA contracts", len(poly_contracts))

    odds_client = OddsClient()
    mapper = MarketMapper(preferred_bookmakers=PREFERRED_BOOKMAKERS)
    adapter = PriceAdapter()
    mapped_markets: list = []

    # Championship odds (skip when trading game markets only)
    if not skip_championship:
        try:
            championship_events = odds_client.get_nba_championship_odds()
            external_odds = []
            for event in championship_events:
                external_odds.extend(
                    parse_event_to_external_odds(event, bookmaker_filter=PREFERRED_BOOKMAKERS)
                )
            logger.info("Fetched %d championship odds from %s", len(external_odds), PREFERRED_BOOKMAKERS)
            mapped_markets.extend(mapper.map_championship(external_odds, poly_contracts))
        except (OddsClientError, ValueError) as e:
            logger.warning("Championship odds unavailable: %s", e)

    # Game odds
    try:
        game_events = odds_client.get_nba_game_odds(regions="us,eu")
        logger.info("Fetched %d game events", len(game_events))
        mapped_markets.extend(mapper.map_all_games(game_events, poly_contracts, skip_in_progress=False))
    except (OddsClientError, ValueError) as e:
        logger.warning("Game odds unavailable: %s", e)

    logger.info("Mapped %d total markets", len(mapped_markets))

    reference_prices = []
    for mm in mapped_markets:
        reference_prices.extend(adapter.adapt(mm))

    logger.info("Generated %d initial reference prices", len(reference_prices))
    return poly_contracts, mapped_markets, reference_prices


# ---------------------------------------------------------------------------
# 4. Reference fetch function (bridge P-001 → P-002)
# ---------------------------------------------------------------------------

def build_reference_fetch_fn(
    poly_contracts,
    token_whitelist: set[str] | None = None,
    include_championship: bool | None = None,
):
    """Return a callable that fetches fresh reference prices as ReferenceUpdates.

    Uses CompositeReferenceFetcher for game odds (Betfair → Kalshi → Odds API
    per-game fallback). Championship odds still come from The Odds API only
    since Kalshi/Betfair don't have NBA futures markets.

    Args:
        poly_contracts: All Polymarket contracts (filtered to whitelist if provided).
        token_whitelist: If set, only fetch odds for contracts matching these token IDs.
            Reduces API calls from ~38 updates to just the traded tokens.
    """
    # Filter contracts to only whitelisted tokens — mapper only matches games
    # that have at least one of these contracts, dramatically reducing API waste.
    if token_whitelist:
        poly_contracts = [c for c in poly_contracts if c.token_id in token_whitelist]
        logger.info("Reference fetch scoped to %d contracts (from whitelist)", len(poly_contracts))

    if include_championship is None:
        # Default: only fetch championship if no filter is applied.
        include_championship = token_whitelist is None
    odds_client = OddsClient()
    mapper = MarketMapper(preferred_bookmakers=PREFERRED_BOOKMAKERS)
    adapter = PriceAdapter(preferred_bookmakers=PREFERRED_BOOKMAKERS)

    # Initialize exchange clients (graceful degradation if credentials missing)
    kalshi_client = KalshiClient()
    betfair_client = BetfairClient()
    draftkings_client = DraftKingsClient()
    fanduel_client = FanDuelClient()

    composite = CompositeReferenceFetcher(
        poly_contracts=poly_contracts,
        mapper=mapper,
        adapter=adapter,
        fanduel_client=fanduel_client,
        draftkings_client=draftkings_client,
        kalshi_client=kalshi_client,
        betfair_client=betfair_client,
        odds_client=odds_client,
    )

    def fetch_fn() -> list[ReferenceUpdate]:
        try:
            all_mapped = []
            # Championship odds: still from The Odds API only
            # (Kalshi/Betfair don't have NBA futures)
            if include_championship:
                try:
                    events = odds_client.get_nba_championship_odds()
                    external_odds = []
                    for event in events:
                        external_odds.extend(
                            parse_event_to_external_odds(event, bookmaker_filter=PREFERRED_BOOKMAKERS)
                        )
                    all_mapped.extend(mapper.map_championship(external_odds, poly_contracts))
                except (OddsClientError, ValueError):
                    logger.debug("Championship odds unavailable in refresh")

            # Game odds: composite Kalshi → Betfair → Odds API per-game fallback
            try:
                game_mapped = composite.fetch_mapped_markets()
                all_mapped.extend(game_mapped)
            except Exception:
                logger.warning("Composite game fetch failed", exc_info=True)

            updates = []
            for m in all_mapped:
                for ref_price in adapter.adapt(m):
                    updates.append(reference_to_update(ref_price))
            logger.info("Reference fetch: %d updates", len(updates))
            return updates
        except Exception:
            logger.exception("Reference fetch failed")
            return []

    return fetch_fn


def reference_to_update(ref: ReferencePrice) -> ReferenceUpdate:
    """Bridge ReferencePrice (P-001) → ReferenceUpdate (P-002).

    C-4: Uses arrival time (time.time()) instead of bookmaker's last_update
    timestamp. The bookmaker timestamp may already be stale when received
    (e.g., Pinnacle last updated 55s ago + 5s poll = 60s old on arrival),
    causing premature staleness detection and order cancellation.
    """
    return ReferenceUpdate(
        token_id=ref.token_id,
        fair_probability=ref.fair_probability,
        source=ref.source,
        timestamp=time.time(),
    )


def user_trade_to_fills(event: UserTradeEvent) -> list[Fill]:
    """Bridge UserTradeEvent (P-002) → list of Fill (P-003).

    A single trade event can contain many maker orders from different users.
    I-6: Emits fill candidates for ALL orders in the event (taker + all makers).
    The engine's on_fill checks tracked order membership inside its lock,
    avoiding the race where orders are cleaned up between a pre-check and
    fill processing.

    I-7: Side enum conversion is wrapped in try/except to prevent crashes
    from unexpected enum values.

    The top-level ``event.size`` is the TAKER's total trade size and must
    NOT be used as our fill size — each maker order has its own
    ``matched_amount``.
    """
    fills: list[Fill] = []

    # Emit taker fill candidate
    # I-5 fix: Taker side parse failure should NOT drop maker fills.
    # Process taker independently — fall through to maker processing regardless.
    if event.taker_order_id:
        try:
            side = Side(event.side.value)
            fills.append(Fill(
                order_id=event.taker_order_id,
                token_id=event.asset_id,
                side=side,
                price=event.price,
                size=event.size,
            ))
        except (ValueError, AttributeError):
            logger.error("Unknown side value in taker: %s — skipping taker, processing makers", event.side)

    # Emit maker fill candidates
    for mo in event.maker_orders:
        try:
            mo_side = Side(mo.side.value)
        except (ValueError, AttributeError):
            logger.error("Unknown side value in maker order: %s", mo.side)
            continue
        fills.append(Fill(
            order_id=mo.order_id,
            token_id=mo.asset_id,
            side=mo_side,
            price=mo.price,
            size=mo.matched_amount,
        ))

    return fills


# ---------------------------------------------------------------------------
# 5. Seed existing positions from trade history
# ---------------------------------------------------------------------------

def _fetch_trade_positions(
    client,
    token_ids: list[str],
    funder: str,
) -> dict[str, tuple[float, float]]:
    """Fetch trade history from the API and compute net positions per token.

    This is phase 1 of position seeding — performs HTTP calls and must be
    called OUTSIDE any engine lock (C-2).

    Returns:
        Dict mapping token_id -> (net_size, avg_price). Only tokens with
        |net_size| >= 0.01 are included.
    """
    # I-6: If funder is empty, we cannot reliably filter trades — "".lower()
    # would match trades with empty/missing addresses, giving false positives.
    # Position seeding requires a valid funder address.
    if not funder:
        logger.warning(
            "POLYMARKET_FUNDER is not set — cannot seed positions from trade history. "
            "Engine will start with zero positions and accumulate naturally from fills."
        )
        return {}

    try:
        from py_clob_client.clob_types import TradeParams
        # I-2: Use time filter to limit fetch volume instead of bailing out at >=100.
        # py_clob_client auto-paginates get_trades(), so we don't need to worry about
        # truncation. Limit to last 24 hours to avoid fetching entire trade history.
        #
        # I-6 KNOWN LIMITATION: TradeParams only supports maker_address filter (no
        # taker_address). Trades where we were the taker (accumulation cross-spread
        # with post_only=False) won't appear in results. In practice this is low-risk:
        # taker fills are processed immediately via the fill callback, so they'd only
        # be missed if the bot crashes between order submission and fill processing
        # (a ~100ms window). The seeding primarily catches maker fills from orders
        # placed before the bot went offline.
        after_ts = int(time.time()) - 86400  # 24 hours ago
        trades_params = TradeParams(maker_address=funder, after=after_ts)
        trades = client.get_trades(params=trades_params)
    except Exception as e:
        logger.warning("Could not fetch trade history for position seeding: %s", e)
        return {}

    trade_count = len(trades) if trades else 0
    logger.info("Fetched %d trades for position seeding (last 24h)", trade_count)

    # I-2: Warn if trade count is very large (but don't bail out — client handles pagination)
    if trade_count > 1000:
        logger.warning(
            "Large trade history: %d trades fetched. Position seeding may take longer than expected.",
            trade_count,
        )

    token_set = set(token_ids)
    funder_lower = funder.lower()

    # Aggregate net position per token: BUY adds, SELL subtracts
    positions: dict[str, float] = {}  # token_id -> net size
    avg_prices: dict[str, float] = {}  # token_id -> volume-weighted avg price
    # C-1: Track cumulative buy volume and total buy cost separately.
    # Do NOT reduce volumes on sells — this prevents avg_price drift when
    # buys and sells interleave. The average reflects cost basis of all buys.
    buy_volumes: dict[str, float] = {}  # token_id -> cumulative buy volume
    buy_costs: dict[str, float] = {}  # token_id -> cumulative buy cost

    # C-2: Deduplicate trades by trade ID to prevent double-counting from
    # paginated API results.
    seen_trade_ids: set[str] = set()

    for trade in trades:
        # C-2: Skip duplicate trades
        trade_id = trade.get("id", "")
        if trade_id and trade_id in seen_trade_ids:
            continue
        if trade_id:
            seen_trade_ids.add(trade_id)

        asset_id = trade.get("asset_id", "")
        if asset_id not in token_set:
            continue

        # C-4 KNOWN LIMITATION: This taker_address check is effectively dead code
        # because TradeParams(maker_address=funder) only returns trades where we are
        # a maker. The CLOB API does not support taker_address as a filter parameter,
        # so taker fills (from accumulation with post_only=False) are never returned.
        # This code is kept for correctness if the API adds taker_address support in
        # the future or if trade data is sourced from a different endpoint.
        if trade.get("taker_address", "").lower() == funder_lower:
            side = trade.get("side", "")
            size = float(trade.get("size", 0))
            price = float(trade.get("price", 0))
            # M-2: Skip zero-size trades to avoid division by zero in avg price calc
            if size <= 0:
                continue
            if side == "BUY":
                positions[asset_id] = positions.get(asset_id, 0) + size
                # C-1: Accumulate buy cost basis
                buy_volumes[asset_id] = buy_volumes.get(asset_id, 0) + size
                buy_costs[asset_id] = buy_costs.get(asset_id, 0) + price * size
                avg_prices[asset_id] = buy_costs[asset_id] / buy_volumes[asset_id]
            elif side == "SELL":
                positions[asset_id] = positions.get(asset_id, 0) - size
                # C-1: Do NOT adjust buy_volumes on sells — avg_price tracks
                # cost basis of all buys, not remaining position.
            continue  # skip maker_orders loop for this trade

        # Check if we are a maker
        for mo in trade.get("maker_orders", []):
            if mo.get("maker_address", "").lower() != funder_lower:
                continue
            mo_asset = mo.get("asset_id", "")
            if mo_asset not in token_set:
                continue
            mo_side = mo.get("side", "")
            mo_size = float(mo.get("matched_amount", 0))
            mo_price = float(mo.get("price", 0))
            # M-2: Skip zero-size maker orders to avoid division by zero
            if mo_size <= 0:
                continue
            if mo_side == "BUY":
                positions[mo_asset] = positions.get(mo_asset, 0) + mo_size
                # C-1: Accumulate buy cost basis
                buy_volumes[mo_asset] = buy_volumes.get(mo_asset, 0) + mo_size
                buy_costs[mo_asset] = buy_costs.get(mo_asset, 0) + mo_price * mo_size
                avg_prices[mo_asset] = buy_costs[mo_asset] / buy_volumes[mo_asset]
            elif mo_side == "SELL":
                positions[mo_asset] = positions.get(mo_asset, 0) - mo_size
                # C-1: Do NOT adjust buy_volumes on sells.

    # Filter to non-trivial positions
    result: dict[str, tuple[float, float]] = {}
    for token_id, net_size in positions.items():
        if abs(net_size) < 0.01:
            continue
        # I-6: If we have a net-negative position (only SELLs in the 24h window,
        # buys were older) there will be no entry in avg_prices. Use 0.0 as
        # avg_price instead of 0.50 — this is conservative for PnL tracking
        # (assumes worst-case cost basis), making the max_loss check trigger
        # sooner rather than later.
        avg_price = avg_prices.get(token_id)
        if avg_price is None:
            avg_price = 0.0
            if net_size < 0:
                logger.warning(
                    "No buy cost data for %s (net_size=%.2f) — using avg_price=0.0 "
                    "(conservative). Buy trades may be older than the 24h seeding window.",
                    token_id[-10:], net_size,
                )
        result[token_id] = (net_size, avg_price)

    return result


def _fetch_onchain_balances(
    client,
    token_ids: list[str],
) -> dict[str, tuple[float, float]]:
    """Query actual on-chain token balances from Polymarket.

    Returns dict mapping token_id -> (balance, avg_price).
    avg_price defaults to 0.50 since we don't know the actual cost basis.
    """
    try:
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
    except ImportError:
        logger.warning("Cannot import BalanceAllowanceParams — skipping on-chain balance query")
        return {}

    positions: dict[str, tuple[float, float]] = {}
    for tid in token_ids:
        try:
            params = BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL,
                token_id=tid,
                signature_type=2,
            )
            result = client.get_balance_allowance(params)
            raw_balance = int(result.get("balance", "0"))
            # Polymarket uses 6 decimal places for token balances
            balance = raw_balance / 1_000_000.0
            if balance >= 0.01:
                # Use 0.50 as default avg_price (unknown cost basis)
                positions[tid] = (balance, 0.50)
                logger.info("On-chain balance for %s: %.2f shares", tid[-10:], balance)
            else:
                logger.info("On-chain balance for %s: zero", tid[-10:])
        except Exception as e:
            logger.warning("Failed to fetch on-chain balance for %s: %s", tid[-10:], e)
    return positions


def _seed_positions_from_trades(
    client,
    engine: MarketMakingEngine,
    token_ids: list[str],
    funder: str,
) -> None:
    """Query on-chain balances (preferred) or past trades to seed engine positions.

    Uses actual on-chain token balances for accuracy. Falls back to trade
    history if balance query fails.

    This prevents the engine from re-accumulating tokens it already holds.

    C-1: Resets positions before re-seeding to avoid double-counting on reconnect.
    C-2: Fetches data outside engine lock, then acquires lock for atomic
    reset+inject to avoid racing with on_fill.
    """
    # Phase 1a: Try on-chain balances first (most accurate)
    computed_positions = _fetch_onchain_balances(client, token_ids)

    # Phase 1b: Fall back to trade history if on-chain query returned nothing
    if not computed_positions:
        logger.info("On-chain balance query returned no results — falling back to trade history")
        computed_positions = _fetch_trade_positions(client, token_ids, funder)

    # C-2 fix: Guard against empty results to avoid wiping live-accumulated
    # inventory. If the API call fails or funder is empty, _fetch_trade_positions
    # returns {}. Calling seed_positions({}) would reset ALL positions to zero,
    # destroying live fills accumulated via on_fill callbacks.
    if not computed_positions:
        logger.warning(
            "Position seeding returned empty results — keeping existing positions "
            "to avoid wiping live-accumulated inventory."
        )
        return

    # Phase 2: Reset positions and inject synthetic fills atomically
    # (C-3: fills arriving between phase 1 and the lock acquisition may be lost)
    # I-NEW-2: Use public method instead of accessing engine._lock and
    # engine._inventory directly.
    engine.seed_positions(computed_positions)


# ---------------------------------------------------------------------------
# 5b. CTF split
# ---------------------------------------------------------------------------

def _run_ctf_split(
    creds: dict,
    condition_to_tokens: dict[str, list[str]],
    token_metadata: dict[str, dict],
    amount_usdc: float,
) -> None:
    """Split USDC into YES+NO token pairs for all conditions via the Relayer.

    Called when ``--split AMOUNT`` is passed on the CLI.  Requires builder
    API credentials in the environment (BUILDER_API_KEY, BUILDER_SECRET,
    BUILDER_PASSPHRASE).
    """
    from src.ctf import CTFClient

    builder_key = os.getenv("BUILDER_API_KEY", "")
    builder_secret = os.getenv("BUILDER_SECRET", "")
    builder_passphrase = os.getenv("BUILDER_PASSPHRASE", "")

    if not all([builder_key, builder_secret, builder_passphrase]):
        logger.error(
            "CTF split requires builder credentials. Set BUILDER_API_KEY, "
            "BUILDER_SECRET, and BUILDER_PASSPHRASE in .env"
        )
        sys.exit(1)

    conditions: list[dict] = []
    for cid, tids in condition_to_tokens.items():
        # Use neg_risk from the first token's metadata
        meta = token_metadata.get(tids[0], {})
        neg_risk = meta.get("neg_risk", False)
        conditions.append({
            "condition_id": cid,
            "amount": amount_usdc,
            "neg_risk": neg_risk,
        })

    if not conditions:
        logger.warning("No conditions to split — skipping CTF split")
        return

    logger.info(
        "CTF split: %.2f USDC × %d conditions = %.2f USDC total",
        amount_usdc, len(conditions), amount_usdc * len(conditions),
    )

    try:
        ctf = CTFClient(
            private_key=creds["private_key"],
            chain_id=creds["chain_id"],
            builder_api_key=builder_key,
            builder_secret=builder_secret,
            builder_passphrase=builder_passphrase,
        )
        response = ctf.split_multiple(conditions)
        logger.info("CTF split submitted (tx_id=%s) — waiting for confirmation...",
                     response.transaction_id)
        result = response.wait()
        if result:
            logger.info("CTF split confirmed: %s", result.get("state", "unknown"))
        else:
            logger.warning(
                "CTF split may not have confirmed in time — continuing with "
                "position seeding (on-chain balances will reflect actual state)"
            )
    except Exception:
        logger.exception(
            "CTF split failed — falling back to accumulation. "
            "Position seeding will use existing on-chain balances."
        )


# ---------------------------------------------------------------------------
# 6–8. Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    live = args.live
    creds = load_credentials(live)

    # Load MM config
    if args.config:
        config = load_config(args.config)
        logger.info("Loaded config from %s", args.config)
    else:
        config = MMConfig()
    if live:
        config = config.model_copy(update={"dry_run": False})
    logger.info("Mode: %s | spread=%d bps | size=%.1f",
                "LIVE" if not config.dry_run else "DRY-RUN",
                config.spread_bps, config.order_size)

    # ClobClient
    client = create_clob_client(creds, live)

    # Market discovery
    try:
        include_championship = not bool(args.markets)
        poly_contracts, mapped_markets, initial_refs = discover_markets(
            skip_championship=not include_championship,
        )
    except Exception:
        logger.exception("Market discovery failed — exiting")
        sys.exit(1)
    if not mapped_markets:
        logger.error(
            "No mapped markets found — exiting. "
            "Possible causes: all games are in-progress (live odds don't match "
            "pre-game Polymarket lines), or no external odds matched any Polymarket contracts."
        )
        sys.exit(1)

    # Filter to selected markets if --markets is provided
    whitelist: set[str] | None = None
    if args.markets:
        whitelist = load_markets_file(args.markets)
        total_before = len(mapped_markets)
        mapped_markets = [
            m for m in mapped_markets
            if any(c.token_id in whitelist for c in m.polymarket_contracts)
        ]
        initial_refs = [r for r in initial_refs if r.token_id in whitelist]
        logger.info("Market filter: kept %d / %d mapped markets (%d token_ids whitelisted)",
                    len(mapped_markets), total_before, len(whitelist))
        if not mapped_markets:
            logger.error(
                "No markets remaining after filter — check %s. "
                "This can happen if all matching games are in-progress and were skipped.",
                args.markets,
            )
            sys.exit(1)

    # C-3/C-2/C-4: Build per-token metadata from contract data for engine registration
    token_metadata: dict[str, dict] = {}
    for m in mapped_markets:
        for c in m.polymarket_contracts:
            token_metadata[c.token_id] = {
                "tick_size": c.minimum_tick_size,
                "neg_risk": c.neg_risk,
                "min_order_size": c.min_order_size,
            }

    # Group tokens by condition_id for pair-aware registration
    condition_to_tokens: dict[str, list[str]] = {}
    condition_ids: set[str] = set()
    for m in mapped_markets:
        for c in m.polymarket_contracts:
            condition_to_tokens.setdefault(c.condition_id, []).append(c.token_id)
            condition_ids.add(c.condition_id)

    # Deduplicate token lists per condition (preserve order)
    for cid in condition_to_tokens:
        condition_to_tokens[cid] = list(dict.fromkeys(condition_to_tokens[cid]))

    token_ids = list(dict.fromkeys(
        tid for tids in condition_to_tokens.values() for tid in tids
    ))
    logger.info("Tracking %d token_ids across %d conditions", len(token_ids), len(condition_to_tokens))

    # Initialize components
    orderbook_mgr = OrderbookManager()
    engine = MarketMakingEngine(config, client, orderbook_manager=orderbook_mgr)
    polymarket_feed = PolymarketFeed()
    reference_feed = ReferenceFeed(
        fetch_fn=build_reference_fetch_fn(
            poly_contracts,
            token_whitelist=whitelist,
            include_championship=include_championship,
        ),
        poll_interval=60.0,  # 60s — responsive enough for live games
    )
    user_feed = None
    if live:
        user_feed = UserFeed(
            api_key=creds["api_key"],
            secret=creds["api_secret"],
            passphrase=creds["api_passphrase"],
        )

    # Register tokens with engine — binary pairs get pair-aware treatment
    # C-3/C-2/C-4: Pass tick_size, neg_risk, and min_order_size from market data
    registered: set[str] = set()
    for cid, tids in condition_to_tokens.items():
        # Use metadata from first token (all tokens in a condition share the same market properties)
        meta = token_metadata.get(tids[0], {})
        tick_size = meta.get("tick_size", 0.01)
        neg_risk = meta.get("neg_risk", False)
        min_order_size = meta.get("min_order_size", 5.0)
        if len(tids) == 2:
            engine.add_token_pair(
                tids[0], tids[1],
                tick_size=tick_size, neg_risk=neg_risk, min_order_size=min_order_size,
            )
            registered.update(tids)
        else:
            for tid in tids:
                if tid not in registered:
                    t_meta = token_metadata.get(tid, {})
                    engine.add_token(
                        tid,
                        tick_size=t_meta.get("tick_size", 0.01),
                        neg_risk=t_meta.get("neg_risk", False),
                        min_order_size=t_meta.get("min_order_size", 5.0),
                    )
                    registered.add(tid)

    # CTF split: convert USDC into YES+NO token pairs before seeding positions
    if args.split is not None:
        if not live:
            logger.error("--split requires --live (splits are real on-chain transactions)")
            sys.exit(1)
        if args.split <= 0:
            logger.error("--split amount must be positive (got %.2f)", args.split)
            sys.exit(1)
        _run_ctf_split(creds, condition_to_tokens, token_metadata, args.split)

    # Seed existing positions from trade history (so engine knows about pre-existing holdings)
    if live and client is not None:
        # C-4: Warn about taker fill seeding limitation when cross-spread accumulation
        # is enabled. TradeParams only supports maker_address filter — there is no
        # taker_address filter in the CLOB API. Taker fills (from accumulation with
        # post_only=False) won't appear in seeded positions on restart.
        if config.accumulation_cross_spread:
            logger.warning(
                "accumulation_cross_spread is enabled but the CLOB API's TradeParams "
                "does not support taker_address filtering. Taker fills from cross-spread "
                "accumulation will NOT be included in position seeding on restart. "
                "Positions may be under-reported until fills arrive via the live feed."
            )
        _seed_positions_from_trades(client, engine, token_ids, creds["funder"])

    # Wire callbacks
    reference_feed.on_update(
        lambda u: engine.fair_value_engine.update(u.token_id, u.fair_probability, u.timestamp)
    )
    # Refresh FV timestamps on every successful poll so unchanged odds
    # don't trigger the stale-data risk check and cancel orders.
    reference_feed.on_poll(engine.fair_value_engine.touch_all)
    polymarket_feed.on_book(orderbook_mgr.handle_snapshot)
    polymarket_feed.on_price_change(orderbook_mgr.handle_update)

    # I-2 fix: Store cancel thread reference for coordination on rapid reconnects.
    _reconnect_cancel_thread: threading.Thread | None = None
    # R16-I2: Generation counter prevents stale reconnect-cancel threads from
    # wiping quotes placed by a newer reconnect cycle. Each reconnect
    # increments the counter; the finally block only acts if its generation
    # matches the current value.
    _reconnect_generation: int = 0

    def handle_orderbook_reconnect():
        nonlocal _reconnect_cancel_thread, _reconnect_generation
        _reconnect_generation += 1
        my_generation = _reconnect_generation
        orderbook_mgr.clear_all()
        # R23-I2: clear_divergence_emas() call removed — clear_quotes() now
        # atomically clears both quotes and divergence EMAs (see R23-I2 fix).
        # R21-I5: Close quoting gate FIRST (before clearing quotes) to prevent
        # a micro-window where the tick loop sees empty _current_quotes with
        # the gate still open and places orders against the stale/empty orderbook.
        engine._quoting_gate.clear()

        # C-3 fix: Clear quotes so the tick loop starts fresh after gate reopens.
        engine.clear_quotes()

        # I-2 fix: If a previous cancel thread is still running (rapid reconnects),
        # wait for it to finish before spawning a new one.
        # RN-4 fix: Scale timeout with token count. Each cancel_all_for_token is
        # an HTTP call (~0.5-1s). With many tokens, 2s was too short, causing
        # overlapping cancel threads that waste HTTP calls.
        if _reconnect_cancel_thread is not None and _reconnect_cancel_thread.is_alive():
            join_timeout = max(5.0, len(token_ids) * 0.5)
            _reconnect_cancel_thread.join(timeout=join_timeout)
            if _reconnect_cancel_thread.is_alive():
                logger.warning("Previous reconnect-cancel thread still running after %.1fs", join_timeout)

        # C-1 fix: Run HTTP cancel calls in a daemon thread to avoid
        # blocking the WebSocket thread (which runs this callback via
        # _resubscribe → _on_open). Blocking the WS thread prevents
        # receiving new snapshots and heartbeats during cancellation.
        def _cancel_on_reconnect(gen: int = my_generation):
            try:
                for tid in token_ids:
                    try:
                        engine.order_manager.cancel_all_for_token(tid)
                    except Exception:
                        logger.exception("Failed to cancel orders for %s on reconnect", tid)
            finally:
                # R16-I2: Only clear quotes and re-open gate if this is still
                # the current reconnect generation. A stale thread's finally
                # block must NOT wipe quotes placed by a newer cycle.
                if gen != _reconnect_generation:
                    logger.info(
                        "Stale reconnect-cancel (gen %d, current %d) — skipping gate/quote cleanup",
                        gen, _reconnect_generation,
                    )
                    return
                # Re-clear quotes after cancel so tick loop starts truly fresh.
                try:
                    engine.clear_quotes()
                except Exception:
                    logger.exception("Error clearing quotes in reconnect-cancel finally")
                # R17-I1: Re-check generation AFTER clear_quotes and right
                # before gate.set().  A new reconnect can increment
                # _reconnect_generation and call gate.clear() between the
                # first gen check above and here; without this second check
                # we'd undo the new cycle's gate.clear().
                if gen != _reconnect_generation:
                    logger.info(
                        "Stale reconnect-cancel (gen %d, current %d) after clear_quotes — skipping gate re-open",
                        gen, _reconnect_generation,
                    )
                    return
                engine._quoting_gate.set()

        t = threading.Thread(target=_cancel_on_reconnect, daemon=True, name="reconnect-cancel")
        _reconnect_cancel_thread = t
        t.start()

    polymarket_feed.on_reconnect(handle_orderbook_reconnect)

    # I-8: Update engine tick_size when the exchange sends a tick_size_change event.
    # I-4: min_order_size is NOT updated here because the tick_size_change event
    # does not include it. If Polymarket changes min_order_size alongside tick_size,
    # a restart would be needed to pick it up (this is very rare in practice).
    def handle_tick_size_change(asset_id: str, new_tick_size: float):
        # I-NEW-1: Use public method instead of accessing engine._lock,
        # engine._tokens, and engine._pairs directly.
        engine.update_tick_size(asset_id, new_tick_size)
    polymarket_feed.on_tick_size_change(handle_tick_size_change)

    if user_feed is not None:
        # I-3: Guard against concurrent position re-seeding on rapid reconnects.
        # If a re-seed is already in progress, skip the new one.
        _reseed_lock = threading.Lock()

        def handle_user_reconnect():
            # I-7: Run re-seeding in a daemon thread so the WebSocket thread
            # returns quickly and can proceed with subscription immediately.
            def _reseed():
                # I-3: Non-blocking trylock — if a re-seed is already running,
                # skip this one to avoid concurrent re-seeds racing.
                acquired = _reseed_lock.acquire(blocking=False)
                if not acquired:
                    logger.warning(
                        "Position re-seed already in progress — skipping duplicate re-seed from rapid reconnect"
                    )
                    return
                try:
                    logger.critical(
                        "UserFeed reconnected — fills during disconnect may be lost. "
                        "Triggering position reconciliation from trade history."
                    )
                    if live and client is not None:
                        try:
                            _seed_positions_from_trades(client, engine, token_ids, creds["funder"])
                            logger.info("Position reconciliation complete after UserFeed reconnect")
                        except Exception:
                            logger.exception("Failed to reconcile positions after UserFeed reconnect")
                finally:
                    _reseed_lock.release()

            t = threading.Thread(target=_reseed, daemon=True, name="reseed-positions")
            t.start()
        user_feed.on_reconnect(handle_user_reconnect)

        # C-M1: Idempotency guard — prevent duplicate CONFIRMED events from
        # double-counting fills (e.g., after WebSocket reconnect replays).
        # R2-2: Use OrderedDict (ordered set) instead of set so that eviction
        # removes the oldest entries, not arbitrary ones.
        # I-3: Protected by _processed_trade_ids_lock since it is accessed from
        # the WebSocket thread (handle_user_fill) without other synchronization.
        from collections import OrderedDict
        _processed_trade_ids: OrderedDict[str, None] = OrderedDict()
        _processed_trade_ids_lock = threading.Lock()
        _MAX_PROCESSED_TRADE_IDS = 1000

        def handle_user_fill(event: UserTradeEvent):
            if event.status != TradeStatus.CONFIRMED:
                return
            # RN-2 fix: Guard against empty/null event.id. An empty string
            # would poison the dedup cache — every subsequent event with a
            # missing ID would match "" and be silently dropped.
            if not event.id:
                logger.warning("Trade event with empty ID — processing without dedup guard")
                fills = user_trade_to_fills(event)
                for fill in fills:
                    engine.on_fill(fill)
                return
            # I-3: All accesses to _processed_trade_ids under lock for thread safety.
            with _processed_trade_ids_lock:
                # C-M1: Skip duplicate trade events
                if event.id in _processed_trade_ids:
                    logger.debug("Skipping duplicate trade event: %s", event.id[:16])
                    return
                _processed_trade_ids[event.id] = None
                # Cleanup: remove oldest entries when exceeding limit
                while len(_processed_trade_ids) > _MAX_PROCESSED_TRADE_IDS:
                    _processed_trade_ids.popitem(last=False)  # Remove oldest

            # I-6: Pass all fills to engine.on_fill unconditionally.
            # The engine checks tracked order membership inside its lock,
            # avoiding the race where orders are cleaned up between the
            # tracked-ID check and the fill processing.
            # R19-I3 fix: Wrap fill processing in try/except. If
            # user_trade_to_fills throws an unexpected exception, remove
            # the trade from the dedup cache so a replay can retry it.
            try:
                fills = user_trade_to_fills(event)
            except Exception:
                logger.exception("Failed to parse fills for trade %s — removing from dedup cache", event.id[:16])
                with _processed_trade_ids_lock:
                    _processed_trade_ids.pop(event.id, None)
                return
            for fill in fills:
                engine.on_fill(fill)
            if not fills:
                logger.debug(
                    "Trade %s: no fill candidates (taker=%s, makers=%d)",
                    event.id[:16], event.taker_order_id[:16] if event.taker_order_id else "none",
                    len(event.maker_orders),
                )
        user_feed.on_trade_update(handle_user_fill)

        def handle_order_cancel(event: UserOrderEvent) -> None:
            if event.type != OrderEventType.CANCELLATION:
                return
            engine.on_order_cancel(event.id)

        user_feed.on_order_update(handle_order_cancel)

    # Seed initial fair values
    for ref in initial_refs:
        engine.fair_value_engine.update(ref.token_id, ref.fair_probability, time.time())

    # Connect feeds — check return values to avoid running with no live data.
    logger.info("Connecting PolymarketFeed...")
    if not polymarket_feed.connect():
        logger.error("Failed to connect PolymarketFeed — aborting")
        raise RuntimeError("PolymarketFeed connection failed")
    polymarket_feed.subscribe(token_ids)

    if user_feed is not None:
        logger.info("Connecting UserFeed...")
        if not user_feed.connect():
            logger.error("Failed to connect UserFeed — aborting")
            raise RuntimeError("UserFeed connection failed")
        user_feed.subscribe(list(condition_ids))

    # Start background polling
    reference_feed.start()

    # C-1: Start CLOB API heartbeat to prevent order auto-cancellation.
    # Polymarket cancels all orders if no heartbeat received within 10s.
    heartbeat: ClobHeartbeat | None = None
    if live and client is not None:
        heartbeat = ClobHeartbeat(client, engine=engine)
        heartbeat.start()

    # Register SIGTERM handler for graceful shutdown (e.g. Docker/systemd stop).
    # SIGINT is handled via KeyboardInterrupt to avoid racing with the signal handler.
    def _signal_handler(signum, _frame):
        logger.info("Received signal %d — stopping engine", signum)
        engine.stop()
    signal.signal(signal.SIGTERM, _signal_handler)

    # Run engine (blocks main thread)
    logger.info("Starting MarketMakingEngine...")
    try:
        engine.run()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received — shutting down")
    finally:
        shutdown(engine, polymarket_feed, user_feed, reference_feed, heartbeat)


# ---------------------------------------------------------------------------
# 8. Graceful shutdown
# ---------------------------------------------------------------------------

_SHUTDOWN_TIMEOUT_SECONDS = 30


def shutdown(engine, polymarket_feed, user_feed, reference_feed, heartbeat=None):
    logger.info("Shutting down...")

    # I-7: Set a safety timer — if shutdown takes longer than 30 seconds,
    # force-stop the heartbeat so Polymarket's server-side auto-cancel kicks
    # in (~10s after last heartbeat). This prevents the scenario where a hung
    # engine.stop() keeps the heartbeat alive indefinitely, leaving stale
    # orders on the book.
    def _force_stop_heartbeat():
        logger.error(
            "Shutdown exceeded %ds timeout — force-stopping heartbeat to "
            "trigger Polymarket auto-cancel",
            _SHUTDOWN_TIMEOUT_SECONDS,
        )
        if heartbeat is not None:
            try:
                heartbeat.stop()
            except Exception:
                logger.exception("Error force-stopping heartbeat")

    shutdown_timer = threading.Timer(_SHUTDOWN_TIMEOUT_SECONDS, _force_stop_heartbeat)
    shutdown_timer.daemon = True
    shutdown_timer.start()

    # I-4: Stop heartbeat AFTER engine stop, not before. The heartbeat must
    # keep running during engine.stop() so that Polymarket does not auto-cancel
    # orders server-side while the engine is also trying to cancel them.
    try:
        for name, fn in [
            ("engine", engine.stop),
            ("heartbeat", lambda: heartbeat.stop() if heartbeat is not None else None),
            ("polymarket_feed", polymarket_feed.close),
            ("user_feed", lambda: user_feed.close() if user_feed is not None else None),
            ("reference_feed", reference_feed.stop),
        ]:
            try:
                fn()
            except Exception:
                # R19-I1 fix: Use Exception, not BaseException.  The old
                # BaseException handler swallowed KeyboardInterrupt, making it
                # impossible to force-quit when a close() call hangs.
                logger.exception("Error shutting down %s", name)
    finally:
        shutdown_timer.cancel()
    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
