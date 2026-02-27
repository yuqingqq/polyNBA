"""Compare sportsbook odds vs Polymarket orderbook for NBA games.

Usage:
    # With real odds (requires ODDS_API_KEY):
    ODDS_API_KEY=your_key python3 -m src.reference.compare_game

    # Without API key (shows Polymarket side only):
    python3 -m src.reference.compare_game
"""

import json
import logging
import os
import re
from datetime import datetime, timezone

import requests

from .vig_removal import american_to_probability, proportional_vig_removal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
ODDS_API = "https://api.the-odds-api.com/v4/sports"

# Slug pattern for NBA game events: nba-XXX-XXX-YYYY-MM-DD
_GAME_SLUG_RE = re.compile(r"^nba-[a-z]{3}-[a-z]{3}-\d{4}-\d{2}-\d{2}$")

# Team name mapping: full name -> short names used in Polymarket questions
TEAM_SHORT_NAMES: dict[str, list[str]] = {
    "Houston Rockets": ["rockets", "houston"],
    "Charlotte Hornets": ["hornets", "charlotte"],
    "Los Angeles Lakers": ["lakers"],
    "Boston Celtics": ["celtics", "boston"],
    "Golden State Warriors": ["warriors", "golden state"],
    "Oklahoma City Thunder": ["thunder", "okc"],
    "New York Knicks": ["knicks"],
    "Milwaukee Bucks": ["bucks"],
    "Denver Nuggets": ["nuggets"],
    "Phoenix Suns": ["suns"],
    "Cleveland Cavaliers": ["cavaliers", "cavs"],
    "Dallas Mavericks": ["mavericks", "mavs"],
    "Minnesota Timberwolves": ["timberwolves", "wolves"],
    "Memphis Grizzlies": ["grizzlies"],
    "Sacramento Kings": ["kings"],
    "Miami Heat": ["heat"],
    "Philadelphia 76ers": ["76ers", "sixers"],
    "Indiana Pacers": ["pacers"],
    "Atlanta Hawks": ["hawks"],
    "New Orleans Pelicans": ["pelicans"],
    "Toronto Raptors": ["raptors"],
    "Brooklyn Nets": ["nets"],
    "Chicago Bulls": ["bulls"],
    "Detroit Pistons": ["pistons"],
    "Orlando Magic": ["magic"],
    "Portland Trail Blazers": ["blazers", "trail blazers"],
    "San Antonio Spurs": ["spurs"],
    "Utah Jazz": ["jazz"],
    "Washington Wizards": ["wizards"],
    "Los Angeles Clippers": ["clippers"],
}


def _parse_json_field(value: str | list) -> list:
    """Parse a JSON string field or return as-is if already a list."""
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


def fetch_polymarket_nba_games() -> list[dict]:
    """Fetch active NBA game events from Polymarket Gamma API.

    Uses the events endpoint with tag_slug=nba, then filters for
    head-to-head game events (not futures, All-Star, draft, etc.).
    Returns a list of dicts with keys: title, slug, end_date,
    moneyline (the h2h market dict), and extra_markets.
    """
    url = f"{GAMMA_API}/events"
    params = {
        "tag_slug": "nba",
        "active": "true",
        "closed": "false",
        "limit": 100,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    events = resp.json()

    games = []
    for event in events:
        slug = event.get("slug", "")
        title = event.get("title", "")

        # Identify game events by slug pattern or title with "vs."
        is_game = bool(_GAME_SLUG_RE.match(slug))
        if not is_game and " vs. " in title:
            # Exclude All-Star, draft, futures by keyword
            title_lower = title.lower()
            skip = ["all-star", "draft", "champion", "mvp", "rookie",
                    "rising", "celebrity", "division", "playoff", "seed",
                    "record", "leader", "coach", "improved", "clutch",
                    "totals", "attend", "dunk", "3-point", "half-court"]
            if not any(w in title_lower for w in skip):
                is_game = True

        if not is_game:
            continue

        markets = event.get("markets", [])
        moneyline = None
        extra = []

        for m in markets:
            q = m.get("question", "")
            outcomes = _parse_json_field(m.get("outcomes", "[]"))
            # Moneyline market: question is "Team vs. Team" with team name outcomes
            # (not spread, O/U, or player props)
            if " vs. " in q and "Spread" not in q and "O/U" not in q:
                # Check outcomes are team names (not Yes/No)
                if len(outcomes) == 2 and "Yes" not in outcomes:
                    moneyline = m
                    continue
            extra.append(m)

        if moneyline:
            end_date = moneyline.get("endDate", "")[:10]
            games.append({
                "title": title,
                "slug": slug,
                "end_date": end_date,
                "moneyline": moneyline,
                "extra_markets": extra,
            })

    # Sort by end date
    games.sort(key=lambda g: g["end_date"])
    return games


def fetch_orderbook(token_id: str) -> dict:
    """Fetch CLOB orderbook for a token."""
    url = f"{CLOB_API}/book"
    resp = requests.get(url, params={"token_id": token_id}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_odds_api_games(api_key: str) -> list[dict]:
    """Fetch NBA game odds from The Odds API."""
    url = f"{ODDS_API}/basketball_nba/odds/"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
    }
    resp = requests.get(url, params=params, timeout=15)

    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    logger.info("Odds API quota: %s remaining, %s used", remaining, used)

    resp.raise_for_status()
    return resp.json()


def parse_orderbook_bbo(book: dict) -> dict:
    """Extract best bid/ask from CLOB orderbook."""
    bids = book.get("bids", [])
    asks = book.get("asks", [])

    best_bid = max((float(b["price"]) for b in bids), default=0.0) if bids else 0.0
    best_ask = min((float(a["price"]) for a in asks), default=1.0) if asks else 1.0
    mid = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask < 1 else None

    bid_depth = sum(float(b["size"]) for b in sorted(
        bids, key=lambda x: -float(x["price"]))[:5])
    ask_depth = sum(float(a["size"]) for a in sorted(
        asks, key=lambda x: float(x["price"]))[:5])

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread": best_ask - best_bid,
        "bid_depth_5": round(bid_depth, 2),
        "ask_depth_5": round(ask_depth, 2),
    }


def _match_team_name(odds_api_name: str, poly_name: str) -> bool:
    """Check if an Odds API team name matches a Polymarket outcome name."""
    odds_lower = odds_api_name.lower()
    poly_lower = poly_name.lower()

    # Direct match
    if odds_lower == poly_lower or poly_lower in odds_lower or odds_lower in poly_lower:
        return True

    # Check short names
    for full, shorts in TEAM_SHORT_NAMES.items():
        full_lower = full.lower()
        if odds_lower in full_lower or full_lower in odds_lower:
            if poly_lower in shorts or any(s == poly_lower for s in shorts):
                return True
        if poly_lower in shorts or poly_lower in full_lower:
            if odds_lower in full_lower or any(s in odds_lower for s in shorts):
                return True

    return False


def _find_odds_game(
    poly_title: str, odds_games: list[dict]
) -> dict | None:
    """Find the matching Odds API game for a Polymarket game title."""
    # Extract team names from title like "Rockets vs. Hornets"
    parts = poly_title.split(" vs. ")
    if len(parts) != 2:
        return None

    team_a, team_b = parts[0].strip(), parts[1].strip()

    for game in odds_games:
        home = game.get("home_team", "")
        away = game.get("away_team", "")

        a_match = _match_team_name(home, team_a) or _match_team_name(away, team_a)
        b_match = _match_team_name(home, team_b) or _match_team_name(away, team_b)

        if a_match and b_match:
            return game

    return None


def run_comparison():
    """Run the full comparison."""
    print("=" * 70)
    print("  NBA ODDS COMPARISON: Sportsbooks vs Polymarket")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    # --- Polymarket side (always free, no API key needed) ---
    print("\n[1] Fetching Polymarket NBA game markets...")
    poly_games = fetch_polymarket_nba_games()
    print(f"    Found {len(poly_games)} game markets on Polymarket")

    if not poly_games:
        print("    No active NBA game markets found.")
        print("    (This is normal during All-Star Weekend or off-season)")
        return

    # --- Odds API side (needs key) ---
    api_key = os.environ.get("ODDS_API_KEY")
    odds_games: list[dict] = []

    if api_key:
        print("\n[2] Fetching sportsbook odds from The Odds API...")
        try:
            odds_games = fetch_odds_api_games(api_key)
            print(f"    Found {len(odds_games)} upcoming NBA games with odds")
        except Exception as e:
            print(f"    ERROR fetching odds: {e}")
    else:
        print("\n[2] ODDS_API_KEY not set -- skipping sportsbook odds")
        print("    Set ODDS_API_KEY env var to compare with real sportsbook lines")
        print("    Get a free key at: https://the-odds-api.com/")

    # --- Display each game ---
    print("\n" + "=" * 70)
    if odds_games:
        print("  COMPARISON: Sportsbook Fair Odds vs Polymarket Orderbook")
    else:
        print("  POLYMARKET NBA GAME ORDERBOOKS")
    print("=" * 70)

    for game in poly_games:
        title = game["title"]
        end_date = game["end_date"]
        ml = game["moneyline"]

        outcomes = _parse_json_field(ml.get("outcomes", "[]"))
        prices = _parse_json_field(ml.get("outcomePrices", "[]"))
        token_ids = _parse_json_field(ml.get("clobTokenIds", "[]"))

        if len(outcomes) != 2:
            continue

        print(f"\n  {title}  ({end_date})")
        print(f"  {'-' * 60}")

        # Listed prices from Gamma API
        for i, name in enumerate(outcomes):
            p = float(prices[i]) if i < len(prices) else 0
            print(f"    {name:<20} Listed: {p:.1%}")

        # Fetch live orderbooks
        print()
        bbos = {}
        for i, tid in enumerate(token_ids):
            name = outcomes[i] if i < len(outcomes) else f"Outcome {i}"
            try:
                book = fetch_orderbook(tid)
                bbo = parse_orderbook_bbo(book)
                bbos[name] = bbo
                mid_str = f"{bbo['mid']:.4f}" if bbo["mid"] else "N/A"
                print(
                    f"    {name:<20} "
                    f"Bid: {bbo['best_bid']:.4f}  "
                    f"Ask: {bbo['best_ask']:.4f}  "
                    f"Mid: {mid_str}  "
                    f"Spread: {bbo['spread']:.4f}  "
                    f"Depth(5): ${bbo['bid_depth_5']:,.0f} / ${bbo['ask_depth_5']:,.0f}"
                )
            except Exception as e:
                print(f"    {name:<20} Orderbook error: {e}")

        # Find matching sportsbook game
        odds_game = _find_odds_game(title, odds_games) if odds_games else None

        if odds_game:
            home = odds_game.get("home_team", "?")
            away = odds_game.get("away_team", "?")
            # Pick reference bookmaker: Pinnacle preferred
            ref_bm = None
            for bm in odds_game.get("bookmakers", []):
                if bm.get("key") == "pinnacle":
                    ref_bm = bm
                    break
            if ref_bm is None and odds_game.get("bookmakers"):
                ref_bm = odds_game["bookmakers"][0]

            if ref_bm:
                for mkt in ref_bm.get("markets", []):
                    if mkt.get("key") != "h2h":
                        continue
                    ext_outcomes = mkt.get("outcomes", [])
                    if len(ext_outcomes) != 2:
                        continue

                    raw_probs = [
                        american_to_probability(round(o["price"])) for o in ext_outcomes
                    ]
                    fair_probs = proportional_vig_removal(raw_probs)
                    overround = sum(raw_probs) - 1.0

                    ref_name = ref_bm.get("title", ref_bm.get("key", "?"))
                    print(f"\n    Reference: {ref_name} (vig-removed)")
                    print(f"    {'Outcome':<20} {'American':>10} {'Raw':>8} {'Fair':>8} {'Vig':>8}")
                    print(f"    {'-'*20} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

                    for j, o_ext in enumerate(ext_outcomes):
                        vig = raw_probs[j] - fair_probs[j]
                        print(
                            f"    {o_ext['name']:<20} {round(o_ext['price']):>+10d} "
                            f"{raw_probs[j]:>8.4f} {fair_probs[j]:>8.4f} {vig:>+8.4f}"
                        )
                    print(f"    Overround: {overround:.4f} ({overround * 100:.2f}%)")

                    # Compare fair prob vs Polymarket mid
                    print(f"\n    {'Outcome':<20} {'Fair':>8} {'Poly Mid':>10} {'Diff':>10} {'bps':>8}")
                    print(f"    {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")

                    for j, o_ext in enumerate(ext_outcomes):
                        # Find matching Polymarket outcome
                        for poly_name, bbo in bbos.items():
                            if _match_team_name(o_ext["name"], poly_name):
                                if bbo["mid"] is not None:
                                    diff = bbo["mid"] - fair_probs[j]
                                    bps = diff * 10000
                                    signal = ""
                                    if abs(bps) > 200:
                                        signal = " <-- DIVERGENCE"
                                    print(
                                        f"    {poly_name:<20} "
                                        f"{fair_probs[j]:>8.4f} "
                                        f"{bbo['mid']:>10.4f} "
                                        f"{diff:>+10.4f} "
                                        f"{bps:>+8.0f}{signal}"
                                    )
                                break

            # Show key bookmaker lines
            print(f"\n    All sportsbook lines ({away} @ {home}):")
            print(f"    {'Bookmaker':<25} {away:>12} {home:>12}")
            print(f"    {'-'*25} {'-'*12} {'-'*12}")

            for bm in odds_game.get("bookmakers", [])[:8]:
                bm_name = bm.get("title", bm.get("key", "?"))
                for mkt in bm.get("markets", []):
                    if mkt.get("key") == "h2h" and len(mkt.get("outcomes", [])) == 2:
                        o1, o2 = mkt["outcomes"]
                        # Order: away first, home second
                        if o1["name"] == home:
                            o1, o2 = o2, o1
                        print(
                            f"    {bm_name:<25} {round(o1['price']):>+12d} {round(o2['price']):>+12d}"
                        )

        # Show extra market count
        n_extra = len(game.get("extra_markets", []))
        if n_extra:
            print(f"\n    + {n_extra} other markets (spreads, totals, player props)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_comparison()
