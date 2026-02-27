"""One-shot test: split $1 USDC into YES+NO, then merge back.

Usage:
    python3 scripts/test_split_merge.py
"""

import logging
import os
import sys
import time

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ctf import CTFClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Test parameters
CONDITION_ID = "0x22e7b5e35423e76842dd3a5e1a21d13793811080d5e7b2896d0c001bd5e97d54"
NEG_RISK = True
AMOUNT_USDC = 1.0  # $1 test


def main() -> None:
    load_dotenv()

    private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    chain_id = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))

    # Try CLOB creds as builder creds (same HMAC format)
    builder_key = os.getenv("BUILDER_API_KEY") or os.getenv("POLYMARKET_API_KEY", "")
    builder_secret = os.getenv("BUILDER_SECRET") or os.getenv("POLYMARKET_API_SECRET", "")
    builder_passphrase = os.getenv("BUILDER_PASSPHRASE") or os.getenv("POLYMARKET_API_PASSPHRASE", "")

    if not private_key:
        logger.error("POLYMARKET_PRIVATE_KEY not set")
        sys.exit(1)

    cred_source = "BUILDER_*" if os.getenv("BUILDER_API_KEY") else "POLYMARKET_API_* (fallback)"
    logger.info("Using credentials from: %s", cred_source)

    ctf = CTFClient(
        private_key=private_key,
        chain_id=chain_id,
        builder_api_key=builder_key,
        builder_secret=builder_secret,
        builder_passphrase=builder_passphrase,
    )

    # Step 1: Split $1
    logger.info("=== STEP 1: Split $%.2f USDC (condition=%s, neg_risk=%s) ===",
                AMOUNT_USDC, CONDITION_ID[:18], NEG_RISK)
    try:
        response = ctf.approve_and_split(CONDITION_ID, AMOUNT_USDC, neg_risk=NEG_RISK)
        logger.info("Split submitted: tx_id=%s", response.transaction_id)
        result = response.wait()
        if result:
            logger.info("Split confirmed! state=%s", result.get("state"))
        else:
            logger.error("Split did not confirm in time")
            sys.exit(1)
    except Exception as e:
        logger.error("Split failed: %s", e)
        sys.exit(1)

    # Brief pause
    logger.info("Waiting 3s before merge...")
    time.sleep(3)

    # Step 2: Merge $1 back
    logger.info("=== STEP 2: Merge $%.2f shares back to USDC ===", AMOUNT_USDC)
    try:
        response = ctf.merge(CONDITION_ID, AMOUNT_USDC, neg_risk=NEG_RISK)
        logger.info("Merge submitted: tx_id=%s", response.transaction_id)
        result = response.wait()
        if result:
            logger.info("Merge confirmed! state=%s", result.get("state"))
        else:
            logger.error("Merge did not confirm in time")
            sys.exit(1)
    except Exception as e:
        logger.error("Merge failed: %s", e)
        sys.exit(1)

    logger.info("=== TEST COMPLETE: Split + merge of $%.2f succeeded ===", AMOUNT_USDC)


if __name__ == "__main__":
    main()
