"""CTF split/merge operations via the Polymarket Relayer.

Splits USDC into YES+NO token pairs (or merges them back) using the CTF
(Conditional Tokens Framework) contract on Polygon.  Transactions are
submitted through the Polymarket Relayer for gasless execution via the
user's Gnosis Safe proxy wallet.

Usage:
    from src.ctf import CTFClient

    ctf = CTFClient(
        private_key="0x...",
        chain_id=137,
        builder_api_key="...",
        builder_secret="...",
        builder_passphrase="...",
    )
    response = ctf.approve_and_split(condition_id="0xabc...", amount_usdc=50)
    result = response.wait()
"""

from __future__ import annotations

import logging
from typing import Any

from eth_abi import encode as abi_encode
from eth_utils import keccak

from py_builder_relayer_client.client import RelayClient
from py_builder_relayer_client.models import OperationType, SafeTransaction
from py_builder_signing_sdk.config import BuilderConfig
from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Contract addresses (Polygon mainnet)
# ---------------------------------------------------------------------------

CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
NEG_RISK_ADAPTER_ADDRESS = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

RELAYER_URL = "https://relayer-v2.polymarket.com"

# USDC.e uses 6 decimals
USDC_DECIMALS = 6

# Approve max to avoid repeated approvals
MAX_UINT256 = (1 << 256) - 1

# Collateral token address used in standard CTF splitPosition calls
COLLATERAL_TOKEN = USDC_ADDRESS

# Parent collection ID for top-level positions (no parent)
PARENT_COLLECTION_ID = b"\x00" * 32

# Partition for binary markets: [1, 2] = outcome slots 0 and 1
BINARY_PARTITION = [1, 2]


# ---------------------------------------------------------------------------
# ABI encoding helpers
# ---------------------------------------------------------------------------

def _function_selector(signature: str) -> bytes:
    """Compute the 4-byte function selector from a Solidity signature."""
    return keccak(text=signature)[:4]


# Pre-computed selectors
_SEL_APPROVE = _function_selector("approve(address,uint256)")

_SEL_SPLIT_STANDARD = _function_selector(
    "splitPosition(address,bytes32,bytes32,uint256[],uint256)"
)
_SEL_SPLIT_NEG_RISK = _function_selector(
    "splitPosition(bytes32,uint256)"
)

_SEL_MERGE_STANDARD = _function_selector(
    "mergePositions(address,bytes32,bytes32,uint256[],uint256)"
)
_SEL_MERGE_NEG_RISK = _function_selector(
    "mergePositions(bytes32,uint256)"
)


def _encode_condition_id(condition_id: str) -> bytes:
    """Convert a hex condition_id string to bytes32."""
    raw = bytes.fromhex(condition_id.replace("0x", ""))
    return raw.rjust(32, b"\x00")


def _usdc_to_raw(amount_usdc: float) -> int:
    """Convert a USDC dollar amount to raw units (6 decimals)."""
    return round(amount_usdc * 10**USDC_DECIMALS)


def encode_approve(spender: str, amount: int) -> str:
    """Encode an ERC-20 approve(address, uint256) call."""
    params = abi_encode(
        ["address", "uint256"],
        [spender, amount],
    )
    return "0x" + (_SEL_APPROVE + params).hex()


def encode_split_standard(
    condition_id: str,
    amount_raw: int,
) -> str:
    """Encode CTF splitPosition for standard (non-neg-risk) markets."""
    cid_bytes = _encode_condition_id(condition_id)
    params = abi_encode(
        ["address", "bytes32", "bytes32", "uint256[]", "uint256"],
        [
            COLLATERAL_TOKEN,
            PARENT_COLLECTION_ID,
            cid_bytes,
            BINARY_PARTITION,
            amount_raw,
        ],
    )
    return "0x" + (_SEL_SPLIT_STANDARD + params).hex()


def encode_split_neg_risk(
    condition_id: str,
    amount_raw: int,
) -> str:
    """Encode NegRiskAdapter splitPosition for neg-risk markets."""
    cid_bytes = _encode_condition_id(condition_id)
    params = abi_encode(
        ["bytes32", "uint256"],
        [cid_bytes, amount_raw],
    )
    return "0x" + (_SEL_SPLIT_NEG_RISK + params).hex()


def encode_merge_standard(
    condition_id: str,
    amount_raw: int,
) -> str:
    """Encode CTF mergePositions for standard (non-neg-risk) markets."""
    cid_bytes = _encode_condition_id(condition_id)
    params = abi_encode(
        ["address", "bytes32", "bytes32", "uint256[]", "uint256"],
        [
            COLLATERAL_TOKEN,
            PARENT_COLLECTION_ID,
            cid_bytes,
            BINARY_PARTITION,
            amount_raw,
        ],
    )
    return "0x" + (_SEL_MERGE_STANDARD + params).hex()


def encode_merge_neg_risk(
    condition_id: str,
    amount_raw: int,
) -> str:
    """Encode NegRiskAdapter mergePositions for neg-risk markets."""
    cid_bytes = _encode_condition_id(condition_id)
    params = abi_encode(
        ["bytes32", "uint256"],
        [cid_bytes, amount_raw],
    )
    return "0x" + (_SEL_MERGE_NEG_RISK + params).hex()


# ---------------------------------------------------------------------------
# CTFClient
# ---------------------------------------------------------------------------

class CTFClient:
    """Client for CTF split/merge operations via the Polymarket Relayer."""

    def __init__(
        self,
        private_key: str,
        chain_id: int,
        builder_api_key: str,
        builder_secret: str,
        builder_passphrase: str,
        relayer_url: str = RELAYER_URL,
    ) -> None:
        builder_config = BuilderConfig(
            local_builder_creds=BuilderApiKeyCreds(
                key=builder_api_key,
                secret=builder_secret,
                passphrase=builder_passphrase,
            ),
        )
        self._client = RelayClient(
            relayer_url=relayer_url,
            chain_id=chain_id,
            private_key=private_key,
            builder_config=builder_config,
        )
        self._chain_id = chain_id

    def split(
        self,
        condition_id: str,
        amount_usdc: float,
        neg_risk: bool = False,
    ) -> Any:
        """Split USDC into YES+NO token pairs for a single condition.

        Returns the relayer transaction response (call .wait() to block
        until confirmed).
        """
        amount_raw = _usdc_to_raw(amount_usdc)
        if neg_risk:
            calldata = encode_split_neg_risk(condition_id, amount_raw)
            target = NEG_RISK_ADAPTER_ADDRESS
        else:
            calldata = encode_split_standard(condition_id, amount_raw)
            target = CTF_ADDRESS

        txn = SafeTransaction(
            to=target,
            operation=OperationType.Call,
            data=calldata,
            value="0",
        )
        logger.info(
            "Splitting %.2f USDC for condition %s (neg_risk=%s)",
            amount_usdc, condition_id[:18], neg_risk,
        )
        return self._client.execute([txn])

    def merge(
        self,
        condition_id: str,
        amount_shares: float,
        neg_risk: bool = False,
    ) -> Any:
        """Merge YES+NO token pairs back into USDC for a single condition.

        Returns the relayer transaction response.
        """
        amount_raw = _usdc_to_raw(amount_shares)
        if neg_risk:
            calldata = encode_merge_neg_risk(condition_id, amount_raw)
            target = NEG_RISK_ADAPTER_ADDRESS
        else:
            calldata = encode_merge_standard(condition_id, amount_raw)
            target = CTF_ADDRESS

        txn = SafeTransaction(
            to=target,
            operation=OperationType.Call,
            data=calldata,
            value="0",
        )
        logger.info(
            "Merging %.2f shares for condition %s (neg_risk=%s)",
            amount_shares, condition_id[:18], neg_risk,
        )
        return self._client.execute([txn])

    def approve_and_split(
        self,
        condition_id: str,
        amount_usdc: float,
        neg_risk: bool = False,
    ) -> Any:
        """Approve USDC spending + split in a single batched transaction.

        Returns the relayer transaction response.
        """
        # Determine the spender (who needs approval to pull USDC)
        spender = NEG_RISK_ADAPTER_ADDRESS if neg_risk else CTF_ADDRESS

        approve_calldata = encode_approve(spender, MAX_UINT256)
        approve_txn = SafeTransaction(
            to=USDC_ADDRESS,
            operation=OperationType.Call,
            data=approve_calldata,
            value="0",
        )

        amount_raw = _usdc_to_raw(amount_usdc)
        if neg_risk:
            split_calldata = encode_split_neg_risk(condition_id, amount_raw)
            target = NEG_RISK_ADAPTER_ADDRESS
        else:
            split_calldata = encode_split_standard(condition_id, amount_raw)
            target = CTF_ADDRESS

        split_txn = SafeTransaction(
            to=target,
            operation=OperationType.Call,
            data=split_calldata,
            value="0",
        )

        logger.info(
            "Approve + split %.2f USDC for condition %s (neg_risk=%s)",
            amount_usdc, condition_id[:18], neg_risk,
        )
        return self._client.execute([approve_txn, split_txn])

    def split_multiple(
        self,
        conditions: list[dict],
    ) -> Any:
        """Batch approve + split for multiple conditions in one transaction.

        Each entry in *conditions* must have:
            - condition_id: str
            - amount: float (USDC)
            - neg_risk: bool

        Returns the relayer transaction response.
        """
        if not conditions:
            raise ValueError("conditions list must not be empty")

        transactions: list[SafeTransaction] = []

        # Collect which spender contracts need approval
        needs_ctf_approval = any(not c.get("neg_risk", False) for c in conditions)
        needs_neg_risk_approval = any(c.get("neg_risk", False) for c in conditions)

        # Add approval transactions first
        if needs_ctf_approval:
            transactions.append(SafeTransaction(
                to=USDC_ADDRESS,
                operation=OperationType.Call,
                data=encode_approve(CTF_ADDRESS, MAX_UINT256),
                value="0",
            ))
        if needs_neg_risk_approval:
            transactions.append(SafeTransaction(
                to=USDC_ADDRESS,
                operation=OperationType.Call,
                data=encode_approve(NEG_RISK_ADAPTER_ADDRESS, MAX_UINT256),
                value="0",
            ))

        # Add split transactions
        for c in conditions:
            cid = c["condition_id"]
            amount_raw = _usdc_to_raw(c["amount"])
            neg_risk = c.get("neg_risk", False)

            if neg_risk:
                calldata = encode_split_neg_risk(cid, amount_raw)
                target = NEG_RISK_ADAPTER_ADDRESS
            else:
                calldata = encode_split_standard(cid, amount_raw)
                target = CTF_ADDRESS

            transactions.append(SafeTransaction(
                to=target,
                operation=OperationType.Call,
                data=calldata,
                value="0",
            ))

        total_usdc = sum(c["amount"] for c in conditions)
        logger.info(
            "Batch split: %d conditions, %.2f USDC total (%d transactions)",
            len(conditions), total_usdc, len(transactions),
        )
        return self._client.execute(transactions)
