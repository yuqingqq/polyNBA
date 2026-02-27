"""Tests for CTF split/merge operations (src/ctf.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.ctf import (
    BINARY_PARTITION,
    COLLATERAL_TOKEN,
    CTF_ADDRESS,
    MAX_UINT256,
    NEG_RISK_ADAPTER_ADDRESS,
    PARENT_COLLECTION_ID,
    USDC_ADDRESS,
    CTFClient,
    _encode_condition_id,
    _function_selector,
    _usdc_to_raw,
    encode_approve,
    encode_merge_neg_risk,
    encode_merge_standard,
    encode_split_neg_risk,
    encode_split_standard,
)
from py_builder_relayer_client.models import OperationType, SafeTransaction


# ------------------------------------------------------------------
# Encoding helpers
# ------------------------------------------------------------------


class TestFunctionSelector:
    """Verify 4-byte function selector computation."""

    def test_approve_selector(self) -> None:
        sel = _function_selector("approve(address,uint256)")
        assert len(sel) == 4
        # ERC-20 approve selector is 0x095ea7b3
        assert sel.hex() == "095ea7b3"

    def test_split_standard_selector(self) -> None:
        sel = _function_selector(
            "splitPosition(address,bytes32,bytes32,uint256[],uint256)"
        )
        assert len(sel) == 4

    def test_split_neg_risk_selector(self) -> None:
        sel = _function_selector("splitPosition(bytes32,uint256)")
        assert len(sel) == 4

    def test_merge_standard_selector(self) -> None:
        sel = _function_selector(
            "mergePositions(address,bytes32,bytes32,uint256[],uint256)"
        )
        assert len(sel) == 4

    def test_merge_neg_risk_selector(self) -> None:
        sel = _function_selector("mergePositions(bytes32,uint256)")
        assert len(sel) == 4


class TestConditionIdEncoding:
    """Verify condition_id hex → bytes32 conversion."""

    def test_with_0x_prefix(self) -> None:
        cid = "0x" + "ab" * 32
        result = _encode_condition_id(cid)
        assert len(result) == 32
        assert result == bytes.fromhex("ab" * 32)

    def test_without_0x_prefix(self) -> None:
        cid = "ab" * 32
        result = _encode_condition_id(cid)
        assert len(result) == 32
        assert result == bytes.fromhex("ab" * 32)

    def test_short_id_left_padded(self) -> None:
        cid = "0x1234"
        result = _encode_condition_id(cid)
        assert len(result) == 32
        assert result == b"\x00" * 30 + b"\x12\x34"


class TestUsdcConversion:
    """Verify USDC dollar → raw (6 decimal) conversion."""

    def test_whole_dollars(self) -> None:
        assert _usdc_to_raw(50) == 50_000_000

    def test_fractional_dollars(self) -> None:
        assert _usdc_to_raw(1.5) == 1_500_000

    def test_zero(self) -> None:
        assert _usdc_to_raw(0) == 0

    def test_small_amount(self) -> None:
        assert _usdc_to_raw(0.01) == 10_000


# ------------------------------------------------------------------
# Calldata encoding
# ------------------------------------------------------------------


class TestEncodeApprove:
    """Verify ERC-20 approve calldata encoding."""

    def test_starts_with_selector(self) -> None:
        calldata = encode_approve(CTF_ADDRESS, MAX_UINT256)
        assert calldata.startswith("0x095ea7b3")

    def test_contains_spender_address(self) -> None:
        calldata = encode_approve(CTF_ADDRESS, MAX_UINT256)
        # Address is left-padded to 32 bytes in ABI encoding
        addr_hex = CTF_ADDRESS[2:].lower()
        assert addr_hex in calldata.lower()

    def test_max_uint256_encoded(self) -> None:
        calldata = encode_approve(CTF_ADDRESS, MAX_UINT256)
        # max_uint256 = 0xff...ff (64 hex chars)
        assert "f" * 64 in calldata


class TestEncodeSplitStandard:
    """Verify standard CTF splitPosition calldata."""

    def test_calldata_length(self) -> None:
        cid = "0x" + "ab" * 32
        calldata = encode_split_standard(cid, 50_000_000)
        # 4 (selector) + 5 params * 32 bytes + dynamic array overhead
        # The uint256[] adds offset + length + 2 elements
        raw = bytes.fromhex(calldata[2:])
        assert len(raw) > 4 + 5 * 32  # at least selector + 5 words

    def test_starts_with_correct_selector(self) -> None:
        cid = "0x" + "ab" * 32
        calldata = encode_split_standard(cid, 50_000_000)
        sel = _function_selector(
            "splitPosition(address,bytes32,bytes32,uint256[],uint256)"
        )
        assert calldata.startswith("0x" + sel.hex())


class TestEncodeSplitNegRisk:
    """Verify NegRiskAdapter splitPosition calldata."""

    def test_calldata_shorter_than_standard(self) -> None:
        cid = "0x" + "ab" * 32
        standard = encode_split_standard(cid, 50_000_000)
        neg_risk = encode_split_neg_risk(cid, 50_000_000)
        # neg_risk encoding has fewer params → shorter calldata
        assert len(neg_risk) < len(standard)

    def test_starts_with_correct_selector(self) -> None:
        cid = "0x" + "ab" * 32
        calldata = encode_split_neg_risk(cid, 50_000_000)
        sel = _function_selector("splitPosition(bytes32,uint256)")
        assert calldata.startswith("0x" + sel.hex())

    def test_fixed_length(self) -> None:
        cid = "0x" + "ab" * 32
        calldata = encode_split_neg_risk(cid, 50_000_000)
        raw = bytes.fromhex(calldata[2:])
        # 4 (selector) + 2 * 32 (bytes32 + uint256)
        assert len(raw) == 4 + 2 * 32


class TestEncodeMerge:
    """Verify merge calldata encoding mirrors split encoding."""

    def test_merge_standard_selector_differs_from_split(self) -> None:
        cid = "0x" + "ab" * 32
        split = encode_split_standard(cid, 50_000_000)
        merge = encode_merge_standard(cid, 50_000_000)
        # Same params but different function selector
        assert split[:10] != merge[:10]
        # Params portion should be identical
        assert split[10:] == merge[10:]

    def test_merge_neg_risk_selector_differs_from_split(self) -> None:
        cid = "0x" + "ab" * 32
        split = encode_split_neg_risk(cid, 50_000_000)
        merge = encode_merge_neg_risk(cid, 50_000_000)
        assert split[:10] != merge[:10]
        assert split[10:] == merge[10:]


# ------------------------------------------------------------------
# CTFClient construction
# ------------------------------------------------------------------


class TestCTFClientConstruction:
    """Verify CTFClient initialises RelayClient correctly."""

    @patch("src.ctf.RelayClient")
    def test_creates_relay_client(self, mock_relay_cls: MagicMock) -> None:
        CTFClient(
            private_key="0x" + "ab" * 32,
            chain_id=137,
            builder_api_key="key",
            builder_secret="secret",
            builder_passphrase="pass",
        )
        mock_relay_cls.assert_called_once()
        call_kwargs = mock_relay_cls.call_args
        assert call_kwargs.kwargs["chain_id"] == 137
        assert call_kwargs.kwargs["private_key"] == "0x" + "ab" * 32

    @patch("src.ctf.RelayClient")
    def test_custom_relayer_url(self, mock_relay_cls: MagicMock) -> None:
        CTFClient(
            private_key="0x" + "ab" * 32,
            chain_id=137,
            builder_api_key="key",
            builder_secret="secret",
            builder_passphrase="pass",
            relayer_url="https://custom-relayer.example.com",
        )
        call_kwargs = mock_relay_cls.call_args
        assert call_kwargs.kwargs["relayer_url"] == "https://custom-relayer.example.com"


# ------------------------------------------------------------------
# CTFClient operations (mocked RelayClient)
# ------------------------------------------------------------------


@pytest.fixture
def mock_relay() -> MagicMock:
    return MagicMock()


@pytest.fixture
def ctf_client(mock_relay: MagicMock) -> CTFClient:
    with patch("src.ctf.RelayClient", return_value=mock_relay):
        client = CTFClient(
            private_key="0x" + "ab" * 32,
            chain_id=137,
            builder_api_key="key",
            builder_secret="secret",
            builder_passphrase="pass",
        )
    return client


class TestCTFClientSplit:
    """Test split() dispatches to the correct contract."""

    def test_standard_split_targets_ctf(
        self, ctf_client: CTFClient, mock_relay: MagicMock
    ) -> None:
        cid = "0x" + "ab" * 32
        ctf_client.split(cid, 50.0, neg_risk=False)
        mock_relay.execute.assert_called_once()
        txns = mock_relay.execute.call_args[0][0]
        assert len(txns) == 1
        assert txns[0].to == CTF_ADDRESS

    def test_neg_risk_split_targets_adapter(
        self, ctf_client: CTFClient, mock_relay: MagicMock
    ) -> None:
        cid = "0x" + "ab" * 32
        ctf_client.split(cid, 50.0, neg_risk=True)
        txns = mock_relay.execute.call_args[0][0]
        assert txns[0].to == NEG_RISK_ADAPTER_ADDRESS

    def test_split_operation_is_call(
        self, ctf_client: CTFClient, mock_relay: MagicMock
    ) -> None:
        cid = "0x" + "ab" * 32
        ctf_client.split(cid, 50.0)
        txns = mock_relay.execute.call_args[0][0]
        assert txns[0].operation == OperationType.Call

    def test_split_value_is_zero(
        self, ctf_client: CTFClient, mock_relay: MagicMock
    ) -> None:
        cid = "0x" + "ab" * 32
        ctf_client.split(cid, 50.0)
        txns = mock_relay.execute.call_args[0][0]
        assert txns[0].value == "0"


class TestCTFClientMerge:
    """Test merge() dispatches to the correct contract."""

    def test_standard_merge_targets_ctf(
        self, ctf_client: CTFClient, mock_relay: MagicMock
    ) -> None:
        cid = "0x" + "ab" * 32
        ctf_client.merge(cid, 50.0, neg_risk=False)
        txns = mock_relay.execute.call_args[0][0]
        assert txns[0].to == CTF_ADDRESS

    def test_neg_risk_merge_targets_adapter(
        self, ctf_client: CTFClient, mock_relay: MagicMock
    ) -> None:
        cid = "0x" + "ab" * 32
        ctf_client.merge(cid, 50.0, neg_risk=True)
        txns = mock_relay.execute.call_args[0][0]
        assert txns[0].to == NEG_RISK_ADAPTER_ADDRESS


class TestCTFClientApproveAndSplit:
    """Test approve_and_split() batches approve + split."""

    def test_two_transactions_in_batch(
        self, ctf_client: CTFClient, mock_relay: MagicMock
    ) -> None:
        cid = "0x" + "ab" * 32
        ctf_client.approve_and_split(cid, 30.0, neg_risk=False)
        txns = mock_relay.execute.call_args[0][0]
        assert len(txns) == 2

    def test_first_is_approve_to_usdc(
        self, ctf_client: CTFClient, mock_relay: MagicMock
    ) -> None:
        cid = "0x" + "ab" * 32
        ctf_client.approve_and_split(cid, 30.0, neg_risk=False)
        txns = mock_relay.execute.call_args[0][0]
        assert txns[0].to == USDC_ADDRESS
        assert txns[0].data.startswith("0x095ea7b3")

    def test_second_is_split(
        self, ctf_client: CTFClient, mock_relay: MagicMock
    ) -> None:
        cid = "0x" + "ab" * 32
        ctf_client.approve_and_split(cid, 30.0, neg_risk=False)
        txns = mock_relay.execute.call_args[0][0]
        assert txns[1].to == CTF_ADDRESS

    def test_neg_risk_approves_adapter(
        self, ctf_client: CTFClient, mock_relay: MagicMock
    ) -> None:
        cid = "0x" + "ab" * 32
        ctf_client.approve_and_split(cid, 30.0, neg_risk=True)
        txns = mock_relay.execute.call_args[0][0]
        # Approve spender should be NegRiskAdapter
        approve_data = txns[0].data
        adapter_hex = NEG_RISK_ADAPTER_ADDRESS[2:].lower()
        assert adapter_hex in approve_data.lower()
        # Split target should also be NegRiskAdapter
        assert txns[1].to == NEG_RISK_ADAPTER_ADDRESS


class TestCTFClientSplitMultiple:
    """Test split_multiple() batches approvals + splits."""

    def test_empty_conditions_raises(self, ctf_client: CTFClient) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            ctf_client.split_multiple([])

    def test_single_standard_condition(
        self, ctf_client: CTFClient, mock_relay: MagicMock
    ) -> None:
        conditions = [
            {"condition_id": "0x" + "ab" * 32, "amount": 50.0, "neg_risk": False},
        ]
        ctf_client.split_multiple(conditions)
        txns = mock_relay.execute.call_args[0][0]
        # 1 approval (CTF) + 1 split
        assert len(txns) == 2
        assert txns[0].to == USDC_ADDRESS  # approve
        assert txns[1].to == CTF_ADDRESS  # split

    def test_mixed_conditions(
        self, ctf_client: CTFClient, mock_relay: MagicMock
    ) -> None:
        conditions = [
            {"condition_id": "0x" + "aa" * 32, "amount": 30.0, "neg_risk": False},
            {"condition_id": "0x" + "bb" * 32, "amount": 20.0, "neg_risk": True},
        ]
        ctf_client.split_multiple(conditions)
        txns = mock_relay.execute.call_args[0][0]
        # 2 approvals (CTF + NegRiskAdapter) + 2 splits
        assert len(txns) == 4
        # First two are approvals to USDC
        assert txns[0].to == USDC_ADDRESS
        assert txns[1].to == USDC_ADDRESS
        # Then splits
        assert txns[2].to == CTF_ADDRESS
        assert txns[3].to == NEG_RISK_ADAPTER_ADDRESS

    def test_all_neg_risk_only_one_approval(
        self, ctf_client: CTFClient, mock_relay: MagicMock
    ) -> None:
        conditions = [
            {"condition_id": "0x" + "aa" * 32, "amount": 10.0, "neg_risk": True},
            {"condition_id": "0x" + "bb" * 32, "amount": 20.0, "neg_risk": True},
        ]
        ctf_client.split_multiple(conditions)
        txns = mock_relay.execute.call_args[0][0]
        # 1 approval (NegRiskAdapter only) + 2 splits
        assert len(txns) == 3
        assert txns[0].to == USDC_ADDRESS
        assert txns[1].to == NEG_RISK_ADAPTER_ADDRESS
        assert txns[2].to == NEG_RISK_ADAPTER_ADDRESS

    def test_neg_risk_defaults_to_false(
        self, ctf_client: CTFClient, mock_relay: MagicMock
    ) -> None:
        conditions = [
            {"condition_id": "0x" + "ab" * 32, "amount": 50.0},
        ]
        ctf_client.split_multiple(conditions)
        txns = mock_relay.execute.call_args[0][0]
        # Should target CTF (standard, not neg_risk)
        assert txns[1].to == CTF_ADDRESS
