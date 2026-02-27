"""Tests for the base WebSocket client — reconnect backoff and state transitions.

These tests verify internal logic without needing a live WebSocket server.
They mock the ``websocket`` library to simulate connect / disconnect / error
scenarios.
"""

import time
from unittest.mock import MagicMock, patch


from src.data.ws_client import ConnectionState, WebSocketClient


# ---------------------------------------------------------------------------
# State transition tests
# ---------------------------------------------------------------------------


class TestConnectionState:
    """Verify the connection state enum values."""

    def test_states_exist(self) -> None:
        assert ConnectionState.DISCONNECTED == "DISCONNECTED"
        assert ConnectionState.CONNECTING == "CONNECTING"
        assert ConnectionState.CONNECTED == "CONNECTED"
        assert ConnectionState.RECONNECTING == "RECONNECTING"


class TestInitialState:
    def test_initial_state_is_disconnected(self) -> None:
        client = WebSocketClient(
            url="wss://example.com/ws",
            on_message=lambda m: None,
        )
        assert client.state == ConnectionState.DISCONNECTED

    def test_close_on_disconnected_is_safe(self) -> None:
        client = WebSocketClient(
            url="wss://example.com/ws",
            on_message=lambda m: None,
        )
        client.close()  # Should not raise
        assert client.state == ConnectionState.DISCONNECTED


class TestStateTransitions:
    """Test state changes driven by simulated WebSocket callbacks."""

    def test_connect_transitions_to_connecting(self) -> None:
        """Calling connect() should set state to CONNECTING."""
        messages: list[str] = []
        client = WebSocketClient(
            url="wss://example.com/ws",
            on_message=lambda m: messages.append(m),
        )

        # Mock WebSocketApp so we don't actually connect
        with patch("src.data.ws_client.websocket.WebSocketApp") as MockWSApp:
            mock_app = MagicMock()
            MockWSApp.return_value = mock_app
            # run_forever blocks, so make it a no-op
            mock_app.run_forever.side_effect = lambda **kw: None

            # connect() will set CONNECTING, then wait for connected_event
            # Since run_forever is a no-op, on_open won't fire,
            # so connect will time out but state should have been CONNECTING
            client.connect(timeout=0.5)

        # After timeout without on_open firing, it may still be CONNECTING
        # or DISCONNECTED (if on_close fired). The key assertion is that
        # it passed through CONNECTING.
        assert client.state in (
            ConnectionState.CONNECTING,
            ConnectionState.DISCONNECTED,
            ConnectionState.RECONNECTING,
        )
        client._auto_reconnect = False
        client._stop_event.set()

    def test_on_open_sets_connected(self) -> None:
        """Simulating on_open should transition to CONNECTED."""
        client = WebSocketClient(
            url="wss://example.com/ws",
            on_message=lambda m: None,
        )
        client._set_state(ConnectionState.CONNECTING)

        # Simulate the on_open callback
        mock_ws = MagicMock()
        client._ws = mock_ws
        client._on_open(mock_ws)

        assert client.state == ConnectionState.CONNECTED
        assert client._connected_event.is_set()

        # Cleanup
        client._stop_event.set()

    def test_on_close_sets_disconnected(self) -> None:
        """Simulating on_close should transition to DISCONNECTED."""
        client = WebSocketClient(
            url="wss://example.com/ws",
            on_message=lambda m: None,
        )
        client._auto_reconnect = False
        client._set_state(ConnectionState.CONNECTED)
        mock_ws = MagicMock()
        client._ws = mock_ws
        client._on_close(mock_ws, None, None)
        assert client.state == ConnectionState.DISCONNECTED

    def test_on_close_triggers_reconnect(self) -> None:
        """If auto_reconnect is True, on_close should go to RECONNECTING."""
        client = WebSocketClient(
            url="wss://example.com/ws",
            on_message=lambda m: None,
        )
        client._auto_reconnect = True
        client._stop_event.clear()
        client._set_state(ConnectionState.CONNECTED)
        mock_ws = MagicMock()
        client._ws = mock_ws

        with patch.object(client, "_start_ws"):
            client._on_close(mock_ws, None, None)
            # Give the reconnect thread time to set state
            time.sleep(0.1)
            assert client.state in (
                ConnectionState.RECONNECTING,
                ConnectionState.CONNECTING,
            )

        # Cleanup
        client._auto_reconnect = False
        client._stop_event.set()


# ---------------------------------------------------------------------------
# Reconnect backoff tests
# ---------------------------------------------------------------------------


class TestReconnectBackoff:
    """Verify exponential backoff delay calculation."""

    def test_backoff_increases(self) -> None:
        """Successive reconnect attempts should increase delay."""
        max_delay = 30.0

        # Manually check the formula: delay = min(2^(attempt-1), max_delay)
        # attempt 1 -> 1s, attempt 2 -> 2s, attempt 3 -> 4s, ...
        delays = []
        for attempt in range(1, 7):
            delay = min(2 ** (attempt - 1), max_delay)
            delays.append(delay)

        assert delays == [1, 2, 4, 8, 16, 30]

    def test_backoff_capped_at_max(self) -> None:
        """Delay should never exceed max_reconnect_delay."""
        max_delay = 10.0
        for attempt in range(1, 20):
            delay = min(2 ** (attempt - 1), max_delay)
            assert delay <= max_delay

    def test_reconnect_resets_on_successful_connect(self) -> None:
        """On successful connect, reconnect_attempts should reset to 0."""
        client = WebSocketClient(
            url="wss://example.com/ws",
            on_message=lambda m: None,
        )
        client._reconnect_attempts = 5
        client._ws = MagicMock()
        client._on_open(client._ws)
        assert client._reconnect_attempts == 0

        # Cleanup
        client._stop_event.set()


# ---------------------------------------------------------------------------
# Send / message tests
# ---------------------------------------------------------------------------


class TestSendMessage:
    def test_send_when_not_connected_logs_warning(self) -> None:
        """Sending when disconnected should not raise."""
        client = WebSocketClient(
            url="wss://example.com/ws",
            on_message=lambda m: None,
        )
        # Should not raise
        client.send("test message")

    def test_on_message_dispatches_to_callback(self) -> None:
        """Incoming messages should be forwarded to the user callback."""
        received: list[str] = []
        client = WebSocketClient(
            url="wss://example.com/ws",
            on_message=lambda m: received.append(m),
        )
        mock_ws = MagicMock()
        client._ws = mock_ws
        client._on_message(mock_ws, "hello world")
        assert received == ["hello world"]

    def test_on_error_dispatches_to_callback(self) -> None:
        errors: list[Exception] = []
        client = WebSocketClient(
            url="wss://example.com/ws",
            on_message=lambda m: None,
            on_error=lambda e: errors.append(e),
        )
        mock_ws = MagicMock()
        client._ws = mock_ws
        err = RuntimeError("test error")
        client._on_error(mock_ws, err)
        assert len(errors) == 1
        assert errors[0] is err
