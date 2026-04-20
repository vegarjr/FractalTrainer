"""Tests for make_local_llm_client. Most tests mock the HTTP layer; one
integration test hits a live llama-server on localhost:8080 if reachable
(otherwise skipped)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from unittest import mock

import pytest

from fractaltrainer.repair.llm_client import make_local_llm_client


# ────────────────────────────────────────────────────────────────────────
#   Mocked unit tests (always run in CI)
# ────────────────────────────────────────────────────────────────────────

def _mock_response(body: dict):
    """Return a context-manager-compatible mock response."""
    cm = mock.MagicMock()
    cm.__enter__.return_value.read.return_value = json.dumps(body).encode()
    cm.__exit__.return_value = False
    return cm


def test_ping_failure_raises_informative_error():
    with mock.patch(
        "fractaltrainer.repair.llm_client.urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        with pytest.raises(RuntimeError, match="Cannot reach llama-server"):
            make_local_llm_client(ping_on_create=True)


def test_ping_wrong_status_raises():
    with mock.patch(
        "fractaltrainer.repair.llm_client.urllib.request.urlopen",
        return_value=_mock_response({"status": "loading"}),
    ):
        with pytest.raises(RuntimeError, match="expected 'ok'"):
            make_local_llm_client(ping_on_create=True)


def test_ping_ok_returns_callable():
    with mock.patch(
        "fractaltrainer.repair.llm_client.urllib.request.urlopen",
        return_value=_mock_response({"status": "ok"}),
    ):
        fn = make_local_llm_client(ping_on_create=True)
    assert callable(fn)


def test_successful_call_returns_response_text():
    responses = [
        _mock_response({"status": "ok"}),  # ping
        _mock_response({"choices": [{"message": {
            "role": "assistant", "content": "hello there"}}]}),
    ]
    with mock.patch(
        "fractaltrainer.repair.llm_client.urllib.request.urlopen",
        side_effect=responses,
    ):
        fn = make_local_llm_client()
        result = fn("system", "user question")
    assert result == "hello there"


def test_empty_choices_raises():
    responses = [
        _mock_response({"status": "ok"}),
        _mock_response({"choices": []}),
    ]
    with mock.patch(
        "fractaltrainer.repair.llm_client.urllib.request.urlopen",
        side_effect=responses,
    ):
        fn = make_local_llm_client()
        with pytest.raises(RuntimeError, match="no choices"):
            fn("sys", "user")


def test_malformed_response_raises():
    responses = [
        _mock_response({"status": "ok"}),
        _mock_response({"choices": [{"message": {"role": "assistant"}}]}),
    ]
    with mock.patch(
        "fractaltrainer.repair.llm_client.urllib.request.urlopen",
        side_effect=responses,
    ):
        fn = make_local_llm_client()
        with pytest.raises(RuntimeError, match="malformed"):
            fn("sys", "user")


def test_http_error_propagates_with_rc():
    class _Err(urllib.error.HTTPError):
        def __init__(self):
            super().__init__("url", 500, "Server Error", {}, None)

        def read(self):  # overriding base class
            return b"internal error detail"

    responses = [
        _mock_response({"status": "ok"}),
    ]
    with mock.patch(
        "fractaltrainer.repair.llm_client.urllib.request.urlopen",
        side_effect=responses + [_Err()],
    ):
        fn = make_local_llm_client()
        with pytest.raises(RuntimeError, match="HTTP 500"):
            fn("sys", "user")


def test_ping_disabled_skips_health_check():
    # Nothing in the mock stack for the ping call — should NOT fail at
    # construction. If it tries to ping, urlopen is not patched and will
    # raise (proving ping is skipped).
    fn = make_local_llm_client(ping_on_create=False)
    assert callable(fn)


# ────────────────────────────────────────────────────────────────────────
#   Integration test against a live local server (auto-skip if not up)
# ────────────────────────────────────────────────────────────────────────

def _server_is_up(url: str = "http://127.0.0.1:8080/health") -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False


@pytest.mark.skipif(not _server_is_up(),
                    reason="no local llama-server on 127.0.0.1:8080")
def test_live_server_hello_world():
    fn = make_local_llm_client(max_tokens=32, temperature=0.0)
    text = fn("You are concise.",
              "Reply with the single word: hello")
    assert isinstance(text, str)
    assert len(text) > 0
    # Model should include 'hello' somewhere in its response (case-insensitive)
    assert "hello" in text.lower()
