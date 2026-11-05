"""Tests for the LiteLLM wrapper: rate limiting and retries, with no network."""

from types import SimpleNamespace

import litellm
import pytest

from analyst_agent.llm import RETRY_BACKOFF_S, LiteLLMClient


def _response():
    message = SimpleNamespace(content="hi", tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


@pytest.fixture
def sleeps(monkeypatch):
    """Record every sleep instead of waiting."""
    calls = []
    monkeypatch.setattr("analyst_agent.llm.time.sleep", calls.append)
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.0)
    return calls


def test_retries_with_backoff_then_succeeds(monkeypatch, sleeps):
    outcomes = [
        litellm.RateLimitError("slow down", "mistral", "m"),
        litellm.InternalServerError("boom", "mistral", "m"),
        _response(),
    ]
    seen = []

    def fake_completion(**kwargs):
        seen.append(kwargs)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(litellm, "completion", fake_completion)
    result = LiteLLMClient(model="m").complete([], [])
    assert result.content == "hi"
    assert sleeps == [2, 4]
    assert all(kw["num_retries"] == 0 for kw in seen)


def test_gives_up_after_five_retries(monkeypatch, sleeps):
    def always_fail(**kwargs):
        raise litellm.RateLimitError("slow down", "mistral", "m")

    monkeypatch.setattr(litellm, "completion", always_fail)
    with pytest.raises(litellm.RateLimitError):
        LiteLLMClient(model="m").complete([], [])
    assert sleeps == list(RETRY_BACKOFF_S)


def test_other_errors_are_not_retried(monkeypatch, sleeps):
    def bad_request(**kwargs):
        raise ValueError("bad")

    monkeypatch.setattr(litellm, "completion", bad_request)
    with pytest.raises(ValueError):
        LiteLLMClient(model="m").complete([], [])
    assert sleeps == []


def test_min_interval_spaces_consecutive_calls(monkeypatch, sleeps):
    clock = iter([100.0, 100.2, 100.2])  # first call, then the gap check and stamp of the second
    monkeypatch.setattr("analyst_agent.llm.time.monotonic", lambda: next(clock))
    monkeypatch.setattr(litellm, "completion", lambda **kw: _response())
    client = LiteLLMClient(model="m", min_interval_s=1.0)
    client.complete([], [])
    client.complete([], [])
    assert sleeps == [pytest.approx(0.8)]


def test_real_litellm_call_path_runs_offline(monkeypatch):
    """Go through the real litellm.completion with a mocked answer (no network).

    The other tests replace litellm.completion, so they cannot see problems inside it,
    such as LiteLLM importing an optional package that is not installed.
    """
    real_completion = litellm.completion
    monkeypatch.setattr(
        litellm, "completion", lambda **kw: real_completion(mock_response="ok", **kw)
    )
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.0)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_tables",
                "description": "List the tables.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]
    client = LiteLLMClient(model="gemini/gemini-3.5-flash-lite")
    response = client.complete([{"role": "user", "content": "hi"}], tools)
    assert response.content == "ok"
