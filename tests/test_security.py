"""API keys and the per-key rate limiter."""

import pytest

from analyst_agent.security import KeyStore, RateLimiter, hash_key, main, new_key, parse_api_keys


def test_keys_are_matched_by_hash():
    key = new_key()
    store = KeyStore(parse_api_keys(f"alice:{hash_key(key)}"))
    assert store.identify(key) == "alice"
    assert store.identify("aa_wrong") is None and store.identify(None) is None


def test_bad_entries_are_refused():
    with pytest.raises(ValueError):
        parse_api_keys("alice:short")
    with pytest.raises(ValueError):
        parse_api_keys("nohash")
    assert parse_api_keys("") == {}


def test_no_keys_fails_closed_unless_disabled():
    assert KeyStore({}).configured is False
    assert KeyStore({}, disabled=True).identify(None) == "local"


def test_rate_limit_per_minute_and_per_day():
    limiter = RateLimiter(per_minute=2, per_day=3)
    t = 1_800_000_000.0
    assert limiter.check("a", t) is None and limiter.check("a", t + 1) is None
    assert limiter.check("a", t + 2) is not None  # third in the same minute
    assert limiter.check("b", t + 2) is None  # other keys are counted apart
    assert limiter.check("a", t + 61) is None  # window moved: third of the day
    assert limiter.check("a", t + 200) is not None  # daily cap reached
    assert limiter.check("a", t + 86_400) is None  # next day


def test_new_key_command(capsys):
    assert main(["new-key", "alice"]) == 0
    out = capsys.readouterr().out
    key = out.split("stored): ")[1].split()[0]
    assert f"alice:{hash_key(key)}" in out
    assert main([]) == 2
