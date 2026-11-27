"""API keys and per-key limits for the HTTP API.

Keys are never stored in clear. ``API_KEYS`` holds ``name:sha256`` pairs, comma-separated:

    API_KEYS=alice:9f86d081...,demo:2c26b46b...

Make a new key with ``python -m analyst_agent.security new-key alice``: it prints the key
(give it to the user once) and the ``name:hash`` entry to add to API_KEYS.

Set ``AUTH_DISABLED=true`` only for local development; without keys and without that flag
the API refuses questions, so a forgotten setting fails closed.

The rate limiter counts requests per key in memory (per minute and per UTC day). That is
enough for one instance; with several instances it moves to Postgres or Redis.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

KEY_PREFIX = "aa_"


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def new_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def parse_api_keys(value: str | None) -> dict[str, str]:
    """'alice:hash,bob:hash' -> {hash: name}. Malformed entries are an error, not ignored."""
    keys: dict[str, str] = {}
    for entry in (value or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, sep, digest = entry.partition(":")
        digest = digest.strip().lower()
        if not sep or not name.strip() or len(digest) != 64:
            raise ValueError(f"Bad API_KEYS entry {entry[:20]!r}: expected name:sha256hex")
        keys[digest] = name.strip()
    return keys


@dataclass
class KeyStore:
    hashes: dict[str, str]  # sha256 -> name
    disabled: bool = False

    @classmethod
    def from_env(cls) -> KeyStore:
        disabled = os.getenv("AUTH_DISABLED", "").lower() in ("1", "true", "yes")
        return cls(parse_api_keys(os.getenv("API_KEYS")), disabled=disabled)

    @property
    def configured(self) -> bool:
        return self.disabled or bool(self.hashes)

    def identify(self, key: str | None) -> str | None:
        """The key owner's name, or None. Constant-time comparison of the hashes."""
        if self.disabled:
            return "local"
        if not key:
            return None
        digest = hash_key(key)
        for known, name in self.hashes.items():
            if hmac.compare_digest(known, digest):
                return name
        return None


@dataclass
class RateLimiter:
    per_minute: int = 10
    per_day: int = 200
    _minute: dict[str, deque] = field(default_factory=dict)
    _day: dict[str, tuple[str, int]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def from_env(cls) -> RateLimiter:
        return cls(
            per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE") or 10),
            per_day=int(os.getenv("DAILY_QUERY_CAP") or 200),
        )

    def check(self, who: str, now: float | None = None) -> int | None:
        """Count one request. Returns None if allowed, else seconds to wait (Retry-After)."""
        now = time.time() if now is None else now
        today = datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%d")
        with self._lock:
            day, count = self._day.get(who, (today, 0))
            if day != today:
                count = 0
            if self.per_day and count >= self.per_day:
                midnight = datetime.fromtimestamp(now, UTC).replace(
                    hour=0, minute=0, second=0, microsecond=0
                ).timestamp() + 86400
                return max(1, int(midnight - now))
            window = self._minute.setdefault(who, deque())
            while window and window[0] <= now - 60:
                window.popleft()
            if self.per_minute and len(window) >= self.per_minute:
                return max(1, int(window[0] + 60 - now) + 1)
            window.append(now)
            self._day[who] = (today, count + 1)
        return None


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] != "new-key":
        print("usage: python -m analyst_agent.security new-key <name>", file=sys.stderr)
        return 2
    key = new_key()
    print(f"key (give it to the user, it is not stored): {key}")
    print(f"add to API_KEYS: {argv[1]}:{hash_key(key)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
