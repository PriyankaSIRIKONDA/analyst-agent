"""API tests with a fake runtime: FakeLLM models, the real MCP server in process."""

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from analyst_agent import api
from analyst_agent.graph import GraphAgent
from analyst_agent.llm import FakeLLM, tool_turn
from analyst_agent.mcp_tools import MCPTools
from analyst_agent.security import KeyStore, RateLimiter, hash_key

KEY = "aa_test_key"


def good_llms(n=1):
    return {
        "router": FakeLLM([tool_turn(("route", {"route": "sql"}))] * n),
        "sql_writer": FakeLLM(
            [tool_turn(("write_sql", {"sql": "SELECT COUNT(*) FROM items",
                                      "tables_used": ["items"]}))] * n
        ),
        "explainer": FakeLLM([tool_turn(("final_answer", {"answer": "20 items"}))] * n),
    }


@pytest.fixture
def make_client(db, docs_file):
    def make(llms=None, keys=None, limiter=None):
        @asynccontextmanager
        async def runtime():
            async with MCPTools.in_process(db.path, docs_file) as tools:
                yield api.Runtime(
                    agent=GraphAgent(llms or good_llms(), tools, checkpointer=InMemorySaver()),
                    keys=keys or KeyStore({hash_key(KEY): "alice"}),
                    limiter=limiter or RateLimiter(per_minute=100, per_day=100),
                    info={"checkpointer": "memory"},
                )

        return TestClient(api.create_app(runtime))

    return make


def test_health_and_ready(make_client):
    with make_client() as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {"status": "ready", "checkpointer": "memory"}


def test_ask_returns_structured_answer(make_client):
    with make_client() as client:
        response = client.post("/ask", json={"question": "How many items?"},
                               headers={"X-API-Key": KEY, "X-Request-ID": "req-1"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == {"answer": "20 items", "sql": "SELECT COUNT(*) FROM items",
                              "tables_used": ["items"]}
    assert body["status"] == "ok" and body["request_id"] == "req-1"
    assert response.headers["X-Request-ID"] == "req-1"
    assert len(body["thread_id"]) == 32


def test_bearer_token_works_too(make_client):
    with make_client() as client:
        response = client.post("/ask", json={"question": "q"},
                               headers={"Authorization": f"Bearer {KEY}"})
    assert response.status_code == 200


def test_missing_or_wrong_key_is_401(make_client):
    with make_client() as client:
        assert client.post("/ask", json={"question": "q"}).status_code == 401
        wrong = client.post("/ask", json={"question": "q"}, headers={"X-API-Key": "nope"})
        assert wrong.status_code == 401


def test_no_keys_configured_is_503(make_client):
    with make_client(keys=KeyStore({})) as client:
        response = client.post("/ask", json={"question": "q"}, headers={"X-API-Key": KEY})
    assert response.status_code == 503


def test_rate_limit_is_429_with_retry_after(make_client):
    with make_client(llms=good_llms(2), limiter=RateLimiter(per_minute=1, per_day=10)) as client:
        headers = {"X-API-Key": KEY}
        assert client.post("/ask", json={"question": "q"}, headers=headers).status_code == 200
        limited = client.post("/ask", json={"question": "q"}, headers=headers)
    assert limited.status_code == 429 and int(limited.headers["Retry-After"]) > 0


def test_follow_up_keeps_the_thread(make_client):
    llms = good_llms(2)
    with make_client(llms=llms) as client:
        headers = {"X-API-Key": KEY}
        first = client.post("/ask", json={"question": "How many items?"}, headers=headers).json()
        second = client.post("/ask", json={"question": "And now?", "thread_id": first["thread_id"]},
                             headers=headers).json()
    assert second["thread_id"] == first["thread_id"]
    assert "How many items?" in llms["sql_writer"].calls[1][1]["content"]


def test_bad_input_is_422(make_client):
    with make_client() as client:
        headers = {"X-API-Key": KEY}
        assert client.post("/ask", json={"question": ""}, headers=headers).status_code == 422
        bad_thread = {"question": "q", "thread_id": "../etc"}
        assert client.post("/ask", json=bad_thread, headers=headers).status_code == 422


def test_failed_model_is_502(make_client):
    llms = {"router": FakeLLM([RuntimeError("quota")])}
    with make_client(llms=llms) as client:
        response = client.post("/ask", json={"question": "q"}, headers={"X-API-Key": KEY})
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["status"] == "llm_error" and "quota" not in detail["error"]  # no internals
    assert detail["request_id"]
