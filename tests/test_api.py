"""API tests with a FakeLLM agent injected, so no real model is called."""

from fastapi.testclient import TestClient

from analyst_agent import api
from analyst_agent.agent import Agent
from analyst_agent.llm import FakeLLM, tool_turn


def test_health():
    assert TestClient(api.app).get("/health").json() == {"status": "ok"}


def test_ask_returns_structured_answer(monkeypatch, toolbox):
    final = (
        "final_answer",
        {"answer": "20 items", "sql": "SELECT COUNT(*) FROM items", "tables_used": ["items"]},
    )
    agent = Agent(FakeLLM([tool_turn(final)]), toolbox)
    monkeypatch.setattr(api, "build_agent", lambda: agent)

    response = TestClient(api.app).post("/ask", json={"question": "How many items?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]["tables_used"] == ["items"]
    assert body["status"] == "ok"


def test_ask_returns_502_when_the_agent_gives_no_answer(monkeypatch, toolbox):
    agent = Agent(FakeLLM([]), toolbox)
    monkeypatch.setattr(api, "build_agent", lambda: agent)
    response = TestClient(api.app).post("/ask", json={"question": "?"})
    assert response.status_code == 502
