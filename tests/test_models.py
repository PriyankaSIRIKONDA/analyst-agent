"""Per-node model configuration."""

from analyst_agent.models import ModelConfig


def test_defaults_to_model_for_every_node(monkeypatch):
    for name in ("MODEL_ROUTER", "MODEL_SQL_WRITER", "MODEL_EXPLAINER", "MODEL_FALLBACK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODEL", "gemini/x")
    config = ModelConfig.from_env()
    assert (config.router, config.sql_writer, config.explainer) == ("gemini/x",) * 3
    assert config.fallback is None


def test_each_node_can_have_its_own_model(monkeypatch):
    monkeypatch.setenv("MODEL", "gemini/x")
    monkeypatch.setenv("MODEL_SQL_WRITER", "mistral/big")
    monkeypatch.setenv("MODEL_FALLBACK", "groq/y")
    monkeypatch.setenv("LLM_TIMEOUT_S", "12")
    clients = ModelConfig.from_env().clients(min_interval_s=1.0)
    assert clients["sql_writer"].model == "mistral/big"
    assert clients["router"].model == "gemini/x"
    assert clients["router"] is clients["explainer"]  # same model, one shared client
    assert clients["router"].fallback_model == "groq/y"
    assert clients["router"].request_timeout_s == 12.0
    assert clients["router"].backoff_s == ()  # graph nodes retry, not the client
