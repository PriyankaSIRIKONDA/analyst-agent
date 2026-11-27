"""One place that decides which model each graph node uses.

Every model call goes through LiteLLM, so a model is one string in LiteLLM format
(``gemini/gemini-3.5-flash-lite``, ``mistral/mistral-small-latest``,
``groq/llama-3.3-70b-versatile``, later ``bedrock/...``). Changing provider is a change
to ``.env``, never to agent code.

Environment variables (all optional):

    MODEL               default model for every node
    MODEL_ROUTER        cheap model that classifies the question
    MODEL_SQL_WRITER    the model that writes SQL (the one worth paying for)
    MODEL_EXPLAINER     the model that turns rows into a short answer
    MODEL_FALLBACK      tried once when a node's model keeps failing (e.g. a Groq model)
    LLM_TIMEOUT_S       HTTP timeout of one model call, in seconds (default 30)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from analyst_agent.llm import DEFAULT_MODEL, LiteLLMClient

ROLES = ("router", "sql_writer", "explainer")
DEFAULT_LLM_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class ModelConfig:
    """Model names per role. Build with from_env() or by hand in tests and evals."""

    router: str
    sql_writer: str
    explainer: str
    fallback: str | None = None
    request_timeout_s: float = DEFAULT_LLM_TIMEOUT_S

    @classmethod
    def from_env(cls, default: str | None = None) -> ModelConfig:
        base = default or os.getenv("MODEL") or DEFAULT_MODEL
        return cls(
            router=os.getenv("MODEL_ROUTER") or base,
            sql_writer=os.getenv("MODEL_SQL_WRITER") or base,
            explainer=os.getenv("MODEL_EXPLAINER") or base,
            fallback=os.getenv("MODEL_FALLBACK") or None,
            request_timeout_s=float(os.getenv("LLM_TIMEOUT_S") or DEFAULT_LLM_TIMEOUT_S),
        )

    def clients(self, min_interval_s: float = 0.0) -> dict[str, LiteLLMClient]:
        """One LiteLLM client per role. No client-side backoff: graph nodes retry themselves.

        `min_interval_s` spaces calls (free-tier rate limits in eval runs); the roles that use
        the same model share one client, so the spacing holds across nodes.
        """
        shared: dict[str, LiteLLMClient] = {}
        for role in ROLES:
            model = getattr(self, role)
            if model not in shared:
                shared[model] = LiteLLMClient(
                    model=model,
                    fallback_model=self.fallback,
                    request_timeout_s=self.request_timeout_s,
                    backoff_s=(),
                    min_interval_s=min_interval_s,
                )
        return {role: shared[getattr(self, role)] for role in ROLES}

    def describe(self) -> dict:
        """Model names, for run files and the /ready endpoint."""
        return {role: getattr(self, role) for role in ROLES} | {"fallback": self.fallback}
