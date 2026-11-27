"""Tracing with Langfuse (self-hosted or cloud), switched on only when its keys are set.

Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST (for a self-hosted server,
for example http://localhost:3000). Then:

* every graph run becomes a Langfuse trace with one span per node (LangGraph callback), and
* every model call is logged by LiteLLM with its tokens, cost and latency (``langfuse_otel``).

Without the keys nothing is imported or sent anywhere.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("analyst_agent")


def langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def setup_tracing() -> list:
    """Turn tracing on if configured. Returns the LangGraph callbacks to pass on each run."""
    if not langfuse_enabled():
        return []
    try:
        import litellm
        from langfuse.langchain import CallbackHandler
    except ImportError:  # the package is optional at run time
        log.warning("Langfuse keys are set but the langfuse package is not installed.")
        return []
    if "langfuse_otel" not in (litellm.callbacks or []):
        litellm.callbacks = [*(litellm.callbacks or []), "langfuse_otel"]
    return [CallbackHandler()]
